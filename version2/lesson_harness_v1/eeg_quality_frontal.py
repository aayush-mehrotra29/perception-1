"""Separate AF7/AF8 experiment after the all-four-channel screen failed.

Original eight-target aggregation is verified before selecting frontal channels.
Thresholds are unchanged; source/channel choice is explicitly a new experiment.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard,select_groups
from brain.worker_contract import file_sha256

POLICY=Path(__file__).with_name('eeg_quality_frontal_policy.json')


def contact_support(times,contact_times,codes,maximum_gap=500):
    times=np.asarray(times);ct=np.asarray(contact_times);codes=np.asarray(codes)
    if ct.ndim!=1 or len(ct)<2 or codes.shape!=(len(ct),2) or np.any(np.diff(ct)<=0):
        raise ValueError('unique ordered frontal-channel contact packets required')
    if not np.isin(codes,[1,2,4]).all():raise ValueError('unverified contact code')
    # Include exact packets only if a neighboring valid interval supports them.
    right=np.searchsorted(ct,times,side='right');left=right-1
    eligible=(left>=0)&(right<len(ct));out=np.zeros(len(times),bool)
    rows=np.flatnonzero(eligible);a=left[rows];b=right[rows]
    out[rows]=((ct[b]-ct[a])<=maximum_gap)&(codes[a]==1).all(1)&(codes[b]==1).all(1)
    return out


def event_mask(times,events,padding):
    times=np.asarray(times);events=np.unique(events)
    if not len(events):return np.zeros(len(times),bool)
    return np.searchsorted(events,times+padding,side='right')>np.searchsorted(events,times-padding,side='left')


def flatline_mask(values,original_rows,minimum):
    values=np.asarray(values);original_rows=np.asarray(original_rows)
    if values.ndim!=2 or values.shape[1]!=2 or len(values)!=len(original_rows) or minimum<2:
        raise ValueError('aligned frontal-channel rows required')
    out=np.zeros(len(values),bool)
    for channel in range(2):
        same=(np.diff(original_rows)==1)&(values[1:,channel]==values[:-1,channel])
        bounds=np.r_[0,np.flatnonzero(~same)+1,len(values)]
        for a,b in zip(bounds[:-1],bounds[1:]):
            if b-a>=minimum:out[a:b]=True
    return out


def read_markers(path):
    contacts={};events={'blink':[],'jaw_clench':[]};previous=None
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            time=int(row['timestamp']);kind=row['marker_type']
            if previous is not None and time<previous:raise ValueError('marker clock moved backwards')
            previous=time
            if kind.startswith('horseshoe_'):
                code=ast.literal_eval(kind[len('horseshoe_'):])
                if len(code)!=4 or not np.isin(code,[1,2,4]).all():raise ValueError('unrecognized contact packet')
                # Conflicting simultaneous packets keep the worse state.
                contacts[time]=np.maximum(contacts.get(time,[1,1,1,1]),code)
            elif kind in events:events[kind].append(time)
            else:raise ValueError('unrecognized marker; explicit audit required')
    times=np.array(sorted(contacts),np.int64)
    return times,np.array([contacts[t] for t in times]),events


def screen(metadata,original_targets,output):
    metadata,original_targets,output=map(Path,(metadata,original_targets,output));guard(output)
    if output.exists():raise ValueError('preserve existing quality audit')
    policy=json.loads(POLICY.read_text());meta=json.loads(metadata.read_text())
    original=json.loads((original_targets/'report.json').read_text())
    if original['metadata_sha256']!=file_sha256(metadata) or not original['source_hashes_checked']:
        raise ValueError('original target metadata identity mismatch')
    pins=meta['input_pins'];raw=Path(pins['eeg']['path']);playback=Path(pins['playback']['path'])
    marker=raw.with_name(raw.name.replace('_EEG.csv','_MARKERS.csv'))
    if not marker.is_file():raise ValueError('no matching received marker stream')
    for key,path in [('eeg',raw),('playback',playback)]:
        if file_sha256(path)!=pins[key]['sha256']:raise ValueError('raw source changed')
    marker_sha=file_sha256(marker)
    keys=['eeg_original_row_indices','eeg_receiver_milliseconds','row_to_group','offset_seconds','query_media_seconds',
          'group_left_source_row','group_right_source_row','group_declared_run_id','group_host_interval_milliseconds']
    with np.load(metadata.with_suffix('.npz'),allow_pickle=False) as archive:arrays={k:archive[k] for k in keys}
    for k,v in arrays.items():
        expected=meta['arrays'][k]
        if list(v.shape)!=expected['shape'] or str(v.dtype)!=expected['dtype'] or hashlib.sha256(v.tobytes()).hexdigest()!=expected['Cbytes_sha256']:
            raise ValueError('observation array identity changed: '+k)
    rows=arrays['eeg_original_row_indices'];times=arrays['eeg_receiver_milliseconds']
    if not len(rows) or np.any(np.diff(rows)<=0) or np.any(np.diff(times)<0):raise ValueError('ordered received rows required')
    values=np.empty((len(rows),4));cursor=0
    with raw.open() as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            if cursor==len(rows):break
            if i==rows[cursor]:
                if int(row['timestamp'])!=times[cursor]:raise ValueError('EEG receiver clock changed')
                values[cursor]=[float(row[c]) for c in policy['source_channels']];cursor+=1
            if i%100000==0:guard(output)
    if cursor!=len(rows):raise ValueError('raw source truncated')
    ct,codes,events=read_markers(marker)
    frontal=values[:,[1,2]]
    contact=contact_support(times,ct,codes[:,[1,2]],policy['maximum_contact_packet_gap_ms'])
    movement={kind:event_mask(times,items,policy['movement_padding_ms'][kind]) for kind,items in events.items()}
    flat=flatline_mask(frontal,rows,policy['flatline_rows'])
    low,high=policy['documented_voltage_range_uv'];range_ok=np.isfinite(frontal).all(1)&(frontal>low).all(1)&(frontal<high).all(1)
    good=contact&range_ok&~flat&~movement['blink']&~movement['jaw_clench']
    with playback.open() as stream:right_clocks=np.array([float(r['video_timestamp_sec']) if r['video_timestamp_sec'] else np.nan for r in csv.DictReader(stream)])
    q=arrays['query_media_seconds'];right_q=right_clocks[arrays['group_right_source_row']]
    primary=np.flatnonzero(arrays['offset_seconds']==0)
    if len(primary)!=1:raise ValueError('unique zero-offset observation required')
    assignment=arrays['row_to_group'][primary[0]]
    target_path=original_targets/'targets.npz'
    if file_sha256(target_path)!=original['targets_sha256']:raise ValueError('original target changed')
    with np.load(target_path,allow_pickle=False) as data:
        queries=data['query_seconds'];supported=data['supported'];old=data['targets'];counts=data['sample_counts']
    cleaned=np.full((len(old),4),np.nan);eligible=np.zeros(len(queries),bool);quality=[]
    for i,query in enumerate(queries):
        if not supported[i]:quality.append({'query':float(query),'eligible':False,'reason':'original_context_unsupported'});continue
        groups,reason=select_groups(q,right_q,arrays['group_left_source_row'],arrays['group_right_source_row'],
            arrays['group_declared_run_id'],arrays['group_host_interval_milliseconds'],query)
        if groups is None:raise ValueError('formerly supported context no longer supported')
        ix=np.flatnonzero(np.isin(assignment,groups));sample=values[ix]
        if len(ix)!=counts[i]:raise ValueError('original received sample count mismatch')
        np.testing.assert_allclose(np.r_[sample.mean(0),np.log1p(sample.var(0))],old[i],rtol=1e-10,atol=1e-10)
        sample=sample[:,[1,2]]
        keep=good[ix];fraction=float(keep.mean());reason=None
        if fraction<1-policy['maximum_rejected_sample_fraction']:reason='insufficient_quality_supported_rows'
        elif keep.sum()<policy['minimum_retained_rows']:reason='too_few_retained_rows'
        elif (sample[keep].var(0)<=0).any():reason='constant_retained_channel'
        else:cleaned[i]=np.r_[sample[keep].mean(0),np.log1p(sample[keep].var(0))];eligible[i]=True
        quality.append({'query':float(query),'eligible':bool(eligible[i]),'reason':reason,'received_rows':len(ix),
            'retained_rows':int(keep.sum()),'retained_fraction':fraction,
            'contact_supported_fraction':float(contact[ix].mean()),'range_supported_fraction':float(range_ok[ix].mean()),
            'blink_flagged_fraction':float(movement['blink'][ix].mean()),'jaw_flagged_fraction':float(movement['jaw_clench'][ix].mean()),
            'flatline_flagged_fraction':float(flat[ix].mean())})
    guard(output)
    if file_sha256(marker)!=marker_sha or file_sha256(raw)!=pins['eeg']['sha256']:raise ValueError('source changed during screen')
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'screened-targets.npz',query_seconds=queries,targets=cleaned,supported=eligible)
    (output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    report={'schema':'perception-received-EEG-frontal-quality-screen-v1','screen_mode':'frontal','target_channels':['AF7','AF8'],
        'complete':True,'participant':meta['public_participant_id'],
        'lecture':meta['video_uid'],'media_sha256':meta['media_sha256'],'original_supported':int(supported.sum()),
        'quality_supported':int(eligible.sum()),'policy_sha256':file_sha256(POLICY),'code_sha256':file_sha256(Path(__file__)),
        'metadata_sha256':file_sha256(metadata),'original_report_sha256':file_sha256(original_targets/'report.json'),
        'marker':{'path':str(marker),'sha256':marker_sha},'EEG_source_sha256':pins['eeg']['sha256'],
        'files':{n:file_sha256(output/n) for n in ['screened-targets.npz','quality.json']},
        'qualification_scope':'specified contact/movement/range/flatline engineering screen only',
        'physical_acquisition_timing_established':False,'artifact_free_claim':False,'human_state_validity_established':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['participant','lecture','original_supported','quality_supported']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metadata',type=Path);p.add_argument('original_targets',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();screen(a.metadata,a.original_targets,a.output)
