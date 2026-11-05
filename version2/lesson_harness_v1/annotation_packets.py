"""Local-only evidence packets, selected without consulting behavioral ratings."""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.auxiliary_windows import authenticated_child
from brain.edugage_targets import guard
from brain.meta_features import ContentSpan
from brain.worker_contract import file_sha256
from .quality_import import validate_qualification
from .rubrics import prompt


def spaced_indices(queries, maximum=8):
    q=np.asarray(queries,float)
    if q.ndim!=1 or not np.isfinite(q).all() or np.any(np.diff(q)<=0) or maximum<1:
        raise ValueError('unique chronologically ordered queries required')
    return np.unique(np.linspace(0,len(q)-1,min(maximum,len(q))).astype(int)) if len(q) else np.empty(0,int)


def contained_spans(spans,start,stop):
    """Never invent word timing or expose a transcript bin beyond the context."""
    return [s for s in spans if s['start']>=start and s['stop']<=stop and s['text'].strip()]


def transcript_spans(path,digest,source_sha256):
    path=Path(path)
    if file_sha256(path)!=digest:raise ValueError('semantic manifest changed')
    manifest=json.loads(path.read_text())
    if manifest['source']['sha256']!=source_sha256:raise ValueError('semantic media mismatch')
    spans=[];seen=set()
    for record in manifest['batches']:
        child=authenticated_child(path.parent,record['manifest'],record['sha256'])
        batch=json.loads(child.read_text())
        if batch['source_sha256']!=source_sha256:raise ValueError('transcript batch media mismatch')
        for value in batch['spans']:
            span=ContentSpan.model_validate(value)
            if span.id in seen:raise ValueError('duplicate transcript span')
            seen.add(span.id);spans.append(span.model_dump())
    return sorted(spans,key=lambda s:(s['start'],s['stop']))


def make_packet(window,target,spans,channels=('TP9','AF7','AF8','TP10')):
    start,stop=window['context_start'],window['query_seconds']
    evidence=[{'id':s['id'],'kind':'transcript','start':s['start'],'stop':s['stop'],'text':s['text'],
               'timing':s.get('timing','original coarse transcript interval wholly inside context; no word timings')}
              for s in contained_spans(spans,start,stop)]
    evidence.append({'id':'coverage','kind':'limitations',
        'text':'Only wholly contained transcript bins are supplied. Missing text is unknown, not evidence of silence. No visual pixels or independent cognitive-state labels are supplied. Abstain when this evidence is insufficient.'})
    target=np.asarray(target,float)
    if tuple(channels) not in [('TP9','AF7','AF8','TP10'),('AF7','AF8')]:raise ValueError('unknown EEG target montage')
    count=len(channels)
    if target.shape!=(count*2,) or not np.isfinite(target).all():raise ValueError('finite screened summaries must match montage')
    evidence.append({'id':'screened-eeg','kind':'exploratory_voltage_summary',
        'channel_order':list(channels),'mean_received_microvolts':target[:count].tolist(),
        'log1p_received_variance_microvolts_squared':target[count:].tolist(),
        'qualification':'fixed contact, movement, range and flatline screen; not artifact-free certification',
        'clock':'host-receipt and reported playback mapping; hardware synchronization unverified',
        'interpretation':'Voltage offsets and variance do not uniquely identify attention, engagement, effort, confusion or comprehension. Do not infer those states from these numbers alone.'})
    packet={'source_sha256':window['source_sha256'],'interval':{'start':start,'stop':stop},
        'reference_learner':'Reference adult novice with introductory high-school algebra and science prerequisites. This is an explicit assumption, not a measured participant profile.',
        'evidence':evidence}
    prompt('evidence_first_v1',packet)
    return packet


def prepare(source,output,word_cache=None):
    source,output=map(Path,(source,output));guard(output)
    if output.exists():raise ValueError('preserve prior annotation packet set')
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['files'].items():authenticated_child(source,name,digest)
    with np.load(source/'screened-eeg.npz',allow_pickle=False) as z:eeg={k:z[k] for k in z.files}
    validate_qualification(manifest,eeg)
    windows=json.loads((source/'windows.json').read_text())
    upstream=Path(manifest['upstream'])/'manifest.json'
    if file_sha256(upstream)!=manifest['upstream_manifest_sha256']:raise ValueError('content manifest changed')
    features=json.loads(upstream.read_text())['feature_sources'];spans={};pins={}
    for lesson in np.unique(eeg['lesson']):
        record=features[str(lesson)];p=Path(record['manifest'])
        if file_sha256(p)!=record['sha256']:raise ValueError('feature source changed')
        feature=json.loads(p.read_text());semantic=feature['source_manifests']['semantics']
        if word_cache:
            from .word_transcripts import load_words
            spans[str(lesson)],word_pin=load_words(word_cache,str(lesson),p,record['sha256'])
            pins[str(lesson)]={'feature':record,'word_cache':word_pin}
        else:
            spans[str(lesson)]=transcript_spans(semantic['path'],semantic['sha256'],feature['source_sha256'])
            pins[str(lesson)]={'feature':record,'semantics':semantic}
    records=[]
    for person,lesson in sorted(set(zip(eeg['person'].tolist(),eeg['lesson'].tolist()))):
        rows=np.flatnonzero((eeg['person']==person)&(eeg['lesson']==lesson))
        rows=sorted(rows,key=lambda i:windows[int(eeg['point'][i])]['query_seconds'])
        queries=[windows[int(eeg['point'][i])]['query_seconds'] for i in rows]
        for selection in spaced_indices(queries):
            row=rows[int(selection)];point=int(eeg['point'][row]);window=windows[point]
            content=spans[lesson]
            if word_cache:
                from .word_transcripts import pack_contained_words
                content=pack_contained_words(content,window['context_start'],window['query_seconds'])
            packet=make_packet(window,eeg['target'][row],content,manifest.get('EEG_target_channels',('TP9','AF7','AF8','TP10')))
            if word_cache:
                packet['evidence'][len(content)]['text']='Only cached ASR words wholly inside this context are supplied. ASR may contain errors. Missing words are unknown, not silence. No visual pixels or measured psychological states are supplied. Abstain when insufficient.'
            records.append({'id':f'edugage-{person:02d}-{lesson}-q{window["query_seconds"]:g}',
                'person':f'edugage:{person}','content_family':window['content_family'],'lesson':lesson,
                'window_index':point,'screened_eeg_row':int(row),'packet':packet})
    if not records:raise ValueError('no eligible EEG-supported annotation packets')
    output.mkdir(parents=True)
    (output/'packets.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    report={'schema':'perception-local-annotation-packets-v1','complete':True,'source':str(source.resolve()),
        'source_manifest_sha256':file_sha256(source/'manifest.json'),'packets_sha256':file_sha256(output/'packets.json'),
        'code_sha256':file_sha256(Path(__file__)),'content_sources':pins,'packet_count':len(records),
        'human_ratings_read':False,'behavioral_labels_in_prompt':False,'sampling':'at most eight evenly spaced chronological row positions per viewing',
        'transcript_policy':('only whole cached ASR word intervals inside preceding 44 seconds; no sentence-field leakage' if word_cache else 'only whole original bins inside the preceding 44 seconds; no future text or invented word clocks'),
        'participant_data_sent_to_cloud':False,'teacher_output_is_ground_truth':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--word-cache',type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output,a.word_cache)),flush=True)
