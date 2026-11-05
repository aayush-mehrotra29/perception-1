"""Bind screened voltage summaries to existing content rows without imputation.

Qualification here is an engineering screen, not validated brain-state supervision.
All original human observations and held-out identities remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .data import load_import
from .eeg_quality import POLICY
from .quality_corpus import screen_configuration


def select_screened(eeg, windows, qualified, target_channels=('TP9','AF7','AF8','TP10')):
    """Exact (person, lesson, query) join; remove singletons for every EEG arm."""
    if tuple(target_channels) not in [('TP9','AF7','AF8','TP10'),('AF7','AF8')]:raise ValueError('unknown target montage')
    width=2*len(target_channels);rows=[];targets=[];excluded=[]
    for i,(person,lesson,point) in enumerate(zip(eeg['person'],eeg['lesson'],eeg['point'])):
        window=windows[int(point)]
        if window['lesson']!=str(lesson):raise ValueError('content row identity mismatch')
        key=(int(person),str(lesson),window['query_seconds'])
        entry=qualified.get(key)
        if entry is None:
            excluded.append({'source_row':i,'reason':'no_quality_supported_exact_query'});continue
        if entry['media_sha256']!=window['source_sha256']:raise ValueError('quality/media identity mismatch')
        y=np.asarray(entry['target'],float)
        if y.shape!=(width,) or not np.isfinite(y).all() or (y[width//2:]<=0).any():
            raise ValueError('invalid screened voltage target')
        rows.append(i);targets.append(y)
    counts={}
    for i in rows:
        key=(int(eeg['person'][i]),str(eeg['lesson'][i]));counts[key]=counts.get(key,0)+1
    keep=[];kept_targets=[]
    for i,y in zip(rows,targets):
        if counts[(int(eeg['person'][i]),str(eeg['lesson'][i]))]<2:
            excluded.append({'source_row':i,'reason':'singleton_viewing_cannot_support_reassignment'})
        else:keep.append(i);kept_targets.append(y)
    selected={k:v[np.asarray(keep,dtype=int)].copy() for k,v in eeg.items()}
    selected['target']=np.asarray(kept_targets,float).reshape(-1,width)
    selected['original_row']=np.asarray(keep,dtype=int)
    return selected,sorted(excluded,key=lambda r:r['source_row'])


def rebuild_folds(folds, observations, windows, eeg):
    result=[]
    for old in folds:
        fold=dict(old)
        train=np.flatnonzero((eeg['family']!=fold['held_family'])&~np.isin(eeg['person'],fold['held_people']))
        fold['train_eeg']=train.tolist()
        fold['train_windows']=sorted({w for i in fold['train_observations'] for w in observations[i].window_indices}
                                     |set(eeg['point'][train].tolist()))
        if any(windows[i]['content_family']==fold['held_family'] for i in fold['train_windows']):
            raise ValueError('quality import leaks held-out family')
        result.append(fold)
    return result


def prepare(source, corpus, output):
    source,corpus,output=map(Path,(source,corpus,output));guard(output)
    if output.exists():raise ValueError('preserve existing import')
    manifest,blocks,observations,windows,folds,eeg=load_import(source)
    summary=json.loads((corpus/'summary.json').read_text())
    mode=summary.get('screen_mode','four_channel');_,policy,code=screen_configuration(mode)
    channels=['AF7','AF8'] if mode=='frontal' else ['TP9','AF7','AF8','TP10'];width=2*len(channels)
    policy_hash=file_sha256(policy);code_hash=file_sha256(code)
    if (not summary['complete'] or len(summary['viewings'])!=32 or summary['policy_sha256']!=policy_hash
            or summary['screen_code_sha256']!=code_hash):raise ValueError('complete source-pinned quality audit required')
    qualified={};pins=[]
    for record in summary['viewings']:
        if record['status']!='screened':continue
        path=Path(record['manifest'])
        if file_sha256(path)!=record['sha256']:raise ValueError('quality manifest changed')
        audit=json.loads(path.read_text())
        if (not audit['complete'] or audit['policy_sha256']!=policy_hash or audit['code_sha256']!=code_hash
                or audit['participant']!=record['participant'] or audit['lecture']!=record['lecture']):
            raise ValueError('quality audit identity mismatch')
        for name,digest in audit['files'].items():
            if (path.parent/name).resolve().parent!=path.parent.resolve() or file_sha256(path.parent/name)!=digest:
                raise ValueError('quality target changed')
        with np.load(path.parent/'screened-targets.npz',allow_pickle=False) as z:
            queries=z['query_seconds'];y=z['targets'];supported=z['supported']
            if supported.dtype!=bool or len(queries)!=len(supported) or y.shape!=(len(queries),width):
                raise ValueError('invalid quality array shapes')
            if int(supported.sum())!=audit['quality_supported']:raise ValueError('quality support count mismatch')
            for query,target in zip(queries[supported],y[supported]):
                key=(audit['participant'],audit['lecture'],float(query))
                if key in qualified:raise ValueError('duplicate quality-supported query')
                qualified[key]={'target':target,'media_sha256':audit['media_sha256']}
        pins.append({'path':str(path.resolve()),'sha256':record['sha256']})
    selected,excluded=select_screened(eeg,windows,qualified,channels)
    new_folds=rebuild_folds(folds,observations,windows,selected)
    output.mkdir(parents=True)
    for name in ['windows.json','observations.json']:
        (output/name).write_bytes((source/name).read_bytes())
    (output/'folds.json').write_text(json.dumps(new_folds,indent=2)+'\n')
    (output/'quality-exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    np.savez(output/'screened-eeg.npz',**selected)
    report={**manifest,'schema':'perception-lesson-quality-import-v1',
        'parent_import':str(source.resolve()),'parent_import_sha256':file_sha256(source/'manifest.json'),
        'eeg_override':'screened-eeg.npz','eeg_row_count':len(selected['point']),'EEG_target_channels':channels,
        'code_sha256':file_sha256(Path(__file__)),
        'EEG_quality_audit':{'screen_mode':mode,'summary_path':str((corpus/'summary.json').resolve()),
            'summary_sha256':file_sha256(corpus/'summary.json'),'policy_sha256':policy_hash,
            'screen_code_sha256':code_hash,'viewing_manifests':pins,
            'scope':'fixed contact/movement/range/flatline engineering screen',
            'physical_acquisition_timing_established':False,'human_state_validity_established':False,
            'artifact_free_claim':False},
        'EEG_arm_eligible':bool(len(selected['point']) and all(f['train_eeg'] for f in new_folds)),
        'target_definition':'quality-screened received-row channel means and log1p variances; no band or brain-state target',
        'scope':'exploratory supervised voltage-summary auxiliary; retains original human-label scopes'}
    report['files']={n:file_sha256(output/n) for n in ['windows.json','observations.json','folds.json','quality-exclusions.json','screened-eeg.npz']}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def validate_qualification(manifest,eeg):
    """A caller-supplied boolean cannot qualify EEG; require pinned audit receipts."""
    audit=manifest.get('EEG_quality_audit')
    if not audit or not manifest.get('EEG_arm_eligible'):
        raise ValueError('EEG signal quality is not qualified for every training fold')
    path=Path(audit['summary_path'])
    if file_sha256(path)!=audit['summary_sha256']:raise ValueError('EEG quality summary changed')
    summary=json.loads(path.read_text())
    mode=audit.get('screen_mode','four_channel');_,policy,code=screen_configuration(mode)
    if (not summary['complete'] or summary.get('screen_mode','four_channel')!=mode or audit['policy_sha256']!=file_sha256(policy)
            or summary['policy_sha256']!=audit['policy_sha256']
            or summary['screen_code_sha256']!=audit['screen_code_sha256']
            or audit['screen_code_sha256']!=file_sha256(code)):
        raise ValueError('EEG quality policy identity changed')
    if not len(eeg['point']) or not np.isfinite(eeg['target']).all():raise ValueError('no finite qualified EEG')
    for pin in audit['viewing_manifests']:
        if file_sha256(Path(pin['path']))!=pin['sha256']:raise ValueError('EEG quality receipt changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('corpus',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();r=prepare(a.source,a.corpus,a.output)
    print(json.dumps({k:r[k] for k in ['eeg_row_count','EEG_arm_eligible']}),flush=True)
