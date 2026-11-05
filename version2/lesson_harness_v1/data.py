"""Read-only audited EduGage import. No held-out dataset acquisition or imputation."""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.eeg import outer_scopes
from brain.fusion_data import checked
from brain.worker_contract import file_sha256
from .labels import Label,human_attention,assert_disjoint
from .training import Observation,validate_observations

RENAMES={'dense_text':'text','dense_audio':'audio','dense_video':'video','predicted_tribe':'tribe'}


def load_arrays(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    result={}
    for name in ['features','attention','eeg']:
        guard(directory)
        with np.load(checked(directory,name+'.npz',manifest['files'][name+'.npz']),allow_pickle=False) as data:
            result[name]={k:data[k] for k in data.files}
    return manifest,result


def prepare(prepared,output):
    prepared,output=Path(prepared),Path(output);guard(output)
    manifest,data=load_arrays(prepared)
    if manifest['schema']!='perception-behavior-prepared-v2':raise ValueError('audited behavior cache required')
    audit=Path(manifest['audit_path'])/'audit.json'
    if file_sha256(audit)!=manifest['audit_sha256']:raise ValueError('behavior audit identity changed')
    if json.loads(audit.read_text())['status']!='passed_for_offline_exploratory_fitting':
        raise ValueError('behavior audit not qualified')
    sources={};family={}
    for lesson,record in manifest['feature_sources'].items():
        path=Path(record['manifest'])
        if file_sha256(path)!=record['sha256']:raise ValueError('feature source identity changed')
        source=json.loads(path.read_text());sources[lesson]=source
        families=set(data['attention']['family'][data['attention']['lesson']==lesson])
        if len(families)!=1:raise ValueError('ambiguous content family')
        family[lesson]=str(next(iter(families)))
        for pin in source['source_manifests'].values():
            if file_sha256(Path(pin['path']))!=pin['sha256']:raise ValueError('upstream feature identity changed')
    features=data['features'];blocks={RENAMES.get(k,k):v for k,v in features.items() if k not in ['lesson','query']}
    windows=[]
    for index,(lesson,query) in enumerate(zip(features['lesson'],features['query'])):
        source=sources[str(lesson)]
        windows.append({'index':index,'lesson':str(lesson),'content_family':family[str(lesson)],
            'source_sha256':source['source_sha256'],'query_seconds':float(query),'context_start':float(query)-44,
            'supported':bool(query>=44 and all(np.isfinite(v[index]).all() for v in blocks.values()))})
    attention=data['attention'];observations=[]
    for row in range(len(attention['y'])):
        i=int(attention['point'][row]);window=windows[i]
        if window['query_seconds']!=attention['query'][row] or window['lesson']!=attention['lesson'][row]:
            raise ValueError('original attention context identity changed')
        label=Label(id=str(attention['id'][row]),person='edugage:'+str(attention['person'][row]),
            content_family=window['content_family'],source_sha256=window['source_sha256'],
            quality='attention_difficulty',origin='human',scope='interval',start=window['context_start'],stop=window['query_seconds'],
            distribution=human_attention(float(attention['y'][row])),evidence_ids=[str(attention['id'][row])],
            provenance_sha256=manifest['files']['attention.npz'])
        observations.append(Observation(label=label,window_indices=[i]))
    validate_observations(observations,windows)
    eeg=data['eeg']
    if eeg['target'].shape!=(len(eeg['point']),8) or not np.isfinite(eeg['target']).all():raise ValueError('invalid EEG targets')
    for i,lesson in zip(eeg['point'],eeg['lesson']):
        if windows[int(i)]['lesson']!=lesson or not windows[int(i)]['supported']:raise ValueError('EEG context mismatch')
    folds=[]
    for scope in outer_scopes(attention['family'],attention['person']):
        train,test=scope['train'],scope['test']
        assert_disjoint([observations[i].label for i in train],[observations[i].label for i in test])
        held_people=list(range(4*(scope['held_group']-1)+1,4*scope['held_group']+1))
        eeg_train=np.flatnonzero((eeg['family']!=scope['held_family'])&~np.isin(eeg['person'],held_people))
        train_windows=sorted({i for r in train for i in observations[r].window_indices}|set(eeg['point'][eeg_train].tolist()))
        if {windows[i]['content_family'] for i in train_windows}&{observations[i].label.content_family for i in test}:
            raise ValueError('EEG/content transform scope leaks held family')
        folds.append({'name':scope['held_family']+'-group'+str(scope['held_group']),
            'train_observations':train.tolist(),'test_observations':test.tolist(),'train_eeg':eeg_train.tolist(),
            'train_windows':train_windows,'held_family':scope['held_family'],'held_people':held_people})
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('windows.json',windows),('observations.json',[o.model_dump() for o in observations]),('folds.json',folds)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    report={'schema':'perception-lesson-edugage-import-v1','complete':True,'upstream':str(prepared.resolve()),
        'upstream_manifest_sha256':file_sha256(prepared/'manifest.json'),'code_sha256':file_sha256(Path(__file__)),
        'files':{name:file_sha256(output/name) for name in ['windows.json','observations.json','folds.json']},
        'window_count':len(windows),'human_label_count':len(observations),'eeg_row_count':len(eeg['point']),
        'fold_count':len(folds),'synthetic_labels_included':False,'previously_evaluated_content':True,
        'clock':'audited receiver-to-reported-position operator; not physical EEG synchronization',
        'historical_media_identity_established':False,'eeg_artifact_free_claim':False,
        'training_features':'original authenticated exact-query pooled features; one-second equivalence audit still required',
        'human_label_resolution':'sporadic responses, each retaining its audited preceding 44-second context',
        'scope':'preliminary human-attention training inputs; not completed five-quality corpus'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['window_count','human_label_count','eeg_row_count','fold_count']}),flush=True)
    return report


def load_import(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():checked(root,name,digest)
    upstream=Path(manifest['upstream'])
    if file_sha256(upstream/'manifest.json')!=manifest['upstream_manifest_sha256']:raise ValueError('upstream corpus identity changed')
    _,arrays=load_arrays(upstream)
    features={RENAMES.get(k,k):v for k,v in arrays['features'].items() if k not in ['lesson','query']}
    eeg=arrays['eeg']
    if manifest.get('eeg_override'):
        name=manifest['eeg_override']
        if name!='screened-eeg.npz' or name not in manifest['files']:
            raise ValueError('unsupported EEG override')
        with np.load(checked(root,name,manifest['files'][name]),allow_pickle=False) as archive:
            eeg={k:archive[k] for k in archive.files}
    return manifest,features,[Observation.model_validate(o) for o in json.loads((root/'observations.json').read_text())],json.loads((root/'windows.json').read_text()),json.loads((root/'folds.json').read_text()),eeg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();prepare(a.prepared,a.output)
