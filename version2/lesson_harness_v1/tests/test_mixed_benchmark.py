"""Orchestration fixtures only: mocked networks are not model-performance evidence."""
import json
import numpy as np
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.annotation_import import crossed_folds
from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation
from lesson_harness_v1 import mixed_benchmark as runner


def test_all_arms_use_common_rows_and_scoring_keeps_missing_heads(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    (source/'manifest.json').write_text('{"software_fixture":true}')
    (source/'label-audit.json').write_text('{"complete":true,"software_fixture":true}')
    windows=[]
    for family,source_sha in [('a','a'*64),('b','b'*64)]:
        for query in [45.,50.]:
            windows.append({'content_family':family,'source_sha256':source_sha,'context_start':query-44,
                'query_seconds':query,'supported':True,'lesson':family})
    observations=[]
    for person in [1,2]:
        for base in [0,2]:
            for extra in ([0,1] if person==2 else [0]):
                point=base+extra;w=windows[point]
                label=Label(id=f'{person}-{point}',person=f'edugage:{person}',content_family=w['content_family'],
                    source_sha256=w['source_sha256'],quality=QUALITIES[extra],origin='synthetic' if extra else 'human',
                    scope='interval',start=w['context_start'],stop=w['query_seconds'],distribution=[1,0,0,0,0],
                    evidence_ids=['software-fixture'],provenance_sha256='c'*64)
                observations.append(Observation(label=label,window_indices=[point]))
    eeg={'person':np.repeat([1,2],4),'family':np.array(['a','a','b','b']*2),
         'lesson':np.array(['a','a','b','b']*2),'point':np.tile(np.arange(4),2),
         'target':np.arange(64,dtype=float).reshape(8,8)}
    base=[{'name':f'{f}-{p}','held_family':f,'held_people':[p]} for f in ['a','b'] for p in [1,2]]
    folds=crossed_folds(base,observations,windows,eeg)
    blocks={'text':np.arange(12,dtype=float).reshape(4,3),'tribe':np.arange(8,dtype=float).reshape(4,2)}
    monkeypatch.setattr(runner,'guard',lambda p:None)
    monkeypatch.setattr(runner,'validate_qualification',lambda *a:None)
    monkeypatch.setattr(runner,'load_import',lambda p:({'schema':'perception-lesson-labeled-import-v1'},blocks,observations,windows,folds,eeg))
    calls=[]
    def artificial_fit(x,train,win,output,*,seed,arm,eeg=None):
        output.mkdir(parents=True)
        calls.append((arm,seed,[o.label.id for o in train]))
        if arm=='training_eeg_reassigned':
            assert (eeg['source_indices']!=eeg['reassigned_source_indices']).all()
    monkeypatch.setattr(runner,'fit_network',artificial_fit)
    monkeypatch.setattr(runner,'load_network',lambda p:None)
    monkeypatch.setattr(runner,'observation_predictions',lambda n,x,obs:np.full((len(obs),5),.2))
    out=tmp_path/'software-fixture-result'
    result=runner.fit(source,out)
    assert result['complete'] and len(calls)==4*4*3
    assert not (out/'benchmark.json').exists()
    report=runner.score(source,out)
    assert report['complete'] and not report['five_heads_fitted_in_every_fold']
    assert len(report['exclusions'])==2
    assert report['endpoints'][QUALITIES[1]]['synthetic']['status']=='not_evaluated'
    assert report['endpoints'][QUALITIES[0]]['human']['evaluated_labels']==4
    assert not report['measured_fmri_superiority_tested']
