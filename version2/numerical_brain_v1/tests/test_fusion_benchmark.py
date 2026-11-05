import json
import numpy as np
from numerical_brain_v1 import fusion_benchmark as runner
from numerical_brain_v1.fusion_inference import preflight
from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation


def test_matched_rows_all_seeds_then_separate_portable_deployment(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'manifest.json').write_text('{"fixture":true}')
    (source/'label-audit.json').write_text('{"fixture":true}')
    sources=tmp_path/'sources.json';sources.write_text('{}')
    windows=[{'source_sha256':letter*64,'content_family':letter,'lesson':letter,'context_start':0.,'query_seconds':44.,'supported':True}
             for letter in ('a','b')]
    obs=[]
    for person in (1,2):
        for i,w in enumerate(windows):
            obs.append(Observation(label=Label(id=f'{person}-{i}',person=f'p{person}',content_family=w['content_family'],
                source_sha256=w['source_sha256'],quality='attention_difficulty',origin='human',scope='interval',start=0,stop=44,
                distribution=[0,1,0,0,0],evidence_ids=['fixture'],provenance_sha256='c'*64),window_indices=[i]))
    folds=[]
    for i,o in enumerate(obs):
        train=[j for j,v in enumerate(obs) if v.label.person!=o.label.person and v.label.content_family!=o.label.content_family]
        folds.append({'name':str(i),'train_observations':train,'test_observations':[i]})
    # Two copies per family are not needed for the orchestration fixture; train
    # transforms need2distinct rows, so add an unlabeled shared training point
    # through an additional observation in each family/person combination.
    windows=[{**w,'context_start':k,'query_seconds':44+k} for w in windows for k in (0,1)]
    obs=[]
    for person in (1,2):
        for i,w in enumerate(windows):
            obs.append(Observation(label=Label(id=f'{person}-{i}',person=f'p{person}',content_family=w['content_family'],
                source_sha256=w['source_sha256'],quality='attention_difficulty',origin='human',scope='interval',start=w['context_start'],stop=w['query_seconds'],
                distribution=[0,1,0,0,0],evidence_ids=['fixture'],provenance_sha256='c'*64),window_indices=[i]))
    folds=[]
    for family in ('a','b'):
        for person in ('p1','p2'):
            folds.append({'name':family+person,'train_observations':[i for i,o in enumerate(obs) if o.label.person!=person and o.label.content_family!=family],
                'test_observations':[i for i,o in enumerate(obs) if o.label.person==person and o.label.content_family==family]})
    blocks={'text':np.arange(12.).reshape(4,3)}
    monkeypatch.setattr(runner,'guard',lambda p:None)
    monkeypatch.setattr(runner,'load_import',lambda p:({'schema':'perception-lesson-labeled-import-v1'},blocks.copy(),obs,windows,folds,{}))
    monkeypatch.setattr(runner,'assemble',lambda *a:({'predicted_eeg':np.ones((4,120)),'predicted_cortex':np.ones((4,32))}, {'channel_order':list(map(str,range(40))),'source_pins':{}}))
    real_train=runner.train;calls=[]
    def mock_train(x,observations,windows,out,*,seed,arm):out.mkdir(parents=True);calls.append((seed,arm))
    monkeypatch.setattr(runner,'train',mock_train);monkeypatch.setattr(runner,'load_network',lambda p:None)
    monkeypatch.setattr(runner,'observed',lambda n,x,obs:np.full((len(obs),5),.2))
    result=runner.fit(source,sources,tmp_path/'benchmark')
    assert result['complete'] and len(calls)==4*4*3
    assert not (tmp_path/'benchmark/benchmark.json').exists()
    report=runner.score(source,tmp_path/'benchmark')
    assert report['endpoints']['attention_difficulty']['human']['count']==8
    assert set(report['endpoints']['attention_difficulty']['human']['scores'])==set(runner.ARMS)
    monkeypatch.setattr(runner,'train',real_train)
    deployed=runner.deploy(source,tmp_path/'benchmark',tmp_path/'deployment')
    assert deployed['input_width']==204
    assert preflight(tmp_path/'deployment')['complete']
