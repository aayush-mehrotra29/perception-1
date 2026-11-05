import copy
import hashlib
import json
import numpy as np
import pytest
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.annotation_import import reconcile,crossed_folds
from lesson_harness_v1.labels import Label
from lesson_harness_v1.rubrics import RUBRICS
from lesson_harness_v1.training import Observation


def fixture():
    window={'source_sha256':'a'*64,'content_family':'physics','context_start':1.,'query_seconds':45.,'supported':True}
    packet={'source_sha256':'a'*64,'interval':{'start':1.,'stop':45.},'reference_learner':'assumed reference novice',
            'evidence':[{'id':'content','text':'an explanation'}]}
    record={'id':'packet1','person':'edugage:1','content_family':'physics','window_index':0,'packet':packet}
    response={q:{'distribution':[0,0,1,0,0],'evidence_ids':['content'],'reason':'content-based synthetic estimate'} for q in QUALITIES}
    responses=[{'id':'packet1','rubric':r,'packet_sha256':hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),
                'status':'valid','raw_response':json.dumps(response),'response':copy.deepcopy(response)} for r in RUBRICS]
    label=Label(id='human1',person='edugage:1',content_family='physics',source_sha256='a'*64,
        quality=QUALITIES[0],origin='human',scope='interval',start=1,stop=45,
        distribution=[0,1,0,0,0],evidence_ids=['human-probe'],provenance_sha256='b'*64)
    return [record],responses,[Observation(label=label,window_indices=[0])],[window]


def test_human_scope_preserved_and_synthetic_labels_keep_provenance():
    records,responses,human,windows=fixture()
    result,audit=reconcile(records,responses,human,windows,'c'*64)
    assert len(result)==5 and result[0]==human[0]
    assert {o.label.quality for o in result}==set(QUALITIES)
    assert all(o.label.origin=='synthetic' and o.label.provenance_sha256=='c'*64 for o in result[1:])
    assert audit[0]['reason']=='existing_human_label_same_scope'
    assert all(o.window_indices==[0] and o.label.stop==45 for o in result)


def test_disagreement_and_abstention_remain_missing_not_zero():
    records,responses,human,windows=fixture()
    a=responses[0]['response'];b=responses[1]['response']
    a[QUALITIES[1]]['distribution']=[1,0,0,0,0];b[QUALITIES[1]]['distribution']=[0,0,0,0,1]
    a[QUALITIES[2]]['distribution']=None
    responses[0]['raw_response']=json.dumps(a);responses[1]['raw_response']=json.dumps(b)
    result,audit=reconcile(records,responses,human,windows,'c'*64)
    assert {o.label.quality for o in result}=={QUALITIES[0],QUALITIES[3],QUALITIES[4]}
    assert audit[1]['reason']=='rubric_disagreement' and audit[2]['reason']=='insufficient_evidence'
    assert audit[1]['distribution'] is None


def test_missing_or_tampered_response_cannot_be_silently_accepted():
    records,responses,human,windows=fixture()
    with pytest.raises(ValueError,match='incomplete'):reconcile(records,responses[:1],human,windows,'c'*64)
    responses[0]['response'][QUALITIES[1]]['distribution']=[1,0,0,0,0]
    with pytest.raises(ValueError,match='differs'):reconcile(records,responses,human,windows,'c'*64)


def test_invalid_generation_is_audited_not_repaired():
    records,responses,human,windows=fixture()
    responses[0].update(status='invalid',raw_response='not json')
    result,audit=reconcile(records,responses,human,windows,'c'*64)
    assert result==human and all(r['reason']=='invalid_rubric_response' for r in audit)


def test_added_labels_keep_frozen_participant_and_family_splits():
    _,_,human,windows=fixture();base_label=human[0].label
    windows.append({**windows[0],'source_sha256':'d'*64,'content_family':'medicine'})
    observations=[]
    for person in [1,2]:
        for point,family in enumerate(['physics','medicine']):
            label=base_label.model_copy(update={'id':f'{person}-{point}','person':f'edugage:{person}',
                'content_family':family,'source_sha256':windows[point]['source_sha256']})
            observations.append(Observation(label=label,window_indices=[point]))
    folds=[{'name':f'{f}-{p}','held_family':f,'held_people':[p]} for f in ['physics','medicine'] for p in [1,2]]
    eeg={'person':np.array([1,1,2,2]),'family':np.array(['physics','medicine']*2),'point':np.array([0,1,0,1])}
    result=crossed_folds(folds,observations,windows,eeg)
    assert sorted(i for f in result for i in f['test_observations'])==list(range(4))
    for f in result:
        assert len(f['train_observations'])==len(f['test_observations'])==1
        assert len(f['train_eeg'])==1
        assert all(windows[i]['content_family']!=f['held_family'] for i in f['train_windows'])
