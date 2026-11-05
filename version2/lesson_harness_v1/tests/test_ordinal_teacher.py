import copy
import math
import hashlib
import json
import pytest
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.ordinal_teacher import LETTERS,response_from_scores,messages


def fixture():
    scores={q:{letter:[{'token_id':i*2+j,'text':text,'log_probability':math.log(p/2)}
            for j,text in enumerate((letter,' '+letter))]
            for i,(letter,p) in enumerate(zip(LETTERS,[.04,.08,.4,.08,.04,.01]))} for q in QUALITIES}
    packet={'source_sha256':'a'*64,'interval':{'start':1,'stop':45},'reference_learner':'novice',
            'evidence':[{'id':'words','kind':'transcript','text':'An explanation.'}]}
    return scores,packet


def test_conditional_distribution_comes_from_raw_likelihoods():
    scores,packet=fixture();result=response_from_scores(scores,packet)
    assert result[QUALITIES[0]]['distribution']==pytest.approx([.0625,.125,.625,.125,.0625])
    assert result[QUALITIES[0]]['evidence_ids']==['words']
    assert 'uncalibrated' in result[QUALITIES[0]]['reason'].lower()


def test_missing_content_abstention_and_low_mass_are_not_repaired():
    scores,packet=fixture()
    packet['evidence'][0]['kind']='exploratory_voltage_summary'
    assert all(v['distribution'] is None for v in response_from_scores(scores,packet).values())
    scores,packet=fixture()
    for raw in scores.values():
        for rows in raw.values():
            for row in rows:row['log_probability']-=10
    assert all(v['distribution'] is None for v in response_from_scores(scores,packet).values())
    scores,packet=fixture()
    for raw in scores.values():
        for row in raw['X']:row['log_probability']=math.log(.3)
        for rows in [raw[k] for k in LETTERS[:-1]]:
            for row in rows:row['log_probability']-=3
    assert all(v['distribution'] is None for v in response_from_scores(scores,packet).values())


def test_invalid_probabilities_and_duplicate_tokens_rejected():
    scores,packet=fixture();scores[QUALITIES[0]]['A'][0]['log_probability']=float('nan')
    with pytest.raises(ValueError):response_from_scores(scores,packet)
    scores,packet=fixture();scores[QUALITIES[0]]['A'][0]['token_id']=2
    with pytest.raises(ValueError,match='overlapping'):response_from_scores(scores,packet)


def test_two_rubrics_share_definitions_but_have_fixed_different_directions():
    _,packet=fixture()
    a=messages('evidence_first_v1',QUALITIES[0],packet)
    b=messages('counterevidence_first_v1',QUALITIES[0],packet)
    assert a[0]==b[0] and a[1]!=b[1]
    assert 'X=insufficient evidence' in a[1]['content']


def test_import_recomputes_scores_and_checks_actual_prompt_hashes():
    from lesson_harness_v1.annotation_import import reconcile
    scores,packet=fixture()
    record={'id':'p','person':'edugage:1','content_family':'physics','window_index':0,'packet':packet}
    window={'source_sha256':'a'*64,'content_family':'physics','context_start':1.,'query_seconds':45.,'supported':True}
    responses=[]
    for rubric in ['evidence_first_v1','counterevidence_first_v1']:
        responses.append({'id':'p','rubric':rubric,'status':'valid','method':'ordinal-next-token-v1',
            'packet_sha256':hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),
            'raw_scores':copy.deepcopy(scores),'response':response_from_scores(scores,packet),
            'prompt_sha256':{q:hashlib.sha256(json.dumps(messages(rubric,q,packet),sort_keys=True).encode()).hexdigest() for q in QUALITIES}})
    observations,audit=reconcile([record],responses,[],[window],'b'*64)
    assert len(observations)==5 and all(r['accepted'] for r in audit)
    responses[0]['prompt_sha256'][QUALITIES[0]]='f'*64
    with pytest.raises(ValueError,match='prompt identity'):reconcile([record],responses,[],[window],'b'*64)
