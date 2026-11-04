import csv
import json
from pathlib import Path
import numpy as np
import pytest
from brain.behavior_audit import parse_reports,parse_key,audit_quizzes
from brain.behavior_models import fit_head,predict_head,loss_rows,Unsupported,Engine
from brain.fusion_data import write_json
from brain.worker_contract import file_sha256


def test_probe_decimal_labels_match_cue_not_neighbor_and_exclude_duplicates():
    rows=[]
    for i in range(612):
        q=min(i/10,60.)
        rows.append({'video_uid':'lesson','video_timestamp_sec':str(q),'system_time_sec':str(i*100),
            'flash_on':'1' if 600<=i<=608 else '0','is_paused':'1' if 600<=i<=608 else '0',
            'engagement_report':'2.0' if i==609 else ''})
    lessons={'lesson':{'features':{'media_duration':100}}}
    events,x,v=parse_reports(rows,1,lessons)
    assert len(events)==1 and events[0]['rating']==2 and events[0]['query']==60 and not x
    rows[610]['engagement_report']='3'
    events,x,v=parse_reports(rows,1,lessons)
    assert not events and x[0]['reason']=='multiple_responses_per_cue'


def test_attention_and_binomial_models_reload_and_loss_definition(tmp_path):
    x=np.linspace(-1,1,50)[:,None];y=np.where(x[:,0]>0,4.,2.);w=np.ones(50)/50
    model=fit_head(x,y,w,.1,'attention');pred=predict_head(model,x,'attention')
    assert np.abs(pred-y).mean()<np.abs(y-y.mean()).mean()
    np.savez(tmp_path/'model.npz',**model)
    with np.load(tmp_path/'model.npz') as z:np.testing.assert_array_equal(pred,predict_head(dict(z),x,'attention'))
    y=(y-2)/2;model=fit_head(x,y,w,.1,'quiz');p=predict_head(model,x,'quiz')
    assert np.isfinite(p).all() and (p>0).all() and (p<1).all()
    np.testing.assert_allclose(loss_rows(np.array([.4]),np.array([.3]),'quiz'),[.4*.7**2+.6*.3**2])
    with pytest.raises(Unsupported,match='single class'):fit_head(x,np.zeros(50),w,.1,'quiz')


def test_report_quiz_scores_match_expanded_individual_trials():
    from brain.behavior_report import errors
    y=np.array([.4,.8]);p=np.array([.3,.7])
    scored=errors(y,p,'quiz')
    for i,correct in enumerate([2,4]):
        trials=np.array([1]*correct+[0]*(5-correct))
        assert scored['Brier'][i]==pytest.approx(np.mean((trials-p[i])**2))
        assert scored['log_loss'][i]==pytest.approx(np.mean(-trials*np.log(p[i])-(1-trials)*np.log1p(-p[i])))


def fixture(root):
    root.mkdir();rng=np.random.default_rng(42)
    lessons=np.repeat(['a','b','c','d'],5);query=np.tile(np.arange(5)*5+45,4)
    names=['dense_text','dense_audio','dense_video','predicted_tribe','progress','sam_visual']
    raw={n:rng.normal(size=(20,6)).astype('float32') for n in names};raw['progress']=query[:,None]/100
    np.savez(root/'features.npz',lesson=lessons,query=query,**raw)
    person=np.repeat(np.arange(1,17),20);point=np.tile(np.arange(20),16)
    target=rng.normal(size=(320,8));target[:,4:]=np.exp(target[:,4:])
    np.savez(root/'eeg.npz',person=person,point=point,lesson=lessons[point],target=target)
    np.savez(root/'attention.npz',person=person,point=point,lesson=lessons[point],y=np.ones(320)*3)
    np.savez(root/'quiz.npz',person=np.arange(1,17),lesson=np.array(['a']*16),pre=np.zeros(16),y=np.ones(16)*.4)
    write_json(root/'quiz-windows.json',{l:np.flatnonzero(lessons==l).tolist() for l in ['a','b','c','d']})
    protocol=json.loads((Path(__file__).parents[1]/'configs/behavior_benchmark_v2.json').read_text())
    write_json(root/'protocol.json',protocol)
    write_json(root/'manifest.json',{'files':{p.name:file_sha256(p) for p in root.iterdir()}})


def test_held_out_EEG_and_features_cannot_change_training_state(tmp_path):
    root=tmp_path/'prepared';fixture(root)
    engine=Engine(root,tmp_path/'models');a=engine.scope(['a','b'],list(range(1,9)))
    held=(engine.eeg['person']>8)|np.isin(engine.eeg['lesson'],['c','d'])
    engine.eeg['target'][held]=999999
    engine.output=tmp_path/'changed';engine.cache={}
    for n in engine.names:engine.raw[n][np.isin(engine.raw['lesson'],['c','d'])]=999999
    b=engine.scope(['a','b'],list(range(1,9)))
    np.testing.assert_array_equal(a['eeg'][:10],b['eeg'][:10])
    np.testing.assert_array_equal(a['base'][:10],b['base'][:10])
    index=engine.mask(engine.tasks['attention'],['a','b'],list(range(1,9)))
    x,scope=engine.matrices('attention',index,['a','b'],list(range(1,9)),True)
    assert x['all_blocks'].shape==x['content_matched'].shape
    assert len(scope['stacked'])==2
    for r in scope['stacked']:
        record=next(s for s in engine.records if s['id']==r['scope'])
        assert all((p-1)//4!=r['group'] for p in record['people'])
