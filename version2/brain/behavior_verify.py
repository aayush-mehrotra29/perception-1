"""Replay nested behavioral fits before opening their outer benchmark scores."""
import argparse
from pathlib import Path
import numpy as np
from .behavior_models import Engine,fit_head,predict_head
from .fusion_data import read_json,write_json,checked
from .fusion_v3_model import load_transforms
from .eeg import equal_viewing_weights
from .worker_contract import file_sha256


def run(evaluation,output):
    evaluation,output=Path(evaluation),Path(output);m=read_json(evaluation/'manifest.json')
    assert m['complete'];output.mkdir(parents=True,exist_ok=False)
    for name,digest in m['code_sha256'].items():assert file_sha256(Path(__file__).parent/name)==digest
    assert file_sha256(Path(m['prepared'])/'manifest.json')==m['prepared_manifest_sha256']
    engine=Engine(m['prepared'],output/'replay');protocol=engine.protocol;count=0
    for task,record in m['tasks'].items():
        data=engine.tasks[task];seen=np.zeros(len(data['y']),int)
        for fold in record['folds']:
            root=evaluation/task/fold['fold'];s=read_json(checked(root,'scope.json',fold['scope_sha256']))
            tr,te=np.array(s['train']),np.array(s['test'])
            assert not set(data['person'][tr])&set(data['person'][te])
            assert not set(data['lesson'][tr])&set(data['lesson'][te]);seen[te]+=1
            for inner in s['inner']:
                it,iv=np.array(inner['training']),np.array(inner['validation'])
                assert set(it).issubset(tr) and set(iv).issubset(tr)
                for field in ['person','lesson']:assert not set(data[field][it])&set(data[field][iv])
            # Scope IDs hash the original numpy string tuple representation.
            # JSON converts those strings to Python str, so restore their type.
            lessons=list(np.asarray(s['lessons']))
            a,scope=engine.matrices(task,tr,lessons,s['people'],True)
            b,_=engine.matrices(task,te,lessons,s['people'],False)
            assert scope==s['training_scopes']
            w=equal_viewing_weights(data['lesson'][tr],data['person'][tr])
            with np.load(checked(root,'forecasts.npz',fold['forecast_sha256']),allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['row_indices'],te)
                for arm in protocol['arms']:
                    scores=np.mean([r['scores'][arm] for r in s['inner']],axis=0)
                    j=min(range(len(scores)),key=lambda i:(scores[i],-protocol['lambda_grid'][i]))
                    assert s['chosen_lambda'][arm]==protocol['lambda_grid'][j]
                    refit=fit_head(a[arm],data['y'][tr],w,s['chosen_lambda'][arm],task)
                    with np.load(checked(root,arm+'.npz',fold['heads_sha256'][arm]),allow_pickle=False) as head:
                        for name in refit:np.testing.assert_allclose(refit[name],head[name],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(predict_head(refit,b[arm],task),saved[arm],rtol=0,atol=1e-12);count+=1
            print(task+' '+fold['fold']+' verified',flush=True)
        assert np.all(seen<=1)
    # Refit every preprocessing/EEG scope, including scopes used only in inner selection.
    for record in m['scopes']:
        engine.scope(list(np.asarray(record['lessons'])),record['people'])
        original=evaluation/'scopes'/record['id'];replay=output/'replay/scopes'/record['id']
        checked(original,'transforms.json',record['transform_sha256'])
        a=load_transforms(original);b=load_transforms(replay)
        for name in a:
            for field in a[name]:np.testing.assert_array_equal(a[name][field],b[name][field])
        with np.load(checked(original,'readout.npz',record['readout_sha256']),allow_pickle=False) as x,np.load(replay/'readout.npz',allow_pickle=False) as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    write_json(output/'verification.json',{'schema':'perception-behavior-verification-v2','status':'passed',
        'evaluation_manifest_sha256':file_sha256(evaluation/'manifest.json'),'heads_refitted_and_replayed':count,
        'training_only_preprocessing_and_EEG_scopes_refitted':len(m['scopes']),
        'outer_participant_and_lesson_disjoint':True,'inner_participant_and_lesson_disjoint':True,
        'training_EEG_features_cross_fitted_by_participant_group':True,'held_out_EEG_not_used':True,
        'quiz_answers_excluded_from_model_inputs':True,'native_TRIBE_parameters_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.evaluation,a.output)
