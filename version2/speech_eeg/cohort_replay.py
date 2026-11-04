"""Independent model reload and stimulus-only reproduction of locked forecasts."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,guard
from .cohort_data import load_features,SLOTS,design
from .cohort_evaluate import verify_lock
from .model import feature_transform,load_model,forecast


def run(run,features,targets):
    run,features,targets=map(Path,[run,features,targets]);guard(read('configs/stage_a.json'))
    lock,c=verify_lock(run,features,targets);torch.set_num_threads(1)
    with np.load(run/'transforms.npz',allow_pickle=False) as z:state={k:z[k] for k in z.files}
    data,_=load_features(features);checks=[];max_error=0.;seed_pins={}
    # Labels are never loaded. Reconstruct masks, ordering, wrong-content and progress.
    for name,r in lock['forecasts'].items():
        own=r['segment'];donor=own
        if name.startswith('wrong'):donor=int(name.split('from')[1])
        s=data[own];d=data[donor];n=min(len(s['feature_supported']),len(d['feature_supported']))
        ids=np.flatnonzero(s['feature_supported'][:n]&d['feature_supported'][:n])
        x,_=feature_transform(d,state,ids);_,p=feature_transform(s,state,ids)
        with np.load(run/r['file'],allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['query_indices'])
            np.testing.assert_allclose(s['query_seconds'][ids],z['query_seconds'],atol=1e-10,rtol=0)
            np.testing.assert_array_equal(z['mean'],np.zeros_like(z['mean']))
            for record in lock['ridges']:
                with np.load(run/record['file'],allow_pickle=False) as m:prediction=design(x,p,SLOTS[record['block']])@m['coef']+m['intercept']
                np.testing.assert_allclose(prediction,z[record['name']],rtol=1e-12,atol=1e-12)
                max_error=max(max_error,float(np.max(np.abs(prediction-z[record['name']]))))
                if record['budget']==8:np.testing.assert_array_equal(z[record['name']],z[record['block']+'_ridge'])
                checks.append({'context':name,'arm':record['name'],'passed':True})
            for block in ['tribe','content','full']:
                predictions=[];slots=SLOTS[block]
                for seed in c['training']['seeds']:
                    folder=run/(block+'_tcn')/f'seed{seed}';model=load_model(folder/'model.npz',len(slots),40,c['architecture'])
                    pred=forecast(model,x[:,slots,:],p)
                    with np.load(folder/(name+'.npz'),allow_pickle=False) as single:
                        np.testing.assert_array_equal(ids,single['query_indices']);np.testing.assert_array_equal(pred,single['prediction'])
                    seed_path=folder/(name+'.npz');seed_pins[str(seed_path.relative_to(run))]=sha(seed_path)
                    predictions.append(pred)
                pred=np.mean(predictions,axis=0);np.testing.assert_array_equal(pred,z[block+'_tcn'])
                checks.append({'context':name,'arm':block+'_tcn','all_seeds_exact':True,'passed':True})
    result={'passed':True,'forecast_lock_sha256':sha(run/'forecast-lock.json'),'checks':checks,
            'maximum_ridge_absolute_error':max_error,'neural_replay_bitwise_equal':True,
            'held_EEG_opened':False,'seed_forecast_sha256':seed_pins,'code_sha256':sha(__file__)}
    write(run/'reload-audit.json',result);print('All locked predictions independently reproduced without opening held EEG.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('features');p.add_argument('targets')
    a=p.parse_args();run(a.run,a.features,a.targets)
