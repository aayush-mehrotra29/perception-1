"""Fixed-budget local temporal training; preserve forecasts before dev scores."""
import argparse
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,guard
from .model import fit_scaling,standardize,TemporalEEG,save_model,load_model,forecast


def linear_design(x,p,rich):
    # all acoustic histories then lag-major speech; no targets in predictors
    a=x[:,:2,:].reshape(len(x),-1)
    return np.c_[p,a,x[:,2:,:].transpose(0,2,1).reshape(len(x),-1)] if rich else np.c_[p,a]


def fit_ridge(segments,rich,lam):
    xx=[];yy=[];weights=[]
    for s in segments:
        x=linear_design(s['x'],s['p'],rich);xx.append(x);yy.append(s['y'])
        weights.append(np.full(len(x),1/(len(segments)*len(x))))
    x=np.concatenate(xx).astype('float64');y=np.concatenate(yy).astype('float64');w=np.concatenate(weights)
    mx=w@x;my=w@y;a=x-mx;b=y-my
    coef=np.linalg.solve(a.T@(w[:,None]*a)+lam*np.eye(a.shape[1]),a.T@(w[:,None]*b))
    return {'coef':coef,'intercept':my-mx@coef}


def scores(y,p):
    mse=np.mean((y-p)**2,axis=0);a=y-y.mean(axis=0);b=p-p.mean(axis=0)
    denominator=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0));supported=denominator>1e-15
    corr=[float(v) if ok else None for v,ok in zip(np.divide(np.sum(a*b,axis=0),denominator,out=np.zeros_like(denominator),where=supported),supported)]
    return {'MSE':float(mse.mean()),'per_channel_MSE':mse.tolist(),'per_channel_r':corr,
        'macro_r':float(np.mean(corr)) if supported.all() else None,'defined_correlations':int(supported.sum()),
        'undefined_reason':None if supported.all() else 'constant observed or predicted series'}


def run(config_path,prepared,output):
    config_path,prepared,output=map(Path,[config_path,prepared,output]);c=read(config_path);guard(c)
    pm=read(prepared/'manifest.json')
    assert pm['complete'] and pm['config_sha256']==sha(config_path)
    torch.set_num_threads(c['resources']['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True,exist_ok=False);write(output/'config.json',c)
    provenance={'prepared_manifest_sha256':sha(prepared/'manifest.json'),'config_sha256':sha(config_path),
        'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
        'checkpoint_selection':'final fixed epoch; no development-driven selection','paid_requests':0,
        'participant':'ALICE S01','independent_benchmark':False,'TRIBE_present':False,'complete':False}
    write(output/'started.json',provenance)
    data={}
    for record in pm['segments']:
        p=prepared/record['file']
        assert sha(p)==record['sha256']
        with np.load(p,allow_pickle=False) as z:data[record['segment']]={k:z[k] for k in z.files}
    state=fit_scaling([data[i] for i in c['training_segments']]);np.savez(output/'transforms.npz',**state)
    train=[]
    for segment in c['training_segments']:
        ids=np.flatnonzero(data[segment]['supported']);x,p,y=standardize(data[segment],state,ids)
        train.append({'x':x,'p':p,'y':y,'segment':segment})
    dev={}
    for segment in c['development_segments']:
        ids=np.flatnonzero(data[segment]['supported']);x,p,y=standardize(data[segment],state,ids)
        dev[segment]={'x':x,'p':p,'y':y,'ids':ids}
    # Controls are fit only on training segments and saved before evaluation.
    controls={}
    for arm,rich in [('acoustic_ridge',False),('speech_acoustic_ridge',True)]:
        guard(c);controls[arm]=fit_ridge(train,rich,c['controls'][arm+'_lambda'])
        np.savez(output/(arm+'.npz'),**controls[arm])
    start=time.monotonic();settings=c['training'];all_predictions={};checkpoint_records=[]
    for arm in settings['arms']:
        inputs=2 if arm=='acoustic_tcn' else 34
        by_segment={i:[] for i in dev}
        for seed in settings['seeds']:
            guard(c);torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=TemporalEEG(inputs,53,c['architecture']);optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            folder=output/arm/f'seed{seed}';folder.mkdir(parents=True)
            history=[]
            for epoch in range(1,settings['epochs']+1):
                guard(c)
                if time.monotonic()-start>c['resources']['max_training_seconds']:raise RuntimeError('Training runtime cap reached; checkpoints retained')
                model.train();losses=[]
                for _ in range(settings['steps_per_epoch']):
                    s=train[int(rng.integers(len(train)))];ix=rng.integers(len(s['x']),size=settings['batch_size'])
                    x=torch.from_numpy(s['x'][ix,:inputs]);p=torch.from_numpy(s['p'][ix]);y=torch.from_numpy(s['y'][ix])
                    optimizer.zero_grad(set_to_none=True);pred=model(x,p);loss=torch.mean((pred-y)**2)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);optimizer.step();losses.append(float(loss.detach()))
                history.append({'epoch':epoch,'sampled_training_MSE':float(np.mean(losses))})
                if epoch%10==0:
                    save_model(model,folder/f'epoch{epoch:03d}.npz');write(folder/'training.json',history)
                    print(f'{arm} seed{seed} epoch{epoch}: training MSE {np.mean(losses):.6f}',flush=True)
            save_model(model,folder/'model.npz');reloaded=load_model(folder/'model.npz',inputs,53,c['architecture'])
            for segment,s in dev.items():
                pred=forecast(model,s['x'][:,:inputs],s['p']);rp=forecast(reloaded,s['x'][:,:inputs],s['p'])
                np.testing.assert_array_equal(pred,rp);by_segment[segment].append(pred)
                np.savez(folder/f'segment{segment:02d}-forecasts.npz',query_indices=s['ids'],prediction_standardized=pred)
            checkpoint_records.append({'arm':arm,'seed':seed,'file':str((folder/'model.npz').relative_to(output)),
                'sha256':sha(folder/'model.npz'),'epochs':settings['epochs'],
                'parameters':sum(p.numel() for p in model.parameters()),'reload_exact':True})
        all_predictions[arm]={i:np.mean(v,axis=0) for i,v in by_segment.items()}
    # Persist all normal predictions before computing any development score.
    forecast_manifest={}
    for segment,s in dev.items():
        preds={arm:v[segment] for arm,v in all_predictions.items()}
        preds['training_mean']=np.zeros_like(s['y'])
        for arm,rich in [('acoustic_ridge',False),('speech_acoustic_ridge',True)]:
            m=controls[arm];preds[arm]=linear_design(s['x'],s['p'],rich)@m['coef']+m['intercept']
        path=output/f'segment{segment:02d}-all-forecasts.npz'
        np.savez(path,query_indices=s['ids'],query_seconds=data[segment]['query_seconds'][s['ids']],**preds)
        forecast_manifest[str(segment)]={'file':path.name,'sha256':sha(path)}
    write(output/'forecast-lock.json',forecast_manifest)
    # Wrong-content checks swap non-progress histories only on supported paired clocks.
    wrong_records={}
    for segment,donor in [(9,10),(10,9)]:
        own=data[segment];other=data[donor];n=min(len(own['supported']),len(other['feature_supported']))
        ids=np.flatnonzero(own['supported'][:n]&other['feature_supported'][:n])
        np.testing.assert_array_equal(own['query_seconds'][ids],other['query_seconds'][ids])
        x,p,_=standardize(other,state,ids);_,op,y=standardize(own,state,ids)
        ownx,_,_=standardize(own,state,ids);wr={}
        for arm in settings['arms']:
            inputs=2 if arm=='acoustic_tcn' else 34;preds=[];normal=[]
            for seed in settings['seeds']:
                model=load_model(output/arm/f'seed{seed}'/'model.npz',inputs,53,c['architecture'])
                preds.append(forecast(model,x[:,:inputs],op));normal.append(forecast(model,ownx[:,:inputs],op))
            wr[arm]=np.mean(preds,axis=0);wr[arm+'_matched']=np.mean(normal,axis=0)
        path=output/f'wrong-{segment}-from-{donor}.npz';np.savez(path,query_indices=ids,**wr)
        wrong_records[f'{segment}_from_{donor}']={'rows':len(ids),'sha256':sha(path),
            'metrics':{arm:scores(y,pred) for arm,pred in wr.items()}}
    results={}
    for segment,s in dev.items():
        record=forecast_manifest[str(segment)];assert sha(output/record['file'])==record['sha256']
        with np.load(output/record['file'],allow_pickle=False) as z:
            results[str(segment)]={k:scores(s['y'],z[k]) for k in z.files if not k.startswith('query_')}
    arms=list(results[str(c['development_segments'][0])])
    macro={a:float(np.mean([r[a]['MSE'] for r in results.values()])) for a in arms}
    write(output/'development-results.json',{'status':'exploratory_development_only','per_segment':results,
        'equal_segment_channel_MSE':macro,'wrong_content':wrong_records,
        'no_population_confidence_interval':True,'no_TRIBE_comparison':True,'limitations':c['limitations']})
    provenance.update(complete=True,checkpoints=checkpoint_records,forecasts=forecast_manifest,
        elapsed_training_seconds=time.monotonic()-start,transform_sha256=sha(output/'transforms.npz'),
        results_sha256=sha(output/'development-results.json'),train_supported_rows=sum(len(s['x']) for s in train),
        development_supported_rows={str(i):len(s['x']) for i,s in dev.items()})
    write(output/'manifest.json',provenance)
    print('Fixed-budget development training complete; no TRIBE superiority claim',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('prepared');p.add_argument('output')
    a=p.parse_args();run(a.config,a.prepared,a.output)
