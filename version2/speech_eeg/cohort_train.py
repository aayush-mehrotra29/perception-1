"""Fixed cohort training with all test forecasts locked before target access."""
import argparse
import shutil
import time
import platform
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,guard
from .cohort_data import load_features,attach_training_targets,SLOTS,design,design_indices,ridge_from_moments
from .model import fit_scaling,standardize,feature_transform,TemporalEEG,save_model,load_model,forecast


def run(config_path,features,targets,output,resume=False):
    config_path,features,targets,output=map(Path,[config_path,features,targets,output])
    c=read(config_path);resource=read('configs/stage_a.json');guard(resource)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prior_wall=0.
    if resume:
        if (output/'forecast-lock.json').exists():raise ValueError('Run is already locked; do not resume it')
        assert read(output/'config.json')==c
        original=read(output/'started.json')
        assert original['config_sha256']==sha(config_path)
        assert original['feature_manifest_sha256']==sha(features/'manifest.json')
        assert original['target_manifest_sha256']==sha(targets/'manifest.json')
        for name in ['model.py','cohort_data.py','common.py']:
            assert sha(Path(__file__).with_name(name))==original['source'][name]
        prior_wall=max(0,time.time()-(output/'started.json').stat().st_mtime)
        record=output/f'resume-{time.time_ns()}.json'
        shutil.copy2(__file__,record.with_suffix('.py'))
        write(record,{'reason':'Original process handle and PID disappeared after turn interruption',
            'policy':'Replay completed final seeds exactly; restart unfinished seeds from fixed seed, not partial weights',
            'prior_wall_seconds_conservative':prior_wall,'code_sha256':sha(__file__),
            'files_before_resume':{str(p.relative_to(output)):sha(p) for p in output.rglob('*.npz')}})
    else:
        output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,output/'source'/p.name)
        write(output/'config.json',c)
    data,fm=load_features(features);tm=attach_training_targets(data,targets,c)
    if fm['model_config_sha256']!=sha(config_path):raise ValueError('Feature configuration changed')
    state=fit_scaling([data[i] for i in c['training_segments']])
    if resume:
        with np.load(output/'transforms.npz',allow_pickle=False) as z:
            for k,v in state.items():np.testing.assert_array_equal(v,z[k])
    else:np.savez(output/'transforms.npz',**state)
    started={'complete':False,'config_sha256':sha(config_path),'feature_manifest_sha256':sha(features/'manifest.json'),
        'target_manifest_sha256':sha(targets/'manifest.json'),'channels':tm['channels'],
        'source':{p.name:sha(p) for p in (output/'source').glob('*.py')},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
        'test_EEG_opened':False,'training_groups':c['training_people'],'training_clips':c['training_segments']}
    if resume:started=original
    else:write(output/'started.json',started)
    start=time.monotonic()
    train=[]
    for i in c['training_segments']:
        ids=np.flatnonzero(data[i]['supported']);x,p,y=standardize(data[i],state,ids)
        train.append({'segment':i,'x':x,'p':p,'y':y,'target':data[i]['target'][ids]})
    # Release large raw training histories after transforms; retain only dev/test stimulus.
    for i in c['training_segments']:del data[i]
    dim=3+82*51;mx=np.zeros(dim);xx=np.zeros((dim,dim));my=np.zeros(40);yy=np.zeros(40);xy=np.zeros((dim,40))
    ridge_records=[];budgets=[len(b) for b in c['label_budget_benchmark']['labeled_training_clips']]
    for count,s in enumerate(train,1):
        guard(resource);x=design(s['x'],s['p'],SLOTS['full']);y=s['target'];n=len(x)
        mx+=x.mean(0);xx+=x.T@x/n;my+=y.mean(0);yy+=(y*y).mean(0);xy+=x.T@y/n
        del x
        if count not in budgets:continue
        for block,slots in SLOTS.items():
            ix=design_indices(slots)
            m=ridge_from_moments(mx[ix]/count,xx[np.ix_(ix,ix)]/count,my/count,yy/count,xy[ix]/count,state,c['ridge_lambda'])
            name=f'{block}_ridge_labels{count}';path=output/(name+'.npz')
            if resume and path.exists():
                with np.load(path,allow_pickle=False) as z:
                    for k,v in m.items():np.testing.assert_array_equal(v,z[k])
            else:np.savez(path,**m)
            ridge_records.append({'name':name,'block':block,'budget':count,'file':path.name,'sha256':sha(path),
                                  'labeled_segments':c['training_segments'][:count]})
        print(f'Fixed ridge label budget {count}/8 saved',flush=True)
    del xx,xy
    forecasts=output/'forecasts';forecasts.mkdir(exist_ok=resume);contexts={}
    for i in c['development_segments']+c['test_segments']:
        s=data[i];ids=np.flatnonzero(s['feature_supported']);x,p=feature_transform(s,state,ids)
        contexts[f'segment{i:02d}']={'segment':i,'ids':ids,'x':x,'p':p,'query_seconds':s['query_seconds'][ids]}
    for own,donor in [(11,12),(12,11)]:
        s=data[own];d=data[donor];n=min(len(s['feature_supported']),len(d['feature_supported']))
        ids=np.flatnonzero(s['feature_supported'][:n]&d['feature_supported'][:n])
        np.testing.assert_allclose(s['query_seconds'][ids],d['query_seconds'][ids],atol=1e-10,rtol=0)
        x,_=feature_transform(d,state,ids);_,p=feature_transform(s,state,ids)
        contexts[f'wrong{own:02d}-from{donor:02d}']={'segment':own,'ids':ids,'x':x,'p':p,'query_seconds':s['query_seconds'][ids]}
    del data
    predictions={name:{'mean':np.zeros((len(s['ids']),40))} for name,s in contexts.items()}
    for r in ridge_records:
        with np.load(output/r['file'],allow_pickle=False) as z:coef=z['coef'];intercept=z['intercept']
        for name,s in contexts.items():
            preds=design(s['x'],s['p'],SLOTS[r['block']])@coef+intercept
            predictions[name][r['name']]=preds
            if r['budget']==8:predictions[name][r['block']+'_ridge']=preds
    settings=c['training'];checkpoints=[]
    for block in ['tribe','content','full']:
        arm=block+'_tcn';slots=SLOTS[block];ensemble={name:[] for name in contexts}
        for seed in settings['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=TemporalEEG(len(slots),40,c['architecture'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            folder=output/arm/f'seed{seed}';completed=resume and (folder/'model.npz').exists()
            if completed:
                assert read(folder/'training.json')[-1]['epoch']==settings['epochs']
                model=load_model(folder/'model.npz',len(slots),40,c['architecture'])
                last=load_model(folder/f'epoch{settings["epochs"]:03d}.npz',len(slots),40,c['architecture'])
                for key,v in model.state_dict().items():np.testing.assert_array_equal(v.numpy(),last.state_dict()[key].numpy())
                print(f'{arm} seed{seed}: preserving and replaying completed final checkpoint',flush=True)
            elif resume and folder.exists():
                saved=output/'interrupted'/f'{arm}-seed{seed}-{time.time_ns()}';saved.parent.mkdir(exist_ok=True)
                folder.rename(saved)
                print(f'{arm} seed{seed}: partial checkpoints archived; restarting fixed seed',flush=True)
            folder.mkdir(parents=True,exist_ok=completed);history=[]
            for epoch in range(1,(0 if completed else settings['epochs'])+1):
                guard(resource)
                if prior_wall+time.monotonic()-start>c['resource_limits']['max_training_seconds']:raise RuntimeError('Fixed training time cap reached; partial results preserved')
                model.train();losses=[]
                for _ in range(settings['steps_per_epoch']):
                    s=train[int(rng.integers(len(train)))];ix=rng.integers(len(s['x']),size=settings['batch_size'])
                    x=torch.from_numpy(s['x'][ix][:,slots,:]);p=torch.from_numpy(s['p'][ix]);y=torch.from_numpy(s['y'][ix])
                    optimizer.zero_grad(set_to_none=True);loss=torch.mean((model(x,p)-y)**2)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);optimizer.step();losses.append(float(loss.detach()))
                history.append({'epoch':epoch,'sampled_training_MSE':float(np.mean(losses))})
                if epoch%10==0:
                    save_model(model,folder/f'epoch{epoch:03d}.npz');write(folder/'training.json',history)
                    print(f'{arm} seed{seed} epoch{epoch}: training MSE {np.mean(losses):.6f}',flush=True)
            if not completed:save_model(model,folder/'model.npz')
            reload=load_model(folder/'model.npz',len(slots),40,c['architecture'])
            for name,s in contexts.items():
                pred=forecast(model,s['x'][:,slots,:],s['p']);again=forecast(reload,s['x'][:,slots,:],s['p'])
                np.testing.assert_array_equal(pred,again);ensemble[name].append(pred)
                if completed and (folder/(name+'.npz')).exists():
                    with np.load(folder/(name+'.npz'),allow_pickle=False) as saved:
                        np.testing.assert_array_equal(pred,saved['prediction']);np.testing.assert_array_equal(s['ids'],saved['query_indices'])
                else:np.savez(folder/(name+'.npz'),prediction=pred,query_indices=s['ids'])
            checkpoints.append({'arm':arm,'seed':seed,'file':str((folder/'model.npz').relative_to(output)),
                'sha256':sha(folder/'model.npz'),'reload_exact':True,'parameters':sum(p.numel() for p in model.parameters())})
        for name,v in ensemble.items():predictions[name][arm]=np.mean(v,axis=0)
    locked={}
    for name,s in contexts.items():
        for arm,p in predictions[name].items():
            if p.shape!=(len(s['ids']),40) or not np.isfinite(p).all():raise ValueError('Invalid forecast')
        path=forecasts/(name+'.npz');np.savez(path,query_indices=s['ids'],query_seconds=s['query_seconds'],**predictions[name])
        locked[name]={'file':str(path.relative_to(output)),'sha256':sha(path),'rows':len(s['ids']),
                      'segment':s['segment'],'arms':list(predictions[name])}
    # Test EEG must not be read until this immutable-by-hash lock has been audited.
    lock={'schema':'perception-v2-cohort-forecast-lock','complete':True,'test_EEG_opened':False,
          'forecasts':locked,'models':checkpoints,'ridges':ridge_records,
          'transforms_sha256':sha(output/'transforms.npz'),'config_sha256':sha(output/'config.json'),
          'feature_manifest_sha256':sha(features/'manifest.json'),'target_manifest_sha256':sha(targets/'manifest.json'),
          'elapsed_seconds':prior_wall+time.monotonic()-start,'source_sha256':started['source'],
          'resume_records':{p.name:sha(p) for p in output.glob('resume-*.json')}}
    write(output/'forecast-lock.json',lock);started.update(complete=True,forecast_lock_sha256=sha(output/'forecast-lock.json'),elapsed_seconds=time.monotonic()-start)
    write(output/'manifest.json',started)
    print('All fixed models and forecasts locked. Reserved test EEG has not been opened.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('features');p.add_argument('targets');p.add_argument('output');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.features,a.targets,a.output,a.resume)
