"""Prespecified v3 repairs; legacy forecasts and native cortical arrays stay fixed."""
import argparse
from pathlib import Path
import time
import shutil
import numpy as np

from .fusion_data import read_json,write_json,checked
from .fusion_experiment import load_prepared,apply_fold_data
from .fusion_model import load_transforms as load_old_transforms,load_network,predict,train,shuffled_targets
from .fusion_v3_model import fit_fold,load_transforms,apply,design,fit_readout,correct,positive_shift
from .eeg import equal_viewing_weights
from .edugage_targets import guard
from .worker_contract import file_sha256


def save_readout(root,coef,scale,mode,bound,scope,anchor_sha):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    np.savez(root/'readout.npz',coef=coef,scale=scale)
    write_json(root/'manifest.json',{'schema':'perception-bounded-readout-v3','mode':mode,'bound':bound,
        'training_scope':scope,'anchor_manifest_sha256':anchor_sha,'sha256':file_sha256(root/'readout.npz')})


def load_readout(root):
    m=read_json(Path(root)/'manifest.json')
    with np.load(checked(root,'readout.npz',m['sha256']),allow_pickle=False) as z:
        return m,z['coef'],z['scale']


def run(config,output):
    import torch
    torch.set_num_threads(1)
    protocol=read_json(config); prepared=Path(protocol['prepared']); old=Path(protocol['previous_evaluation'])
    meta,rows,grids,v2,folds=load_prepared(prepared)
    previous=read_json(old/'manifest.json')
    if not previous['complete'] or previous['prepared_manifest_sha256']!=file_sha256(prepared/'manifest.json'):
        raise ValueError('matching complete baseline required')
    output=Path(output); guard(output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',protocol)
    report={'schema':'perception-fusion-evaluation-v3','complete':False,'folds':[],
        'prepared_manifest_sha256':file_sha256(prepared/'manifest.json'),
        'old_evaluation_sha256':file_sha256(old/'manifest.json'),'protocol_sha256':file_sha256(config),
        'code_sha256':{n:file_sha256(Path(__file__).parent/n) for n in ['fusion_v3_model.py','fusion_v3_experiment.py','fusion_model.py','fusion_experiment.py']}}
    write_json(output/'started.json',report)
    np.savez(output/'evaluation-rows.npz',**{k:rows[k] for k in ['target','lesson','person','family','content_id','query']})
    for scope in folds:
        guard(output); start=time.monotonic()
        name=f"{scope['held_family']}-group{scope['held_group']}"; root=output/name; origin=old/name
        tr,te=np.array(scope['train']),np.array(scope['test'])
        for field in ['person','lesson','content_id']:
            if set(rows[field][tr]) & set(rows[field][te]): raise ValueError('fold leakage')
        _,(seq,aux)=fit_fold(rows,grids,tr,root,protocol['preprocessing'])
        write_json(root/'scope.json',scope)
        source=next(r for r in previous['folds'] if r['fold']==name)
        with np.load(checked(origin,'forecasts.npz',source['forecast_sha256']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['row_indices'],te)
            scale,mean=z['training_scale'],z['training_mean']
            forecasts={'mean':z['mean'],'tribe_temporal':z['tribe_temporal'],'historical_v2_fusion':z['fusion']}
        old_seq,old_aux=apply_fold_data(rows,grids,load_old_transforms(origin))
        weight=equal_viewing_weights(rows['lesson'][tr],rows['person'][tr])
        x=design(seq,aux); no_aux=design(seq,aux,True)
        seed_predictions={k:[] for k in ['fusion','without_aux','shuffled_correction']}
        for seed in protocol['seeds']:
            anchor_path=origin/'tribe_temporal'/str(seed); anchor=load_network(anchor_path)
            base_train=predict(anchor,old_seq[tr],old_aux[tr],'tribe_temporal')
            base_test=predict(anchor,old_seq[te],old_aux[te],'tribe_temporal')
            with np.load(anchor_path/'forecasts.npz') as saved: np.testing.assert_array_equal(base_test,saved['predictions'])
            for arm in seed_predictions:
                target=rows['target'][tr]
                if arm=='shuffled_correction': target,permutation=shuffled_targets(target,rows['lesson'][tr],rows['person'][tr],seed)
                else: permutation=np.arange(len(tr))
                features=no_aux if arm=='without_aux' else x
                residual=np.clip((target-base_train)/scale,-protocol['residual']['training_residual_clip'],protocol['residual']['training_residual_clip'])
                coef=fit_readout(features[tr],residual,weight,protocol['residual']['ridge_lambda'])
                path=root/arm/str(seed); bound=protocol['residual']['maximum_normalized_correction']
                save_readout(path,coef,scale,arm,bound,scope,file_sha256(anchor_path/'manifest.json'))
                np.save(path/'permutation.npy',permutation,allow_pickle=False)
                pred=correct(features[te],base_test,coef,scale,bound)
                _,reloaded,rs=load_readout(path)
                np.testing.assert_array_equal(pred,correct(features[te],base_test,reloaded,rs,bound))
                if np.max(np.abs(pred-base_test)/scale)>bound+1e-9: raise ValueError('correction bound violated')
                if not np.isfinite(pred).all() or np.any(pred[:,4:]<0): raise ValueError('invalid correction')
                np.savez(path/'forecasts.npz',row_indices=te,predictions=pred)
                seed_predictions[arm].append(pred)
        for arm,values in seed_predictions.items(): forecasts[arm]=np.mean(values,axis=0)
        coef=fit_readout(x[tr],(rows['target'][tr]-mean)/scale,weight,protocol['simple_ridge']['lambda'])
        save_readout(root/'simple_ridge',coef,scale,'simple_ridge',None,scope,None)
        forecasts['simple_ridge']=positive_shift(np.broadcast_to(mean,(len(te),8)).copy(),np.c_[np.ones(len(te)),x[te]]@coef,scale)
        for arm,internal in [('robust_tribe','tribe_temporal'),('robust_fusion','fusion')]:
            values=[]
            for seed in protocol['seeds']:
                guard(output)
                network,_=train(seq[tr],aux[tr],rows['target'][tr],rows['lesson'][tr],rows['person'][tr],internal,seed,v2['training'],root/arm/str(seed))
                pred=predict(network,seq[te],aux[te],internal)
                np.testing.assert_array_equal(pred,predict(load_network(root/arm/str(seed)),seq[te],aux[te],internal))
                np.savez(root/arm/str(seed)/'forecasts.npz',row_indices=te,predictions=pred)
                values.append(pred)
                print(f'{name} {arm} seed {seed} forecast saved',flush=True)
            forecasts[arm]=np.mean(np.stack(values).astype(float),axis=0)
        if set(forecasts)!=set(protocol['arms']): raise ValueError('incomplete arms')
        np.savez(root/'forecasts.npz',row_indices=te,training_mean=mean,training_scale=scale,**forecasts)
        record={'fold':name,'rows':len(te),'seconds':time.monotonic()-start,'forecast_sha256':file_sha256(root/'forecasts.npz')}
        write_json(root/'complete.json',record); report['folds'].append(record)
    report['complete']=True; write_json(output/'manifest.json',report)
    print('All v3 forecasts saved before scoring',flush=True)


def deploy(config,evaluation,benchmark,output):
    protocol=read_json(config); prepared=Path(protocol['prepared'])
    meta,rows,grids,v2,_=load_prepared(prepared)
    scored=read_json(Path(benchmark)/'report.json'); evaluated=read_json(Path(evaluation)/'manifest.json')
    if not evaluated['complete'] or scored['evaluation_manifest_sha256']!=file_sha256(Path(evaluation)/'manifest.json'):
        raise ValueError('complete benchmark required')
    output=Path(output); guard(output)
    _,(seq,aux)=fit_fold(rows,grids,np.arange(len(rows['target'])),output,protocol['preprocessing'])
    anchor_root=output/'anchor'; anchor_root.mkdir()
    original=Path('.local/brain-runs/fusion-v2-deployment')
    # The old deployment transforms were fitted to this exact full-data corpus.
    om=read_json(original/'manifest.json')
    if om['prepared_manifest_sha256']!=file_sha256(prepared/'manifest.json'): raise ValueError('full-data transform source mismatch')
    load_old_transforms(original)
    for f in ['transforms.json','transforms.npz']: shutil.copy2(original/f,anchor_root/f)
    old_seq,old_aux=apply_fold_data(rows,grids,load_old_transforms(anchor_root))
    x=design(seq,aux); w=equal_viewing_weights(rows['lesson'],rows['person']); mean=w@rows['target']
    scale=np.sqrt(w@((rows['target']-mean)**2)); scale[scale==0]=1
    ensemble=[]
    for seed in protocol['seeds']:
        guard(output); netroot=anchor_root/str(seed)
        net,_=train(old_seq,old_aux,rows['target'],rows['lesson'],rows['person'],'tribe_temporal',seed,v2['training'],netroot)
        base=predict(net,old_seq,old_aux,'tribe_temporal')
        y=np.clip((rows['target']-base)/scale,-protocol['residual']['training_residual_clip'],protocol['residual']['training_residual_clip'])
        coef=fit_readout(x,y,w,protocol['residual']['ridge_lambda'])
        rr=output/'ensemble'/str(seed)
        save_readout(rr,coef,scale,'fusion',protocol['residual']['maximum_normalized_correction'],{'full_data':True},file_sha256(netroot/'manifest.json'))
        ensemble.append({'seed':seed,'path':f'ensemble/{seed}','anchor':f'anchor/{seed}',
            'manifest_sha256':file_sha256(rr/'manifest.json'),'anchor_manifest_sha256':file_sha256(netroot/'manifest.json')})
        print('Deployment seed fitted: '+str(seed),flush=True)
    write_json(output/'benchmark.json',scored); write_json(output/'protocol.json',protocol)
    write_json(output/'manifest.json',{'schema':'perception-fusion-deployment-v3','model_version':'fusion-v3-bounded-residual',
        'ensemble':ensemble,'transforms_manifest_sha256':file_sha256(output/'transforms.json'),
        'anchor_transforms_manifest_sha256':file_sha256(anchor_root/'transforms.json'),
        'benchmark_sha256':file_sha256(output/'benchmark.json'),'protocol_sha256':file_sha256(output/'protocol.json'),
        'benchmark_reference':str((Path(benchmark)/'report.json').resolve()),
        'evaluation_status':'exploratory_reanalysis_after_v2_diagnosis',
        'benchmark_success_target_met':scored['success_target_point_estimate_met'],
        'measured_fmri_superiority_tested':False,'TRIBE_cortical_head_changed':False,
        'prepared_manifest_sha256':file_sha256(prepared/'manifest.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('run'); a.add_argument('config',type=Path); a.add_argument('output',type=Path)
    a=s.add_parser('deploy')
    for n in ['config','evaluation','benchmark','output']: a.add_argument(n,type=Path)
    a=p.parse_args()
    if a.cmd=='run': run(a.config,a.output)
    else: deploy(a.config,a.evaluation,a.benchmark,a.output)
