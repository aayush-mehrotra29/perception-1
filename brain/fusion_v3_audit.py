"""Independent persisted v3 checks before scoring."""
import argparse
from pathlib import Path
import numpy as np
from .fusion_data import read_json,write_json,checked,MODALITIES,AUXILIARY
from .fusion_experiment import load_prepared,apply_fold_data
from .fusion_model import load_transforms as old_transforms,load_network,predict,shuffled_targets
from .fusion_v3_model import load_transforms,fit_transform,apply,design,fit_readout,correct,positive_shift
from .fusion_v3_experiment import load_readout
from .eeg import equal_viewing_weights
from .worker_contract import file_sha256
from .edugage_targets import guard


def run(source,output):
    import torch
    torch.set_num_threads(1)
    source,output=Path(source),Path(output);guard(output)
    m=read_json(source/'manifest.json');protocol=read_json(source/'protocol.json')
    if not m['complete'] or len(m['folds'])!=8:raise ValueError('complete matrix required')
    for name,digest in m['code_sha256'].items():assert file_sha256(Path(__file__).parent/name)==digest
    meta,rows,grids,v2,folds=load_prepared(protocol['prepared']);old=Path(protocol['previous_evaluation'])
    assert file_sha256(Path(protocol['prepared'])/'manifest.json')==m['prepared_manifest_sha256']
    with np.load(source/'evaluation-rows.npz',allow_pickle=False) as exported:
        for key in exported.files:np.testing.assert_array_equal(exported[key],rows[key])
    seen=np.zeros(len(rows['target']),int);checks=[]
    for fold,record in zip(folds,m['folds']):
        guard(output);root=source/record['fold'];origin=old/record['fold'];tr,te=np.array(fold['train']),np.array(fold['test'])
        assert read_json(root/'scope.json')==fold
        for field in ['person','lesson','content_id']:assert not set(rows[field][tr])&set(rows[field][te])
        seen[te]+=1;fitted=load_transforms(root);scope=read_json(root/'transforms.json')['training_scope']
        _,first=np.unique(rows['content_id'][tr],return_index=True);unique=tr[first]
        assert scope['content_ids']==rows['content_id'][unique].tolist()
        assert scope['people']==np.unique(rows['person'][tr]).tolist()
        from .fusion_data import grid_support
        expected_bins={}
        for lesson in np.unique(rows['lesson'][tr]):
            q=np.unique(rows['query'][unique[rows['lesson'][unique]==lesson]])
            ix,good=grid_support(grids[lesson],q);assert good.all();expected_bins[lesson]=np.unique(ix).tolist()
        assert scope['grid_bins']==expected_bins
        for name in [*MODALITIES,*('aux_'+n for n in AUXILIARY)]:
            x=rows[name][unique] if name.startswith('aux_') else np.concatenate([grids[l][name][ix] for l,ix in expected_bins.items()])
            reference=fit_transform(x,protocol['preprocessing'])
            for key in fitted[name]:np.testing.assert_allclose(fitted[name][key],reference[key],rtol=1e-6,atol=1e-7)
        seq,aux=apply(rows,grids,fitted,protocol['preprocessing']['auxiliary_gain'])
        assert np.abs(seq).max()<=4 and np.abs(aux).max()<=1
        os,oa=apply_fold_data(rows,grids,old_transforms(origin));x=design(seq,aux);nx=design(seq,aux,True)
        w=equal_viewing_weights(rows['lesson'][tr],rows['person'][tr]);mean=w@rows['target'][tr]
        scale=np.sqrt(w@((rows['target'][tr]-mean)**2));scale[scale==0]=1
        with np.load(checked(root,'forecasts.npz',record['forecast_sha256'])) as saved,np.load(origin/'forecasts.npz') as legacy:
            np.testing.assert_array_equal(saved['row_indices'],te);np.testing.assert_array_equal(saved['training_scale'],scale)
            for key,oldkey in [('mean','mean'),('tribe_temporal','tribe_temporal'),('historical_v2_fusion','fusion')]:np.testing.assert_array_equal(saved[key],legacy[oldkey])
            for arm in ['robust_tribe','robust_fusion']:
                pred=[];mode='tribe_temporal' if arm=='robust_tribe' else 'fusion'
                for seed in protocol['seeds']:
                    path=root/arm/str(seed);r=read_json(path/'manifest.json')
                    assert r['seed']==seed and r['training']==v2['training']
                    value=predict(load_network(path),seq[te],aux[te],mode)
                    with np.load(path/'forecasts.npz') as z:np.testing.assert_array_equal(value,z['predictions']);np.testing.assert_array_equal(z['row_indices'],te)
                    pred.append(value)
                np.testing.assert_array_equal(saved[arm],np.mean(np.stack(pred).astype(float),axis=0))
            for arm in ['fusion','without_aux','shuffled_correction']:
                preds=[]
                for seed in protocol['seeds']:
                    ap=origin/'tribe_temporal'/str(seed);anchor=load_network(ap)
                    bt=predict(anchor,os[tr],oa[tr],'tribe_temporal');be=predict(anchor,os[te],oa[te],'tribe_temporal')
                    y=rows['target'][tr];permutation=np.arange(len(tr))
                    if arm=='shuffled_correction':y,permutation=shuffled_targets(y,rows['lesson'][tr],rows['person'][tr],seed)
                    path=root/arm/str(seed);rm,coef,rs=load_readout(path)
                    assert rm['training_scope']==fold and rm['anchor_manifest_sha256']==file_sha256(ap/'manifest.json')
                    np.testing.assert_array_equal(np.load(path/'permutation.npy'),permutation);np.testing.assert_array_equal(rs,scale)
                    features=nx if arm=='without_aux' else x
                    refit=fit_readout(features[tr],np.clip((y-bt)/scale,-3,3),w,10.)
                    np.testing.assert_allclose(coef,refit,rtol=1e-10,atol=1e-10)
                    value=correct(features[te],be,coef,scale,.1)
                    assert np.max(np.abs(value-be)/scale)<=.100000001 and (value[:,4:]>=0).all()
                    with np.load(path/'forecasts.npz') as z:np.testing.assert_array_equal(value,z['predictions']);np.testing.assert_array_equal(z['row_indices'],te)
                    preds.append(value)
                np.testing.assert_array_equal(saved[arm],np.mean(preds,axis=0))
            _,coef,rs=load_readout(root/'simple_ridge')
            np.testing.assert_allclose(coef,fit_readout(x[tr],(rows['target'][tr]-mean)/scale,w,10.),rtol=1e-10,atol=1e-10)
            pred=positive_shift(np.broadcast_to(mean,(len(te),8)).copy(),np.c_[np.ones(len(te)),x[te]]@coef,scale)
            np.testing.assert_array_equal(pred,saved['simple_ridge'])
            for arm in protocol['arms']:assert np.isfinite(saved[arm]).all()
        checks.append({'fold':record['fold'],'replayed':True,'training_only_transforms_refitted':True,
            'bounded_inputs':True,'maximum_normalized_correction':.1,'dropped_audio_feature_2':bool(not fitted['aux_audio_descriptors']['active'][2])})
        print(record['fold']+' audited',flush=True)
    assert (seen==1).all();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'audit.json',{'schema':'perception-revision-audit-v3','status':'passed','folds':checks,
        'evaluation_manifest_sha256':file_sha256(source/'manifest.json'),'rows':len(seen),
        'temporal_networks_replayed':48,'residual_readouts_refitted_and_replayed':72,'simple_readouts_refitted_and_replayed':8})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
