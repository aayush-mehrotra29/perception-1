import numpy as np
import pytest
import torch
from speech_eeg.prepare import history_from_cache
from speech_eeg.model import fit_scaling,feature_transform,TemporalEEG,save_model,load_model,forecast
from speech_eeg.train import linear_design,scores


def test_history_order_and_missing_coverage_preserved():
    speech=np.tile(np.arange(51)[:,None],(1,32)).reshape(1,-1).astype(float)
    acoustic=np.r_[np.arange(51),100+np.arange(51)][None].astype(float)
    x=history_from_cache(speech,acoustic)
    np.testing.assert_array_equal(x[0,0],np.arange(51)[::-1])
    np.testing.assert_array_equal(x[0,1],100+np.arange(51)[::-1])
    np.testing.assert_array_equal(x[0,2],np.arange(51)[::-1])
    speech[0,0]=np.nan;x=history_from_cache(speech,acoustic)
    assert np.isnan(x[0,2,-1])
    with pytest.raises(ValueError):history_from_cache(speech[:,:32],acoustic)


def test_scaling_training_only_equal_segments_and_target_free_inference():
    rng=np.random.default_rng(8);data=[]
    for n,shift in [(10,0),(40,3)]:
        data.append({'history':rng.normal(size=(n,34,51))+shift,'progress':rng.normal(size=(n,3)),
            'target':rng.normal(size=(n,53))+shift,'supported':np.ones(n,bool)})
    a=fit_scaling(data)
    np.testing.assert_allclose(a['target_mean'],np.mean([s['target'].mean(0) for s in data],0))
    query={'history':np.ones((2,34,51))*1e6,'progress':np.ones((2,3))*1e6}
    x,p=feature_transform(query,a,np.arange(2))
    assert np.max(x)==5 and np.max(p)==5
    b=fit_scaling(data)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    data[0]['target'][:]=0;data[1]['target'][:]=0
    with pytest.raises(ValueError):fit_scaling(data)


def test_temporal_training_learns_and_reloads(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    arch={'width':8,'kernel':3,'dilations':[1,3,9]}
    model=TemporalEEG(2,2,arch);rng=np.random.default_rng(3)
    x=rng.normal(size=(32,2,51)).astype('float32');p=rng.normal(size=(32,3)).astype('float32')
    y=np.c_[p[:,0],p[:,1]].astype('float32');opt=torch.optim.AdamW(model.parameters(),lr=.02)
    initial=np.mean((forecast(model,x,p)-y)**2)
    for _ in range(100):
        opt.zero_grad();loss=((model(torch.from_numpy(x),torch.from_numpy(p))-torch.from_numpy(y))**2).mean()
        loss.backward();opt.step()
    pred=forecast(model,x,p);assert np.mean((pred-y)**2)<initial*.2
    save_model(model,tmp_path/'weights.npz');replay=load_model(tmp_path/'weights.npz',2,2,arch)
    np.testing.assert_array_equal(pred,forecast(replay,x,p))
    assert linear_design(x,p,False).shape==(32,105)


def test_metric_constant_prediction_not_spurious_correlation():
    y=np.arange(50,dtype=float).reshape(25,2)
    m=scores(y,np.zeros_like(y))
    assert m['macro_r'] is None and m['defined_correlations']==0
    assert scores(y,y)['MSE']==0
    assert scores(y,y)['macro_r']==pytest.approx(1.)
