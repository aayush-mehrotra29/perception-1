import json
from pathlib import Path
import numpy as np
import pytest
from brain.fusion_v3_model import fit_transform,transform,save_transforms,load_transforms,fit_readout,correct,positive_shift,design

CONFIG=json.loads((Path(__file__).parents[1]/'configs/brain_fusion_v3.json').read_text())['preprocessing']

def test_near_constant_audio_is_dropped_and_unseen_values_stay_bounded(tmp_path):
    rng=np.random.default_rng(10)
    x=rng.normal(size=(100,24)); x[:,2]=rng.uniform(0,1e-5,100)
    fitted=fit_transform(x,CONFIG)
    assert not fitted['active'][2]
    assert fitted['axes'].shape[1]<=16
    held=x[:5].copy(); held[:,2]=.0037; held[:,3]=1e20
    out=transform(held,fitted)
    assert np.isfinite(out).all() and np.abs(out).max()<=4
    save_transforms(tmp_path,{'audio':fitted},{'training_only':True})
    np.testing.assert_array_equal(out,transform(held,load_transforms(tmp_path)['audio']))
    before={k:v.copy() for k,v in fitted.items()}
    transform(np.ones((3,24))*-1e20,fitted)
    for k in fitted: np.testing.assert_array_equal(before[k],fitted[k])

def test_residual_can_never_exceed_bound_or_make_variances_negative():
    rng=np.random.default_rng(1); x=rng.normal(size=(40,9)); anchor=rng.normal(size=(40,8)); anchor[:,4:]=np.exp(anchor[:,4:])
    scale=np.exp(rng.normal(size=8)); coef=rng.normal(size=(10,8))*10000
    pred=correct(x,anchor,coef,scale,.1)
    assert np.isfinite(pred).all() and (pred[:,4:]>0).all()
    assert np.max(np.abs(pred-anchor)/scale)<=.1000000001
    zero=correct(x,anchor,np.zeros_like(coef),scale,.1)
    np.testing.assert_allclose(zero,anchor,rtol=1e-12,atol=1e-12)

def test_design_missing_auxiliary_is_zero_and_linear_readout_learns():
    rng=np.random.default_rng(5); seq=rng.normal(size=(50,88,4,16)); aux=rng.normal(size=(50,96))
    x=design(seq,aux,True)
    assert not x[:,-96:].any() and x.shape==(50,288)
    y=np.tile(x[:,0,None],(1,8)); coef=fit_readout(x,y,np.ones(50)/50,regularization=.01)
    assert np.mean((np.c_[np.ones(50),x]@coef-y)**2)<np.mean(y*y)

def test_all_constant_features_and_missing_time_are_not_invented():
    fit=fit_transform(np.ones((30,5)),CONFIG)
    assert fit['axes'].shape==(5,0)
    test=np.ones((2,5)); test[1,0]=np.nan
    out=transform(test,fit)
    assert not out[0].any() and np.isnan(out[1]).all()


def test_v3_dispatch_rejects_wrong_presentation_before_loading_models(tmp_path,monkeypatch):
    from brain import fusion_inference
    model=tmp_path/'model';model.mkdir()
    (model/'manifest.json').write_text(json.dumps({'schema':'perception-fusion-deployment-v3'}))
    native=tmp_path/'native';native.mkdir()
    (native/'manifest.json').write_text(json.dumps({'assumed_presentation_schedule':{'input_format':'text','duration_seconds':65}}))
    monkeypatch.setattr(fusion_inference,'preflight',lambda _:pytest.fail('Model/runtime loading must follow clock validation'))
    with pytest.raises(ValueError,match='presentation mismatch'):
        fusion_inference.predict_native(native,model,tmp_path/'out',duration=60,input_format='text')
    assert not (tmp_path/'out').exists()
