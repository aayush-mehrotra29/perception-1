import numpy as np
from speech_eeg.cohort_data import design,design_indices,ridge_from_moments,score_masked
from speech_eeg.model import fit_scaling,feature_transform


def test_subset_label_ridge_matches_direct_weighted_fit_in_reference_units():
    rng=np.random.default_rng(5);x=rng.normal(size=(90,8));y=rng.normal(size=(90,3))*np.array([1,2,4])+np.array([5,-3,9])
    w=np.r_[np.full(30,1/60),np.full(60,1/120)];mx=w@x;my=w@y
    ref={'target_mean':np.array([-1,2,7]),'target_scale':np.array([3,4,2])}
    m=ridge_from_moments(mx,x.T@(w[:,None]*x),my,w@(y*y),x.T@(w[:,None]*y),ref,.1)
    a=x-mx;b=y-my
    direct=np.linalg.solve(a.T@(w[:,None]*a)+.1*np.eye(8),a.T@(w[:,None]*b))
    expected=((x-mx)@direct+my-ref['target_mean'])/ref['target_scale']
    np.testing.assert_allclose(x@m['coef']+m['intercept'],expected,atol=1e-12)


def test_slot_mapping_and_masked_metrics():
    rng=np.random.default_rng(9);x=rng.normal(size=(4,82,51));p=rng.normal(size=(4,3));slots=[0,1,66,81]
    np.testing.assert_array_equal(design(x,p,slots),design(x,p,list(range(82)))[:,design_indices(slots)])
    y=np.array([[1,2,np.nan],[2,3,np.nan],[3,4,np.nan]],float);pred=np.zeros_like(y);mask=np.isfinite(y)
    s=score_masked(y,pred,mask)
    assert s['evaluated_channels']==2 and s['macro_r'] is None
    assert s['per_channel_MSE'][2] is None
    np.testing.assert_allclose(s['MSE'],np.mean(y[:,:2]**2),atol=1e-14)


def test_held_feature_transform_never_refits_and_equal_clip_scaling():
    rng=np.random.default_rng(2);segments=[]
    for n in [9,31]:
        segments.append({'history':rng.normal(size=(n,3,51)), 'progress':rng.normal(size=(n,3)),
                         'target':rng.normal(size=(n,4)), 'supported':np.ones(n,bool)})
    state=fit_scaling(segments);copy={k:v.copy() for k,v in state.items()}
    np.testing.assert_allclose(state['target_mean'],np.mean([s['target'].mean(0) for s in segments],0))
    held={'history':np.full((3,3,51),1e6),'progress':np.full((3,3),1e6)}
    x,p=feature_transform(held,state,np.arange(3))
    assert np.max(x)<=5 and np.max(p)<=5
    for k in state:np.testing.assert_array_equal(state[k],copy[k])
