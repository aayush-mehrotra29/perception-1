import numpy as np
from speech_eeg.cohort_features import fit_pca,project,bin_history


def test_training_pca_and_no_future_bin_or_padding_assignment():
    rng=np.random.default_rng(4);a=rng.normal(size=(80,40));b=rng.normal(size=(90,40));state=fit_pca([a,b])
    before={k:v.copy() for k,v in state.items()};project(np.ones((20,40))*99999,state)
    for k in state:np.testing.assert_array_equal(before[k],state[k])
    z=np.repeat(np.arange(200)[:,None],16,axis=1)
    h,m=bin_history(z,np.array([.98,1.,1.5,1.52,2.]),duration=2.)
    assert m.tolist()==[False,True,True,True,False]
    assert h[1,0,-1]==2 and h[2,0,-1]==3 and h[3,0,-1]==3
    assert h[1,0,0]==0 and np.isnan(h[0,0,0]) and np.isnan(h[-1,0,-1])
