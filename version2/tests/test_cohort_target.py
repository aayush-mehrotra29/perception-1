import numpy as np
from speech_eeg.cohort_targets import kernel,filtered_target
from speech_eeg.common import read


def test_constant_rejection_clock_and_missing_context():
    c=read('configs/cohort_target_v1.json');h=kernel(c)
    assert abs(h.sum())<1e-12
    raw=np.ones((2,4000));raw[1,2000]=np.nan
    y,mask,q=filtered_target(raw,h)
    assert mask[:,0].sum()==200 and np.max(np.abs(y[mask[:,0],0]))<1e-12
    assert not mask[(q>=1000)&(q<=3000),1].any()
    assert np.isnan(y[~mask]).all()
    np.testing.assert_array_equal(q,np.arange(0,4000,10))
