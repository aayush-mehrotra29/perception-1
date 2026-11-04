import numpy as np
from speech_eeg.cohort_evaluate import macro,improvement
from speech_eeg.cohort_data import score_masked


def test_equal_clip_channel_aggregation_not_independent_frame_scoring():
    short={'a':{'MSE':1.,'MAE':.4,'macro_r':.1},'b':{'MSE':2.,'MAE':.8,'macro_r':None}}
    long={'a':{'MSE':3.,'MAE':.6,'macro_r':.3},'b':{'MSE':4.,'MAE':1.2,'macro_r':None}}
    m=macro({'short':short,'long':long})
    assert m['a']['MSE']==2. and m['b']['MSE']==3.
    np.testing.assert_allclose(m['a']['macro_r'],.2)
    assert m['b']['macro_r'] is None
    assert improvement(1.9,2.)>4.99


def test_bad_channels_excluded_identically_and_correlation_signed():
    x=np.arange(10,dtype=float);y=np.c_[x,x*1e6,x];pred=np.c_[-x,np.zeros(10),x]
    mask=np.ones_like(y,bool);mask[:,1]=False
    s=score_masked(y,pred,mask)
    np.testing.assert_allclose(s['per_channel_r'][0],-1.)
    np.testing.assert_allclose(s['per_channel_r'][2],1.)
    np.testing.assert_allclose(s['MSE'],2*np.mean(x*x))
    assert s['rows_per_channel']==[10,0,10] and s['per_channel_r'][1] is None
