import numpy as np
import pytest
from lesson_harness_v1.eeg_quality import contact_support,event_mask,flatline_mask,read_markers


def test_contact_requires_two_good_packets_and_does_not_bridge_gap():
    ct=np.array([0,100,200,1000,1100]);codes=np.ones((5,4));codes[1,0]=2
    times=np.array([-1,0,50,100,150,200,500,1000,1050,1100])
    assert contact_support(times,ct,codes).tolist()==[False,False,False,False,False,False,False,True,True,False]
    codes[1,0]=3
    with pytest.raises(ValueError,match='unverified'):contact_support(times,ct,codes)


def test_movement_windows_are_explicit_and_inclusive():
    assert event_mask([499,500,1000,1500,1501],[1000],500).tolist()==[False,True,True,True,False]
    assert not event_mask([1,2],[],500).any()


def test_flatline_does_not_join_unobserved_original_rows():
    values=np.repeat(np.arange(8,dtype=float)[:,None],4,axis=1);values[:,0]=10
    assert flatline_mask(values,np.arange(8),4).all()
    rows=np.array([0,1,2,10,11,12,20,21])
    assert not flatline_mask(values,rows,4).any()


def test_conflicting_simultaneous_contact_packets_keep_worse_state(tmp_path):
    p=tmp_path/'markers.csv';p.write_text('timestamp,marker_type\n0,"horseshoe_(1,1,1,1)"\n0,"horseshoe_(2,1,1,1)"\n100,"horseshoe_(1,1,1,1)"\n150,blink\n')
    times,codes,events=read_markers(p)
    assert times.tolist()==[0,100] and codes[0].tolist()==[2,1,1,1] and events['blink']==[150]
