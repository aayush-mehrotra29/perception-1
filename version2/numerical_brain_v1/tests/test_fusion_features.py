import numpy as np
import json
import pytest
from numerical_brain_v1 import fusion_features as f
from numerical_brain_v1.fusion_benchmark import reassign


def test_eeg_stats_use_valid_samples_and_never_bridge_gap():
    t=np.arange(2200)/50;v=np.ones((2200,40))*1e-6;valid=np.ones(2200,bool)
    valid[1000:1100]=False;v[1100:]=1000e-6
    out,audit=f.eeg_windows(t,v,valid,[44],return_audit=True)
    assert np.isfinite(out).all() and audit[0]['valid_samples']==2100
    np.testing.assert_allclose(out[0,:40],(1000+1100*1000)/2100)
    np.testing.assert_allclose(out[0,80:],0,atol=1e-7)
    valid[:221]=False
    assert np.isnan(f.eeg_windows(t,v,valid,[44])).all()


def test_eeg_alignment_and_channels_fail_closed():
    t=np.arange(2200)/50;v=np.ones((2200,40));valid=np.ones(2200,bool)
    assert np.isnan(f.eeg_windows(t,v,valid,[43])).all()
    with pytest.raises(ValueError):f.eeg_windows(t,v[:,:39],valid,[44])
    t[100]+=.001
    with pytest.raises(ValueError,match='regular'):f.eeg_windows(t,v,valid,[44])


def test_fractional_queries_keep_the_same_44s_interval():
    times=np.arange(4000)/50.;values=np.broadcast_to(times[:,None],(len(times),40))*1e-6
    support=np.ones(len(times),bool)
    queries=[44.02,57.151,57.15,57.4,58.001]
    out,audit=f.eeg_windows(times,values,support,queries,return_audit=True)
    assert np.isfinite(out).all()
    for query,row,stats in zip(queries,audit,out):
        assert row['query_seconds']==query and row['context_start']==query-44
        assert row['valid_samples']==2200 and row['supported']
        assert query-44-1e-8<=row['first_sample_seconds']<query-44+.02+1e-8
        assert query-.02-1e-8<=row['last_sample_seconds']<query+1e-8
        selected=times[row['sample_start_index']:row['sample_stop_index']]
        np.testing.assert_allclose(stats[:40],selected.mean(),rtol=1e-6)


def test_offset_grid_and_eof_context_boundaries():
    times=np.arange(4000)/50.+.011;values=np.ones((4000,40))*1e-6;valid=np.ones(4000,bool)
    out,audit=f.eeg_windows(times,values,valid,[57.151,44.,80.1],return_audit=True)
    assert audit[0]['supported'] and audit[0]['valid_samples']==2200
    assert np.isfinite(out[0]).all()
    assert not audit[1]['supported'] and audit[1]['reason']=='context_outside_prediction_clock'
    assert not audit[2]['supported'] and np.isnan(out[2]).all()


def test_measured_eeg_cannot_enter_predicted_feature_adapter(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'complete':True,'source_sha256':'a'*64,'schema':'measured-eeg',
        'unit':'V','rate_hz':50,'filter_hz':[.5,8.]}))
    with pytest.raises(ValueError,match='content identity'):
        f.neural_blocks(tmp_path,tmp_path,'a'*64,[44])


def test_cortex_is_duration_weighted_and_full_coverage_required():
    values=np.stack([np.ones(f.CORTEX_WIDTH),np.ones(f.CORTEX_WIDTH)*3])
    out=f.cortical_windows([0,30],[30,44],values,[44])
    np.testing.assert_allclose(out,(30+14*3)/44)
    assert np.isnan(f.cortical_windows([0,31],[30,44],values,[44])).all()
    values[1,5]=np.nan
    assert np.isnan(f.cortical_windows([0,30],[30,44],values,[44])).all()


def test_204_slots_train_only_pca_and_explicit_missing_mask(tmp_path):
    rng=np.random.default_rng(4)
    blocks={'text':rng.normal(size=(8,3)),'predicted_eeg':rng.normal(size=(8,120)),
            'predicted_cortex':rng.normal(size=(8,200))}
    train=np.arange(5);states=f.fit_transforms(blocks,train)
    changed={k:v.copy() for k,v in blocks.items()};changed['predicted_cortex'][5:]=1e6
    other=f.fit_transforms(changed,train)
    for name,state in states.items():
        for key,value in state.items():np.testing.assert_array_equal(value,other[name][key])
    blocks['predicted_eeg'][6]=np.nan;x=f.transform(blocks,states)
    assert x.shape==(8,204) and x[6,186]==0 and x[5,186]==1
    assert np.isfinite(x).all()
    f.save_transforms(states,tmp_path/'transform',train);loaded=f.load_transforms(tmp_path/'transform')
    np.testing.assert_array_equal(x,f.transform(blocks,loaded))


def test_reassignment_only_changes_training_eeg_within_lesson():
    x=np.zeros((6,204),np.float32);slot=f.BLOCKS.index('predicted_eeg')*17
    x[:,slot]=np.arange(6);x[:,slot+16]=1
    windows=[{'lesson':'a' if i<3 else 'b'} for i in range(6)]
    out,audit=reassign(x,[0,1,3,4],windows)
    np.testing.assert_array_equal(out[[2,5]],x[[2,5]])
    assert set(out[[0,1],slot])=={0,1} and set(out[[3,4],slot])=={3,4}
    assert not (out[[0,1,3,4],slot]==x[[0,1,3,4],slot]).any()
