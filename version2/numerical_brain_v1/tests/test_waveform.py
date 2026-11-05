import numpy as np
import pytest
from numerical_brain_v1.waveform import acoustic_points,base_history,feature_batches,native_points

def test_acoustics_match_audited_integer_operator():
    from av_eeg_encoding.features.alice_sensory import prepare_stimulus
    rng=np.random.default_rng(71);pcm=rng.integers(-31000,31000,size=44100*3,dtype=np.int16)
    q,a,p=acoustic_points(pcm);old=prepare_stimulus(pcm)
    np.testing.assert_array_equal(q,old.query_native_indices/500)
    np.testing.assert_array_equal(a[:,0],old.rms)
    np.testing.assert_array_equal(a[:,1],old.onset)
    np.testing.assert_array_equal(p,old.progress)

def test_exact_native_midpoint_lookup_preserves_chunk_gap():
    q=np.arange(1601)/50;a=np.ones((len(q),2))
    starts=np.r_[np.arange(1499)*320,480000+np.arange(99)*320]
    sp={'ticks':starts*4+798,'values':np.repeat((starts/16000)[:,None],32,axis=1)}
    ids=np.array([1499,1500,1501,1550,1551])
    history,mask,ticks=base_history(q,a,sp,ids)
    assert mask.tolist()==[True,False,False,False,True]
    assert history.shape==(5,34,51)
    assert np.isnan(history[1]).any()
    assert history[-1,2,-1]>history[-1,2,0]

def test_native_hidden_missing_is_rejected(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps({'input_modalities':['text','audio'],'hidden_features':[]}))
    with pytest.raises(ValueError,match='qualified native audio hidden'):native_points(tmp_path)

def test_reloaded_endpoint_matches_frozen_predictor():
    from numerical_brain_v1.waveform import EndpointPredictor,ROOT,load
    from speech_eeg.endpoint_inference import predict_features
    s=load(ROOT/'data/cohort-features/segment01.npz')
    f={k:v[:128] for k,v in s.items()}
    a,ma,_=EndpointPredictor(ROOT/'runs/endpoint-v1').predict(f)
    b,mb,_=predict_features(ROOT/'runs/endpoint-v1',f)
    np.testing.assert_array_equal(ma,mb)
    np.testing.assert_array_equal(a,b)
    c,mc,_=EndpointPredictor(ROOT/'runs/endpoint-v1').predict(f)
    np.testing.assert_array_equal(a,c)

def test_native_features_follow_parent_video_offset(tmp_path):
    q=np.arange(100)/50;ac=np.ones((100,2));prog=np.zeros((100,3))
    starts=np.arange(100)*320;sp={'ticks':starts*4+798,'values':np.ones((100,32))}
    native={b:np.arange(10,dtype=float)[:,None] for b in ('native_audio','native_text','tribe_hidden')}
    for b in native:np.savez(tmp_path/(b+'-pca.npz'),mean=np.array([0.]),scale=np.array([1.]),center=np.array([0.]),components=np.ones((1,1)))
    _,a=next(feature_batches(q,ac,prog,sp,native,np.arange(10)*250,tmp_path))
    _,b=next(feature_batches(q,ac,prog,sp,native,np.arange(10)*250,tmp_path,native_offset_seconds=.5))
    # Original acoustic/speech time unchanged; parent native bins move by one.
    np.testing.assert_array_equal(a['history'][:,:34],b['history'][:,:34])
    np.testing.assert_array_equal(b['history'][75,34:,-1],a['history'][75,34:,-1]+1)
