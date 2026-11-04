import json
import numpy as np
import pytest
from speech_eeg.inference import validate_features,json_array,verify_feature_source
from speech_eeg.common import sha


def test_unsupported_is_null_and_cannot_silently_fill_missing_modalities():
    s={'query_seconds':np.arange(3)/50,'history':np.ones((3,82,51)),
       'progress':np.ones((3,3)),'feature_supported':np.array([True,False,True])}
    s['history'][1]=np.nan
    np.testing.assert_array_equal(validate_features(s),s['feature_supported'])
    assert json.loads(json.dumps(json_array(np.array([[1.,np.nan],[np.nan,2.]])),allow_nan=False))==[[1.,None],[None,2.]]
    s['history'][0]=np.nan
    with pytest.raises(ValueError):validate_features(s)


def test_feature_source_pairing_is_enforced(tmp_path):
    p=tmp_path/'sample.npz';np.savez(p,example=np.arange(3))
    side=p.with_suffix('.provenance.json');side.write_text(json.dumps({'sha256':sha(p),'media_sha256':'correct'}))
    verify_feature_source(p,{'source':{'sha256':'correct'}})
    with pytest.raises(ValueError):verify_feature_source(p,{'source':{'sha256':'different'}})
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_feature_source(p,{'source':{'sha256':'correct'}})
