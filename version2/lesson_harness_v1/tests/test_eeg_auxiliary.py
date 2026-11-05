import numpy as np
import pytest
from lesson_harness_v1.eeg_auxiliary import prepare


def fixture():
    return {'point':np.arange(8),'target':np.arange(64,dtype=float).reshape(8,8),
            'person':np.array([1,1,1,1,2,2,2,2]),'lesson':np.array(['a']*4+['b']*4)}


def test_eeg_transform_uses_training_scope_only():
    data=fixture();a=prepare(data,[0,1,2,3]);data['target'][4:]=1e12;b=prepare(data,[0,1,2,3])
    np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_array_equal(a['target'],b['target'])
    np.testing.assert_allclose(a['weights']@a['target'],0,atol=1e-7)


def test_eeg_negative_control_stays_within_viewing_and_changes_assignment():
    data=fixture();result=prepare(data,np.arange(8),reassigned=True);p=result['reassigned_source_indices']
    assert np.all(p!=np.arange(8))
    np.testing.assert_array_equal(data['person'][p],data['person'])
    np.testing.assert_array_equal(data['lesson'][p],data['lesson'])
    np.testing.assert_array_equal(p,prepare(data,np.arange(8),reassigned=True)['reassigned_source_indices'])
    with pytest.raises(ValueError):prepare(data,[0],reassigned=True)
