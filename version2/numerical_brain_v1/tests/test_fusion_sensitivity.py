import numpy as np
from numerical_brain_v1.fusion_sensitivity import intervene


def test_fixed_numeric_intervention_retains_availability_and_other_inputs():
    x=np.ones((3,204),np.float32);x[:,:186]=2
    out=intervene(x,170,'zero_numeric_coordinates_keep_mask')
    assert not out[:,170:186].any()
    np.testing.assert_array_equal(out[:,186:],x[:,186:])
    np.testing.assert_array_equal(out[:,:170],x[:,:170])
    np.testing.assert_array_equal(x[:,170:186],2)
    masked=intervene(x,170,'mask_entire_block')
    assert not masked[:,170:187].any()
