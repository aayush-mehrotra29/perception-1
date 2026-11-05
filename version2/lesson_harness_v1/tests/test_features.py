import numpy as np
import pytest

from lesson_harness_v1.features import fit_transforms, transform, save_transforms, load_transforms, WIDTH


def test_held_out_features_cannot_influence_transform_and_masks(tmp_path):
    rng=np.random.default_rng(17)
    blocks={'audio':rng.normal(size=(12,20)),'text':rng.normal(size=(12,4))}
    train=np.arange(8)
    fitted=fit_transforms(blocks,train)
    changed={n:x.copy() for n,x in blocks.items()}
    for x in changed.values():x[8:]=1e8
    other=fit_transforms(changed,train)
    for n,state in fitted.items():
        for k,v in state.items():np.testing.assert_array_equal(v,other[n][k])
    assert fitted['audio']['components'].shape==(7,20)
    x=transform(blocks,fitted)
    assert x.shape==(12,WIDTH) and np.isfinite(x).all()
    save_transforms(fitted,tmp_path/'transforms',train)
    np.testing.assert_array_equal(x,transform(blocks,load_transforms(tmp_path/'transforms')))
    changed['audio'][9]=np.nan
    y=transform(changed,fitted)
    assert y[9,33]==0 and y[9,16]==1
    np.testing.assert_array_equal(y[9,17:34],0)


def test_ablated_modalities_have_zero_values_and_zero_masks():
    rng=np.random.default_rng(29)
    blocks={'text':rng.normal(size=(4,3)),'tribe':rng.normal(size=(4,2))}
    fitted=fit_transforms(blocks,np.arange(3))
    full=transform(blocks,fitted);ablated=transform(blocks,fitted,include=['text'])
    assert (full[:,67]==1).all()
    np.testing.assert_array_equal(ablated[:,51:68],0)
    np.testing.assert_array_equal(full[:,:17],ablated[:,:17])


def test_missing_training_block_not_fitted_from_test():
    blocks={'text':np.arange(12,dtype=float).reshape(4,3),'audio':np.ones((4,2))}
    blocks['audio'][:3]=np.nan
    fitted=fit_transforms(blocks,np.arange(3))
    assert 'audio' not in fitted
    assert transform(blocks,fitted)[3,33]==0
    with pytest.raises(ValueError):fit_transforms(blocks,np.array([0,0]))
