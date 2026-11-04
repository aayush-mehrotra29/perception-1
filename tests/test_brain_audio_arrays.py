import numpy as np
import pytest
from brain.audio_arrays import mono_array


class TensorFixture:
    def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def test_native_list_and_tensor_waveforms_preserve_samples():
    expected = np.arange(8, dtype=np.float32)[None, :]
    for value in (TensorFixture(expected), [TensorFixture(expected[0])]):
        np.testing.assert_array_equal(mono_array(value), expected)
    for value in ([TensorFixture(expected)] * 2, TensorFixture(np.ones((2, 8))), TensorFixture([np.nan])):
        with pytest.raises(ValueError): mono_array(value)
