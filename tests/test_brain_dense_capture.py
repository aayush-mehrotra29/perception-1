from types import SimpleNamespace
import numpy as np
import pytest

from brain.dense_capture import DenseFeatureCapture


class TensorFixture:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def test_capture_preserves_native_features_without_mutating_or_inventing_modalities(tmp_path):
    array = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    original = array.copy()
    batch = SimpleNamespace(segments=[SimpleNamespace(start=10, duration=2, timeline="lesson")],
                            data={"text": TensorFixture(array)})
    capture = DenseFeatureCapture(tmp_path, 2)
    assert capture(None, (batch,)) is None
    np.testing.assert_array_equal(array, original)
    with np.load(tmp_path / "dense-0000.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["text"], original)
        assert data.files == ["text"]
    assert capture.records[0]["missing_modalities"] == ["audio", "video"]
    assert capture.records[0]["brain_trained_transform_applied"] is False


def test_capture_rejects_native_time_mismatch_and_large_outputs(tmp_path):
    batch = SimpleNamespace(segments=[SimpleNamespace(start=0, duration=3, timeline="lesson")],
                            data={"video": TensorFixture(np.ones((1, 2, 4)))})
    with pytest.raises(ValueError, match="clock"):
        DenseFeatureCapture(tmp_path, 2).capture(batch)
    batch.segments[0].duration = 2
    with pytest.raises(ValueError, match="bound"):
        DenseFeatureCapture(tmp_path, 2, maximum_bytes=1).capture(batch)
    assert list(tmp_path.iterdir()) == []
