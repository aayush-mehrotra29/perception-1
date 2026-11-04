import json
import numpy as np
import pytest

from brain.representations import native_window_blocks
from brain.worker_contract import file_sha256


def test_native_pooling_keeps_layers_and_preserves_tail_and_start_missingness(tmp_path):
    # A padded 100s native segment, but only 60s of actual media.
    value = np.broadcast_to(np.arange(6).reshape(1, 2, 3, 1), (1, 2, 3, 200)).copy().astype(float)
    np.savez(tmp_path / "dense-0000.npz", text=value)
    np.savez(tmp_path / "fmri.npz", predicted_cortical_response=np.ones((100, 4)))
    m = {"schema": "perception-native-tribe-v1", "source": {"sha256": "a" * 64},
         "shape": [100, 4], "array_sha256": file_sha256(tmp_path / "fmri.npz"),
         "segments": [{"start": i, "duration": 1, "timeline": "default"} for i in range(100)],
         "dense_features": [{"file": "dense-0000.npz", "sha256": file_sha256(tmp_path / "dense-0000.npz"),
             "frequency_hz": 2, "shapes": {"text": [1, 2, 3, 200]},
             "segments": [{"start": 0, "duration": 100, "timeline": "default"}],
             "timing_qualified_against_extractor": False}]}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="not been qualified"):
        native_window_blocks(tmp_path, [0, 45, 65], 60)
    blocks, meta = native_window_blocks(tmp_path, [0, 45, 65], 60, allow_unqualified_timing=True)
    np.testing.assert_allclose(blocks["dense_text"][1], np.arange(6))
    assert np.isnan(blocks["dense_text"][[0, 2]]).all()
    assert np.isnan(blocks["predicted_tribe"][[0, 2]]).all()
    assert meta["missing_dense_modalities"] == ["audio", "video"]
    assert not meta["fitted_transforms_applied"]
