import json

import numpy as np
import pytest

from brain.eeg import EEGModel
from brain.transfer import predict, select_blocks
from brain.worker_contract import file_sha256


def test_mode_selection_never_fills_absent_modalities():
    blocks = {"dense_text": np.ones((2, 3)), "predicted_tribe": np.ones((2, 4))}
    assert set(select_blocks(blocks, "text")) == {"dense_text"}
    with pytest.raises(ValueError, match="missing native"):
        select_blocks(blocks, "audio")


def test_real_numeric_transfer_head_bundle_and_cross_modality_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.transfer.guard", lambda _: None)
    native, fitted = tmp_path / "native", tmp_path / "model"
    native.mkdir()
    content = np.arange(12, dtype=float).reshape(4, 3)
    target = np.arange(32, dtype=float).reshape(4, 8)
    model = EEGModel.fit({"dense_text": content}, target, list("abcd"), np.array(["lesson"] * 4), np.ones(4))
    model.save(fitted)
    (fitted / "transfer.json").write_text(json.dumps({"schema": "perception-transfer-head-v1", "mode": "text",
        "blocks": ["dense_text"], "model_manifest_sha256": file_sha256(fitted / "manifest.json"),
        "claim_limit": "Synthetic numeric test; not measured neural accuracy."}))
    value = np.broadcast_to(content[0].reshape(1, 1, 3, 1), (1, 1, 3, 200)).copy()
    np.savez(native / "dense-0000.npz", text=value)
    np.savez(native / "fmri.npz", predicted_cortical_response=np.ones((60, 4)))
    manifest = {"schema": "perception-native-tribe-v1", "source": {"sha256": "a" * 64},
        "input_modalities": ["text"], "assumed_presentation_schedule": {
            "duration_seconds": 60, "clock_basis": "assumed_constant_reading_pace"},
        "checkpoint_revision": "synthetic", "shape": [60, 4], "array_sha256": file_sha256(native / "fmri.npz"),
        "segments": [{"start": i, "duration": 1, "timeline": "default"} for i in range(60)],
        "dense_features": [{"file": "dense-0000.npz", "sha256": file_sha256(native / "dense-0000.npz"),
            "frequency_hz": 2, "shapes": {"text": [1, 1, 3, 200]},
            "segments": [{"start": 0, "duration": 100, "timeline": "default"}],
            "timing_qualified_against_extractor": False}]}
    (native / "manifest.json").write_text(json.dumps(manifest))
    presentation = {"input_format": "text", "schedule_basis": "assumed 180 words/minute",
                    "transfer_status": "experimental_transfer"}
    result = predict(native, fitted, tmp_path / "result", mode="text", duration=60, presentation=presentation)
    assert result["eeg"]["supported_rows"] == 4
    assert result["observed_participant_measurements"] is False
    with np.load(tmp_path / "result/predictions.npz") as values:
        supported = values["eeg_supported"]
        assert np.isnan(values["eeg_features"][~supported]).all()
        expected = model.predict({"dense_text": np.repeat(content[:1], 4, axis=0)})
        np.testing.assert_allclose(values["eeg_features"][supported], expected)
    manifest["input_modalities"] = ["text", "audio", "video"]
    (native / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different modalities"):
        predict(native, fitted, tmp_path / "wrong", mode="text", duration=60, presentation=presentation)
    assert not (tmp_path / "wrong").exists()
