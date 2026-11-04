import json
import numpy as np
import pytest

from brain.bundle import write_bundle
from brain.worker_contract import file_sha256


def inputs(tmp_path):
    fmri_dir = tmp_path / "fmri"
    fmri_dir.mkdir()
    np.savez(fmri_dir / "fmri.npz", predicted_cortical_response=np.ones((2, 3)))
    m = {"schema": "perception-native-tribe-v1", "source": {"sha256": "a" * 64},
         "array_sha256": file_sha256(fmri_dir / "fmri.npz"), "shape": [2, 3],
         "checkpoint_revision": "test", "segments": [
             {"start": 5, "duration": 1, "timeline": "native"},
             {"start": 7, "duration": 1, "timeline": "native"}]}
    (fmri_dir / "manifest.json").write_text(json.dumps(m))
    return dict(source_sha256="a" * 64, eeg_model_sha256="b" * 64,
                eeg_query_seconds=[0, 45, 50], eeg_features=np.ones((3, 8)),
                eeg_supported=np.array([False, True, False]), fmri_directory=fmri_dir,
                presentation={"input_format": "video", "schedule_basis": "native media timestamps",
                              "transfer_status": "video_training_domain"},
                evaluation_status="synthetic_contract_test")


def test_bundle_keeps_independent_native_clock_and_masks_unsupported_eeg(tmp_path):
    kwargs = inputs(tmp_path)
    output = tmp_path / "bundle"
    result = write_bundle(output, **kwargs)
    with np.load(output / "predictions.npz") as data:
        np.testing.assert_array_equal(data["fmri_segment_starts_native"], [5, 7])
        assert np.isnan(data["eeg_features"][[0, 2]]).all()
        assert np.isfinite(data["eeg_features"][1]).all()
    assert result["observed_participant_measurements"] is False
    assert result["attention_or_retention_predictions_included"] is False


def test_bundle_rejects_different_sources_and_unsupported_transfer_claim(tmp_path):
    kwargs = inputs(tmp_path)
    kwargs["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identity"):
        write_bundle(tmp_path / "wrong", **kwargs)
    kwargs["presentation"]["input_format"] = "pdf"
    with pytest.raises(ValueError, match="experimental transfer"):
        write_bundle(tmp_path / "wrong", **kwargs)


def test_json_export_retains_clocks_and_uses_null_for_unsupported_rows(tmp_path):
    from brain.fusion_export import export
    kwargs = inputs(tmp_path)
    bundle = tmp_path / "bundle"
    write_bundle(bundle, **kwargs)
    result = export(bundle, bundle / "app-result.json")
    assert result["eeg"]["timeline"][0]["features"] == [None] * 8
    assert result["eeg"]["timeline"][1]["features"] == [1.] * 8
    assert result["fmri"]["segment_starts_native"] == [5., 7.]
    parsed = json.loads((bundle / "app-result.json").read_text(),
                        parse_constant=lambda x: pytest.fail("Non-JSON numeric constant: " + x))
    assert parsed == result
