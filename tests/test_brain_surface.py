import json
import numpy as np
import pytest

from brain.surface import export, load_surface, VERTICES, SOURCE_REVISION
from brain.remote_inference import TRIBE_REVISION
from brain.worker_contract import file_sha256


def fixture(tmp_path):
    surface, native = tmp_path / "surface", tmp_path / "native"
    surface.mkdir(); native.mkdir()
    arrays = {}
    for hemi in ("left", "right"):
        for kind in ("pial", "infl"):
            arrays[f"{kind}_{hemi}_coordinates"] = np.zeros((VERTICES, 3), dtype=np.float32)
            arrays[f"{kind}_{hemi}_faces"] = np.array([[0, 1, VERTICES - 1]], dtype=np.int32)
    np.savez(surface / "surface.npz", **arrays)
    metadata = {"schema": "perception-template-surface-v1", "mesh": "fsaverage5",
        "vertices_per_hemisphere": VERTICES, "prediction_columns": 2 * VERTICES,
        "column_order": ["left", "right"], "upstream_git_revision": SOURCE_REVISION,
        "native_projector_index_tests_passed": True, "geometry_role": "synthetic contract test fixture",
        "arrays_sha256": file_sha256(surface / "surface.npz")}
    (surface / "manifest.json").write_text(json.dumps(metadata))
    values = np.arange(2 * 2 * VERTICES, dtype=np.float32).reshape(2, 2 * VERTICES)
    np.savez(native / "fmri.npz", predicted_cortical_response=values)
    (native / "manifest.json").write_text(json.dumps({"schema": "perception-native-tribe-v1",
        "checkpoint_revision": TRIBE_REVISION, "source": {"sha256": "a" * 64},
        "spatial_metadata": {"mesh": "fsaverage5"}, "shape": list(values.shape),
        "array_sha256": file_sha256(native / "fmri.npz"), "neuro_offset_seconds": 5,
        "segments": [{"start": t, "duration": 1, "timeline": "native"} for t in (3, 7)]}))
    return surface, native, values, arrays, metadata


def test_export_preserves_vertex_values_hemispheres_and_discontinuous_native_clock(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.surface.guard", lambda _: None)
    surface, native, values, _, _ = fixture(tmp_path)
    result = export(native, surface, tmp_path / "view")
    with np.load(tmp_path / "view/surface-prediction.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["predicted_cortical_response"], values)
        np.testing.assert_array_equal(data["segment_starts_native"], [3, 7])
    assert result["hemisphere_column_ranges_half_open"]["right"] == [VERTICES, 2 * VERTICES]
    assert not result["scientific_accuracy_validated"]
    assert not result["checkpoint_anatomical_correspondence_independently_validated"]


def test_rejects_changed_geometry_or_inconsistent_inflated_topology(tmp_path):
    surface, _, _, arrays, metadata = fixture(tmp_path)
    arrays["infl_left_faces"] = np.array([[0, 2, VERTICES - 1]], dtype=np.int32)
    np.savez(surface / "surface.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_surface(surface)
    metadata["arrays_sha256"] = file_sha256(surface / "surface.npz")
    (surface / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="topology"):
        load_surface(surface)
