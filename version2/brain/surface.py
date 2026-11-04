"""Package native cortical predictions with an audited generic display surface.

Template geometry and upstream projector indexing are not an independent
validation of checkpoint anatomy, individual anatomy, or neural accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .edugage_targets import guard
from .remote_inference import TRIBE_REVISION
from .worker_contract import file_sha256

VERTICES = 10242
SOURCE_REVISION = "af58661791a351a448a489042a28f6c37e1c14b7"


def load_surface(directory: Path):
    meta = json.loads((directory / "manifest.json").read_text())
    if (meta.get("schema") != "perception-template-surface-v1" or meta.get("mesh") != "fsaverage5"
            or meta.get("vertices_per_hemisphere") != VERTICES
            or meta.get("prediction_columns") != 2 * VERTICES
            or meta.get("column_order") != ["left", "right"]
            or meta.get("upstream_git_revision") != SOURCE_REVISION
            or meta.get("native_projector_index_tests_passed") is not True):
        raise ValueError("qualified native surface indexing required")
    path = directory / "surface.npz"
    if file_sha256(path) != meta["arrays_sha256"]:
        raise ValueError("surface checksum mismatch")
    expected = {f"{kind}_{hemi}_{field}" for kind in ("pial", "infl")
                for hemi in ("left", "right") for field in ("coordinates", "faces")}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("unexpected surface arrays")
        arrays = {key: archive[key] for key in archive.files}
    for hemi in ("left", "right"):
        for kind in ("pial", "infl"):
            coordinates, faces = (arrays[f"{kind}_{hemi}_{field}"] for field in ("coordinates", "faces"))
            if (coordinates.shape != (VERTICES, 3) or not np.isfinite(coordinates).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                    or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= VERTICES):
                raise ValueError("invalid template mesh")
        if not np.array_equal(arrays[f"pial_{hemi}_faces"], arrays[f"infl_{hemi}_faces"]):
            raise ValueError("pial/inflated mesh topology differs")
    return meta, arrays


def export(native: Path, surface: Path, output: Path):
    guard(output)
    geometry, arrays = load_surface(surface)
    meta = json.loads((native / "manifest.json").read_text())
    if (meta.get("schema") != "perception-native-tribe-v1"
            or meta.get("checkpoint_revision") != TRIBE_REVISION
            or meta.get("spatial_metadata", {}).get("mesh") != "fsaverage5"):
        raise ValueError("matching native fsaverage5 predictions required")
    path = native / "fmri.npz"
    if file_sha256(path) != meta["array_sha256"]:
        raise ValueError("native prediction checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["predicted_cortical_response"]
    segments = meta["segments"]
    starts = np.array([s["start"] for s in segments], dtype=float)
    durations = np.array([s["duration"] for s in segments], dtype=float)
    if (values.ndim != 2 or values.shape != (len(segments), 2 * VERTICES) or not len(values)
            or list(values.shape) != meta["shape"] or not np.isfinite(values).all()
            or not np.isfinite(starts).all() or not np.isfinite(durations).all()
            or np.any(durations <= 0) or len({s["timeline"] for s in segments}) != 1
            or np.any(starts[1:] < starts[:-1] + durations[:-1] - 1e-8)):
        raise ValueError("invalid cortical prediction shape or native clock")
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "surface-prediction.npz", **arrays, predicted_cortical_response=values,
             segment_starts_native=starts, segment_durations_native=durations)
    result = {"schema": "perception-surface-prediction-v1", "source_sha256": meta["source"]["sha256"],
        "native_manifest_sha256": file_sha256(native / "manifest.json"),
        "surface_manifest_sha256": file_sha256(surface / "manifest.json"),
        "arrays_sha256": file_sha256(output / "surface-prediction.npz"),
        "checkpoint_revision": TRIBE_REVISION, "mesh": "fsaverage5", "shape": list(values.shape),
        "hemisphere_column_ranges_half_open": {"left": [0, VERTICES], "right": [VERTICES, 2 * VERTICES]},
        "clock": "unchanged native TRIBE stimulus-segment clock",
        "native_neuro_offset_seconds": meta.get("neuro_offset_seconds"),
        "values": "unchanged native model predictions; no per-frame rescaling, thresholding or interpolation",
        "vertex_order_basis": "pinned upstream projector source and synthetic index qualification",
        "geometry_role": geometry["geometry_role"],
        "checkpoint_anatomical_correspondence_independently_validated": False,
        "anatomical_region_labels_included": False, "individual_anatomy": False,
        "observed_participant_measurements": False, "attention_or_retention_predictions": False,
        "scientific_accuracy_validated": False}
    guard(output)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("native", "surface", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.native, args.surface, args.output)))
