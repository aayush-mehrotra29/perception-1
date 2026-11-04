"""Check a complete real TRIBE artifact before accepting it for EEG training.

This verifies file integrity, coverage, tensor layout and recorded native replay.
It does not validate the model against measured participant fMRI or EEG.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .auxiliary_windows import authenticated_child
from .edugage_targets import guard
from .representations import native_window_blocks
from .remote_inference import TRIBE_REVISION
from .worker_contract import PublicMediaRequest, file_sha256


def qualify(directory: Path, request: PublicMediaRequest, media_duration: float, output: Path):
    guard(output)
    if output.exists():
        raise ValueError("qualification output exists")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "perception-native-tribe-v1" or manifest["source"] != request.model_dump()
            or manifest["checkpoint_revision"] != TRIBE_REVISION
            or manifest["checkpoint_sha256"] != "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"
            or not manifest.get("capture_equivalence_verified")
            or not manifest.get("content_encoder_revisions_verified")):
        raise ValueError("authenticated native inference and replay evidence required")
    audit = manifest["event_preprocessing_audit"]
    if (not audit["no_duplicate_words"] or not audit["words_within_media_extent"]
            or audit["word_count"] <= 0 or abs(audit["media_stop"] - media_duration) > 0.2):
        raise ValueError("native word/media event audit failed")
    segments = manifest["segments"]
    starts = np.array([r["start"] for r in segments])
    durations = np.array([r["duration"] for r in segments])
    expected_count = int(np.ceil(audit["media_stop"]))
    if (not np.array_equal(starts, np.arange(expected_count)) or not np.all(durations == 1)
            or len({r["timeline"] for r in segments}) != 1):
        raise ValueError("native per-second cortical output has gaps, overlap or unexpected clock")
    path = authenticated_child(directory, "fmri.npz", manifest["array_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        fmri = archive["predicted_cortical_response"]
    if (fmri.shape != (expected_count, 20484) or list(fmri.shape) != manifest["shape"]
            or manifest["spatial_metadata"]["mesh"] != "fsaverage5" or not np.isfinite(fmri).all()):
        raise ValueError("invalid native cortical output")
    query = np.arange(0, media_duration + 1e-8, 5.0)
    blocks, metadata = native_window_blocks(directory, query, media_duration, allow_unqualified_timing=True)
    if set(blocks) != {"dense_text", "dense_audio", "dense_video", "predicted_tribe"}:
        raise ValueError("complete native multimodal controls required")
    supports = {name: np.isfinite(block).all(axis=1) for name, block in blocks.items()}
    if not all(np.array_equal(mask, query >= 44) for mask in supports.values()):
        raise ValueError("native representations do not cover every complete analysis window")
    result = {"schema": "perception-real-native-artifact-qualification-v1",
        "source_sha256": request.sha256, "native_manifest_sha256": file_sha256(manifest_path),
        "integrity_and_full_coverage_passed": True, "fmri_shape": list(fmri.shape),
        "cortical_value_range": [float(fmri.min()), float(fmri.max())],
        "blocks": {k: list(v.shape) for k, v in blocks.items()},
        "supported_queries_per_block": {k: int(v.sum()) for k, v in supports.items()},
        "capture_replay_max_absolute_error": manifest["capture_max_absolute_error"],
        "clock": "native stimulus-segment bins, whole-lesson offline inference; not physical EEG acquisition time",
        "native_neuro_offset_seconds": manifest["neuro_offset_seconds"],
        "cortical_layout_source_review": "pinned TribeSurfaceProjector.apply stacks left then right; 10242 vertices per hemisphere",
        "mesh_geometry_independently_verified": False,
        "feature_timing_independently_qualified": metadata["timing_qualified"],
        "historical_media_identity_established": False,
        "measured_neural_accuracy_validated": False, "attention_or_learning_validated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("media_duration", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.directory, PublicMediaRequest.model_validate_json(args.request.read_text()),
                             args.media_duration, args.output)))
