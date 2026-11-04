"""Assemble frozen content representations; no participant inputs or fitted PCA."""
import argparse
import json
from pathlib import Path

import numpy as np

from .auxiliary_windows import semantic_windows, sam_visual_windows, sam_audio_windows
from .edugage_targets import guard
from .local_baseline import load_sensory, SOURCES
from .representations import native_window_blocks
from .worker_contract import file_sha256


def assemble(spec: dict, queries, output: Path, *, allow_prototype_timing=False):
    """Spec references completed local artifacts for a single authorized lesson."""
    if spec.get("schema") != "perception-lesson-feature-sources-v1":
        raise ValueError("explicit lesson feature source specification required")
    queries = np.asarray(queries, dtype=np.float64)
    if queries.ndim != 1 or not len(queries) or not np.isfinite(queries).all() or np.any(np.diff(queries) <= 0):
        raise ValueError("strictly ordered unique content queries required")
    sha = spec["source_sha256"]
    blocks, native = native_window_blocks(Path(spec["tribe"]), queries, spec["media_duration"],
                                          allow_unqualified_timing=allow_prototype_timing)
    if native["source_sha256"] != sha or set(blocks) != {"dense_text", "dense_audio", "dense_video", "predicted_tribe"}:
        raise ValueError("all native video representations from the same source required")
    names = {}
    blocks["semantics"], names["semantics"] = semantic_windows(Path(spec["semantics"]), queries, sha)
    blocks["sam_visual"], names["sam_visual"] = sam_visual_windows(Path(spec["sam_visual"]), Path(spec["frames"]), queries, sha)
    blocks["sam_audio"], names["sam_audio"] = sam_audio_windows(Path(spec["sam_audio"]), Path(spec["audio_derivative"]), queries, sha)
    if ("sensory_data_root" in spec) == ("sensory_directory" in spec):
        raise ValueError("exactly one authenticated sensory source is required")
    if "sensory_directory" in spec:
        from .sensory import window_blocks
        sensory, sensory_manifest = window_blocks(Path(spec["sensory_directory"]), queries, sha)
        if sensory_manifest["qualification_prefix_only"] or sensory_manifest["duration_seconds"] != spec["media_duration"]:
            raise ValueError("complete matching-duration sensory features required")
        blocks.update(sensory)
    if "sensory_data_root" in spec:
        # This compatibility adapter is only for the already audited EduGage
        # native measurements, never a substitute for an absent new upload.
        sensory, duration, _ = load_sensory(Path(spec["sensory_data_root"]), spec["lecture"], queries)
        summary = Path(spec["sensory_data_root"]) / "runs" / SOURCES[spec["lecture"]][1] / "summary.json"
        pins = json.loads(summary.read_text())["input_pins"]
        media_pins = [value for path, value in pins.items() if Path(path).suffix.lower() == ".mp4"]
        if duration != spec["media_duration"] or media_pins != [sha]:
            raise ValueError("sensory/native media identity or duration mismatch")
        blocks.update(sensory)
    blocks["progress"] = queries[:, None] / spec["media_duration"]
    supported = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in blocks.values()])
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "features.npz", query_seconds=queries, supported=supported,
             **{name: value.astype(np.float32) for name, value in blocks.items()})
    provenance = {}
    for branch in ("tribe", "semantics", "sam_visual", "frames", "sam_audio", "audio_derivative"):
        path = Path(spec[branch]) / "manifest.json"
        provenance[branch] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    sensory_path = (Path(spec["sensory_directory"]) / "manifest.json" if "sensory_directory" in spec else
        Path(spec["sensory_data_root"]) / "runs" / SOURCES[spec["lecture"]][1] / "summary.json")
    provenance["native_sensory"] = {"path": str(sensory_path.resolve()), "sha256": file_sha256(sensory_path)}
    manifest = {"schema": "perception-lesson-features-v1", "source_sha256": sha,
        "lecture": spec["lecture"], "media_duration": spec["media_duration"], "blocks": {k: v.shape[1] for k, v in blocks.items()},
        "feature_names": names, "source_manifests": provenance,
        "features_sha256": file_sha256(output / "features.npz"), "native_representation_metadata": native,
        "queries": len(queries), "supported": int(supported.sum()), "window_seconds": 44,
        "fitted_transforms_applied": False, "participant_data_used": False,
        "historical_stimulus_identity_established": False, "physical_EEG_synchronization_established": False,
        "prototype_timing_explicitly_allowed": allow_prototype_timing,
        "feature_clock": "native media position; sparse visual observations and coarse 30-second semantic bins",
        "scientific_validity_established": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-prototype-timing", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    queries = np.arange(0, spec["media_duration"] + 1e-8, 5.0)
    print(json.dumps(assemble(spec, queries, args.output, allow_prototype_timing=args.allow_prototype_timing)))
