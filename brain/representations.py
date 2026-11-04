"""Window frozen content and TRIBE representations without learning transforms.

Native TRIBE output is indexed by its stimulus-segment clock here, as a content
representation. This does not assert simultaneous EEG/fMRI physical acquisition.
"""
from pathlib import Path
import json

import numpy as np

from .content_windows import interval_means
from .worker_contract import file_sha256


def native_window_blocks(directory: Path, queries, media_duration: float, *, allow_unqualified_timing=False):
    if not np.isfinite(media_duration) or media_duration <= 0:
        raise ValueError("verified media duration required")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "perception-native-tribe-v1":
        raise ValueError("native TRIBE artifact required")
    records = manifest.get("dense_features", [])
    if not records:
        raise ValueError("missing native encoder controls")
    if not allow_unqualified_timing and not all(r["timing_qualified_against_extractor"] for r in records):
        raise ValueError("native feature timing has not been qualified")
    pieces = {name: [] for name in ("text", "audio", "video")}
    for record in records:
        path = directory / record["file"]
        if path.parent != directory or file_sha256(path) != record["sha256"]:
            raise ValueError("dense source identity mismatch")
        rate = record["frequency_hz"]
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("invalid native feature frequency")
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                if name not in pieces:
                    raise ValueError("unexpected dense modality")
                values = archive[name]
                if (list(values.shape) != record["shapes"][name] or values.ndim not in {3, 4}
                        or values.shape[0] != len(record["segments"]) or not np.isfinite(values).all()):
                    raise ValueError("invalid native dense tensor")
                for i, segment in enumerate(record["segments"]):
                    n = values.shape[-1]
                    if abs(n - segment["duration"] * rate) > 1e-6:
                        raise ValueError("native sample count/clock mismatch")
                    starts = segment["start"] + np.arange(n) / rate
                    stops = starts + 1 / rate
                    # Preserve all selected layers; reduction happens inside each
                    # training fold, never in this extraction function.
                    matrix = values[i].reshape(-1, n).T
                    selected = (starts >= 0) & (starts < media_duration)
                    pieces[name].append((starts[selected], np.minimum(stops[selected], media_duration), matrix[selected]))
    blocks = {}
    for name, chunks in pieces.items():
        if not chunks:
            continue  # Audio-only transfer must not invent a video representation.
        starts, stops, values = (np.concatenate([c[i] for c in chunks]) for i in range(3))
        order = np.argsort(starts, kind="stable")
        blocks[f"dense_{name}"], _ = interval_means(starts[order], stops[order], values[order],
            np.ones_like(values[order], dtype=bool), queries, minimum_coverage=1 - 1e-10)
    fmri_path = directory / "fmri.npz"
    if file_sha256(fmri_path) != manifest["array_sha256"]:
        raise ValueError("fMRI source identity mismatch")
    with np.load(fmri_path, allow_pickle=False) as archive:
        fmri = archive["predicted_cortical_response"]
    if list(fmri.shape) != manifest["shape"] or len(fmri) != len(manifest["segments"]) or not np.isfinite(fmri).all():
        raise ValueError("invalid native fMRI array")
    starts = np.array([s["start"] for s in manifest["segments"]])
    stops = starts + np.array([s["duration"] for s in manifest["segments"]])
    selected = (starts >= 0) & (starts < media_duration)
    blocks["predicted_tribe"], _ = interval_means(starts[selected], np.minimum(stops[selected], media_duration),
        fmri[selected], np.ones_like(fmri[selected], dtype=bool), queries, minimum_coverage=1 - 1e-10)
    return blocks, {"schema": "perception-native-window-representations-v1",
                    "source_sha256": manifest["source"]["sha256"],
                    "upstream_manifest_sha256": file_sha256(directory / "manifest.json"),
                    "timing_qualified": all(r["timing_qualified_against_extractor"] for r in records),
                    "window_seconds": 44, "layer_reduction_applied": False,
                    "fitted_transforms_applied": False,
                    "tribe_clock": "native stimulus-segment index; native neuro lag retained in model",
                    "missing_dense_modalities": sorted(set(pieces) - {k.removeprefix("dense_") for k in blocks if k.startswith("dense_")})}
