"""Authenticated, interval-covered 2 Hz content sequences; no EEG imputation."""
import json
from pathlib import Path

import numpy as np

from .edugage_targets import guard
from .worker_contract import file_sha256

MODALITIES = ("text", "audio", "video", "tribe")
AUXILIARY = ("audio_descriptors", "progress", "sam_audio", "sam_visual", "semantics", "visual_descriptors")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(directory, name, digest):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file() or file_sha256(path) != digest:
        raise ValueError("artifact path or checksum mismatch")
    return path


def native_intervals(directory, modality):
    directory = Path(directory)
    meta = read_json(directory / "manifest.json")
    if meta.get("schema") != "perception-native-tribe-v1":
        raise ValueError("native TRIBE source required")
    if modality == "tribe":
        path = checked(directory, "fmri.npz", meta["array_sha256"])
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["predicted_cortical_response"], dtype=np.float32)
        starts = np.array([s["start"] for s in meta["segments"]])
        stops = starts + np.array([s["duration"] for s in meta["segments"]])
        if list(values.shape) != meta["shape"]:
            raise ValueError("cortical array shape changed")
        return starts, stops, values
    chunks = []
    for record in meta.get("dense_features", []):
        path = checked(directory, record["file"], record["sha256"])
        with np.load(path, allow_pickle=False) as data:
            if modality not in data:
                continue
            values = data[modality]
            if list(values.shape) != record["shapes"][modality] or values.shape[0] != len(record["segments"]):
                raise ValueError("native encoder shape changed")
            rate = record["frequency_hz"]
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("invalid native rate")
            for i, segment in enumerate(record["segments"]):
                count = values.shape[-1]
                if abs(count - segment["duration"] * rate) > 1e-6:
                    raise ValueError("native interval count changed")
                starts = segment["start"] + np.arange(count) / rate
                chunks.append((starts, starts + 1 / rate, values[i].reshape(-1, count).T.copy()))
    if not chunks:
        return None
    starts, stops, values = (np.concatenate([c[i] for c in chunks]) for i in range(3))
    order = np.argsort(starts, kind="stable")
    return starts[order], stops[order], values[order].astype(np.float32, copy=False)


def interval_grid(starts, stops, values, duration, output=None):
    """Coverage-weighted representations; 1 Hz TRIBE intervals may span two bins.

    This is a content representation operation, not measured fMRI resampling.
    A genuine gap in a present input rejects its bin instead of being filled.
    """
    starts, stops, values = map(np.asarray, (starts, stops, values))
    if (starts.ndim != 1 or stops.shape != starts.shape or values.ndim != 2
            or len(values) != len(starts) or not len(starts)
            or not np.isfinite(starts).all() or not np.isfinite(stops).all()
            or not np.isfinite(values).all() or np.any(stops <= starts)
            or np.any(starts[1:] < stops[:-1] - 1e-8)
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("finite ordered nonoverlapping native intervals required")
    count = int(np.floor(duration * 2 + 1e-8))
    shape = (count, values.shape[1])
    result = (np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=shape)
              if output else np.empty(shape, dtype=np.float32))
    result[:] = np.nan
    coverage = np.zeros(count)
    for i in range(count):
        left, right = i / 2, (i + 1) / 2
        first = np.searchsorted(stops, left, side="right")
        last = np.searchsorted(starts, right, side="left")
        weights = np.maximum(0, np.minimum(stops[first:last], right) - np.maximum(starts[first:last], left))
        mass = weights.sum()
        coverage[i] = mass / 0.5
        if mass >= 0.5 - 1e-8 and mass <= 0.5 + 1e-8:
            result[i] = weights @ values[first:last] / mass
    if output:
        result.flush()
    return result, coverage


def prepare_native(directory, duration, destination):
    guard(destination)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    meta = read_json(Path(directory) / "manifest.json")
    if not meta.get("capture_equivalence_verified") or not meta.get("content_encoder_revisions_verified"):
        raise ValueError("qualified native capture and encoder revisions required")
    blocks = {}
    for modality in MODALITIES:
        guard(destination)
        intervals = native_intervals(directory, modality)
        if intervals is None:
            continue
        filename = modality + ".npy"
        values, coverage = interval_grid(*intervals, duration, destination / filename)
        supported = np.isfinite(values).all(axis=1)
        np.save(destination / (modality + "-coverage.npy"), coverage, allow_pickle=False)
        blocks[modality] = {"file": filename, "sha256": file_sha256(destination / filename),
            "shape": list(values.shape), "supported_bins": int(supported.sum()),
            "coverage_file": modality + "-coverage.npy",
            "coverage_sha256": file_sha256(destination / (modality + "-coverage.npy"))}
        del values, intervals
    if not blocks:
        raise ValueError("no native content modalities")
    report = {"schema": "perception-fusion-native-grid-v2", "blocks": blocks,
        "duration_seconds": duration, "frequency_hz": 2,
        "source_sha256": meta["source"]["sha256"],
        "native_manifest_sha256": file_sha256(Path(directory) / "manifest.json"),
        "native_directory": str(Path(directory).resolve()),
        "clock": "coverage-weighted content intervals; original cortical clock remains unchanged",
        "physical_EEG_alignment_qualified": False}
    write_json(destination / "manifest.json", report)
    return report


def load_grid(root):
    root = Path(root)
    meta = read_json(root / "manifest.json")
    if meta.get("schema") != "perception-fusion-native-grid-v2":
        raise ValueError("unsupported grid")
    blocks = {}
    for name, block in meta["blocks"].items():
        path = checked(root, block["file"], block["sha256"])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != block["shape"] or name not in MODALITIES:
            raise ValueError("grid shape or modality changed")
        blocks[name] = values
    return meta, blocks


def sequence_indices(queries, bins, steps=88):
    queries = np.asarray(queries, dtype=float)
    if queries.ndim != 1 or not np.isfinite(queries).all() or np.any(np.diff(queries) <= 0):
        raise ValueError("finite strictly ordered queries required")
    if not np.allclose(queries * 2, np.rint(queries * 2), atol=1e-8, rtol=0):
        raise ValueError("query outside 2 Hz grid")
    ends = np.rint(queries * 2).astype(int)
    indices = ends[:, None] - steps + np.arange(steps)
    valid = (ends >= steps) & (ends <= bins)
    return indices, valid


def grid_support(blocks, queries):
    count = min(len(value) for value in blocks.values())
    indices, valid = sequence_indices(queries, count)
    for value in blocks.values():
        observed = np.isfinite(value).all(axis=1)
        rows = np.flatnonzero(valid)
        valid[rows] &= observed[indices[rows]].all(axis=1)
    return indices, valid
