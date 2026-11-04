"""Coarse targets under the existing receiver→reported-position observation rule.

A received EEG sample retains its accepted poll interval's LEFT reported video
position. Windows select whole groups by that assigned position. This is an exact
aggregation of that discrete observation operator, not a reconstruction of physical
sample times. Repeated positions and burst arrivals remain intact. No interpolation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import time

import numpy as np

from .worker_contract import file_sha256

ARRAYS = ("query_media_seconds", "group_left_source_row", "group_right_source_row",
          "group_declared_run_id", "group_host_interval_milliseconds", "offset_seconds",
          "voltage_n", "voltage_mean", "voltage_centered_sse", "row_to_group",
          "eeg_original_row_indices", "eeg_receiver_milliseconds")

RESEARCH_MOUNT = Path("/Volumes/EEG Research Workspace")
RESEARCH_UUID = "DFEAA5DF-CFE0-4327-9D3E-900405DA4695"
_mount_check = None


def guard(root: Path, *, internal_reserve_gib=12):
    if internal_reserve_gib not in {6, 12}:
        raise ValueError("unsupported internal reserve profile")
    global _mount_check
    if root.absolute().is_relative_to(RESEARCH_MOUNT):
        if (not os.path.ismount(RESEARCH_MOUNT) or RESEARCH_MOUNT.resolve() != RESEARCH_MOUNT
                or not root.resolve().is_relative_to(RESEARCH_MOUNT)):
            raise RuntimeError("research volume is not mounted at its authenticated path")
        device, now = RESEARCH_MOUNT.stat().st_dev, time.monotonic()
        if _mount_check is None or _mount_check[0] != device or now - _mount_check[1] >= 5:
            result = subprocess.run(["/usr/sbin/diskutil", "info", "-plist", str(RESEARCH_MOUNT)],
                                    capture_output=True, check=True, timeout=5)
            if plistlib.loads(result.stdout).get("VolumeUUID") != RESEARCH_UUID:
                raise RuntimeError("research volume identity changed")
            _mount_check = (device, now)
    for path, reserve in ((Path.home(), internal_reserve_gib), (root, 12)):
        while not path.exists():
            if path == path.parent:
                raise RuntimeError("cannot locate destination filesystem")
            path = path.parent
        if shutil.disk_usage(path).free < reserve * 1024**3:
            message = "12 GiB disk reserve reached" if reserve == 12 else "6 GiB internal transfer reserve reached"
            raise RuntimeError(message)


def transfer_guard(root: Path):
    """Bounded cloud transport only; numerical analysis retains its 12 GiB guard.

Worker requests/results are limited to 256 MiB and results are written to the
authenticated external workspace. Retain 6 GiB internally for RPC copies and
system swap, and the unchanged 12 GiB reserve on the destination filesystem.
Do not use this profile to authorize EEG fitting or local model inference.
"""
    guard(root, internal_reserve_gib=6)


def aggregate(counts, means, sse):
    active = counts > 0
    n = counts[active].sum()
    if n < 2:
        raise ValueError("insufficient_received_samples")
    counts, means, sse = counts[active], means[active], sse[active]
    if not np.isfinite(means).all() or not np.isfinite(sse).all() or np.any(sse < 0):
        raise ValueError("invalid_group_moments")
    mean = (counts[:, None] * means).sum(axis=0) / n
    variance = (sse + counts[:, None] * (means - mean) ** 2).sum(axis=0) / n
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("constant_or_invalid_channel")
    return np.r_[mean, np.log1p(variance)], int(n)


def select_groups(q, right_q, left_row, right_row, runs, host_intervals, query, seconds=44.0):
    """Select one continuous playback chain covering the reported-position window."""
    start = query - seconds
    if start < 0 or len(q) == 0:
        return None, "insufficient_context"
    joins = ((right_row[:-1] == left_row[1:]) & (runs[:-1] == runs[1:])
             & (host_intervals[:-1, 1] == host_intervals[1:, 0]) & (q[1:] >= q[:-1]))
    cuts = np.r_[0, np.flatnonzero(~joins) + 1, len(q)]
    eligible = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if q[a] <= start and right_q[b - 1] >= query]
    if len(eligible) != 1:
        return None, "unsupported_gap_or_multiple_playbacks"
    a, b = eligible[0]
    groups = np.arange(a, b)[(q[a:b] >= start) & (q[a:b] < query)]
    if len(groups) == 0:
        return None, "no_associated_groups"
    return groups, "supported"


def build_viewing(metadata_path: Path, output: Path, verify_raw=False):
    guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    metadata_bytes = metadata_path.read_bytes()
    meta = json.loads(metadata_bytes)
    if meta["observation_report"]["channels"] != ["tp9", "af7", "af8", "tp10"]:
        raise ValueError("unexpected target montage")
    pins = meta["input_pins"]
    # Hash raw sources before any target reuse. No source files are rewritten.
    for key in ("eeg", "playback", "feature_cache"):
        source = Path(pins[key]["path"])
        if source.stat().st_size != pins[key]["size_bytes"] or file_sha256(source) != pins[key]["sha256"]:
            raise ValueError(f"source identity mismatch: {key}")
        guard(output)
    with np.load(metadata_path.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ARRAYS}
    for key, value in arrays.items():
        expected = meta["arrays"][key]
        if (list(value.shape) != expected["shape"] or value.dtype != np.dtype(expected["dtype"])
                or hashlib.sha256(value.tobytes(order="C")).hexdigest() != expected["Cbytes_sha256"]):
            raise ValueError(f"array identity mismatch: {key}")
    primary = np.flatnonzero(arrays["offset_seconds"] == 0)
    if len(primary) != 1:
        raise ValueError("unique fixed zero-offset association required")
    primary = int(primary[0])
    # Read only timing/identity fields; discard engagement responses without use.
    with Path(pins["playback"]["path"]).open() as stream:
        clocks = [(row["video_uid"], float(row["video_timestamp_sec"]) if row["video_timestamp_sec"] else np.nan)
                  for row in csv.DictReader(stream)]
    q = arrays["query_media_seconds"]
    right_q = np.array([clocks[int(row)][1] for row in arrays["group_right_source_row"]])
    for row, expected in zip(arrays["group_left_source_row"], q):
        uid, position = clocks[int(row)]
        if uid != meta["video_uid"] or position != expected:
            raise ValueError("saved group clock differs from original playback log")
    for row, left_position in zip(arrays["group_right_source_row"], q):
        uid, position = clocks[int(row)]
        if uid != meta["video_uid"] or not np.isfinite(position) or position < left_position:
            raise ValueError("invalid original right playback boundary")
    queries = np.arange(0, pins["feature_cache"]["duration_seconds"] + 1e-9, 5.0)
    targets = np.full((len(queries), 8), np.nan)
    supported = np.zeros(len(queries), dtype=bool)
    counts = np.zeros(len(queries), dtype=np.int64)
    groups_per_window, reasons = [], []
    for i, query in enumerate(queries):
        groups, reason = select_groups(q, right_q, arrays["group_left_source_row"], arrays["group_right_source_row"],
                                       arrays["group_declared_run_id"], arrays["group_host_interval_milliseconds"], query)
        if groups is not None:
            try:
                targets[i], counts[i] = aggregate(arrays["voltage_n"][primary, groups],
                    arrays["voltage_mean"][primary, groups], arrays["voltage_centered_sse"][primary, groups])
                supported[i] = True
            except ValueError as error:
                reason = str(error)
        groups_per_window.append(groups)
        reasons.append(reason)
    raw_verification = {"performed": False}
    if verify_raw and supported.any():
        indices = arrays["eeg_original_row_indices"]
        raw_values = np.empty((len(indices), 4), dtype=float)
        cursor = 0
        with Path(pins["eeg"]["path"]).open() as stream:
            for row_index, row in enumerate(csv.DictReader(stream)):
                if cursor == len(indices):
                    break
                if row_index == indices[cursor]:
                    if int(row["timestamp"]) != arrays["eeg_receiver_milliseconds"][cursor]:
                        raise ValueError("raw EEG receiver timestamp mismatch")
                    raw_values[cursor] = [float(row[key]) for key in ("tp9", "af7", "af8", "tp10")]
                    cursor += 1
                if row_index % 100000 == 0:
                    guard(output)
        if cursor != len(indices):
            raise ValueError("raw EEG source truncated")
        assignment = arrays["row_to_group"][primary]
        largest_error = 0.0
        for i in np.flatnonzero(supported):
            sample = raw_values[np.isin(assignment, groups_per_window[i])]
            if len(sample) != counts[i]:
                raise ValueError("raw/group sample count mismatch")
            direct = np.r_[sample.mean(axis=0), np.log1p(sample.var(axis=0))]
            np.testing.assert_allclose(targets[i], direct, rtol=1e-10, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(targets[i] - direct))))
        raw_verification = {"performed": True, "windows": int(supported.sum()), "max_abs_error": largest_error}
    # Bookend the inputs: do not publish results if a source changed during extraction.
    for key in ("eeg", "playback", "feature_cache"):
        if file_sha256(Path(pins[key]["path"])) != pins[key]["sha256"]:
            raise ValueError(f"source changed during extraction: {key}")
    if metadata_path.read_bytes() != metadata_bytes:
        raise ValueError("metadata changed during extraction")
    np.savez(output / "targets.npz", query_seconds=queries, targets=targets, supported=supported, sample_counts=counts)
    report = {"schema": "perception-reported-position-targets-v1", "participant": meta["public_participant_id"],
              "lecture": meta["video_uid"], "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "media_sha256": meta["media_sha256"], "targets_sha256": file_sha256(output / "targets.npz"),
              "window_definition": "received samples assigned to original left-reported positions in [query-44,query), within one continuous playback chain",
              "window_seconds": 44, "hop_seconds": 5, "supported_windows": int(supported.sum()),
              "scheduled_windows": len(queries), "reasons": reasons, "raw_verification": raw_verification,
              "physical_acquisition_timing_established": False, "signal_quality_qualification_complete": False,
              "neural_or_attention_validity_established": False, "source_hashes_checked": True}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    report = build_viewing(args.metadata, args.output, args.verify_raw)
    print(json.dumps({k: v for k, v in report.items() if k != "reasons"}, indent=2))
