"""Descriptive raw-recording checks; no filtering or new target exclusions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .edugage_targets import guard
from .eeg import CHANNELS
from .worker_contract import file_sha256


def scan(path: Path):
    counts = np.zeros(4, np.int64)
    mean, sse = np.zeros(4), np.zeros(4)
    minimum, maximum = np.full(4, np.inf), np.full(4, -np.inf)
    nonfinite, equal_neighbors = np.zeros(4, np.int64), np.zeros(4, np.int64)
    rows, equal_times, reversed_times, invalid_times = 0, 0, 0, 0
    last_values, last_time, buffer = None, None, []
    def consume(batch):
        nonlocal counts, mean, sse, minimum, maximum, nonfinite
        x = np.asarray(batch)
        for j in range(4):
            valid = x[np.isfinite(x[:, j]), j]
            nonfinite[j] += len(x) - len(valid)
            if not len(valid):
                continue
            n, m = len(valid), valid.mean()
            total = counts[j] + n
            sse[j] += np.sum((valid - m) ** 2) + (m - mean[j]) ** 2 * counts[j] * n / total
            mean[j] += (m - mean[j]) * n / total
            counts[j] = total
            minimum[j], maximum[j] = min(minimum[j], valid.min()), max(maximum[j], valid.max())
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["timestamp", *CHANNELS]:
            raise ValueError("unexpected raw EEG columns")
        for row in reader:
            time = float(row["timestamp"])
            values = np.array([float(row[name]) for name in CHANNELS])
            if not np.isfinite(time):
                invalid_times += 1
            elif last_time is not None:
                equal_times += int(time == last_time)
                reversed_times += int(time < last_time)
            last_time = time if np.isfinite(time) else None
            if last_values is not None:
                equal_neighbors += np.isfinite(values) & np.isfinite(last_values) & (values == last_values)
            last_values = values
            buffer.append(values)
            rows += 1
            if len(buffer) == 4096:
                consume(buffer)
                buffer.clear()
            if rows % 100000 == 0:
                guard(path)
        if buffer:
            consume(buffer)
    return {"rows": rows, "channels": list(CHANNELS), "finite_samples": counts.tolist(),
        "nonfinite_samples": nonfinite.tolist(), "means": [float(m) if n else None for m, n in zip(mean, counts)],
        "population_standard_deviations": [float(np.sqrt(v / n)) if n else None for v, n in zip(sse, counts)],
        "minima": [float(x) if n else None for x, n in zip(minimum, counts)],
        "maxima": [float(x) if n else None for x, n in zip(maximum, counts)],
        "equal_consecutive_record_values": equal_neighbors.tolist(), "invalid_timestamps": invalid_times,
        "equal_neighbor_receiver_timestamps": equal_times, "decreasing_receiver_timestamps": reversed_times,
        "consecutive_record_rows_are_not_verified_hardware_sample_times": True}


def run(corpus: Path, output: Path):
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in sorted(corpus.glob("participant-*/*.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("artifact_kind") != "edugage-recorded-voltage-content-viewing-v2":
            continue
        pin = metadata["input_pins"]["eeg"]
        record = sources.setdefault(pin["sha256"], {"pin": pin, "participants": set(), "lectures": set()})
        record["participants"].add(metadata["public_participant_id"])
        record["lectures"].add(metadata["video_uid"])
    results = []
    for checksum, item in sources.items():
        path = Path(item["pin"]["path"])
        guard(output)
        if file_sha256(path) != checksum:
            raise ValueError("raw EEG source checksum mismatch")
        record = scan(path)
        if file_sha256(path) != checksum:
            raise ValueError("raw EEG changed during audit")
        record.update(source_sha256=checksum, participants=sorted(item["participants"]), lectures=sorted(item["lectures"]))
        if record["rows"] and len(item["participants"]) > 1:
            raise ValueError("same nonempty raw EEG source assigned to multiple participants")
        results.append(record)
        print(json.dumps({"sources_scanned": len(results), "total": len(sources)}), flush=True)
    report = {"schema": "perception-raw-signal-descriptive-audit-v1", "sources": results,
        "source_count": len(results), "no_cross_participant_recording_duplicates": True,
        "raw_columns_have_contact_or_motion_quality": False, "acquisition_sample_clock_verified": False,
        "voltage_calibration_verified": False, "neural_specificity_established": False,
        "new_target_filtering_or_exclusions_applied": False,
        "permitted_interpretation": "Exploratory prediction of released voltage summaries. These checks cannot establish neural attention, retention, or artifact-free EEG."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.corpus, args.output)
