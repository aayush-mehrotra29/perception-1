"""Exploratory local engineering baseline, not the final multimodal brain model.

Uses coarse receiver-associated recorded voltages and candidate media versions.
It cannot establish neural validity, historical synchronization, or attention.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .content_windows import interval_means, point_means
from .edugage_targets import guard
from .evaluate import evaluate_crossed
from .worker_contract import file_sha256

SOURCES = {
    "ideal_rocket_equation": ("aerospace", "edugage-rocket-native-sensory-v1-2026-10-26"),
    "atmosphere_pressure_composition": ("aerospace", "edugage-atmosphere_pressure_composition-native-sensory-v1-2026-10-26"),
    "healing_radiation": ("xray", "edugage-healing_radiation-native-sensory-v1-2026-10-26"),
    "xray_diffraction": ("xray", "edugage-xray_diffraction-native-sensory-v1-2026-10-26"),
}


def load_sensory(root: Path, lecture, queries):
    path = root / "runs" / SOURCES[lecture][1]
    meta = json.loads((path / "summary.json").read_text())
    if file_sha256(path / "features.npz") != meta["features_npz_sha256"]:
        raise ValueError("sensory source checksum mismatch")
    blocks = {}
    with np.load(path / "features.npz", allow_pickle=False) as data:
        for index, stream in enumerate(meta["streams"]):
            arrays = {}
            for field, state in stream["array_states"].items():
                array = data[f"stream_{index}_{field}"]
                if (list(array.shape) != state["shape"] or array.dtype.str != state["dtype"]
                        or hashlib.sha256(array.tobytes()).hexdigest() != state["sha256"]):
                    raise ValueError("sensory array identity mismatch")
                arrays[field] = array
            if stream["name"] == "visual_descriptors":
                blocks[stream["name"]] = point_means(arrays["times_seconds"], arrays["values"], arrays["observed_mask"], queries)
            else:
                blocks[stream["name"]], _ = interval_means(arrays["window_start_seconds"], arrays["window_stop_seconds"],
                    arrays["values"], arrays["observed_mask"], queries)
    return blocks, meta["duration_seconds"], meta["features_npz_sha256"]


def run(data_root: Path, qualified: Path, output: Path):
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads((qualified / "summary.json").read_text())
    if not corpus["complete"] or any(v["status"] == "failed" for v in corpus["viewings"]):
        raise ValueError("complete target extraction audit required")
    cache, rows, exclusions = {}, [], []
    for viewing in corpus["viewings"]:
        report_path = qualified / viewing["report"]
        if file_sha256(report_path) != viewing["report_sha256"]:
            raise ValueError("target report checksum mismatch")
        report = json.loads(report_path.read_text())
        target_path = report_path.parent / "targets.npz"
        if file_sha256(target_path) != report["targets_sha256"]:
            raise ValueError("target data checksum mismatch")
        with np.load(target_path, allow_pickle=False) as data:
            query, target, supported = data["query_seconds"], data["targets"], data["supported"]
        lecture = report["lecture"]
        key = (lecture, query.tobytes())
        if key not in cache:
            cache[key] = load_sensory(data_root, lecture, query)
        blocks, duration, source_hash = cache[key]
        valid_content = np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in blocks.values()])
        usable = supported & valid_content
        exclusions.append({"participant": report["participant"], "lecture": lecture,
                           "scheduled": len(query), "eeg_supported": int(supported.sum()),
                           "content_and_eeg_supported": int(usable.sum()), "sensory_sha256": source_hash})
        for i in np.flatnonzero(usable):
            rows.append({"person": report["participant"], "lesson": lecture, "family": SOURCES[lecture][0],
                         "content_id": f"{lecture}:{query[i].hex()}", "target": target[i],
                         "blocks": {k: v[i] for k, v in blocks.items()}, "progress": query[i] / duration})
    if not rows:
        raise ValueError("no supported paired engineering rows")
    y = np.array([r["target"] for r in rows])
    ids, lessons, families, people = (np.array([r[key] for r in rows]) for key in ("content_id", "lesson", "family", "person"))
    blocks = {key: np.array([r["blocks"][key] for r in rows]) for key in rows[0]["blocks"]}
    progress = np.array([r["progress"] for r in rows])[:, None]
    # Frozen engineering baselines only; no tuning or full-data deployment model.
    for name, inputs in (("progress", {"progress": progress}), ("native_sensory", blocks)):
        guard(output)
        evaluate_crossed(inputs, y, ids, lessons, families, people, output / name)
    manifest = {"schema": "perception-exploratory-local-baseline-v1", "rows": len(rows),
                "distinct_content_windows": len(np.unique(ids)), "viewings": exclusions,
                "input_audit_sha256": file_sha256(qualified / "summary.json"),
                "status": "engineering_baselines_fitted_not_neurally_validated",
                "signal_quality_qualification_complete": False, "historical_stimulus_identity_established": False,
                "physical_acquisition_timing_established": False, "attention_or_learning_prediction": False,
                "full_multimodal_model_fitted": False, "deployment_model_saved": False,
                "coverage_policy": "44s: audio duration-weighted >=99% coverage; visual measured-frame mean >=99% observed frames with <=0.1s gaps. Visual descriptors retain native trailing context. No gap repair."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "viewings"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("qualified", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.data_root, args.qualified, args.output)
