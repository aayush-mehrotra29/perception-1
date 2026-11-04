"""Local EEG-head inference plus existing native TRIBE output; no EEG upload."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bundle import write_bundle
from .edugage_targets import guard
from .eeg import EEGModel
from .train_integrated import arm_inputs
from .worker_contract import file_sha256


def predict(features: Path, model_directory: Path, tribe: Path, output: Path):
    guard(output)
    meta = json.loads((features / "manifest.json").read_text())
    if meta.get("schema") != "perception-lesson-features-v1":
        raise ValueError("authenticated complete content features required")
    path = features / "features.npz"
    if file_sha256(path) != meta["features_sha256"]:
        raise ValueError("content feature checksum mismatch")
    model = EEGModel.load(model_directory)
    with np.load(path, allow_pickle=False) as data:
        queries, supported = data["query_seconds"], data["supported"]
        blocks = {name: data[name] for name in meta["blocks"]}
    if (supported.dtype != bool or supported.shape != queries.shape
            or not np.array_equal(supported, np.logical_and.reduce([np.isfinite(b).all(axis=1) for b in blocks.values()]))):
        raise ValueError("content coverage mismatch")
    if not supported.any():
        raise ValueError("no supported windows; do not invent a prediction")
    inputs = arm_inputs({k: v[supported] for k, v in blocks.items()})["all_blocks"]
    predictions = np.full((len(queries), 8), np.nan)
    predictions[supported] = model.predict(inputs)
    result = write_bundle(output, source_sha256=meta["source_sha256"],
        eeg_model_sha256=file_sha256(model_directory / "manifest.json"), eeg_query_seconds=queries,
        eeg_features=predictions, eeg_supported=supported, fmri_directory=tribe,
        presentation={"input_format": "video", "schedule_basis": "authenticated native media timeline; whole-lesson offline analysis",
                      "transfer_status": "video_training_domain"}, evaluation_status="unvalidated_prototype")
    diagnostics = {"schema": "perception-eeg-inference-diagnostics-v1",
        "feature_manifest_sha256": file_sha256(features / "manifest.json"),
        "negative_predicted_log1p_variances": int(np.sum(predictions[supported, 4:] < 0)),
        "output_clipping_applied": False, "participant_EEG_at_inference": False,
        "historical_alignment_and_signal_quality_unqualified": True}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("features", "model", "tribe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(predict(args.features, args.model, args.tribe, args.output)))
