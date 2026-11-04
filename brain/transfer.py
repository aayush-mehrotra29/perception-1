"""Available-modality EEG heads and joint native fMRI bundles.

Training uses restricted representations of the original lecture videos. It
cannot establish how people respond to silent reading or audio-only lessons.
These separate exploratory heads never impute absent modalities and never alter
the frozen primary six-arm comparison. Their EEG branch uses content features;
their parallel fMRI branch is the native pretrained TRIBE model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .bundle import write_bundle
from .edugage_targets import guard
from .eeg import EEGModel
from .evaluate import evaluate_crossed
from .representations import native_window_blocks
from .train_integrated import load_rows
from .worker_contract import file_sha256

MODES = {
    "text": ("dense_text",),
    "audio": ("dense_text", "dense_audio"),
    "visual": ("dense_video",),
    "document": ("dense_text", "dense_video"),
}
FORMATS = {"text": {"text"}, "audio": {"audio"}, "visual": {"image", "pdf"}, "document": {"pdf"}}


def select_blocks(blocks, mode):
    if mode not in MODES or not set(MODES[mode]).issubset(blocks):
        raise ValueError("requested mode has missing native content modalities")
    return {name: blocks[name] for name in MODES[mode]}


def train(qualified: Path, features: dict, output: Path,
          design=Path("configs/brain_transfer_evaluation_v1.json")):
    protocol = json.loads(design.read_text())
    if (protocol.get("schema") != "perception-modality-transfer-design-v1"
            or protocol["modes"] != {k: list(v) for k, v in MODES.items()}
            or protocol["fixed_ridge_lambda"] != 1.0 or protocol["pca_max_per_block"] != 16):
        raise ValueError("fixed transfer-head design required")
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, blocks, exclusions, provenance = load_rows(qualified, features)
    (output / "design.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for mode in MODES:
        guard(output)
        inputs = select_blocks(blocks, mode)
        report = evaluate_crossed(inputs, arrays["target"], arrays["content_id"], arrays["lesson"],
                                  arrays["family"], arrays["person"], output / mode / "evaluation")
        if len(report["cells"]) != 8 or any(c["status"] != "scored" for c in report["cells"]):
            raise ValueError("incomplete transfer evaluation; model export withheld")
        guard(output)
        model = EEGModel.fit(inputs, arrays["target"], arrays["content_id"].tolist(), arrays["lesson"], arrays["person"])
        destination = output / mode / "exploratory-deployment"
        model.save(destination)
        report = {"schema": "perception-transfer-head-v1", "mode": mode, "blocks": list(MODES[mode]),
            "model_manifest_sha256": file_sha256(destination / "manifest.json"),
            "design_sha256": file_sha256(design), "training_domain": "audiovisual_lecture_viewing",
            "actual_nonvideo_response_validation": False, "model_selection_by_scores": False,
            "fmri_used_as_EEG_predictor": False, "claim_limit": protocol["claim_limit"]}
        (destination / "transfer.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"transfer_head": mode, "eight_video_domain_cells_scored": True}), flush=True)
    result = {"schema": "perception-transfer-training-v1", "modes": list(MODES),
        "rows": len(arrays["target"]), "distinct_content_windows": len(np.unique(arrays["content_id"])),
        "viewing_support": exclusions, "source_features": provenance,
        "target_audit_sha256": file_sha256(qualified / "summary.json"),
        "primary_evaluation_changed": False, "actual_nonvideo_response_validation": False}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def predict(native: Path, model_directory: Path, output: Path, *, mode, duration, presentation):
    if mode not in MODES or presentation.get("input_format") not in FORMATS[mode]:
        raise ValueError("presentation format does not match the fitted transfer mode")
    if presentation.get("transfer_status") != "experimental_transfer":
        raise ValueError("nonvideo predictions must declare experimental transfer")
    head = json.loads((model_directory / "transfer.json").read_text())
    if (head.get("schema") != "perception-transfer-head-v1" or head.get("mode") != mode
            or head.get("blocks") != list(MODES[mode])
            or head.get("model_manifest_sha256") != file_sha256(model_directory / "manifest.json")):
        raise ValueError("transfer model identity mismatch")
    model = EEGModel.load(model_directory)
    native_meta = json.loads((native / "manifest.json").read_text())
    expected = {name.removeprefix("dense_") for name in MODES[mode]}
    if set(native_meta.get("input_modalities", [])) != expected:
        raise ValueError("native inference includes different modalities than the transfer head")
    if mode == "text":
        schedule = native_meta.get("assumed_presentation_schedule") or {}
        if (schedule.get("duration_seconds") != duration
                or schedule.get("clock_basis") != "assumed_constant_reading_pace"):
            raise ValueError("text duration must match its native assumed presentation")
    guard(output)
    queries = np.arange(0, duration + 1e-8, 5.0)
    blocks, clock = native_window_blocks(native, queries, duration, allow_unqualified_timing=True)
    inputs = select_blocks(blocks, mode)
    supported = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in inputs.values()])
    if not supported.any():
        raise ValueError("no complete supported 44-second context; no EEG prediction invented")
    prediction = np.full((len(queries), 8), np.nan)
    prediction[supported] = model.predict({key: value[supported] for key, value in inputs.items()})
    guard(output)
    result = write_bundle(output, source_sha256=native_meta["source"]["sha256"],
        eeg_model_sha256=file_sha256(model_directory / "manifest.json"), eeg_query_seconds=queries,
        eeg_features=prediction, eeg_supported=supported, fmri_directory=native,
        presentation=presentation, evaluation_status="unvalidated_prototype")
    (output / "transfer-diagnostics.json").write_text(json.dumps({
        "schema": "perception-transfer-inference-v1", "mode": mode,
        "head_receipt_sha256": file_sha256(model_directory / "transfer.json"),
        "native_representation_metadata": clock, "missing_modalities_imputed": False,
        "fmri_used_as_EEG_predictor": False, "actual_nonvideo_response_validation": False,
        "negative_predicted_log1p_variances": int(np.sum(prediction[supported, 4:] < 0)),
        "output_clipping_applied": False, "claim_limit": head["claim_limit"]}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("qualified", type=Path)
    parser.add_argument("feature_registry", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.qualified, json.loads(args.feature_registry.read_text()), args.output)))
