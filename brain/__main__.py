"""Read-only readiness and metadata inventory; no implicit cloud execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from .credentials import load_credentials
from .worker_contract import file_sha256


def saved_evidence(project: Path):
    """Inspect artifacts; a status flag alone never establishes a fitted model."""
    runs = project / ".local/brain-runs"
    native = []
    for path in runs.glob("*-native-qualification-v1.json"):
        record = json.loads(path.read_text())
        if record.get("schema") == "perception-real-native-artifact-qualification-v1":
            native.append({"report": str(path), "report_sha256": file_sha256(path),
                "source_sha256": record["source_sha256"],
                "integrity_and_full_coverage_passed": record["integrity_and_full_coverage_passed"],
                "measured_neural_accuracy_validated": record["measured_neural_accuracy_validated"]})
    model_root = runs / "integrated-eeg-training-v1"
    training = model_root / "manifest.json"
    artifact = model_root / "exploratory-deployment/manifest.json"
    fitted = False
    if training.exists() and artifact.exists():
        metadata = json.loads(artifact.read_text())
        report = json.loads(training.read_text())
        arrays = artifact.parent / "arrays.npz"
        fitted = (report.get("schema") == "perception-integrated-eeg-training-v1"
            and metadata.get("schema") == "perception-eeg-ridge-v1" and arrays.is_file()
            and file_sha256(arrays) == metadata.get("arrays_sha256"))
    target_report = runs / "corpus-target-qualification-v1/summary.json"
    target = json.loads(target_report.read_text()) if target_report.exists() else {}
    return {"native_lecture_reports": native, "integrated_model_fitted": fitted,
        "raw_target_aggregation_audit_complete": bool(target.get("complete")),
        "physical_EEG_synchronization_established": False,
        "signal_contact_quality_established": False,
        "gate_access_previously_demonstrated": {
            "llama_via_tribe": (runs / "rocket-native-brain-v4/manifest.json").exists(),
            "sam_audio": (runs / "ideal_rocket_equation-sam-audio-v1/manifest.json").exists()},
        "gate_access_live_checked": False,
        "scientific_validity_established": False}


def preflight(data_root: Path):
    load_credentials()
    project = Path(__file__).resolve().parents[1]
    corpus = data_root / "runs/edugage-recorded-voltage-corpus-v2-2026-10-26"
    viewings = []
    for path in sorted(corpus.glob("participant-*/*.json")):
        if path.name == "summary.json":
            continue
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("artifact_kind") != "edugage-recorded-voltage-content-viewing-v2":
            continue
        pins = record["input_pins"]
        viewings.append({"participant": record["public_participant_id"], "lecture": record["video_uid"],
                         "metadata_sha256": hashlib.sha256(raw).hexdigest(), "metadata_path": str(path),
                         "media_sha256": record["media_sha256"],
                         "associated_rows": record["primary0_associated_received_rows"],
                         "raw_eeg_file_present": Path(pins["eeg"]["path"]).is_file(),
                         "raw_eeg_expected_sha256": pins["eeg"]["sha256"],
                         "playback_file_present": Path(pins["playback"]["path"]).is_file(),
                         "raw_sample_hashes_checked": False})
    evidence = saved_evidence(project)
    return {"schema": "perception-brain-preflight-v1", "project": str(project),
            "credentials": {name: bool(os.environ.get(name)) for name in
                            ("MODEL_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN")},
            "modal_configuration_file_present": Path.home().joinpath(".modal.toml").is_file(),
            "internal_free_bytes": shutil.disk_usage(Path.home()).free,
            "external_free_bytes": shutil.disk_usage(data_root).free,
            "required_disk_reserve_bytes": 12 * 1024**3,
            "viewings": viewings, "EEG_window_qualification_complete": False,
            "integrated_model_fitted": evidence["integrated_model_fitted"], "saved_evidence": evidence,
            "next_gates": (["complete four-lecture native extraction and feature checks", "fit and evaluate real-data EEG head"]
                if not evidence["integrated_model_fitted"] else ["verify held-out metrics and joint inference bundle"])
                + ["non-video inputs require explicit presentation clocks and transfer qualification",
                   "validate against independent, properly synchronized neural recordings before scientific benefit claims"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight"])
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/EEG Research Workspace"))
    args = parser.parse_args()
    print(json.dumps(preflight(args.data_root), indent=2))


if __name__ == "__main__":
    main()
