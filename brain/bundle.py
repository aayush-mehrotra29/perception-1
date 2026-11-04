"""Versioned model outputs with separate EEG/content and native fMRI clocks.

This serialization contract never turns neural predictions into attention,
retention, diagnoses or observed participant measurements.
"""
from pathlib import Path
import json
import re

import numpy as np

from .eeg import TARGETS
from .worker_contract import file_sha256


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 provenance required")
    return value


def write_bundle(destination: Path, *, source_sha256: str, eeg_model_sha256: str,
                 eeg_query_seconds, eeg_features, eeg_supported, fmri_directory: Path,
                 presentation: dict, evaluation_status: str):
    """Save predictions without filling gaps or resampling the fMRI response.

Caller must supply the exact source digest used by both branches. Synthetic
unit-test artifacts are allowed only when explicitly marked as such in status.
"""
    _digest(source_sha256)
    _digest(eeg_model_sha256)
    if evaluation_status not in {"unvalidated_prototype", "held_out_evaluated", "synthetic_contract_test"}:
        raise ValueError("explicit evaluation status required")
    if set(presentation) != {"input_format", "schedule_basis", "transfer_status"}:
        raise ValueError("explicit presentation assumptions required")
    if presentation["input_format"] not in {"video", "audio", "text", "image", "pdf"}:
        raise ValueError("unsupported presentation format")
    if not isinstance(presentation["schedule_basis"], str) or not presentation["schedule_basis"].strip():
        raise ValueError("presentation schedule cannot be unspecified")
    if presentation["transfer_status"] not in {"video_training_domain", "experimental_transfer"}:
        raise ValueError("explicit transfer status required")
    if presentation["input_format"] != "video" and presentation["transfer_status"] != "experimental_transfer":
        raise ValueError("non-video inputs must be marked experimental transfer")
    queries = np.asarray(eeg_query_seconds, dtype=np.float64)
    eeg = np.asarray(eeg_features, dtype=np.float64)
    supported = np.asarray(eeg_supported)
    if (queries.ndim != 1 or not len(queries) or not np.isfinite(queries).all()
            or np.any(queries < 0) or np.any(np.diff(queries) <= 0)
            or eeg.shape != (len(queries), 8) or supported.shape != queries.shape
            or supported.dtype != np.bool_ or not np.isfinite(eeg[supported]).all()
            or np.any(supported & (queries < 44))):
        raise ValueError("invalid EEG prediction coverage or clock")
    # Unsupported slots cannot accidentally be rendered as real predictions.
    eeg = eeg.copy()
    eeg[~supported] = np.nan
    manifest_path = fmri_directory / "manifest.json"
    fmri_meta = json.loads(manifest_path.read_text())
    if (fmri_meta.get("schema") != "perception-native-tribe-v1"
            or fmri_meta.get("source", {}).get("sha256") != source_sha256):
        raise ValueError("EEG/fMRI source identity mismatch")
    path = fmri_directory / "fmri.npz"
    if file_sha256(path) != fmri_meta["array_sha256"]:
        raise ValueError("native fMRI checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        fmri = archive["predicted_cortical_response"]
    segments = fmri_meta["segments"]
    starts = np.array([s["start"] for s in segments], dtype=float)
    durations = np.array([s["duration"] for s in segments], dtype=float)
    if (fmri.ndim != 2 or not all(fmri.shape) or len(fmri) != len(segments)
            or list(fmri.shape) != fmri_meta["shape"] or not np.isfinite(fmri).all()
            or not np.isfinite(starts).all() or not np.isfinite(durations).all()
            or np.any(durations <= 0) or len({s["timeline"] for s in segments}) != 1
            or np.any(starts[1:] < starts[:-1] + durations[:-1] - 1e-8)):
        raise ValueError("invalid or overlapping native fMRI segments")
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "predictions.npz", eeg_query_seconds=queries, eeg_features=eeg,
             eeg_supported=supported, fmri_segment_starts_native=starts,
             fmri_segment_durations_native=durations, predicted_cortical_response=fmri)
    result = {"schema": "perception-brain-prediction-bundle-v1", "source_sha256": source_sha256,
              "eeg_model_sha256": eeg_model_sha256, "fmri_manifest_sha256": file_sha256(manifest_path),
              "arrays_sha256": file_sha256(destination / "predictions.npz"),
              "presentation": presentation, "evaluation_status": evaluation_status,
              "eeg": {"targets": TARGETS, "window_seconds": 44, "query_unit": "content_seconds",
                      "supported_rows": int(supported.sum()), "scheduled_rows": len(queries),
                      "units": "training-source feature units; not calibrated raw EEG waveforms"},
              "fmri": {"shape": list(fmri.shape), "clock": "unmodified native TRIBE segment clock",
                       "checkpoint_revision": fmri_meta["checkpoint_revision"],
                       "spatial_metadata": fmri_meta.get("spatial_metadata"),
                       "neuro_offset_seconds": fmri_meta.get("neuro_offset_seconds"),
                       "native_equivalence_independently_checked": fmri_meta.get("native_equivalence_independently_checked", False)},
              "observed_participant_measurements": False,
              "attention_or_retention_predictions_included": False}
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
