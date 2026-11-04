"""JSON-safe app handoff for existing prediction bundles; large cortex stays NPZ."""
import argparse
from pathlib import Path

import numpy as np

from .fusion_data import read_json, write_json, checked
from .worker_contract import file_sha256


def export(bundle, destination):
    bundle, destination = Path(bundle), Path(destination)
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema") != "perception-brain-prediction-bundle-v1":
        raise ValueError("versioned prediction bundle required")
    arrays = checked(bundle, "predictions.npz", manifest["arrays_sha256"])
    with np.load(arrays, allow_pickle=False) as data:
        queries, predictions, supported = (data[k] for k in ("eeg_query_seconds", "eeg_features", "eeg_supported"))
        if (predictions.shape != (len(queries), 8) or supported.shape != queries.shape
                or supported.dtype != np.bool_ or not np.isfinite(queries).all()
                or not np.isfinite(predictions[supported]).all()):
            raise ValueError("invalid supported EEG output")
        timeline = [{"query_seconds": float(t), "supported": bool(ok),
                     "features": values.tolist() if ok else [None] * 8}
                    for t, values, ok in zip(queries, predictions, supported)]
        cortex = {"array_file": "predictions.npz", "array_key": "predicted_cortical_response",
                  "array_sha256": manifest["arrays_sha256"],
                  "segment_starts_native": data["fmri_segment_starts_native"].tolist(),
                  "segment_durations_native": data["fmri_segment_durations_native"].tolist(),
                  "metadata": manifest["fmri"]}
    result = {"schema": "perception-brain-app-result-v2", "bundle_manifest_sha256": file_sha256(bundle / "manifest.json"),
        "source_sha256": manifest["source_sha256"], "eeg_model_sha256": manifest["eeg_model_sha256"],
        "presentation": manifest["presentation"], "evaluation_status": manifest["evaluation_status"],
        "fusion": manifest.get("fusion"), "eeg": {**manifest["eeg"], "timeline": timeline},
        "fmri": cortex, "observed_participant_measurements": False,
        "attention_or_retention_predictions_included": False}
    write_json(destination, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path); p.add_argument("destination", type=Path)
    a = p.parse_args(); export(a.bundle, a.destination)
