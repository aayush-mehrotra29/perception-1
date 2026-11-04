"""Replay existing content caches locally, verifying EEG serialization and native cortex."""
import argparse
from pathlib import Path
import time

import numpy as np

from .edugage_targets import guard
from .fusion_data import read_json, write_json
from .fusion_export import export
from .fusion_inference import predict_native
from .worker_contract import file_sha256


def qualify(model, cache_root, output):
    model, cache_root, output = map(Path, (model, cache_root, output))
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for mode in ("video", "audio", "text", "image", "pdf"):
        guard(output)
        start = time.monotonic()
        old = cache_root / f"full-{mode}-qualification-v1"
        native = old / "tribe"
        parent, nm = read_json(old / "manifest.json"), read_json(native / "manifest.json")
        pinned = {str(path): file_sha256(path) for path in
                  [old / "manifest.json", native / "manifest.json", native / "fmri.npz",
                   *(native / record["file"] for record in nm["dense_features"])]}
        features = None
        if mode == "video":
            features = old / "features"
            duration = read_json(features / "manifest.json")["media_duration"]
            presentation = {"input_format": mode, "schedule_basis": "verified cached educational-video media clock",
                            "transfer_status": "video_training_domain"}
            for path in (features / "manifest.json", features / "features.npz"):
                pinned[str(path)] = file_sha256(path)
        elif mode == "audio":
            am = read_json(old / "audio/manifest.json")
            duration = am["samples"] / am["sample_rate"]
        else:
            duration = nm["assumed_presentation_schedule"]["duration_seconds"]
        if mode != "video":
            presentation = {"input_format": mode, "schedule_basis": parent["schedule_basis"],
                            "transfer_status": "experimental_transfer"}
        case = output / mode
        case.mkdir()
        first = predict_native(native, model, case / "prediction", duration=duration, input_format=mode,
                               features=features, presentation=presentation)
        predict_native(native, model, case / "replay", duration=duration, input_format=mode,
                       features=features, presentation=presentation)
        with (np.load(case / "prediction/predictions.npz", allow_pickle=False) as a,
              np.load(case / "replay/predictions.npz", allow_pickle=False) as b,
              np.load(native / "fmri.npz", allow_pickle=False) as cortex):
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a["predicted_cortical_response"], cortex["predicted_cortical_response"])
            np.testing.assert_array_equal(a["fmri_segment_starts_native"], [s["start"] for s in nm["segments"]])
            np.testing.assert_array_equal(a["fmri_segment_durations_native"], [s["duration"] for s in nm["segments"]])
            assert np.isnan(a["eeg_features"][~a["eeg_supported"]]).all()
            assert (a["eeg_features"][a["eeg_supported"], 4:] >= 0).all()
        for path, digest in pinned.items():
            assert file_sha256(Path(path)) == digest, "source cache mutated"
        app = export(case / "prediction", case / "prediction/app-result.json")
        assert all(row["features"] == [None] * 8 for row in app["eeg"]["timeline"] if not row["supported"])
        case_report = {"format": mode, "duration_seconds": duration, "supported_queries": first["eeg"]["supported_rows"],
                       "active_modalities": first["fusion"]["active_modalities"], "seconds_including_replay": time.monotonic() - start,
                       "prediction_manifest_sha256": file_sha256(case / "prediction/manifest.json"),
                       "app_result_sha256": file_sha256(case / "prediction/app-result.json"),
                       "source_files_unchanged": pinned, "reloaded_predictions_exact": True,
                       "native_cortical_arrays_and_clocks_unchanged": True, "new_paid_requests": 0,
                       "interpretation": "Content-to-output integration test; not an additional neural-accuracy benchmark"}
        write_json(case / "qualification.json", case_report)
        cases.append(case_report)
        print(f"{mode}: {first['eeg']['supported_rows']} supported queries, exact replay, native cortex unchanged", flush=True)
    result = {"schema": "perception-fusion-qualification-v2", "status": "passed", "cases": cases,
              "model_manifest_sha256": file_sha256(model / "manifest.json"), "new_paid_requests": 0}
    write_json(output / "qualification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("model", "cache_root", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args(); qualify(a.model, a.cache_root, a.output)
