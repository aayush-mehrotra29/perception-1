"""Sequential real-data extraction, qualification and local EEG fitting.

Explicit execution plans reference existing authenticated artifacts. Completed
stages are checked before reuse; an incomplete paid stage stops the pipeline.
No automatic retry, target upload, media replacement or mock inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .credentials import load_credentials
from .edugage_targets import guard
from .lesson_features import assemble
from .local_baseline import SOURCES
from .modal_client import submit
from .qualify_native import qualify
from .train_integrated import run as train
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


def execute(plan_path: Path):
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "perception-corpus-execution-v1"
            or set(plan["lessons"]) != set(SOURCES)):
        raise ValueError("four-lecture execution plan required")
    load_credentials()
    status_path = Path(plan["status"])
    guard(status_path)
    status = {"plan_sha256": file_sha256(plan_path), "stages": [], "complete": False}
    def completed(lesson, stage):
        status["stages"].append({"lecture": lesson, "stage": stage, "status": "complete"})
        guard(status_path)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status["stages"][-1]), flush=True)
    features = {}
    lesson, stage = None, "preflight"
    try:
        for lesson, entry in plan["lessons"].items():
            guard(status_path)
            media, request_path = Path(entry["media"]), Path(entry["request"])
            request = PublicMediaRequest.model_validate_json(request_path.read_text())
            validate_media(media, request)
            spec = entry["features"]
            if spec["lecture"] != lesson or spec["source_sha256"] != request.sha256:
                raise ValueError("execution-plan media identity mismatch")
            stage = "native_tribe"
            tribe = Path(spec["tribe"])
            if not tribe.exists():
                profile = entry.get("tribe_runtime_profile", "standard")
                submit("tribe", media, request, tribe, entry["tribe_request_id"], use_cached_media=True,
                       runtime_profile=profile,
                       reservation_microdollars={"standard": 4000000, "long": 6000000, "resume": 2000000}[profile])
            qualification_path = Path(entry["native_qualification"])
            if not qualification_path.exists():
                qualify(tribe, request, spec["media_duration"], qualification_path)
            else:
                verified = json.loads(qualification_path.read_text())
                if (not verified["integrity_and_full_coverage_passed"]
                        or verified["native_manifest_sha256"] != file_sha256(tribe / "manifest.json")
                        or verified["source_sha256"] != request.sha256):
                    raise ValueError("native qualification provenance mismatch")
            completed(lesson, stage)
            stage = "native_sam_audio"
            sam = Path(spec["sam_audio"])
            if not sam.exists():
                submit("sam_audio", media, request, sam, entry["sam_request_id"],
                    use_cached_media=True, cached_audio_manifest=Path(spec["audio_derivative"]) / "manifest.json",
                    reservation_microdollars=2000000)
            # Assembly authenticates all arrays and requires every audio chunk.
            stage = "feature_assembly"
            feature_dir = Path(entry["feature_output"])
            if not feature_dir.exists():
                assemble(spec, np.arange(0, spec["media_duration"] + 1e-8, 5), feature_dir,
                         allow_prototype_timing=True)
            features[lesson] = str(feature_dir.resolve())
            completed(lesson, stage)
        stage = "fixed_eeg_evaluation_and_separate_refit"
        # train() authenticates every saved feature/target archive before fitting.
        result = train(Path(plan["qualified_targets"]), features, Path(plan["training_output"]), Path(plan["evaluation_design"]))
        status["training_manifest"] = str(Path(plan["training_output"]) / "manifest.json")
        status["complete"] = True
        completed("all_four", stage)
        return result
    except Exception as error:
        status["incomplete_stage"] = {"lecture": lesson, "stage": stage, "error_type": type(error).__name__}
        guard(status_path)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan)))
