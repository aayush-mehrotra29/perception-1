"""One video → real content/brain features → optional locally fitted EEG head.

Remote stages receive only the authenticated permitted media. Participant data,
labels, and fitted EEG weights are never part of a remote request.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from .audio_derivative import extract as extract_audio
from .credentials import load_credentials
from .credits import CreditLedger
from .edugage_targets import guard
from .eeg import EEGModel
from .lesson_features import assemble
from .lesson_meta import run_sam, run_voice_semantics
from .media_frames import extract as extract_frames
from .meta_voice import run as run_voice
from .modal_client import submit
from .predict_bundle import predict
from .sensory import extract as extract_sensory, inspect_media
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


def run(media: Path, request: PublicMediaRequest, output: Path, request_prefix: str, *, model_directory=None, fusion_model=None):
    if model_directory is not None and fusion_model is not None:
        raise ValueError("select either the legacy EEG model or fusion_model")
    if fusion_model is not None:
        from .fusion_inference import preflight
        preflight(fusion_model)
    validate_media(media, request)
    if request.media_kind != "video":
        raise ValueError("v1 requires a timed video presentation; non-video transfer adapters are not yet qualified")
    # Reject an absent/corrupt EEG artifact before paid inference is started.
    if model_directory is not None:
        EEGModel.load(model_directory)
    guard(output)
    _, duration, _, _ = inspect_media(media)
    if (model_directory is not None or fusion_model is not None) and duration < 45:
        raise ValueError("this EEG head requires at least one complete 44-second context at a 5-second query")
    output.mkdir(parents=True, exist_ok=False)
    load_credentials()
    extract_audio(media, request, output / "audio")
    extract_frames(media, request, output / "frames")
    extract_sensory(media, request, output / "sensory")
    ledger = CreditLedger(Path(".local/brain-credits.sqlite3"))

    def gpu_branch():
        profile = "long" if duration >= 600 else "standard"
        submit("tribe", media, request, output / "tribe", request_prefix + "-tribe",
               runtime_profile=profile, reservation_microdollars=6000000 if profile == "long" else 4000000)
        audio_request = PublicMediaRequest.model_validate_json((output / "audio/request.json").read_text())
        submit("sam_audio", output / "audio/audio.wav", audio_request, output / "sam_audio",
               request_prefix + "-sam-audio", reservation_microdollars=2000000)

    def visual_branch():
        asyncio.run(run_sam(output / "frames", output / "sam_visual", request_prefix, ledger))

    def speech_branch():
        run_voice(output / "audio", output / "voice", request_prefix)
        run_voice_semantics(output / "voice", output / "semantics", request_prefix, ledger)

    stage_status = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(function): name for name, function in
                   (("gpu", gpu_branch), ("visual", visual_branch), ("speech_semantics", speech_branch))}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                stage_status[name] = {"status": "complete"}
            except Exception as error:
                # Preserve failed job identity and completed sibling artifacts;
                # never substitute a mock output or resubmit a paid request.
                stage_status[name] = {"status": "incomplete", "error_type": type(error).__name__}
            (output / "stage-status.json").write_text(json.dumps(stage_status, indent=2) + "\n")
    if any(value["status"] != "complete" for value in stage_status.values()):
        raise RuntimeError("pipeline incomplete; inspect saved stages and cloud dispatch handles before recovery")
    spec = {"schema": "perception-lesson-feature-sources-v1", "source_sha256": request.sha256,
        "lecture": request.source_id, "media_duration": duration,
        **{name: str((output / folder).resolve()) for name, folder in
           (("tribe", "tribe"), ("semantics", "semantics"), ("sam_visual", "sam_visual"),
            ("frames", "frames"), ("sam_audio", "sam_audio"), ("audio_derivative", "audio"), ("sensory_directory", "sensory"))}}
    (output / "feature-sources.json").write_text(json.dumps(spec, indent=2) + "\n")
    assemble(spec, np.arange(0, duration + 1e-8, 5.0), output / "features", allow_prototype_timing=True)
    if model_directory is not None:
        predict(output / "features", model_directory, output / "tribe", output / "prediction")
    elif fusion_model is not None:
        from .fusion_inference import predict_native
        predict_native(output / "tribe", fusion_model, output / "prediction", duration=duration,
                       input_format="video", features=output / "features")
    selected_model = fusion_model if fusion_model is not None else model_directory
    result = {"schema": "perception-video-brain-pipeline-v1", "source": request.model_dump(),
        "content_features_complete": True, "eeg_prediction_emitted": selected_model is not None,
        "feature_manifest_sha256": file_sha256(output / "features/manifest.json"),
        "model_manifest_sha256": file_sha256(Path(selected_model) / "manifest.json") if selected_model is not None else None,
        "prediction_bundle": "prediction" if selected_model is not None else None,
        "EEG_model_family": ("bounded_residual_fusion_v3" if json.loads((Path(fusion_model) / "manifest.json").read_text()).get("schema") == "perception-fusion-deployment-v3" else "temporal_fusion_v2") if fusion_model is not None else "legacy_v1",
        "scientific_accuracy_validated": False, "participant_EEG_sent_to_cloud": False,
        "analysis_mode": "whole_lesson_offline", "timing_and_domain_status": "unvalidated_prototype"}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("media", "request", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--eeg-model", type=Path)
    parser.add_argument("--fusion-model", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.media, PublicMediaRequest.model_validate_json(args.request.read_text()), args.output,
                         args.request_prefix, model_directory=args.eeg_model, fusion_model=args.fusion_model)))
