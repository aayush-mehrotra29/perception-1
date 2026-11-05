"""One entry point for timed videos/audio and explicitly timed text/documents.

Optional lesson_model adds experimental five-quality timelines. Legacy EEG paths
remain selectable. Neither path executes synthetic students or predicts retention.
"""
import argparse
import asyncio
import json
from pathlib import Path

from .audio_derivative import extract as extract_audio
from .credentials import load_credentials
from .credits import CreditLedger
from .document_presentation import prepare
from .edugage_targets import guard
from .eeg import EEGModel
from .lesson_meta import _run_semantic_spans, run_sam, run_voice_semantics
from .media_frames import extract as extract_frames
from .meta_features import ContentSpan
from .meta_voice import run as run_voice
from .modal_client import submit
from .pipeline import run as run_video
from .presentation import text_schedule, page_words
from .transfer import MODES, predict as predict_transfer
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


def _text_spans(text, schedule):
    bins = {}
    for word in schedule["words"]:
        bins.setdefault(int(word["start"] // 30), []).append(word)
    return [ContentSpan(id=f"span-{index:04d}", start=index * 30,
        stop=min((index + 1) * 30, schedule["duration_seconds"]),
        text=text[words[0]["character_start"]:words[-1]["character_stop"]])
        for index, words in sorted(bins.items())]


def _load_head(root, mode):
    if root is None:
        return None
    path = root / mode / "exploratory-deployment"
    model = EEGModel.load(path)
    metadata = json.loads((path / "transfer.json").read_text())
    if (set(model.blocks) != set(MODES[mode]) or metadata.get("mode") != mode
            or metadata.get("model_manifest_sha256") != file_sha256(path / "manifest.json")):
        raise ValueError("matching trained transfer head required before paid inference")
    return path


def _run_content(source: Path, request: PublicMediaRequest, output: Path, request_prefix: str, *,
        video_model=None, transfer_models=None, fusion_model=None, words_per_minute=None, seconds_per_page=None):
    if fusion_model is not None:
        if video_model is not None or transfer_models is not None:
            raise ValueError("fusion_model is exclusive with legacy video_model and transfer_models")
        from .fusion_inference import preflight
        preflight(fusion_model)
    validate_media(source, request)
    if request.media_kind == "video":
        if words_per_minute is not None or seconds_per_page is not None:
            raise ValueError("video already has a native clock")
        return run_video(source, request, output, request_prefix, model_directory=video_model, fusion_model=fusion_model)
    if request.media_kind != "text" and words_per_minute is not None:
        raise ValueError("reading pace applies only to text")
    if request.media_kind not in {"pdf", "image"} and seconds_per_page is not None:
        raise ValueError("page timing applies only to PDF/image inputs")
    if request.media_kind == "text" and words_per_minute is None:
        raise ValueError("text requires an explicit assumed reading pace")
    if request.media_kind in {"image", "pdf"} and seconds_per_page is None:
        raise ValueError("documents require an explicit duration for each page")
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    load_credentials()
    ledger = CreditLedger(Path(".local/brain-credits.sqlite3"))
    media, native_request, presentation_manifest = source, request, None
    if request.media_kind == "text":
        text = source.read_bytes().decode("utf-8")
        schedule = text_schedule(text, words_per_minute=words_per_minute)
        duration, mode = schedule["duration_seconds"], "text"
        basis = f"assumed constant reading pace: {words_per_minute} words/minute; not measured gaze"
        (output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    elif request.media_kind == "audio":
        audio = extract_audio(source, request, output / "audio")
        native_request = PublicMediaRequest.model_validate(audio["derivative"])
        media = output / "audio/audio.wav"
        duration, mode = audio["samples"] / audio["sample_rate"], "audio"
        basis = "authenticated decoded audio timeline, beginning at sample zero; parent clock in audio manifest"
    else:
        prepared = prepare(source, request, output / "presentation", seconds_per_page=seconds_per_page)
        native_request = PublicMediaRequest.model_validate(prepared["derivative"])
        media = output / "presentation/presentation.mp4"
        presentation_manifest = output / "presentation/manifest.json"
        presentation = {"input_format": request.media_kind, "schedule": prepared["schedule"], "pages": prepared["pages"]}
        duration = prepared["schedule"]["duration_seconds"]
        mode = "document" if page_words(presentation) else "visual"
        basis = "explicit assumed page display times; extracted PDF word order and uniform within-page pace are unverified"
    model = Path(fusion_model) if fusion_model is not None else _load_head(transfer_models, mode)
    if model is not None and duration < 45:
        raise ValueError("this EEG head requires at least one complete 44-second context at a 5-second query")
    stage = "native_tribe"
    try:
        submit("tribe", media, native_request, output / "tribe", request_prefix + "-tribe",
            reservation_microdollars=2000000 if mode == "text" else 4000000,
            reading_words_per_minute=words_per_minute, presentation_manifest=presentation_manifest)
        stage = "applicable_content_features"
        branches = {"tribe": "complete", "sam_visual": "not_applicable", "sam_audio": "not_applicable",
                    "meta_semantics": "not_applicable", "meta_voice": "not_applicable"}
        if mode == "text":
            _run_semantic_spans(_text_spans(text, schedule), request, duration, output / "semantics",
                request_prefix, ledger, {"span_policy": basis, "source_text_sha256": request.sha256})
            branches["meta_semantics"] = "complete"
        elif mode == "audio":
            submit("sam_audio", media, native_request, output / "sam_audio", request_prefix + "-sam-audio",
                   reservation_microdollars=2000000)
            run_voice(output / "audio", output / "voice", request_prefix)
            run_voice_semantics(output / "voice", output / "semantics", request_prefix, ledger)
            branches.update(sam_audio="complete", meta_voice="complete", meta_semantics="complete")
        else:
            extract_frames(media, native_request, output / "frames")
            asyncio.run(run_sam(output / "frames", output / "sam_visual", request_prefix, ledger))
            branches["sam_visual"] = "complete"
            spans = [ContentSpan(id=f"page-{page['source_page_index']:04d}", start=slot["start"],
                                stop=slot["start"] + slot["duration"], text=page["text"])
                     for page, slot in zip(prepared["pages"], prepared["schedule"]["pages"]) if page["text"].strip()]
            if spans:
                _run_semantic_spans(spans, request, duration, output / "semantics", request_prefix, ledger,
                                    {"span_policy": basis, "source_document_sha256": request.sha256})
                branches["meta_semantics"] = "complete"
        stage = "local_eeg_and_joint_bundle"
        if model is not None:
            presentation_info = {"input_format": request.media_kind, "schedule_basis": basis,
                                 "transfer_status": "experimental_transfer"}
            if fusion_model is not None:
                from .fusion_inference import predict_native
                predict_native(output / "tribe", model, output / "prediction", input_format=request.media_kind,
                               duration=duration, presentation=presentation_info)
            else:
                predict_transfer(output / "tribe", model, output / "prediction", mode=mode, duration=duration,
                                 presentation=presentation_info)
        guard(output)
        result = {"schema": "perception-content-brain-pipeline-v1", "source": request.model_dump(),
            "native_inference_source": native_request.model_dump(), "input_format": request.media_kind,
            "mode": mode, "branches": branches, "eeg_prediction_emitted": model is not None,
            "prediction_bundle": "prediction" if model is not None else None,
            "model_manifest_sha256": file_sha256(model / "manifest.json") if model is not None else None,
            "native_manifest_sha256": file_sha256(output / "tribe/manifest.json"),
            "transfer_status": "experimental_transfer", "schedule_basis": basis,
            "EEG_predictors": list(MODES[mode]) + (["predicted_tribe_cortex"] if fusion_model is not None else []),
            "EEG_model_family": ("bounded_residual_fusion_v3" if json.loads((Path(fusion_model) / "manifest.json").read_text()).get("schema") == "perception-fusion-deployment-v3" else "temporal_fusion_v2") if fusion_model is not None else "legacy_v1",
            "supplemental_Meta_features_used_by_transfer_EEG_head": False,
            "fMRI_used_by_transfer_EEG_head": fusion_model is not None, "participant_EEG_sent_to_cloud": False,
            "attention_or_retention_predictions": False, "scientific_accuracy_validated": False}
        (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as error:
        guard(output)
        (output / "incomplete.json").write_text(json.dumps({"stage": stage,
            "error_type": type(error).__name__, "paid_requests_automatically_retried": False}, indent=2) + "\n")
        raise


def run(source: Path, request: PublicMediaRequest, output: Path, request_prefix: str, *,
        video_model=None, transfer_models=None, fusion_model=None, lesson_model=None, composite_model=None,
        words_per_minute=None, seconds_per_page=None, narration_captions=None,
        narration_voice='Samantha (English (US))', narration_rate=180):
    if composite_model is not None:
        if any(x is not None for x in [video_model,transfer_models,fusion_model,lesson_model]):
            raise ValueError('composite_model is exclusive with other model selectors')
        if words_per_minute is not None or seconds_per_page is not None:
            raise ValueError('composite narration uses actual audio duration, not an assumed silent reading clock')
        from numerical_brain_v1.pipeline import run as run_composite
        return run_composite(source,request,output,request_prefix,composite_model=composite_model,
            caption_records=narration_captions,voice=narration_voice,narration_rate=narration_rate)
    if lesson_model is not None:
        if any(x is not None for x in [video_model,transfer_models,fusion_model]):
            raise ValueError("lesson_model is exclusive with legacy EEG model selection")
        from lesson_harness_v1.artifact import preflight
        from lesson_harness_v1.inference import preflight_runtime
        preflight(lesson_model)
        guard(output)
        preflight_runtime()  # Fail before paid calls if local inference cannot run.
    result=_run_content(source,request,output,request_prefix,video_model=video_model,
        transfer_models=transfer_models,fusion_model=fusion_model,words_per_minute=words_per_minute,
        seconds_per_page=seconds_per_page)
    if lesson_model is None:return result
    from lesson_harness_v1.integration import attach
    try:
        return attach(output,request,lesson_model,result)
    except Exception as error:
        (Path(output)/'lesson-incomplete.json').write_text(json.dumps({'stage':'lesson_quality_inference',
            'error_type':type(error).__name__,'content_and_TRIBE_outputs_preserved':True,
            'paid_calls_automatically_retried':False},indent=2)+'\n')
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "request", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--video-model", type=Path)
    parser.add_argument("--transfer-models", type=Path)
    parser.add_argument("--fusion-model", type=Path)
    parser.add_argument("--lesson-model", type=Path)
    parser.add_argument("--composite-model", type=Path)
    parser.add_argument("--narration-captions", type=Path)
    parser.add_argument("--narration-voice", default='Samantha (English (US))')
    parser.add_argument("--narration-rate", type=int, default=180)
    parser.add_argument("--words-per-minute", type=float)
    parser.add_argument("--seconds-per-page", nargs="+", type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.source, PublicMediaRequest.model_validate_json(args.request.read_text()), args.output,
        args.request_prefix, video_model=args.video_model, transfer_models=args.transfer_models,
        fusion_model=args.fusion_model, lesson_model=args.lesson_model, composite_model=args.composite_model,
        narration_captions=json.loads(args.narration_captions.read_text()) if args.narration_captions else None,
        narration_voice=args.narration_voice, narration_rate=args.narration_rate,
        words_per_minute=args.words_per_minute, seconds_per_page=args.seconds_per_page)))
