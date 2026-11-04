"""Native inference entry points for isolated CUDA environments.

No model is imported or downloaded on module import. These routines require a
budgeted caller. Both native paths have passed small real GPU runs. All outputs are private
run artifacts, never a claim of measured participant responses.
"""
from pathlib import Path
import importlib.metadata
import json

from .worker_contract import PublicMediaRequest, file_sha256, segment_record, validate_media

TRIBE_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
SAM_AUDIO_REVISION = "20b65f56888142eebe7c37448c6f6b3b32600e9b"
ENCODER_REVISIONS = {
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "facebook/w2v-bert-2.0": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    "facebook/vjepa2-vitg-fpc64-256": "875c192b7b704b87d1e1d99345769632dd5f739a",
}


def run_tribe(media: Path, request: PublicMediaRequest, output: Path, cache: Path, *, presentation=None):
    validate_media(media, request)
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download, HfApi
    from tribev2 import TribeModel
    from .dense_capture import DenseFeatureCapture
    from .tribe_events import build_events
    from .video_frame_cache import install as install_frame_cache
    from .presentation import text_schedule, native_text_events, page_words, native_page_events

    if request.media_kind == "text":
        if not isinstance(presentation, dict) or set(presentation) != {"input_format", "words_per_minute"} or presentation["input_format"] != "text":
            raise ValueError("text requires an explicit assumed reading pace")
        source_text = media.read_bytes().decode("utf-8")
        schedule = text_schedule(source_text, words_per_minute=presentation["words_per_minute"])
        if schedule["duration_seconds"] > 1800:
            raise ValueError("text inference exceeds the 30-minute artifact transport limit")
        modalities = ("text",)
    elif request.media_kind == "video" and presentation is not None:
        words = page_words(presentation)
        schedule = presentation["schedule"]
        modalities = ("text", "video") if words else ("video",)
    else:
        if presentation is not None:
            raise ValueError("timed audio/video uses its authenticated native clock")
        schedule = None
        modalities = ("text", "audio") if request.media_kind == "audio" else ("text", "audio", "video")
    repositories = {"text": "meta-llama/Llama-3.2-3B", "audio": "facebook/w2v-bert-2.0",
                    "video": "facebook/vjepa2-vitg-fpc64-256"}
    used_revisions = {repositories[m]: ENCODER_REVISIONS[repositories[m]] for m in modalities}

    if not torch.cuda.is_available():
        raise RuntimeError("TRIBE worker requires qualified CUDA execution")
    # Reuse only identical timestamp requests within the same subclip. Real
    # native pixel inputs were compared before enabling this decoding cache;
    # sampling, resizing, encoder weights and neural features are unchanged.
    install_frame_cache()
    # Upstream extractors resolve main internally rather than accepting a
    # revision option. Fail on a changed revision, and verify the actual cache
    # refs again after inference instead of silently mixing encoder versions.
    api = HfApi()
    for repository, revision in used_revisions.items():
        if api.model_info(repository).sha != revision:
            raise ValueError(f"encoder revision changed: {repository}")
    checkpoint = snapshot_download("facebook/tribev2", revision=TRIBE_REVISION,
                                   allow_patterns=["best.ckpt", "config.yaml"], cache_dir=str(cache / "hub"))
    execution_overrides = {"data.num_workers": 0, "data.batch_size": 1}
    if presentation is not None or request.media_kind == "audio":
        execution_overrides["data.features_to_use"] = list(modalities)
    model = TribeModel.from_pretrained(checkpoint, cache_folder=str(cache / "features"), device="cuda",
                                      config_update=execution_overrides)
    if request.media_kind == "text":
        events, event_audit = native_text_events(source_text, schedule)
    elif presentation is not None:
        events, event_audit = native_page_events(media, presentation)
    else:
        events, event_audit = build_events(media, request.media_kind)
    capture = DenseFeatureCapture(output, float(model.data.frequency))
    handle = model._model.register_forward_pre_hook(capture)
    try:
        predictions, segments = model.predict(events)
    finally:
        handle.remove()
    # Replay the unmodified upstream inference path against the same cached
    # features to establish that capture itself did not alter predictions.
    native_predictions, native_segments = model.predict(events, verbose=False)
    if [segment_record(s) for s in segments] != [segment_record(s) for s in native_segments]:
        raise ValueError("capture changed native output segment identities")
    np.testing.assert_allclose(predictions, native_predictions, rtol=1e-5, atol=1e-6)
    capture_error = float(np.max(np.abs(predictions - native_predictions)))
    observed_revisions = {}
    for repository, revision in used_revisions.items():
        ref = cache / "huggingface/hub" / ("models--" + repository.replace("/", "--")) / "refs/main"
        observed_revisions[repository] = ref.read_text().strip()
        if observed_revisions[repository] != revision or api.model_info(repository).sha != revision:
            raise ValueError(f"encoder revision mismatch after extraction: {repository}")
    if predictions.ndim != 2 or len(predictions) != len(segments) or not np.isfinite(predictions).all():
        raise ValueError("invalid native TRIBE output")
    np.savez(output / "fmri.npz", predicted_cortical_response=predictions)
    report = {"schema": "perception-native-tribe-v1", "source": request.model_dump(),
              "checkpoint_revision": TRIBE_REVISION, "checkpoint_sha256": file_sha256(Path(checkpoint) / "best.ckpt"),
              "upstream_config_sha256": file_sha256(Path(checkpoint) / "config.yaml"),
              "segments": [segment_record(s) for s in segments], "shape": list(predictions.shape),
              "array_sha256": file_sha256(output / "fmri.npz"),
              "torch_version": torch.__version__, "tribe_version": importlib.metadata.version("tribev2"),
              "subject": "native average-subject prediction", "analysis_mode": "whole_lesson_offline",
              "input_modalities": list(modalities), "presentation": presentation,
              "assumed_presentation_schedule": schedule,
              "missing_modalities": sorted({"text", "audio", "video"} - set(modalities)),
              "missing_modality_policy": "omitted from native batch; pretrained TRIBE's own missing-modality path is retained",
              "transfer_status": "experimental_transfer" if schedule is not None else "native_timed_media",
              "native_equivalence_independently_checked": False,
              "event_preprocessing_audit": event_audit,
              "pinned_demo_preprocessing_unmodified": False,
              "capture_equivalence_verified": True,
              "capture_max_absolute_error": capture_error,
              "capture_equivalence_scope": "same checkpoint/events/execution overrides, capture hook versus unmodified upstream forward; not independent scientific replication",
              "execution_overrides": execution_overrides,
              "video_decode_cache": {"method": "exact_timestamp_per_subclip_lru",
                                     "maximum_bytes_per_subclip": 256 * 1024**2,
                                     "qualification": "perception-exact-frame-cache-qualification-v1",
                                     "qualification_pixels_identical": True,
                                     "full_model_equivalence_independently_verified": False},
              "dense_features": capture.records,
              "neuro_offset_seconds": float(model.data.neuro.offset),
              "spatial_metadata": {"mesh": model.data.neuro.projection.mesh,
                                   "vertex_order_qualified": False},
              "runtime_versions": {name: importlib.metadata.version(name) for name in
                  ("neuralset", "neuraltrain", "transformers", "huggingface_hub", "exca")},
              "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
              "subordinate_encoder_revisions_qualified": False,
              "content_encoder_revisions_verified": True,
              "asr_and_alignment_environment_fully_pinned": False,
              "observed_encoder_revisions": observed_revisions,
              "worker_source_sha256": file_sha256(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_sam_audio(media: Path, request: PublicMediaRequest, output: Path, cache: Path, commit_cache=None):
    validate_media(media, request)
    if request.media_kind != "audio":
        raise ValueError("SAM Audio receives a separately authenticated decoded audio derivative")
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download
    from sam_audio import SAMAudio, SAMAudioProcessor
    from .sam_compat import install
    from .audio_arrays import mono_array
    from .separation_features import summarize, FEATURES
    import hashlib
    import shutil
    import time

    if not torch.cuda.is_available():
        raise RuntimeError("SAM Audio worker requires qualified CUDA execution")
    checkpoint = snapshot_download("facebook/sam-audio-small", revision=SAM_AUDIO_REVISION, cache_dir=str(cache / "hub"))
    install()
    model = SAMAudio.from_pretrained(checkpoint).eval().cuda()
    processor = SAMAudioProcessor.from_pretrained(checkpoint)
    audio, rate = torchaudio.load(str(media))
    if audio.shape[0] != 1 or rate != processor.audio_sampling_rate:
        raise ValueError("caller must provide an authenticated mono derivative at the processor rate")
    settings = {"checkpoint_revision": SAM_AUDIO_REVISION, "prompt": "speech", "chunk_seconds": 30,
        "predict_spans": False, "reranking_candidates": 1, "torch_version": torch.__version__,
        "descriptor_code_sha256": file_sha256(Path(__file__).with_name("separation_features.py"))}
    cache_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    chunk_cache = cache / "sam-audio-chunks-v1" / cache_key / request.sha256
    chunk_cache.mkdir(parents=True, exist_ok=True)
    rows, reused = [], 0
    for index, start in enumerate(range(0, audio.shape[1], rate * 30)):
        stop = min(start + rate * 30, audio.shape[1])
        seed = (int(request.sha256[:16], 16) ^ start) % (2**63)
        path = output / f"chunk-{index:04d}.npz"
        saved_array, saved_meta = chunk_cache / path.name, chunk_cache / f"chunk-{index:04d}.json"
        if saved_meta.exists():
            row = json.loads(saved_meta.read_text())
            if (row["start_sample"] != start or row["stop_sample"] != stop or row["sample_rate"] != rate
                    or row["seed"] != seed or row["valid_sample_count"] != stop - start
                    or file_sha256(saved_array) != row["array_sha256"]):
                raise ValueError("cached SAM Audio chunk failed integrity check")
            shutil.copyfile(saved_array, path)
            rows.append(row)
            reused += 1
            continue
        started = time.monotonic()
        chunk = audio[:, start:stop]
        batch = processor(audios=[chunk], descriptions=["speech"]).to("cuda")
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)
        target, residual = mono_array(result.target), mono_array(result.residual)
        if target.shape[1] < stop - start or residual.shape[1] < stop - start:
            raise ValueError("separated output shorter than original chunk")
        # Full-lesson waveforms can exceed the transport limit. Return fixed,
        # lossily summarized content features; preserve native waveforms only
        # for short qualification inputs, not for full lecture uploads.
        arrays = summarize(target, residual, rate, start, stop - start)
        if audio.shape[1] <= rate * 30:
            arrays.update(target=target, residual=residual)
        np.savez(path, **arrays)
        row = {"start_sample": start, "stop_sample": stop, "sample_rate": rate,
                     "seed": seed, "valid_sample_count": stop - start,
                     "array_sha256": file_sha256(path), "target_shape": list(target.shape),
                     "residual_shape": list(residual.shape), "extraction_seconds": time.monotonic() - started}
        temporary_array = saved_array.with_suffix(".npz.partial")
        shutil.copyfile(path, temporary_array)
        temporary_array.replace(saved_array)
        temporary_meta = saved_meta.with_suffix(".json.partial")
        temporary_meta.write_text(json.dumps(row, indent=2) + "\n")
        temporary_meta.replace(saved_meta)
        if commit_cache is not None:
            commit_cache()
        rows.append(row)
        print(json.dumps({"stage": "sam_audio_chunk", "chunk": index, "seconds": row["extraction_seconds"]}), flush=True)
        del batch, result, target, residual
    report = {"schema": "perception-native-sam-audio-v1", "source": request.model_dump(),
              "checkpoint_revision": SAM_AUDIO_REVISION, "chunks": rows,
              "chunk_cache_settings": settings, "reused_verified_chunks": reused,
              "prompt": "speech", "predict_spans": False, "reranking_candidates": 1,
              "hub_compatibility": "restore neutral proxies=None and resume_download=False at SAM BaseModel loader boundary",
              "native_waveform_dtype": "float32", "descriptor_dtype": "float64",
              "batch_outputs": "one native mono item; trailing model padding excluded from descriptors and bounded by valid_sample_count",
              "feature_names": FEATURES, "feature_interval_seconds": 0.5,
              "waveforms_retained": audio.shape[1] <= rate * 30,
              "features_exclude_trailing_padding": True,
              "chunk_boundary_policy": "independent 30-second separation; no cross-chunk context or blending",
              "runtime_qualified": False}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
