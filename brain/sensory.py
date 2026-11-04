"""Portable sensory extraction with the same definitions as the training corpus."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .content_windows import point_means, interval_means
from .edugage_targets import guard
from .sensory_core import (extract_audio_descriptors, extract_visual_descriptors,
    iter_local_audio_blocks, iter_local_video_frames)
from .worker_contract import PublicMediaRequest, validate_media, file_sha256


def inspect_media(media: Path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(media)],
                            capture_output=True, check=True, text=True)
    header = json.loads(result.stdout)
    streams = {kind: [s for s in header["streams"] if s["codec_type"] == kind] for kind in ("audio", "video")}
    if any(len(value) != 1 for value in streams.values()):
        raise ValueError("one audio and one video track required by the video-trained model")
    selected = [streams["video"][0], streams["audio"][0]]
    starts = [float(s["start_time"]) for s in selected] + [float(header["format"]["start_time"])]
    if any(not math.isfinite(s) or s != 0 for s in starts):
        raise ValueError("nonzero media origins require an explicitly audited normalization; no silent time shift")
    durations = [float(s["duration"]) for s in selected] + [float(header["format"]["duration"])]
    if any(not math.isfinite(d) or d <= 0 for d in durations):
        raise ValueError("finite declared media durations required")
    duration = math.floor(min(durations) * 64) / 64
    return header, duration, selected[0]["index"], selected[1]["index"]


def extract(media: Path, request: PublicMediaRequest, output: Path, *, qualification_prefix=None):
    validate_media(media, request)
    if request.media_kind != "video":
        raise ValueError("explicit video presentation required")
    header, full_duration, video_index, audio_index = inspect_media(media)
    duration = full_duration if qualification_prefix is None else float(qualification_prefix)
    if not 0 < duration <= full_duration or duration * 64 != round(duration * 64):
        raise ValueError("qualification prefix must lie on the native descriptor clock")
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    def checked_frames():
        for i, frame in enumerate(iter_local_video_frames(media, stream_index=video_index,
                media_presentation_offset_seconds=0, duration_seconds=duration, maximum_width_pixels=320)):
            if i % 300 == 0:
                guard(output)
            yield frame
    def checked_audio():
        for i, block in enumerate(iter_local_audio_blocks(media, stream_index=audio_index,
                media_presentation_offset_seconds=0, duration_seconds=duration, sample_rate_hz=16000)):
            if i % 1000 == 0:
                guard(output)
            yield block
    streams = [extract_visual_descriptors(checked_frames(), scene_change_threshold=0.3),
               extract_audio_descriptors(checked_audio(), maximum_absolute_sample=2.0)]
    arrays, descriptions = {}, []
    for i, stream in enumerate(streams):
        for field in ("times_seconds", "window_start_seconds", "window_stop_seconds", "values", "observed_mask"):
            arrays[f"s{i}_{field}"] = getattr(stream, field)
        descriptions.append({"name": stream.name, "columns": list(stream.columns), "provenance": dict(stream.provenance)})
    if file_sha256(media) != request.sha256:
        raise ValueError("media changed during sensory extraction")
    np.savez_compressed(output / "features.npz", **arrays)
    (output / "header.json").write_text(json.dumps(header, indent=2) + "\n")
    manifest = {"schema": "perception-sensory-v1", "source": request.model_dump(), "duration_seconds": duration,
        "declared_common_duration": full_duration, "qualification_prefix_only": qualification_prefix is not None,
        "header_sha256": file_sha256(output / "header.json"), "features_sha256": file_sha256(output / "features.npz"),
        "streams": descriptions, "implementation_sha256": file_sha256(Path(__file__)),
        "core_sha256": file_sha256(Path(__file__).with_name("sensory_core.py")),
        "duration_policy": "floor(minimum declared audio/video/container duration * 64) / 64",
        "historical_media_identity_established": False, "is_neural_or_attention_measure": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def window_blocks(directory: Path, queries, source_sha256):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "perception-sensory-v1" or manifest["source"]["sha256"] != source_sha256:
        raise ValueError("sensory source identity mismatch")
    path = directory / "features.npz"
    if file_sha256(path) != manifest["features_sha256"]:
        raise ValueError("sensory feature checksum mismatch")
    blocks = {}
    with np.load(path, allow_pickle=False) as archive:
        for i, stream in enumerate(manifest["streams"]):
            values, observed = archive[f"s{i}_values"], archive[f"s{i}_observed_mask"]
            if stream["name"] == "visual_descriptors":
                blocks[stream["name"]] = point_means(archive[f"s{i}_times_seconds"], values, observed, queries)
            elif stream["name"] == "audio_descriptors":
                blocks[stream["name"]], _ = interval_means(archive[f"s{i}_window_start_seconds"],
                    archive[f"s{i}_window_stop_seconds"], values, observed, queries)
            else:
                raise ValueError("unexpected sensory representation")
    if set(blocks) != {"visual_descriptors", "audio_descriptors"}:
        raise ValueError("both required sensory representations must be present")
    return blocks, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("media", "request", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.media, PublicMediaRequest.model_validate_json(args.request.read_text()), args.output)))
