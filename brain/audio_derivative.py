"""Authenticate and decode one continuous audio track for the SAM worker."""
import argparse
import json
from pathlib import Path
import subprocess
import wave

from .edugage_targets import guard
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


def extract(media: Path, request: PublicMediaRequest, output: Path):
    import av
    validate_media(media, request)
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    first, stop, count, rate, last_frame_samples = None, None, 0, None, 0
    with av.open(str(media)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("one unambiguous audio track required")
        stream = container.streams.audio[0]
        declared_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        stream.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("missing audio clock")
            t = float(frame.pts * frame.time_base)
            if rate is not None and (frame.sample_rate != rate or abs(t - stop) > 2 / rate):
                raise ValueError("discontinuous audio; no automatic gap repair")
            rate = frame.sample_rate
            first = t if first is None else first
            stop = t + frame.samples / rate
            count += frame.samples
            last_frame_samples = frame.samples
    if first is None or not count:
        raise ValueError("empty audio track")
    path = output / "audio.wav"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0", "-vn",
               "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", "-n", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    with wave.open(str(path)) as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 48000 or audio.getsampwidth() != 2:
            raise ValueError("unexpected decoded audio format")
        samples = audio.getnframes()
    decoded_duration = samples / 48000
    discarded_tail_seconds = count / rate - decoded_duration
    if abs(discarded_tail_seconds) > 2 / 48000:
        # AAC may declare a trimmed final frame. FFmpeg honors that duration
        # while PyAV exposes padding. Accept only a documented end trim.
        if (declared_duration is None or abs(decoded_duration - declared_duration) > 2 / 48000
                or not 0 <= discarded_tail_seconds <= last_frame_samples / rate):
            raise ValueError("decoded duration changed without a declared final-frame trim")
    if file_sha256(media) != request.sha256:
        raise ValueError("source changed during decoding")
    derivative = request.model_copy(update={"source_id": request.source_id + "-mono48k",
        "sha256": file_sha256(path), "size_bytes": path.stat().st_size, "media_kind": "audio"})
    # Validate the copy against the same strict public-media contract.
    derivative = PublicMediaRequest.model_validate(derivative.model_dump())
    (output / "request.json").write_text(derivative.model_dump_json(indent=2) + "\n")
    manifest = {"schema": "perception-audio-derivative-v1", "parent": request.model_dump(),
        "derivative": derivative.model_dump(), "first_native_audio_seconds": first,
        "last_native_audio_stop_seconds": stop, "native_rate": rate, "native_samples": count,
        "declared_audio_duration": declared_duration, "discarded_decoded_tail_seconds": discarded_tail_seconds,
        "samples": samples, "sample_rate": 48000, "downmix": "ffmpeg mono default",
        "pcm": "signed 16-bit", "audio_clock_continuity_checked": True,
        "historical_EEG_synchronization_established": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.media, PublicMediaRequest.model_validate_json(args.request.read_text()), args.output)
    print(json.dumps({"samples": result["samples"], "output": str(args.output)}))
