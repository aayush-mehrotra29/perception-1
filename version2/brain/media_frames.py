"""Stream native video frames for SAM; preserve measured PTS and file identity."""
import argparse
import json
from pathlib import Path

from .edugage_targets import guard
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


def extract(media: Path, request: PublicMediaRequest, output: Path, every_seconds=10.0):
    import av
    if request.media_kind != "video" or not 1 <= every_seconds <= 60:
        raise ValueError("video and an explicit 1–60 second sampling interval required")
    validate_media(media, request)
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    records, previous, next_sample, origin = [], None, None, None
    with av.open(str(media)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("one unambiguous video stream required")
        stream = container.streams.video[0]
        stream.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("native presentation timestamp missing")
            timestamp = float(frame.pts * frame.time_base)
            if previous is not None and timestamp <= previous:
                raise ValueError("non-increasing native presentation timestamps")
            previous = timestamp
            if origin is None:
                origin = timestamp
                next_sample = timestamp
            if timestamp < next_sample:
                continue
            guard(output)
            name = f"frame-{len(records):04d}.png"
            frame.to_image().save(output / name)
            records.append({"file": name, "sha256": file_sha256(output / name),
                            "native_pts": int(frame.pts), "time_base": str(frame.time_base),
                            "native_timestamp_seconds": timestamp,
                            "seconds_from_first_video_frame": timestamp - origin,
                            "width": frame.width, "height": frame.height})
            # Choose the first actually decoded frame at/after each scheduled
            # timestamp. Do not manufacture a frame or a nominal frame clock.
            next_sample += every_seconds
            while next_sample <= timestamp:
                next_sample += every_seconds
    if not records or file_sha256(media) != request.sha256:
        raise ValueError("empty media or source changed during extraction")
    manifest = {"schema": "perception-native-frames-v1", "source": request.model_dump(),
                "sampling_seconds": every_seconds, "origin_native_seconds": origin,
                "frames": records, "pyav_version": av.__version__,
                "last_decoded_frame_seconds": previous, "video_EOF_reached": True,
                "mapping_to_EEG_clock_established": False,
                "selection": "first decoded frame at/after each scheduled timestamp; original resolution"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = extract(args.media, PublicMediaRequest.model_validate_json(args.request.read_text()), args.output)
    print(json.dumps({"frames": len(manifest["frames"]), "output": str(args.output)}))
