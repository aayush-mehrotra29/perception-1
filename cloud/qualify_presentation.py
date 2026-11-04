"""CPU-only qualification of assumed presentation timing in the native library."""
import modal
from cloud.modal_brain import tribe_image

app = modal.App("perception-text-event-qualification")


@app.function(image=tribe_image, cpu=2, memory=8192, timeout=300, retries=0, max_containers=1)
def qualify(text: str, words_per_minute: float):
    from brain.presentation import text_schedule, native_text_events
    schedule = text_schedule(text, words_per_minute=words_per_minute)
    events, audit = native_text_events(text, schedule)
    return {"schema": "perception-native-text-preprocessing-qualification-v1",
        "schedule": schedule, "audit": audit,
        "event_counts": {k: int(v) for k, v in events.type.value_counts().items()},
        "neural_inference_performed": False, "source": "original synthetic educational fixture"}


@app.function(image=tribe_image, cpu=2, memory=8192, timeout=300, retries=0, max_containers=1)
def qualify_pages():
    """Exercise real native transforms with original synthetic page presentations.

    No model weights, participant records, original PDFs, or paid GPU are used.
    The 65-second silent video tests native chunk boundaries and absent audio.
    """
    from pathlib import Path
    import tempfile
    import subprocess
    import importlib.metadata
    from brain.presentation import page_schedule, page_words, native_page_events, native_text_events, text_schedule
    from brain.worker_contract import file_sha256

    text = "Gravity changes motion. A force changes the velocity of an object."
    text_plan = text_schedule(text, words_per_minute=120)
    text_events, text_audit = native_text_events(text, text_plan)
    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        media = Path(scratch) / "original-synthetic-pages.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=white:s=320x180:r=25", "-t", "65", "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", str(media)],
                       check=True, timeout=60)
        for kind in ("pdf", "image"):
            durations = [20, 15, 30] if kind == "pdf" else [65]
            texts = ["Gravity changes motion.", "", "Mass resists acceleration."] if kind == "pdf" else [""]
            schedule = page_schedule("a" * 64, len(durations), seconds_per_page=durations, input_format=kind)
            presentation = {"input_format": kind, "schedule": schedule,
                            "pages": [{"source_page_index": i, "text": value} for i, value in enumerate(texts)]}
            events, audit = native_page_events(media, presentation)
            counts = {k: int(v) for k, v in events.type.value_counts().items()}
            if "Audio" in counts or counts.get("Word", 0) != len(page_words(presentation)):
                raise ValueError("native events introduced audio or changed source word count")
            intervals = sorted((float(row.start), float(row.start + row.duration))
                               for row in events[events.type == "Video"].itertuples())
            covered = 0.0
            for start, stop in intervals:
                if start > covered + 1e-6 or stop > 65 + 1e-6:
                    raise ValueError("native video chunks have a gap or exceed the presentation")
                covered = max(covered, stop)
            if not intervals or abs(covered - 65) > 1e-6:
                raise ValueError("native video chunks omit presentation time")
            results[kind] = {"audit": audit, "event_counts": counts, "video_intervals": intervals}
        return {"schema": "perception-native-presentation-preprocessing-qualification-v1",
                "text": {"audit": text_audit,
                         "event_counts": {k: int(v) for k, v in text_events.type.value_counts().items()}},
                "pages": results, "synthetic_video_sha256": file_sha256(media),
                "presentation_adapter_sha256": file_sha256(Path(__import__('brain.presentation', fromlist=['']).__file__)),
                "native_versions": {name: importlib.metadata.version(name) for name in ("tribev2", "neuralset")},
                "neural_inference_performed": False, "real_PDF_extraction_qualified": False,
                "source": "original synthetic page event fixture; placeholder original-document digest"}
