"""Explicit assumed clocks for non-video content; never observed reading/gaze.

Text retains exact UTF-8 source spans. Image/PDF units retain source indices.
These schedules describe an experimental presentation, not a person's behavior.
No attention labels, semantic annotations, synthetic speech or missing modalities
are generated here. Native inference adapters consume the resulting schedule.
"""
import hashlib
import math
import re


def text_schedule(text: str, *, words_per_minute: float):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty educational text required")
    if not math.isfinite(words_per_minute) or not 30 <= words_per_minute <= 600:
        raise ValueError("explicit assumed reading pace from 30 to 600 words/minute required")
    matches = list(re.finditer(r"\S+", text))
    seconds_per_word = 60 / words_per_minute
    duration = len(matches) * seconds_per_word
    if duration > 3600:
        raise ValueError("presentation exceeds the one-hour prototype limit")
    words = [{"text": match.group(), "character_start": match.start(), "character_stop": match.end(),
              "start": i * seconds_per_word, "duration": seconds_per_word}
             for i, match in enumerate(matches)]
    return {"schema": "perception-assumed-presentation-v1", "input_format": "text",
        "source_text_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "clock_basis": "assumed_constant_reading_pace", "words_per_minute": words_per_minute,
        "tokenization": "non-whitespace spans in source order; not a linguistic reading-time model",
        "duration_seconds": duration, "words": words,
        "source_character_offsets": "Python Unicode code-point indices",
        "observed_reading_or_gaze": False, "transfer_status": "experimental_transfer",
        "synthetic_audio_generated": False, "scientific_validity_established": False}


def page_schedule(source_sha256: str, page_count: int, *, seconds_per_page: list[float], input_format="pdf"):
    if (re.fullmatch(r"[0-9a-f]{64}", source_sha256 or "") is None
            or input_format not in {"pdf", "image"} or type(page_count) is not int or page_count < 1
            or (input_format == "image" and page_count != 1)
            or len(seconds_per_page) != page_count):
        raise ValueError("authenticated source, actual page count and an explicit duration for every page required")
    if any(not math.isfinite(t) or not 1 <= t <= 600 for t in seconds_per_page) or sum(seconds_per_page) > 3600:
        raise ValueError("each page must last 1–600 seconds; total must be at most one hour")
    pages, start = [], 0.0
    for index, duration in enumerate(seconds_per_page):
        pages.append({"source_page_index": index, "start": start, "duration": float(duration)})
        start += duration
    return {"schema": "perception-assumed-presentation-v1", "input_format": input_format,
        "source_sha256": source_sha256, "clock_basis": "explicit_assumed_page_display_times",
        "duration_seconds": start, "pages": pages, "observed_reading_or_gaze": False,
        "transfer_status": "experimental_transfer", "synthetic_audio_generated": False,
        "scientific_validity_established": False}


def native_text_events(text: str, schedule: dict):
    """Build the pinned text/context events without converting the text to audio.

Requires the native neuralset/tribev2 environment. This is a preprocessing
adapter only; text-only EEG transfer still requires its own fitted head.
"""
    expected = text_schedule(text, words_per_minute=schedule["words_per_minute"])
    if schedule != expected:
        raise ValueError("text schedule does not match source and declared pace")
    source_words = [{k: word[k] for k in ("text", "start", "duration")} for word in schedule["words"]]
    events, audit = _native_words(source_words, schedule["duration_seconds"])
    audit.update(schema="perception-assumed-text-event-audit-v1",
        clock_basis=schedule["clock_basis"], observed_reading_or_gaze=False,
        source_text_utf8_sha256=schedule["source_text_utf8_sha256"],
        synthetic_audio_generated=False, transfer_status="experimental_transfer")
    return events, audit


def page_words(presentation: dict):
    """Evenly schedule extracted words within each explicitly timed page."""
    if set(presentation) != {"input_format", "schedule", "pages"}:
        raise ValueError("exact document presentation fields required")
    schedule, pages = presentation["schedule"], presentation["pages"]
    durations = [slot["duration"] for slot in schedule["pages"]]
    expected = page_schedule(schedule["source_sha256"], len(pages), seconds_per_page=durations,
                             input_format=presentation["input_format"])
    if schedule != expected:
        raise ValueError("page schedule is not internally consistent")
    words = []
    for index, (page, slot) in enumerate(zip(pages, schedule["pages"])):
        if page["source_page_index"] != index or not isinstance(page["text"], str):
            raise ValueError("page text identity/order mismatch")
        if presentation["input_format"] == "image" and page["text"]:
            raise ValueError("image OCR has not been qualified")
        matches = list(re.finditer(r"\S+", page["text"]))
        for i, match in enumerate(matches):
            words.append({"text": match.group(), "start": slot["start"] + i * slot["duration"] / len(matches),
                          "duration": slot["duration"] / len(matches)})
    return words


def _native_words(source_words, duration):
    import pandas as pd
    from neuralset.events.utils import standardize_events
    from neuralset.events.transforms import AddText, AddSentenceToWords, AddContextToWords, RemoveMissing
    events = standardize_events(pd.DataFrame([{"type": "Word", "text": word["text"],
        "start": word["start"], "duration": word["duration"], "timeline": "default", "subject": "default"}
        for word in source_words]))
    for transform in (AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),
                      AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""), RemoveMissing()):
        events = transform(events)
    events = standardize_events(events)
    words = events[events.type == "Word"][["text", "start", "duration"]].to_dict(orient="records")
    from .tribe_events import audit_word_rows
    audit = audit_word_rows(words, duration, expected_words=source_words)
    if set(events.type) - {"Word", "Sentence", "Text"}:
        raise ValueError("text presentation unexpectedly introduced another modality")
    return events, audit


def native_page_events(media, presentation):
    import pandas as pd
    from neuralset.events.utils import standardize_events
    from neuralset.events.transforms import ChunkEvents
    words = page_words(presentation)
    schedule = presentation["schedule"]
    video = standardize_events(pd.DataFrame([{"type": "Video", "filepath": str(media),
        "start": 0, "timeline": "default", "subject": "default"}]))
    if len(video) != 1 or abs(float(video.iloc[0].duration) - schedule["duration_seconds"]) > 1e-6:
        raise ValueError("rendered video duration differs from assumed document clock")
    video = ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30)(video)
    if words:
        events, audit = _native_words(words, schedule["duration_seconds"])
        events = standardize_events(pd.concat([events, video], ignore_index=True))
    else:
        events = video
        audit = {"word_count": 0, "media_stop": schedule["duration_seconds"]}
    audit.update(schema="perception-assumed-page-event-audit-v1", clock_basis=schedule["clock_basis"],
        observed_reading_or_gaze=False, text_order_and_uniform_within_page_pace_are_assumptions=True,
        source_sha256=schedule["source_sha256"], synthetic_audio_generated=False,
        transfer_status="experimental_transfer")
    return events, audit
