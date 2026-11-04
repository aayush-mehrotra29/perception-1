"""Audited media events for the pinned TRIBE checkpoint.

The pinned demo chunks Audio before extracting words. Its word transform reads
the full-source transcript for each chunk and adds start + offset, duplicating
and shifting the words. Extract words once BEFORE chunking instead. Preserve
the upstream transforms, contextualization, feature extractors and checkpoint.
"""
import math
from pathlib import Path


def audit_word_rows(words, media_stop, expected_words=None):
    if not math.isfinite(media_stop) or media_stop <= 0:
        raise ValueError("positive finite media extent required")
    seen, maximum = set(), 0.0
    for row in words:
        start, duration = float(row["start"]), float(row["duration"])
        text = str(row["text"])
        if (not math.isfinite(start + duration) or start < 0 or duration < 0
                or start + duration > media_stop + 0.1 or not text.strip()):
            raise ValueError("transcript word outside authenticated media extent")
        key = (text, start, duration)
        if key in seen:
            raise ValueError("duplicate transcript word event")
        seen.add(key)
        maximum = max(maximum, start + duration)
    if expected_words is not None:
        expected = {(str(r["text"]), float(r["start"]), float(r["duration"])) for r in expected_words}
        if len(expected) != len(expected_words) or seen != expected:
            raise ValueError("contextualization changed transcript word identities")
    return {"word_count": len(seen), "maximum_word_stop": maximum,
            "media_stop": media_stop, "no_duplicate_words": True, "words_within_media_extent": True}


def build_events(media: Path, media_kind: str):
    import pandas as pd
    from neuralset.events.utils import standardize_events
    from neuralset.events.transforms import (ExtractAudioFromVideo, ChunkEvents, AddText,
        AddSentenceToWords, AddContextToWords, RemoveMissing)
    from tribev2.eventstransforms import ExtractWordsFromAudio

    if media_kind not in {"audio", "video"}:
        raise ValueError("native timed audio/video required")
    events = standardize_events(pd.DataFrame([{"type": "Video" if media_kind == "video" else "Audio",
        "filepath": str(media), "start": 0, "timeline": "default", "subject": "default"}]))
    events = ExtractAudioFromVideo()(events)
    media_rows = events[events.type.isin(["Audio", "Video"])]
    media_stop = float((media_rows.start + media_rows.duration).max())
    # There is exactly one unchunked source audio event, so the upstream word
    # transform applies the full transcript only once at source position zero.
    audio = events[events.type == "Audio"]
    if len(audio) != 1 or float(audio.iloc[0].start) != 0 or float(audio.iloc[0].offset) != 0:
        raise ValueError("one zero-origin unchunked audio source required")
    events = ExtractWordsFromAudio()(events)
    original = events[events.type == "Word"][["text", "start", "duration"]].to_dict(orient="records")
    audit_word_rows(original, media_stop)
    for transform in [ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
                      ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30), AddText(),
                      AddSentenceToWords(max_unmatched_ratio=0.05),
                      AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")]:
        events = transform(events)
    before_filter = events[events.type == "Word"][["text", "start", "duration"]].to_dict(orient="records")
    audit_word_rows(before_filter, media_stop, expected_words=original)
    events = RemoveMissing()(events)
    events = standardize_events(events)
    words = events[events.type == "Word"][["text", "start", "duration"]].to_dict(orient="records")
    audit = audit_word_rows(words, media_stop)
    audit.update(schema="perception-tribe-event-audit-v1",
        preprocessing="ExtractAudio → ExtractWords once on full source → ChunkAudio/Video → original text/context transforms",
        deviation_from_pinned_demo="Word extraction moved before chunking to prevent repeated full-source transcription and doubled chunk offsets",
        event_counts={k: int(v) for k, v in events.type.value_counts().items()},
        original_transcript_words=len(original), words_removed_by_native_missing_context_filter=len(original) - len(words),
        checkpoint_or_feature_extractors_modified=False, independently_validated_neural_accuracy=False)
    return events, audit
