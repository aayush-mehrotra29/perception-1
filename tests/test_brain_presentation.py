import copy

import pytest

from brain.presentation import text_schedule, page_schedule, native_text_events


def test_text_clock_keeps_source_spans_and_declares_assumption():
    text = "First paragraph.\n\nCafé — energy = mc²."
    plan = text_schedule(text, words_per_minute=120)
    for i, word in enumerate(plan["words"]):
        assert text[word["character_start"]:word["character_stop"]] == word["text"]
        assert word["start"] == i * 0.5
        assert word["duration"] == 0.5
    assert plan["observed_reading_or_gaze"] is False
    assert plan["synthetic_audio_generated"] is False
    assert plan["transfer_status"] == "experimental_transfer"
    changed = copy.deepcopy(plan)
    changed["words"][0]["text"] = "invented"
    with pytest.raises(ValueError, match="does not match"):
        native_text_events(text, changed)  # Reject before importing GPU/native dependencies.
    with pytest.raises(ValueError, match="reading pace"):
        text_schedule(text, words_per_minute=float("nan"))


def test_page_clock_has_explicit_complete_nonoverlapping_display_schedule():
    plan = page_schedule("a" * 64, 3, seconds_per_page=[30, 45, 15])
    assert plan["pages"] == [
        {"source_page_index": 0, "start": 0, "duration": 30},
        {"source_page_index": 1, "start": 30, "duration": 45},
        {"source_page_index": 2, "start": 75, "duration": 15}]
    assert plan["duration_seconds"] == 90
    with pytest.raises(ValueError, match="every page"):
        page_schedule("a" * 64, 3, seconds_per_page=[30])
    with pytest.raises(ValueError, match="every page"):
        page_schedule("a" * 64, 2, seconds_per_page=[30, 30], input_format="image")
def test_page_words_follow_explicit_page_clock_without_inventing_blank_page_text():
    from brain.presentation import page_schedule, page_words
    schedule = page_schedule("a" * 64, 3, seconds_per_page=[20, 10, 30], input_format="pdf")
    presentation = {"input_format": "pdf", "schedule": schedule, "pages": [
        {"source_page_index": 0, "text": "First page"},
        {"source_page_index": 1, "text": ""},
        {"source_page_index": 2, "text": "Final page"}]}
    words = page_words(presentation)
    assert [w["start"] for w in words] == [0, 10, 30, 45]
    assert words[-1]["start"] + words[-1]["duration"] == 60
    assert not schedule["observed_reading_or_gaze"]

