import pytest

from brain.tribe_events import audit_word_rows


def test_repeated_or_shifted_long_video_transcripts_are_rejected():
    words = [{"text": "hello", "start": 0.5, "duration": 0.2},
             {"text": "world", "start": 70.0, "duration": 0.5}]
    result = audit_word_rows(words, 80, expected_words=words)
    assert result["word_count"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        audit_word_rows(words + words, 80)
    with pytest.raises(ValueError, match="outside"):
        audit_word_rows(words + [{"text": "world", "start": 190.0, "duration": 0.5}], 80)
    with pytest.raises(ValueError, match="identities"):
        audit_word_rows(words[:-1], 80, expected_words=words)
