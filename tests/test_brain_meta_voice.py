import pytest

from brain.meta_voice import validate_response


def test_voice_uses_measured_audio_duration_without_word_time_invention():
    text, cost = validate_response({"transcript": " A simple lesson. ", "audioDurationMs": 20500, "turns": []}, 492000)
    assert text == "A simple lesson."
    assert cost == 1000
    with pytest.raises(ValueError, match="duration"):
        validate_response({"transcript": "Wrong duration", "audioDurationMs": 50000}, 492000)
    with pytest.raises(ValueError, match="turns"):
        validate_response({"transcript": "Unexpected", "audioDurationMs": 20500, "turns": [{"startMs": 0}]}, 492000)
