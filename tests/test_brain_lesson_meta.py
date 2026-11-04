import json

import pytest

from brain.lesson_meta import transcript_spans
from brain.worker_contract import file_sha256


def test_native_transcript_identity_and_word_clock(tmp_path):
    transcript = tmp_path / "words.tsv"
    transcript.write_text("text\tstart\tduration\nHello\t0.5\t0.5\nworld\t31\t0.5\n")
    receipt = tmp_path / "provenance.json"
    receipt.write_text(json.dumps({"source_sha256": "a" * 64, "transcript_sha256": file_sha256(transcript)}))
    spans = transcript_spans(transcript, receipt, "a" * 64, 45)
    assert [(s.start, s.stop, s.text) for s in spans] == [(0, 30, "Hello"), (30, 45, "world")]
    with pytest.raises(ValueError, match="mismatch"):
        transcript_spans(transcript, receipt, "b" * 64, 45)
    with pytest.raises(ValueError, match="clock"):
        transcript_spans(transcript, receipt, "a" * 64, 30)
