import numpy as np
import pytest

from brain.content_pipeline import _text_spans, run
from brain.presentation import text_schedule
from brain.worker_contract import PublicMediaRequest, file_sha256


def text_input(tmp_path):
    source = tmp_path / "lesson.txt"
    source.write_bytes("First sentence.\r\nSecond paragraph: café and π.".encode())
    request = PublicMediaRequest(source_id="original-test", media_kind="text", source_origin="project_generated",
        sha256=file_sha256(source), size_bytes=source.stat().st_size,
        cloud_use_basis="Original synthetic local-only test fixture.")
    return source, request


def test_assumed_semantic_spans_keep_exact_source_substrings():
    text = "One\r\ntwo.  Café π! " * 50
    schedule = text_schedule(text, words_per_minute=60)
    spans = _text_spans(text, schedule)
    assert len(spans) > 1
    assert all(span.text in text for span in spans)
    assert "\r\n" in spans[0].text
    assert spans[-1].stop == schedule["duration_seconds"]


def test_missing_clock_and_missing_model_fail_before_paid_inference(tmp_path, monkeypatch):
    source, request = text_input(tmp_path)
    monkeypatch.setattr("brain.content_pipeline.guard", lambda _: None)
    monkeypatch.setattr("brain.content_pipeline.load_credentials", lambda: None)
    monkeypatch.setattr("brain.content_pipeline.CreditLedger", lambda _: None)
    calls = []
    monkeypatch.setattr("brain.content_pipeline.submit", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="reading pace"):
        run(source, request, tmp_path / "no-clock", "no-clock")
    assert not (tmp_path / "no-clock").exists()
    with pytest.raises(FileNotFoundError):
        run(source, request, tmp_path / "no-model", "no-model", words_per_minute=180,
            transfer_models=tmp_path / "missing-models")
    assert calls == []
