from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from brain.credits import CreditLedger
from brain.worker_contract import PublicMediaRequest, file_sha256, segment_record, validate_media


def test_reservations_preserve_unknown_cost_and_block_duplicate_submission(tmp_path):
    ledger = CreditLedger(tmp_path / "credits.db")
    ledger.reserve("job1", "modal", 25000000)
    with pytest.raises(ValueError, match="ceiling"):
        ledger.reserve("job2", "modal", 6000000)
    # A second instance observes persisted liability after a process restart.
    assert CreditLedger(tmp_path / "credits.db").snapshot()["modal"]["committed_microdollars"] == 25000000
    with pytest.raises(ValueError, match="already"):
        ledger.reserve("job1", "modal", 100)
    ledger.settle("job1", 1000000)
    ledger.reserve("job2", "modal", 6000000)
    assert ledger.snapshot()["modal"]["committed_microdollars"] == 7000000


def test_actual_overrun_is_retained(tmp_path):
    ledger = CreditLedger(tmp_path / "credits.db")
    ledger.reserve("job", "modal", 1000000)
    ledger.settle("job", 31000000)
    with pytest.raises(ValueError, match="ceiling"):
        ledger.reserve("next", "modal", 1)


def test_app_level_billing_total_is_counted_once_and_allocation_is_explicit(tmp_path):
    import json
    import sqlite3
    ledger = CreditLedger(tmp_path / "credits.db")
    ledger.reserve("one", "modal", 4000000)
    ledger.reserve("two", "modal", 2000000)
    ledger.settle_group("completed-app", ["one", "two"], 3500001, "a" * 64)
    assert CreditLedger(tmp_path / "credits.db").snapshot()["modal"]["committed_microdollars"] == 3500001
    with sqlite3.connect(ledger.path) as db:
        row = db.execute("SELECT actual, allocations FROM settlement_groups").fetchone()
    assert sum(json.loads(row[1]).values()) == row[0] == 3500001
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle_group("completed-app", ["one", "two"], 3500001, "a" * 64)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle("one", 10)
    ledger.reserve("three", "modal", 26000000)
    with pytest.raises(ValueError, match="ceiling"):
        ledger.reserve("over", "modal", 1000000)


def test_content_payload_rejects_participant_fields_and_wrong_bytes(tmp_path):
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"synthetic-contract-fixture-not-decodable-media")
    request = PublicMediaRequest(source_id="synthetic", sha256=file_sha256(media),
                                source_url="https://example.org/fixture.mp4", cloud_use_basis="Synthetic contract test fixture only",
                                media_kind="video", size_bytes=media.stat().st_size)
    validate_media(media, request)
    with pytest.raises(ValidationError):
        PublicMediaRequest(**request.model_dump(), eeg_targets=[1, 2])
    media.write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_media(media, request)


def test_native_timeline_keeps_offset_and_rejects_invalid_duration():
    assert segment_record(SimpleNamespace(start=5.5, duration=1, timeline="original"))["start"] == 5.5
    with pytest.raises(ValueError):
        segment_record(SimpleNamespace(start=0, duration=float("nan"), timeline="original"))
