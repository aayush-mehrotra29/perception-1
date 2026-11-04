import hashlib
import io
import json
from pathlib import Path
import zipfile
import sys
from types import SimpleNamespace

import pytest

from brain.modal_client import unpack_result, submit, single_gpu_job, pending_dispatches
from brain.credits import CreditLedger
from brain.worker_contract import PublicMediaRequest


@pytest.fixture(autouse=True)
def isolated_remote_observation_guard(monkeypatch):
    monkeypatch.setattr("brain.observe_call.guard", lambda _: None)


def fixture_request(tmp_path):
    media = tmp_path / "test.mp4"
    media.write_bytes(b"test transport only")
    request = PublicMediaRequest(source_id="synthetic", sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
        source_url="https://example.org/fixture", cloud_use_basis="Synthetic unit test, never sent over a network.",
        media_kind="video", size_bytes=media.stat().st_size)
    return media, request


def archive(request, extra=None, presentation=None):
    buffer = io.BytesIO()
    data = b"opaque-array-bytes-for-transport-test"
    manifest = {"schema": "perception-native-tribe-v1", "source": request.model_dump(),
                "array_sha256": hashlib.sha256(data).hexdigest()}
    if presentation is not None:
        manifest["presentation"] = presentation
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("fmri.npz", data)
        z.writestr("manifest.json", json.dumps(manifest))
        if extra:
            z.writestr(extra, "bad")
    return buffer.getvalue()


def test_worker_archive_provenance_and_path_safety(tmp_path):
    _, request = fixture_request(tmp_path)
    unpack_result(archive(request), tmp_path / "valid", "tribe", request)
    assert (tmp_path / "valid/fmri.npz").exists()
    with pytest.raises(ValueError, match="invalid worker archive"):
        unpack_result(archive(request, "../escape"), tmp_path / "invalid", "tribe", request)
    assert not (tmp_path / "invalid").exists()


def test_destination_disappearing_during_rpc_does_not_publish(tmp_path, monkeypatch):
    mounted = True
    def check(_):
        if not mounted:
            raise RuntimeError("research volume disconnected")
    monkeypatch.setattr("brain.modal_client.guard", check)
    media, request = fixture_request(tmp_path)
    def remote(*_):
        nonlocal mounted
        mounted = False
        return archive(request)
    with pytest.raises(RuntimeError, match="disconnected"):
        submit("tribe", media, request, tmp_path / "result", "disconnect", tmp_path, remote)
    assert not (tmp_path / "result").exists()
    assert CreditLedger(tmp_path / "brain-credits.sqlite3").snapshot()["modal"]["committed_microdollars"] == 4000000


def test_text_requires_explicit_pace_and_authenticates_returned_assumptions(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    media = tmp_path / "lesson.txt"
    media.write_bytes("A lesson.\r\nTwo lines with café.".encode())
    request = PublicMediaRequest(source_id="synthetic-text", media_kind="text",
        source_url="https://example.org/synthetic-text", cloud_use_basis="Original synthetic unit-test text.",
        sha256=hashlib.sha256(media.read_bytes()).hexdigest(), size_bytes=media.stat().st_size)
    with pytest.raises(ValueError, match="explicit reading pace"):
        submit("tribe", media, request, tmp_path / "bad", "no-pace", tmp_path)
    presentation = {"input_format": "text", "words_per_minute": 180}
    calls = []
    class Function:
        @staticmethod
        def from_name(app, name):
            assert (app, name) == ("perception-brain", "tribe_text")
            return Function()
        def spawn(self, *args):
            calls.append(args)
            return SimpleNamespace(object_id="fc-text", get=lambda **_: archive(request, presentation=presentation))
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=Function))
    submit("tribe", media, request, tmp_path / "text-output", "text", tmp_path,
           reading_words_per_minute=180, reservation_microdollars=2000000)
    assert calls == [(media.read_bytes(), request.model_dump(), presentation)]
    assert pending_dispatches(tmp_path) == []
    with pytest.raises(ValueError, match="presentation assumptions"):
        unpack_result(archive(request, presentation=presentation), tmp_path / "wrong-pace", "tribe", request,
                      expected_presentation={"input_format": "text", "words_per_minute": 240})
    assert not (tmp_path / "wrong-pace").exists()


def test_failed_remote_call_retains_reservation_and_never_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    media, request = fixture_request(tmp_path)
    calls = []
    def failure(*args):
        calls.append(1)
        raise TimeoutError("test")
    with pytest.raises(TimeoutError):
        submit("tribe", media, request, tmp_path / "output", "id", tmp_path, failure)
    with pytest.raises(ValueError, match="already reserved"):
        submit("tribe", media, request, tmp_path / "output", "id", tmp_path, failure)
    assert len(calls) == 1
    assert CreditLedger(tmp_path / "brain-credits.sqlite3").snapshot()["modal"]["committed_microdollars"] == 4000000


def test_gpu_lock_serializes_both_workers(tmp_path):
    with single_gpu_job(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="another project"):
            with single_gpu_job(tmp_path / "lock"):
                pass


def test_runtime_profiles_enforce_reservation_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    media, request = fixture_request(tmp_path)
    with pytest.raises(ValueError, match="reservation"):
        submit("tribe", media, request, tmp_path / "bad", "long-underfunded", tmp_path,
               use_cached_media=True, runtime_profile="long", reservation_microdollars=2000000)
    with pytest.raises(ValueError, match="cached media"):
        submit("tribe", media, request, tmp_path / "bad", "no-cache", tmp_path,
               runtime_profile="resume", reservation_microdollars=2000000)
    calls = []
    class Function:
        @staticmethod
        def from_name(app, name):
            assert name == "tribe_cached_resume"
            return Function()
        def spawn(self, *args):
            calls.append(args)
            return SimpleNamespace(object_id="fc-bounded-resume", get=lambda **_: archive(request))
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=Function))
    submit("tribe", media, request, tmp_path / "resumed", "resume-bounded", tmp_path,
           use_cached_media=True, runtime_profile="resume", reservation_microdollars=2000000)
    assert calls == [(request.model_dump(),)]


def test_disconnected_remote_dispatch_blocks_another_paid_request(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    dispatches = tmp_path / "brain-dispatches"
    dispatches.mkdir()
    (dispatches / "saved.json").write_text(json.dumps({"schema": "perception-modal-dispatch-v1",
        "request_id": "still-running", "output": str(tmp_path / "missing-result")}))
    media, request = fixture_request(tmp_path)
    calls = []
    with pytest.raises(RuntimeError, match="unresolved GPU dispatch"):
        submit("tribe", media, request, tmp_path / "another", "new-id", tmp_path, lambda *args: calls.append(1))
    assert calls == []
    assert pending_dispatches(tmp_path) == ["still-running"]


def test_cached_resume_sends_only_authenticated_request_and_records_call(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    media, request = fixture_request(tmp_path)
    calls = []
    class Function:
        @staticmethod
        def from_name(app, function):
            assert (app, function) == ("perception-brain", "tribe_cached")
            return Function()
        def spawn(self, *args):
            calls.append(args)
            return SimpleNamespace(object_id="fc-test", get=lambda **_: archive(request))
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=Function))
    submit("tribe", media, request, tmp_path / "output", "resume", tmp_path, use_cached_media=True)
    assert calls == [(request.model_dump(),)]
    record = json.loads(next((tmp_path / "brain-dispatches").glob("*.json")).read_text())
    assert record["function_call_id"] == "fc-test"
    assert record["used_authenticated_cloud_media_cache"] is True
    assert pending_dispatches(tmp_path) == []
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity/size"):
        submit("tribe", media, request, tmp_path / "another", "bad", tmp_path, use_cached_media=True)
    assert len(calls) == 1


def test_cloud_audio_resume_authenticates_local_parent_and_derivative_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr("brain.modal_client.guard", lambda _: None)
    media, parent = fixture_request(tmp_path)
    audio = parent.model_copy(update={"media_kind": "audio", "sha256": "a" * 64, "size_bytes": 1000})
    receipt = tmp_path / "audio.json"
    metadata = {"schema": "perception-audio-derivative-v1", "parent": parent.model_dump(),
        "derivative": audio.model_dump(), "execution_location": "isolated_cloud_CPU",
        "cached_file_identity_verified": True, "audio_clock_continuity_checked": True,
        "sample_rate": 48000, "samples": 400}
    receipt.write_text(json.dumps(metadata))
    calls = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema": "perception-native-sam-audio-v1",
            "source": audio.model_dump(), "chunks": []}))  # Transport fixture only, never accepted as lesson features.
    class Function:
        @staticmethod
        def from_name(app, function):
            assert function == "sam_audio_cached"
            return Function()
        def spawn(self, *args):
            calls.append(args)
            return SimpleNamespace(object_id="fc-audio-test", get=lambda **_: buffer.getvalue())
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=Function))
    submit("sam_audio", media, parent, tmp_path / "result", "audio", tmp_path,
           use_cached_media=True, cached_audio_manifest=receipt, reservation_microdollars=2000000)
    assert calls == [(audio.model_dump(),)]
    metadata["parent"]["sha256"] = "b" * 64
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="matching local parent"):
        submit("sam_audio", media, parent, tmp_path / "bad", "bad-audio", tmp_path,
               use_cached_media=True, cached_audio_manifest=receipt)
    assert len(calls) == 1
