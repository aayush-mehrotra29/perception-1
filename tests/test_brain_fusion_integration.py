import json
from pathlib import Path

import numpy as np
import pytest

from brain import pipeline, content_pipeline, fusion_inference
from brain.worker_contract import PublicMediaRequest, file_sha256
from brain.fusion_report import cluster_comparison


def media_fixture(tmp_path, kind="video"):
    path = tmp_path / ("lesson." + {"video": "mp4", "audio": "wav", "text": "txt"}.get(kind, kind))
    path.write_bytes(b"Synthetic fixture; no cloud calls allowed.")
    request = PublicMediaRequest(source_id="fusion-fixture", media_kind=kind,
        source_origin="project_generated", sha256=file_sha256(path), size_bytes=path.stat().st_size,
        cloud_use_basis="Synthetic local contract test, never sent to cloud.")
    return path, request


def test_conflicting_or_invalid_fusion_artifacts_fail_before_remote_calls(tmp_path, monkeypatch):
    source, request = media_fixture(tmp_path)
    def forbidden(*a, **k):
        pytest.fail("No remote work or credentials may be accessed for an invalid model")
    monkeypatch.setattr(pipeline, "submit", forbidden)
    monkeypatch.setattr(pipeline, "load_credentials", forbidden)
    for function, kwargs in [(pipeline.run, {"model_directory": tmp_path}),
                             (content_pipeline.run, {"video_model": tmp_path}),
                             (content_pipeline.run, {"transfer_models": tmp_path})]:
        with pytest.raises(ValueError, match="either|exclusive"):
            function(source, request, tmp_path / "output", "fixture", fusion_model=tmp_path, **kwargs)
    with pytest.raises(FileNotFoundError):
        content_pipeline.run(source, request, tmp_path / "output", "fixture", fusion_model=tmp_path / "missing")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize('version', [2,3])
def test_video_dispatch_keeps_fusion_model_and_eeg_local(tmp_path, monkeypatch, version):
    source, request = media_fixture(tmp_path)
    model = tmp_path / "private-fusion-model"
    model.mkdir(); (model / "manifest.json").write_text(json.dumps({'schema':f'perception-fusion-deployment-v{version}'}))
    monkeypatch.setattr(fusion_inference, "preflight", lambda _: {})
    monkeypatch.setattr(pipeline, "guard", lambda _: None)
    monkeypatch.setattr(pipeline, "inspect_media", lambda _: (None, 60., None, None))
    for name in ("load_credentials", "extract_frames", "extract_sensory", "run_voice", "run_voice_semantics"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "CreditLedger", lambda _: None)
    def audio(media, request, folder):
        folder.mkdir()
        (folder / "audio.wav").write_bytes(b"synthetic audio")
        derivative = request.model_copy(update={"media_kind": "audio", "sha256": file_sha256(folder / "audio.wav"),
                                               "size_bytes": (folder / "audio.wav").stat().st_size})
        (folder / "request.json").write_text(derivative.model_dump_json())
    monkeypatch.setattr(pipeline, "extract_audio", audio)
    async def sam(*a, **k):
        return None
    monkeypatch.setattr(pipeline, "run_sam", sam)
    sent = []
    monkeypatch.setattr(pipeline, "submit", lambda *a, **k: sent.append((a, k)))
    def assemble(spec, query, output, **kwargs):
        output.mkdir(); (output / "manifest.json").write_text('{"test_only":true}')
    monkeypatch.setattr(pipeline, "assemble", assemble)
    local = []
    monkeypatch.setattr(fusion_inference, "predict_native", lambda *a, **k: local.append((a, k)))
    output = tmp_path / "output"
    result = content_pipeline.run(source, request, output, "fixture", fusion_model=model)
    expected='bounded_residual_fusion_v3' if version==3 else 'temporal_fusion_v2'
    assert result["eeg_prediction_emitted"] and result["EEG_model_family"] == expected
    assert len(local) == 1 and local[0][0][1] == model
    assert len(sent) == 2
    for args, kwargs in sent:
        assert isinstance(args[2], PublicMediaRequest)
        assert str(model) not in repr((args, kwargs))
        assert not set(args[2].model_dump()) & {"eeg", "eeg_targets", "model_weights", "token", "credentials"}


def test_transfer_clock_mismatch_is_rejected_without_inference(tmp_path, monkeypatch):
    native = tmp_path / "native"; native.mkdir()
    (native / "manifest.json").write_text(json.dumps({"assumed_presentation_schedule":
        {"input_format": "text", "duration_seconds": 65}}))
    monkeypatch.setattr(fusion_inference, "preflight", lambda _: pytest.fail("must reject clock before loading model"))
    with pytest.raises(ValueError, match="native assumed presentation"):
        fusion_inference.predict_native(native, tmp_path, tmp_path / "out", duration=60, input_format="text")


def test_cluster_uncertainty_keeps_paired_lecture_and_person_observations():
    lessons = np.repeat(["a", "b", "c", "d"], 8)
    people = np.tile(np.repeat([1, 2, 3, 4], 2), 4)
    baseline = np.arange(32)[:, None] / 100 + np.ones((32, 8))
    result = cluster_comparison(baseline * .9, baseline, lessons, people, repeats=200)
    assert result["relative_error_reduction"] == pytest.approx(.1)
    np.testing.assert_allclose(result["reduction_interval_95"], [.1, .1])
    assert result["difference_interval_95"][1] < 0
