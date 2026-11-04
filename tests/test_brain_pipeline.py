from brain import pipeline
from brain.worker_contract import PublicMediaRequest, file_sha256
import pytest


def test_short_video_rejected_before_paid_work_or_output_creation(tmp_path, monkeypatch):
    source = tmp_path / 'short.mp4'
    source.write_bytes(b'local metadata-validation fixture')
    request = PublicMediaRequest(source_id='short-fixture', media_kind='video', source_origin='project_generated',
        sha256=file_sha256(source), size_bytes=source.stat().st_size,
        cloud_use_basis='Local unit-test fixture; never sent to a service.')
    monkeypatch.setattr(pipeline.EEGModel, 'load', lambda _: object())
    monkeypatch.setattr(pipeline, 'guard', lambda _: None)
    monkeypatch.setattr(pipeline, 'inspect_media', lambda _: (None, 44.9, None, None))
    def unexpected(*args, **kwargs):
        pytest.fail('Short video must be rejected before credentials, media extraction, or paid calls')
    for name in ['load_credentials', 'extract_audio', 'extract_frames', 'extract_sensory', 'submit']:
        monkeypatch.setattr(pipeline, name, unexpected)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='44-second context'):
        pipeline.run(source, request, output, 'short-fixture', model_directory=tmp_path / 'model')
    assert not output.exists()
