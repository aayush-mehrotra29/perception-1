import json
import shutil

import numpy as np
import pytest

from brain.document_presentation import prepare
from brain.worker_contract import PublicMediaRequest, file_sha256


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg media integration test")
def test_image_render_preserves_aspect_and_exact_declared_clock(tmp_path, monkeypatch):
    from PIL import Image
    import av
    monkeypatch.setattr("brain.document_presentation.guard", lambda _: None)
    source = tmp_path / "image.png"
    Image.new("RGB", (300, 600), (255, 0, 0)).save(source)
    request = PublicMediaRequest(source_id="original-fixture", source_origin="project_generated",
        media_kind="image", sha256=file_sha256(source), size_bytes=source.stat().st_size,
        cloud_use_basis="Original solid-color test image; local rendering only.")
    with pytest.raises(ValueError, match="multiples"):
        prepare(source, request, tmp_path / "invalid", seconds_per_page=[1.01])
    assert not (tmp_path / "invalid").exists()
    result = prepare(source, request, tmp_path / "render", seconds_per_page=[1.2])
    assert result["rendering"]["frame_count"] == 30
    assert result["rendering"]["audio_tracks"] == 0
    assert result["schedule"]["observed_reading_or_gaze"] is False
    assert result["pages"][0]["text"] == ""
    with av.open(str(tmp_path / "render/presentation.mp4")) as container:
        image = next(container.decode(video=0)).to_ndarray(format="rgb24")
    np.testing.assert_allclose(image[540, 960], [255, 0, 0], atol=3)
    np.testing.assert_allclose(image[540, 100], [255, 255, 255], atol=3)
    assert json.loads((tmp_path / "render/request.json").read_text())["sha256"] == file_sha256(tmp_path / "render/presentation.mp4")
