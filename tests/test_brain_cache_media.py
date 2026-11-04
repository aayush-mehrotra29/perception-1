import hashlib
import wave

import pytest

from brain.cache_media import wav_blueprint, stage_public_media
from brain.worker_contract import PublicMediaRequest, file_sha256


def test_waveform_blueprint_preserves_exact_file_and_rejects_changed_source(tmp_path):
    path = tmp_path / "source.wav"
    pcm = b"\x12\x01\x56\x02" * 480
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)
    request = PublicMediaRequest(source_id="test", sha256=file_sha256(path), source_origin="project_generated",
        cloud_use_basis="Synthetic local test, no cloud call", media_kind="audio", size_bytes=path.stat().st_size)
    plan = wav_blueprint(path, request)
    assert bytes.fromhex(plan["prefix_hex"]) + pcm + bytes.fromhex(plan["suffix_hex"]) == path.read_bytes()
    assert plan["pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
    # Non-owner URLs and generated media never trigger an implicit fetch.
    with pytest.raises(ValueError, match="owner-hosted"):
        stage_public_media(tmp_path / "cache", request, request, plan)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity/size"):
        wav_blueprint(path, request)
