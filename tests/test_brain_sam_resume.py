"""Fault injection for caching only; tensor fixtures are not neural validation."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from brain.remote_inference import run_sam_audio
from brain.worker_contract import PublicMediaRequest, file_sha256


def test_interrupted_separation_reuses_only_checksum_verified_chunks(tmp_path, monkeypatch):
    path = tmp_path / "fixture.wav"
    path.write_bytes(b"mock decoder input, not a real waveform")
    request = PublicMediaRequest(source_id="fixture", sha256=file_sha256(path), size_bytes=path.stat().st_size,
        source_origin="project_generated", cloud_use_basis="Synthetic cache fault-injection test only", media_kind="audio")
    calls, commits = [], []
    class Tensor:
        def __init__(self, values): self.values = values
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.values
    class Processor:
        audio_sampling_rate = 48000
        @classmethod
        def from_pretrained(cls, *args): return cls()
        def __call__(self, audios, descriptions):
            return SimpleNamespace(to=lambda device: audios[0])
    class Model:
        @classmethod
        def from_pretrained(cls, *args): return cls()
        def eval(self): return self
        def cuda(self): return self
        def separate(self, batch, **kwargs):
            calls.append(batch.shape[1])
            if len(calls) == 2:
                raise RuntimeError("simulated second-chunk interruption")
            return SimpleNamespace(target=[Tensor(batch)], residual=[Tensor(np.zeros_like(batch))])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test-fixture", cuda=SimpleNamespace(is_available=lambda: True),
        manual_seed=lambda seed: None, inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=lambda *args: (np.ones((1, 48000 * 40), dtype=np.float32), 48000)))
    monkeypatch.setitem(sys.modules, "sam_audio", SimpleNamespace(SAMAudio=Model, SAMAudioProcessor=Processor))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(tmp_path)))
    monkeypatch.setattr("brain.sam_compat.install", lambda: None)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="second-chunk"):
        run_sam_audio(path, request, tmp_path / "interrupted", cache, lambda: commits.append(1))
    assert len(commits) == 1
    result = run_sam_audio(path, request, tmp_path / "resumed", cache, lambda: commits.append(1))
    assert result["reused_verified_chunks"] == 1
    assert calls == [48000 * 30, 48000 * 10, 48000 * 10]
    assert len(commits) == 2
    np.testing.assert_array_equal(np.load(tmp_path / "interrupted/chunk-0000.npz")["values"],
                                  np.load(tmp_path / "resumed/chunk-0000.npz")["values"])
    next(cache.rglob("chunk-0000.npz")).write_bytes(b"damaged cache")
    with pytest.raises(ValueError, match="integrity"):
        run_sam_audio(path, request, tmp_path / "rejected", cache)
    assert len(calls) == 3
