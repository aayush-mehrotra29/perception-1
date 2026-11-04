"""Read-only capture of native TRIBE inputs before any brain-trained projection.

Preserves layer, hidden and sample axes. It does not pool across layers, fit a
transform, infer missing modalities, or change the native forward computation.
"""
from pathlib import Path
import math

import numpy as np

from .worker_contract import file_sha256, segment_record


class DenseFeatureCapture:
    def __init__(self, destination: Path, frequency: float, modalities=("text", "audio", "video"), maximum_bytes=128 * 1024**2):
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("native feature sample frequency required")
        self.destination = destination
        self.frequency = frequency
        self.modalities = tuple(modalities)
        self.maximum_bytes = maximum_bytes
        self.bytes_written = 0
        self.records = []

    def __call__(self, model, args):
        if len(args) != 1:
            raise ValueError("unexpected native model call signature")
        self.capture(args[0])
        return None  # PyTorch pre-hook: do not replace or mutate the input.

    def capture(self, batch):
        segments = [segment_record(s) for s in batch.segments]
        arrays, shapes = {}, {}
        for name in self.modalities:
            if name not in batch.data:
                continue
            tensor = batch.data[name]
            value = tensor.detach().cpu().numpy()
            if value.ndim not in {3, 4} or value.shape[0] != len(segments) or not np.isfinite(value).all():
                raise ValueError("invalid native content tensor")
            for segment in segments:
                expected = segment["duration"] * self.frequency
                if abs(expected - value.shape[-1]) > 1e-6:
                    raise ValueError("native dense sample count differs from segment clock")
            arrays[name] = np.array(value, dtype=np.float32, copy=True)
            shapes[name] = list(value.shape)
        if not arrays:
            raise ValueError("no native content blocks captured")
        projected_bytes = sum(v.nbytes for v in arrays.values())
        if projected_bytes + self.bytes_written > self.maximum_bytes:
            raise ValueError("dense capture exceeds configured output bound")
        name = f"dense-{len(self.records):04d}.npz"
        path = self.destination / name
        # Exclusive open prevents accidental overwrite of an earlier batch.
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        self.bytes_written += projected_bytes
        self.records.append({"file": name, "sha256": file_sha256(path), "shapes": shapes,
                             "segments": segments, "frequency_hz": self.frequency,
                             "axes": "batch,layer,hidden,sample (or batch,hidden,sample)",
                             "missing_modalities": sorted(set(self.modalities) - set(arrays)),
                             "brain_trained_transform_applied": False,
                             "timing_qualified_against_extractor": False})
