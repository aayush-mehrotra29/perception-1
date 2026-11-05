"""Small five-distribution network; numeric serialization, no pickle loading."""
import json
from pathlib import Path

import numpy as np

from brain.worker_contract import file_sha256
from . import LEVELS, QUALITIES


def build_network(input_width):
    import torch
    if type(input_width) != int or input_width < 1:
        raise ValueError("positive input width required")

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_width = input_width
            self.shared = torch.nn.Sequential(torch.nn.Linear(input_width, 64), torch.nn.GELU(),
                torch.nn.Linear(64, 32), torch.nn.GELU())
            self.heads = torch.nn.ModuleList([torch.nn.Linear(32, 5) for _ in QUALITIES])

        def forward(self, x):
            if x.ndim != 2 or x.shape[1] != self.input_width:
                raise ValueError("wrong feature width")
            hidden = self.shared(x)
            return torch.stack([head(hidden) for head in self.heads], dim=1)

    return Network()


def save_network(network, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    state = {name: value.detach().cpu().numpy() for name, value in network.state_dict().items()}
    np.savez(directory / "weights.npz", **state)
    meta = {"schema": "perception-lesson-network-v1", "input_width": network.input_width,
            "hidden_widths": [64, 32], "activation": "gelu", "qualities": list(QUALITIES),
            "levels": list(LEVELS), "weights_sha256": file_sha256(directory / "weights.npz")}
    (directory / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_network(directory):
    import torch
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    if (meta.get("schema") != "perception-lesson-network-v1" or meta.get("qualities") != list(QUALITIES)
            or meta.get("levels") != list(LEVELS) or meta.get("hidden_widths") != [64, 32]
            or meta.get("activation") != "gelu"
            or file_sha256(directory / "weights.npz") != meta["weights_sha256"]):
        raise ValueError("network identity/checksum mismatch")
    network = build_network(meta["input_width"])
    with np.load(directory / "weights.npz", allow_pickle=False) as saved:
        expected = network.state_dict()
        if set(saved.files) != set(expected):
            raise ValueError("network keys changed")
        state = {}
        for name, prototype in expected.items():
            value = saved[name]
            if value.shape != tuple(prototype.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid network parameters")
            state[name] = torch.from_numpy(value.copy())
    network.load_state_dict(state, strict=True); network.eval()
    return network


def predict(network, features, batch_size=32):
    import torch
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all() or type(batch_size) != int or batch_size < 1:
        raise ValueError("finite matrix and positive batch size required")
    network.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = network(torch.from_numpy(features[start:start + batch_size].copy()))
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 5, 5), dtype=np.float32)


def masked_soft_loss(logits, target, weights):
    """Weights are prebalanced across quality/viewing, including origin weights.

    Zero-weight targets are absent; their contents cannot affect the loss.
    Pool whole-video predictions before calling this function for video labels.
    """
    import torch
    if logits.ndim != 3 or logits.shape[1:] != (5, 5) or target.shape != logits.shape or weights.shape != logits.shape[:2]:
        raise ValueError("invalid loss shapes")
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid logits/weights")
    valid = weights > 0
    if not valid.any():
        return logits.sum() * 0
    y = target[valid]
    if not torch.isfinite(y).all() or (y < 0).any() or not torch.allclose(y.sum(-1), torch.ones_like(y[:, 0]), atol=1e-6, rtol=0):
        raise ValueError("invalid supported distributions")
    loss = -(y * torch.log_softmax(logits[valid], dim=-1)).sum(-1)
    return (loss * weights[valid]).sum() / weights[valid].sum()

