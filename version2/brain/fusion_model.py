"""Small local EEG temporal model. Pretrained cortical predictions are not trained."""
import json
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .eeg import equal_viewing_weights
from .fusion_data import MODALITIES, AUXILIARY, checked, read_json, write_json
from .worker_contract import file_sha256

RANK = 16


def fit_transform(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite training matrix required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = ((x - mean) / scale).astype(np.float32)
    _, s, vt = randomized_svd(z, n_components=min(RANK, *z.shape), n_iter=5,
                             n_oversamples=10, random_state=1729, flip_sign=True)
    rank = int((s > (s[0] if len(s) else 0) * max(z.shape) * np.finfo(np.float32).eps).sum())
    axes = vt[:rank].T.copy()
    return {"mean": mean, "scale": scale, "axes": axes}


def transform(values, fitted):
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != len(fitted["mean"]):
        raise ValueError("feature width changed")
    result = np.zeros((len(x), RANK), dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)
    result[~finite] = np.nan
    rank = fitted["axes"].shape[1]
    result[finite, :rank] = ((x[finite] - fitted["mean"]) / fitted["scale"]) @ fitted["axes"]
    return result


def save_transforms(destination, fitted, provenance):
    arrays = {f"{name}__{key}": value for name, fields in fitted.items() for key, value in fields.items()}
    np.savez(Path(destination) / "transforms.npz", **arrays)
    write_json(Path(destination) / "transforms.json", {"schema": "perception-fusion-transforms-v2",
        "blocks": list(fitted), "sha256": file_sha256(Path(destination) / "transforms.npz"),
        "training_scope": provenance})


def load_transforms(source):
    meta = read_json(Path(source) / "transforms.json")
    path = checked(source, "transforms.npz", meta["sha256"])
    with np.load(path, allow_pickle=False) as arrays:
        return {name: {key: arrays[f"{name}__{key}"] for key in ("mean", "scale", "axes")}
                for name in meta["blocks"]}


def build_network(aux_width=len(AUXILIARY) * RANK, width=32):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Residual(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
            self.norm = nn.LayerNorm(width)

        def forward(self, x):
            return x + F.gelu(self.norm(self.conv(x).transpose(1, 2))).transpose(1, 2)

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(4 * RANK + 4, width, 1)
            self.temporal = nn.Sequential(*(Residual(d) for d in (1, 2, 4)))
            self.head = nn.Sequential(nn.Linear(width + aux_width + len(AUXILIARY), 32), nn.GELU(), nn.Linear(32, 8))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            self.register_buffer("target_mean", torch.zeros(8))
            self.register_buffer("target_scale", torch.ones(8))
            self.register_buffer("variance_offset", torch.ones(4))

        def set_targets(self, mean, scale):
            self.target_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
            self.target_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
            relative = np.maximum(np.asarray(mean)[4:] / np.asarray(scale)[4:], 1e-7)
            offset = relative.copy()
            low = relative < 20
            offset[low] = np.log(np.expm1(relative[low]))
            self.variance_offset.copy_(torch.as_tensor(offset, dtype=torch.float32))

        def forward(self, sequence, auxiliary, mask, auxiliary_mask):
            # sequence B,T,M,R; masks describe availability, not confidence.
            x = (sequence * mask[:, None, :, None]).flatten(2)
            x = torch.cat((x, mask[:, None, :].expand(-1, x.shape[1], -1)), dim=-1)
            x = self.temporal(self.project(x.transpose(1, 2))).mean(dim=-1)
            a = (auxiliary.reshape(-1, len(AUXILIARY), RANK) * auxiliary_mask[:, :, None]).flatten(1)
            z = self.head(torch.cat((x, a, auxiliary_mask), dim=1))
            means = z[:, :4] * self.target_scale[:4] + self.target_mean[:4]
            variances = F.softplus(z[:, 4:] + self.variance_offset) * self.target_scale[4:]
            return torch.cat((means, variances), dim=1)

    return Fusion()


def arm_masks(count, arm, available=None, aux_available=None):
    mask = np.ones((count, 4), dtype=np.float32) if available is None else np.array(available, dtype=np.float32, copy=True)
    aux = np.ones((count, len(AUXILIARY)), dtype=np.float32) if aux_available is None else np.array(aux_available, dtype=np.float32, copy=True)
    if arm == "tribe_temporal":
        mask[:, :3] = 0
        aux[:] = 0
    elif arm == "without_tribe":
        mask[:, 3] = 0
    elif arm not in {"fusion", "shuffled_fusion"}:
        raise ValueError("unknown temporal arm")
    if not (mask.sum(axis=1) > 0).all():
        raise ValueError("no available model input")
    return mask, aux


def dropout_masks(mask, auxiliary_mask, probability=0.2):
    import torch
    keep = (torch.rand_like(mask) >= probability) & (mask > 0)
    empty = keep.sum(dim=1) == 0
    for index in torch.nonzero(empty).flatten():
        choices = torch.nonzero(mask[index] > 0).flatten()
        pick = choices[torch.randint(len(choices), (1,))]
        keep[index, pick] = True
    dropped = ((keep.float() < mask) & (mask > 0)).any(dim=1)
    aux = auxiliary_mask.clone()
    aux[dropped] = 0
    return keep.float(), aux


def shuffled_targets(target, lessons, people, seed):
    rng = np.random.default_rng(seed)
    shuffled = target.copy()
    permutation = np.arange(len(target))
    for lesson in np.unique(lessons):
        for person in np.unique(people[lessons == lesson]):
            rows = np.flatnonzero((lessons == lesson) & (people == person))
            if len(rows) > 1:
                permutation[rows] = np.roll(rows, int(rng.integers(1, len(rows))))
    shuffled[:] = target[permutation]
    return shuffled, permutation


def train(sequence, auxiliary, target, lessons, people, arm, seed, config, output):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    target = np.asarray(target, dtype=np.float64)
    if (sequence.shape != (len(target), 88, 4, RANK) or auxiliary.shape != (len(target), len(AUXILIARY) * RANK)
            or target.shape[1:] != (8,) or not all(np.isfinite(x).all() for x in (sequence, auxiliary, target))
            or np.any(target[:, 4:] < 0)):
        raise ValueError("finite supported training rows and physical variance targets required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    weights = equal_viewing_weights(lessons, people)
    mean = weights @ target
    scale = np.sqrt(weights @ ((target - mean) ** 2))
    scale[scale == 0] = 1
    permutation = np.arange(len(target))
    if arm == "shuffled_fusion":
        target, permutation = shuffled_targets(target, lessons, people, seed)
    network = build_network()
    network.set_targets(mean, scale)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x, a, y = (torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (sequence, auxiliary, target))
    w = torch.from_numpy(weights.astype(np.float32))
    m, am = (torch.from_numpy(v) for v in arm_masks(len(target), arm))
    history = []
    network.train()
    for epoch in range(config["epochs"]):
        order = rng.permutation(len(target))
        total = 0.0
        for start in range(0, len(target), config["batch_size"]):
            rows = order[start:start + config["batch_size"]]
            mm, aa = dropout_masks(m[rows], am[rows])
            prediction = network(x[rows], a[rows], mm, aa)
            losses = F.huber_loss((prediction - y[rows]) / network.target_scale, torch.zeros_like(prediction), reduction="none", delta=1).mean(dim=1)
            loss = (losses * w[rows]).sum() * len(target) / len(rows)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config["gradient_clip"])
            optimizer.step()
            total += float((losses.detach() * w[rows]).sum())
        history.append(total)
    network.eval()
    np.savez(output / "weights.npz", **{k: v.detach().cpu().numpy() for k, v in network.state_dict().items()})
    np.save(output / "target-permutation.npy", permutation, allow_pickle=False)
    report = {"schema": "perception-fusion-network-v2", "arm": arm, "seed": seed,
        "training": config, "training_loss": history, "weights_sha256": file_sha256(output / "weights.npz"),
        "permutation_sha256": file_sha256(output / "target-permutation.npy"),
        "parameters": sum(p.numel() for p in network.parameters()), "torch_version": torch.__version__,
        "training_mean": mean.tolist(), "training_scale": scale.tolist()}
    write_json(output / "manifest.json", report)
    return network, report


def load_network(source):
    import torch
    meta = read_json(Path(source) / "manifest.json")
    if meta.get("schema") != "perception-fusion-network-v2":
        raise ValueError("unsupported fusion network")
    path = checked(source, "weights.npz", meta["weights_sha256"])
    network = build_network()
    with np.load(path, allow_pickle=False) as weights:
        network.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
    return network.eval()


def predict(network, sequence, auxiliary, arm="fusion", available=None, aux_available=None):
    import torch
    mask, am = arm_masks(len(sequence), arm, available, aux_available)
    results = []
    with torch.inference_mode():
        for start in range(0, len(sequence), 32):
            sl = slice(start, start + 32)
            tensors = [torch.from_numpy(np.asarray(v[sl], dtype=np.float32)) for v in (sequence, auxiliary, mask, am)]
            results.append(network(*tensors).numpy())
    return np.concatenate(results) if results else np.empty((0, 8))
