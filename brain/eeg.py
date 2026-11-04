"""Training-only block PCA and weighted multi-output EEG ridge regression.

Inputs are already audited, window-aligned content features and EEG targets.
This module does not certify the media/EEG alignment or signal quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = ("tp9", "af7", "af8", "tp10")
TARGETS = tuple(f"mean_{c}" for c in CHANNELS) + tuple(f"log1p_variance_{c}" for c in CHANNELS)


def matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def equal_viewing_weights(lessons: np.ndarray, people: np.ndarray) -> np.ndarray:
    """Equal lecture, then equal participant, then equal window within viewing."""
    if len(lessons) == 0 or len(lessons) != len(people):
        raise ValueError("invalid viewing identifiers")
    result = np.zeros(len(lessons), dtype=np.float64)
    lectures = np.unique(lessons)
    for lecture in lectures:
        subjects = np.unique(people[lessons == lecture])
        for subject in subjects:
            rows = (lessons == lecture) & (people == subject)
            result[rows] = 1 / (len(lectures) * len(subjects) * rows.sum())
    return result


def outer_scopes(families: np.ndarray, people: np.ndarray):
    """Eight fixed topic × participant-group folds; no random window split."""
    families, people = np.asarray(families), np.asarray(people)
    if families.shape != people.shape or families.ndim != 1:
        raise ValueError("aligned one-dimensional family/participant identifiers required")
    if not set(families).issubset({"aerospace", "xray"}) or not np.isin(people, range(1, 17)).all():
        raise ValueError("unexpected competition cohort")
    for family in ("aerospace", "xray"):
        for group in range(4):
            held = np.isin(people, np.arange(4 * group + 1, 4 * group + 5))
            yield {
                "held_family": family, "held_group": group + 1,
                "train": np.flatnonzero((families != family) & ~held),
                "test": np.flatnonzero((families == family) & held),
            }


@dataclass
class BlockTransform:
    mean: np.ndarray
    scale: np.ndarray
    axes: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = matrix(x, "content block")
        if x.shape[1] != len(self.mean):
            raise ValueError("feature width changed")
        return ((x - self.mean) / self.scale) @ self.axes


@dataclass
class EEGModel:
    blocks: dict[str, BlockTransform]
    target_mean: np.ndarray
    target_scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    fit_content_ids: tuple[str, ...]
    regularization: float

    @classmethod
    def fit(cls, blocks: dict[str, np.ndarray], targets: np.ndarray, content_ids: list[str],
            lessons: np.ndarray, people: np.ndarray, regularization: float = 1.0) -> "EEGModel":
        """Caller supplies only this scope's training rows, never held-out arrays."""
        y = matrix(targets, "EEG targets")
        n = len(y)
        if y.shape[1] != 8 or len(content_ids) != n or len(lessons) != n or len(people) != n:
            raise ValueError("aligned training rows and eight EEG targets required")
        if not blocks or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("feature blocks and positive regularization required")
        ids = np.asarray(content_ids, dtype=str)
        unique, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
        if any(not key for key in unique):
            raise ValueError("content identifiers cannot be empty")
        transforms, design = {}, []
        for name in sorted(blocks):
            x = matrix(blocks[name], name)
            if len(x) != n or not np.array_equal(x, x[first][inverse]):
                raise ValueError("content rows must agree across duplicate viewing identifiers")
            distinct = x[first]
            if len(distinct) < 2:
                raise ValueError("at least two distinct training-content windows required")
            mean, scale = distinct.mean(axis=0), distinct.std(axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            centered = (distinct - mean) / scale
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            tol = (singular[0] if len(singular) else 0) * max(centered.shape) * np.finfo(float).eps
            rank = min(16, int(np.sum(singular > tol)))
            axes = vt[:rank].T.copy()
            for column in range(rank):
                pivot = np.argmax(np.abs(axes[:, column]))
                if axes[pivot, column] < 0:
                    axes[:, column] *= -1
            transform = BlockTransform(mean, scale, axes)
            transforms[name] = transform
            design.append(transform.transform(x))
        x = np.concatenate(design, axis=1)
        weights = equal_viewing_weights(np.asarray(lessons), np.asarray(people))
        ym = weights @ y
        ys = np.sqrt(weights @ ((y - ym) ** 2))
        ys = np.where(ys > 0, ys, 1.0)
        z = (y - ym) / ys
        xm, zm = weights @ x, weights @ z
        xc, zc = x - xm, z - zm
        coef = np.linalg.solve(xc.T @ (weights[:, None] * xc) + regularization * np.eye(x.shape[1]),
                               xc.T @ (weights[:, None] * zc))
        return cls(transforms, ym, ys, zm - xm @ coef, coef, tuple(unique), regularization)

    def predict(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        if set(blocks) != set(self.blocks):
            raise ValueError("missing or unexpected feature block")
        x = np.concatenate([block.transform(blocks[name]) for name, block in self.blocks.items()], axis=1)
        return (x @ self.coefficients + self.intercept) * self.target_scale + self.target_mean

    def save(self, destination: Path) -> None:
        """No pickle, no overwrite; incomplete writes lack a valid manifest."""
        destination.mkdir(parents=True, exist_ok=False)
        arrays = {name: getattr(self, name) for name in ("target_mean", "target_scale", "intercept", "coefficients")}
        for i, block in enumerate(self.blocks.values()):
            for field in ("mean", "scale", "axes"):
                arrays[f"b{i}_{field}"] = getattr(block, field)
        path = destination / "arrays.npz"
        np.savez(path, **arrays)
        metadata = {"schema": "perception-eeg-ridge-v1", "targets": TARGETS,
                    "blocks": list(self.blocks), "fit_content_ids": self.fit_content_ids,
                    "regularization": self.regularization,
                    "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, source: Path) -> "EEGModel":
        meta = json.loads((source / "manifest.json").read_text())
        path = source / "arrays.npz"
        if meta["schema"] != "perception-eeg-ridge-v1" or tuple(meta["targets"]) != TARGETS:
            raise ValueError("unsupported EEG artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["arrays_sha256"]:
            raise ValueError("model array checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            blocks = {name: BlockTransform(*(arrays[f"b{i}_{field}"] for field in ("mean", "scale", "axes")))
                      for i, name in enumerate(meta["blocks"])}
            return cls(blocks, *(arrays[field] for field in ("target_mean", "target_scale", "intercept", "coefficients")),
                       tuple(meta["fit_content_ids"]), meta["regularization"])
