"""Auditable supervision. Synthetic agreement never constitutes human accuracy."""
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from . import LEVELS, QUALITIES
from .contract import StrictModel


class Label(StrictModel):
    id: str = Field(min_length=1)
    person: str = Field(min_length=1)
    content_family: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality: str
    origin: Literal["human", "synthetic"]
    scope: Literal["interval", "whole_video", "assessment"]
    start: float = Field(ge=0)
    stop: float = Field(gt=0)
    distribution: list[float]
    evidence_ids: list[str] = Field(min_length=1)
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid(self):
        p = np.asarray(self.distribution)
        if (self.quality not in QUALITIES or self.stop <= self.start or p.shape != (5,)
                or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or abs(float(p.sum()) - 1) > 1e-6):
            raise ValueError("invalid label quality, interval or probability distribution")
        return self


def ordinal_distribution(score):
    """Linear soft binning preserves expected score; creates no extra precision claim."""
    if not np.isfinite(score) or not 0 <= score <= 10:
        raise ValueError("score must be finite in 0–10")
    position = score / 2.5
    lower = int(np.floor(position)); upper = min(4, lower + 1)
    out = np.zeros(5, dtype=float)
    if upper == lower:
        out[lower] = 1
    else:
        out[lower] = upper - position; out[upper] = position - lower
    return out.tolist()


def human_attention(rating):
    if not np.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
        raise ValueError("released integer attention-difficulty rating must be 1–5")
    return ordinal_distribution((rating - 1) * 2.5)


def reconcile_synthetic(first, second):
    """Two rubric distributions, or null on abstention/disagreement. No cloud calls."""
    if first is None or second is None:
        return {"accepted": False, "reason": "insufficient_evidence", "distribution": None}
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    for p in (a, b):
        if p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(p.sum() - 1) > 1e-6:
            raise ValueError("invalid rubric probabilities")
    disagreement = float(abs(a @ LEVELS - b @ LEVELS))
    accepted = disagreement <= 2.5
    return {"accepted": accepted, "reason": None if accepted else "rubric_disagreement",
            "distribution": ((a + b) / 2).tolist() if accepted else None,
            "expected_score_disagreement": disagreement,
            "disagreement_is_calibrated_confidence": False}


def resolve_labels(labels):
    """Only identical scopes override. Never expand a whole-video label into seconds."""
    selected = {}
    for label in labels:
        key = (label.person, label.source_sha256, label.quality, label.scope, label.start, label.stop)
        old = selected.get(key)
        if old is None or (old.origin == "synthetic" and label.origin == "human"):
            selected[key] = label
        elif old.origin == label.origin and old.model_dump() != label.model_dump():
            raise ValueError("conflicting duplicate labels require an explicit audit")
    return list(selected.values())


def assert_disjoint(train, test):
    if {x.person for x in train} & {x.person for x in test}:
        raise ValueError("participant leakage")
    if {x.content_family for x in train} & {x.content_family for x in test}:
        raise ValueError("content-family leakage")
    if {x.source_sha256 for x in train} & {x.source_sha256 for x in test}:
        raise ValueError("source-media leakage")

