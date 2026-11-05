"""Validated public output: missing evidence is null, never a fabricated score."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import LEVELS, QUALITIES, VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")


class QualityValue(StrictModel):
    distribution: list[float] | None = None
    value: float | None = Field(default=None, ge=0, le=10)
    ensemble_sd: float | None = Field(default=None, ge=0)
    evidence_status: Literal["human_supported", "synthetic_proxy", "experimental_transfer", "unavailable"]
    supervision_origins: list[Literal["human", "synthetic"]] = Field(default_factory=list)
    validation_status: Literal["not_evaluated", "exploratory", "human_evaluated"] = "not_evaluated"
    training_label_resolution_seconds: float | None = Field(default=None, gt=0)
    reason: str | None = None

    @model_validator(mode="after")
    def probabilities(self):
        if self.evidence_status == "unavailable":
            if self.value is not None or self.distribution is not None or self.ensemble_sd is not None or not self.reason:
                raise ValueError("unavailable values require a reason and null predictions")
        else:
            p = self.distribution
            if p is None or len(p) != 5 or any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
                raise ValueError("five finite probabilities required")
            if abs(sum(p) - 1) > 1e-6 or self.value is None:
                raise ValueError("normalized distribution and expected score required")
            if abs(sum(x * y for x, y in zip(p, LEVELS)) - self.value) > 1e-5:
                raise ValueError("score must equal the expected rating")
            if not self.supervision_origins:
                raise ValueError("supported output requires supervision provenance")
            if self.evidence_status == "human_supported" and "human" not in self.supervision_origins:
                raise ValueError("human-supported output requires human supervision")
        return self


class SourceReference(StrictModel):
    source_id: str = Field(min_length=1)
    kind: Literal["media_interval", "text_span", "page", "image"]
    start: float | None = Field(default=None, ge=0)
    stop: float | None = Field(default=None, ge=0)
    page_index: int | None = Field(default=None, ge=0)
    character_start: int | None = Field(default=None, ge=0)
    character_stop: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def boundaries(self):
        if self.kind == "media_interval":
            if self.start is None or self.stop is None or self.stop <= self.start:
                raise ValueError("media reference needs a positive interval")
        if self.kind == "page" and self.page_index is None:
            raise ValueError("page reference needs an index")
        if self.kind == "text_span":
            if self.character_start is None or self.character_stop is None or self.character_stop <= self.character_start:
                raise ValueError("text reference needs source offsets")
        return self


class TimelinePoint(StrictModel):
    second: int = Field(ge=0)
    context_start: float | None = Field(default=None, ge=0)
    context_stop: float | None = Field(default=None, ge=0)
    qualities: dict[str, QualityValue]
    active_modalities: list[Literal["text", "audio", "video", "tribe"]] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    references: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def context(self):
        if set(self.qualities) != set(QUALITIES):
            raise ValueError("all five quality slots must be explicit")
        supported = any(v.evidence_status != "unavailable" for v in self.qualities.values())
        if supported:
            if self.context_stop != self.second or self.context_start != self.second - 44:
                raise ValueError("predictions require the complete preceding 44 seconds")
            if not set(self.active_modalities) & {"text", "audio", "video"}:
                raise ValueError("at least one content modality required")
            if not self.references:
                raise ValueError("supported prediction must reference source content")
        if len(set(self.active_modalities)) != len(self.active_modalities):
            raise ValueError("duplicate modality")
        return self


class ChangeFlag(StrictModel):
    quality: str
    baseline_second: int = Field(ge=0)
    onset_second: int = Field(ge=0)
    confirmed_second: int = Field(ge=0)
    direction: Literal["increase", "decrease"]
    change: float
    interpretation: Literal["predicted_interval_not_observed_student_transition"] = "predicted_interval_not_observed_student_transition"

    @model_validator(mode="after")
    def valid(self):
        if self.quality not in QUALITIES or not self.baseline_second < self.onset_second < self.confirmed_second:
            raise ValueError("invalid flagged interval")
        if abs(self.change) < 2 or (self.change > 0) != (self.direction == "increase"):
            raise ValueError("invalid flagged direction or magnitude")
        return self


class Timeline(StrictModel):
    schema_version: Literal["perception-lesson-timeline-v1"] = "perception-lesson-timeline-v1"
    model_version: str = VERSION
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_format: Literal["video", "audio", "text", "pdf", "image"]
    duration_seconds: float = Field(gt=0)
    clock_basis: str = Field(min_length=1)
    query_step_seconds: Literal[1] = 1
    context_seconds: Literal[44] = 44
    psychological_resolution_validated_seconds: None = None
    observed_student_state: Literal[False] = False
    analysis_mode: Literal["whole_lesson_offline"] = "whole_lesson_offline"
    causal_realtime_prediction_validated: Literal[False] = False
    reference_population: str = Field(min_length=1)
    points: list[TimelinePoint]
    flags: list[ChangeFlag]

    @model_validator(mode="after")
    def grid(self):
        if [p.second for p in self.points] != list(range(math.floor(self.duration_seconds) + 1)):
            raise ValueError("one point at every integer second including unsupported positions")
        return self

