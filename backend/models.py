from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranscriptSegment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str


class LessonSection(BaseModel):
    id: str
    title: str
    start: float | None = None
    end: float | None = None
    text: str
    added_by_revision: bool = False


class AssessmentItem(BaseModel):
    id: str
    prompt: str
    options: list[str] = Field(min_length=2, max_length=6)
    correct_index: int


class LessonVersion(BaseModel):
    id: str
    project_id: str
    version: int
    parent_id: str | None = None
    title: str
    source_hash: str
    content_hash: str = ""
    media_path: str | None = None
    duration_seconds: float | None = None
    transcript: list[TranscriptSegment] = []
    sections: list[LessonSection] = []
    concepts: list[str] = []
    concepts_reviewed: bool = False
    assessment: list[AssessmentItem] = []
    status: str = "queued"
    analysis_provenance: str = "pending"
    created_at: str = Field(default_factory=now)


class LearnerProfile(BaseModel):
    id: str
    name: str
    prior_knowledge: Literal["low", "medium", "high"]
    misconceptions: list[str] = []
    provenance: str = "educator-authored template"


class PredictionBundle(BaseModel):
    endpoint: str
    model_version: str | None = None
    estimates: dict[str, float] | None = None
    uncertainty: dict[str, float] | None = None
    supported_population: str | None = None
    status: Literal["unavailable", "available"] = "unavailable"
    claim_type: Literal["model prediction", "agent-generated hypothesis"] = "agent-generated hypothesis"
    reason: str | None = None


class StudentAction(BaseModel):
    kind: Literal["read_section", "review_section", "ask_question", "attempt_item"]
    section_id: str | None = None
    item_id: str | None = None
    text: str | None = None
    selected_option: int | None = None
    source_start: float | None = None


class SimulationRun(BaseModel):
    id: str
    lesson_id: str
    profile: LearnerProfile
    seed: int
    actions: list[StudentAction]
    prediction: PredictionBundle
    cost_usd: float = 0
    tokens_used: int = 0
    status: str = "complete"
    created_at: str = Field(default_factory=now)


class Finding(BaseModel):
    id: str
    lesson_id: str
    section_id: str
    source_start: float | None = None
    source_excerpt: str
    title: str
    explanation: str
    suggested_repair: str
    evidence_label: Literal["observed human data", "model prediction", "agent-generated hypothesis"] = "agent-generated hypothesis"


class EvidenceReport(BaseModel):
    id: str
    lesson_id: str
    dataset_id: str | None = None
    split_id: str | None = None
    metrics: dict[str, float | None] = {}
    comparisons: list[dict] = []
    limitations: list[str]
    prediction: PredictionBundle
    created_at: str = Field(default_factory=now)


class SimulationRequest(BaseModel):
    lesson_id: str
    seeds_per_profile: int = Field(default=3, ge=1, le=100)
    profile_count: int = Field(default=12, ge=1, le=12)
    profiles: list[LearnerProfile] | None = Field(default=None, min_length=1, max_length=12)
    max_actions_per_run: int = Field(default=20, ge=1, le=100)
    max_tokens_per_run: int = Field(default=500, ge=0, le=10000)
    max_cost_usd_per_run: float = Field(default=5, ge=0, le=5)
