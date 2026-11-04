from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from uuid import uuid4

from .models import (EvidenceReport, Finding, LearnerProfile, LessonSection,
                     LessonVersion, PredictionBundle, SimulationRun, StudentAction,
                     TranscriptSegment, now)


STOP = set("a an and are as at be by can for from has have in into is it its of on or our that the their then there these this to was were what when where which with you your".split())
TIME_LINE = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[|:]\s*(.+)$")


def seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    return float(sum(part * 60**index for index, part in enumerate(reversed(parts))))


def parse_transcript(raw: str) -> list[TranscriptSegment]:
    raw = raw.strip()
    if not raw:
        return []
    segments = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = TIME_LINE.match(line)
        if match:
            start, end = seconds(match[1]), seconds(match[2])
            if end <= start:
                raise ValueError("Transcript end time must follow start time")
            segments.append(TranscriptSegment(start=start, end=end, text=match[3].strip()))
        else:
            segments.append(TranscriptSegment(text=line.strip()))
    if any(s.start is not None for s in segments) and any(s.start is None for s in segments):
        raise ValueError("Use timing on every transcript line or on none")
    if all(s.start is not None for s in segments):
        for previous, current in zip(segments, segments[1:]):
            if current.start < previous.end:
                raise ValueError("Transcript segments overlap or are out of order")
    return segments


def concepts_for(segments: list[TranscriptSegment]) -> list[str]:
    text = " ".join(segment.text.lower() for segment in segments)
    words = re.findall(r"[a-z][a-z'-]{3,}", text)
    counts = Counter(word for word in words if word not in STOP)
    return [word for word, _ in counts.most_common(8)]


def analyze(lesson: LessonVersion) -> LessonVersion:
    if not lesson.transcript:
        lesson.status = "needs_transcript"
        lesson.analysis_provenance = "unavailable: no supplied transcript"
        return lesson
    lesson.concepts = concepts_for(lesson.transcript)
    lesson.concepts_reviewed = False
    lesson.sections = []
    for index, segment in enumerate(lesson.transcript):
        title = "Section " + str(index + 1)
        if segment.text:
            first = re.split(r"[.!?]", segment.text)[0].strip()
            title = (first[:64] + ("…" if len(first) > 64 else "")) or title
        lesson.sections.append(LessonSection(
            id=f"s{index+1}", title=title, start=segment.start,
            end=segment.end, text=segment.text))
    lesson.status = "ready"
    lesson.analysis_provenance = "deterministic transcript extraction; educator review required"
    return lesson


def default_profiles(concepts: list[str]) -> list[LearnerProfile]:
    profiles = []
    for index in range(12):
        level = ("low", "medium", "high")[index % 3]
        misconception = [concepts[index % len(concepts)]] if concepts and index % 4 == 0 else []
        profiles.append(LearnerProfile(
            id=f"profile-{index+1:02}", name=f"Profile {index+1:02} · {level} prior knowledge",
            prior_knowledge=level, misconceptions=misconception,
            provenance="authored template; no individual or demographic inference"))
    return profiles


def unavailable_prediction() -> PredictionBundle:
    return PredictionBundle(endpoint="attention difficulty / quiz correctness", reason=
        "No behavioral model has been fitted and validated for this lesson. Actions are authored simulation assumptions.")


def content_hash(lesson: LessonVersion) -> str:
    payload = {"source_hash": lesson.source_hash,
               "transcript": [s.model_dump() for s in lesson.transcript],
               "concepts": lesson.concepts,
               "concepts_reviewed": lesson.concepts_reviewed,
               "sections": [s.model_dump() for s in lesson.sections if s.added_by_revision],
               "assessment": [a.model_dump() for a in lesson.assessment]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fork(lesson: LessonVersion, version_number: int | None = None) -> LessonVersion:
    version = lesson.model_copy(deep=True)
    version.id = uuid4().hex
    version.parent_id = lesson.id
    version.version = version_number or lesson.version + 1
    version.created_at = now()
    return version


def simulate_one(lesson: LessonVersion, profile: LearnerProfile, seed: int, max_actions: int,
                 run_id: str | None = None) -> SimulationRun:
    rng = random.Random(seed + int(hashlib.sha256(profile.id.encode()).hexdigest()[:8], 16))
    actions: list[StudentAction] = []
    for section in lesson.sections:
        if len(actions) >= max_actions:
            break
        actions.append(StudentAction(kind="read_section", section_id=section.id,
                                     source_start=section.start, text="Read the exposed section."))
        if profile.prior_knowledge == "low" and len(actions) < max_actions:
            actions.append(StudentAction(kind="ask_question", section_id=section.id,
                source_start=section.start, text="Could you explain this step with an example?"))
        if profile.misconceptions and len(actions) < max_actions and any(
            word.lower() in section.text.lower() for word in profile.misconceptions):
            actions.append(StudentAction(kind="review_section", section_id=section.id,
                source_start=section.start, text="Review a concept flagged in this authored profile."))
    for item in lesson.assessment:
        if len(actions) >= max_actions:
            break
        # The renderer receives only prompt/options. It never sees the answer key.
        actions.append(StudentAction(kind="attempt_item", item_id=item.id,
            text="Selected an option in a scripted run; correctness is not inferred.",
            selected_option=rng.randrange(len(item.options))))
    return SimulationRun(id=run_id or uuid4().hex, lesson_id=lesson.id, profile=profile,
                         seed=seed, actions=actions, prediction=unavailable_prediction())


def findings_for(lesson: LessonVersion) -> list[Finding]:
    if not lesson.sections:
        return []
    findings = []
    # This is a transparent content check, not a measured learner difficulty.
    transcript_text = " ".join(section.text for section in lesson.sections).lower()
    if lesson.assessment and not any(mark in transcript_text for mark in ("?", "try", "practice", "checkpoint")):
        source = lesson.sections[-1]
        findings.append(Finding(id=uuid4().hex, lesson_id=lesson.id, section_id=source.id,
            source_start=source.start, source_excerpt=source.text[:220],
            title="No explicit practice before assessment",
            explanation="The supplied transcript has no question or practice cue before the assessment. This is a content check, not evidence that learners struggled.",
            suggested_repair="Add a short checkpoint that asks learners to apply the last concept, then explain the reasoning."))
    for item in lesson.assessment:
        words = [w for w in re.findall(r"[a-z]{5,}", item.prompt.lower()) if w not in STOP]
        if words and not any(word in transcript_text for word in words):
            source = lesson.sections[-1]
            findings.append(Finding(id=uuid4().hex, lesson_id=lesson.id, section_id=source.id,
                source_start=source.start, source_excerpt=source.text[:220],
                title=f"Assessment item {item.id} lacks an obvious transcript match",
                explanation="Prompt terms were not found in the supplied transcript. Wording differences may explain this; educator review is needed.",
                suggested_repair=f"Add an explanation or example preparing learners for: {item.prompt}"))
    return findings[:8]


def report_for(lesson: LessonVersion) -> EvidenceReport:
    report_id = hashlib.sha256(("report:" + lesson.id).encode()).hexdigest()[:32]
    return EvidenceReport(id=report_id, lesson_id=lesson.id,
        metrics={"attention_rating_mae": None, "quiz_brier": None, "quiz_log_loss": None},
        limitations=[
            "No observed human outcome labels were used for this lesson.",
            "Student trajectories are authored simulations, not independent people.",
            "No validated attention or quiz prediction is available; revision comparisons are content hypotheses.",
        ], prediction=unavailable_prediction())


def revise(lesson: LessonVersion, finding: Finding, version_number: int | None = None) -> LessonVersion:
    revised = fork(lesson, version_number)
    revised.title = lesson.title + " · revision " + str(revised.version)
    revised.analysis_provenance = "educator-reviewable authored repair; original media retained"
    revised.sections.append(LessonSection(id=f"repair-{revised.version}", title="Proposed checkpoint",
        text=finding.suggested_repair, added_by_revision=True))
    revised.content_hash = content_hash(revised)
    return revised
