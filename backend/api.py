from __future__ import annotations

import asyncio
import hashlib
import html
import json
import mimetypes
import subprocess
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import store
from .adapter import readiness
from .logic import content_hash, default_profiles, fork, parse_transcript, report_for, revise
from .models import AssessmentItem, Finding, LearnerProfile, LessonVersion, SimulationRequest

app = FastAPI(title="Classroom Lab", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def public_lesson(raw: dict) -> dict:
    lesson = dict(raw)
    lesson.pop("media_path", None)
    lesson["assessment"] = [{"id": a["id"], "prompt": a["prompt"], "options": a["options"]}
                             for a in raw.get("assessment", [])]
    return lesson


def require(kind: str, id: str) -> dict:
    value = store.get(kind, id)
    if value is None:
        raise HTTPException(404, f"{kind} record not found")
    return value


def next_version(project_id: str) -> int:
    return 1 + max((item["version"] for item in store.list_kind("lessons")
                    if item["project_id"] == project_id), default=0)


def parse_assessment(raw: str) -> list[AssessmentItem]:
    try:
        items = [AssessmentItem.model_validate(item) for item in json.loads(raw)]
        if not items:
            raise ValueError("Add at least one assessment item")
        if len({i.id for i in items}) != len(items):
            raise ValueError("Assessment item IDs must be unique")
        if any(i.correct_index < 0 or i.correct_index >= len(i.options) for i in items):
            raise ValueError("Answer index is outside the options")
        return items
    except (ValueError, TypeError, ValidationError) as exc:
        raise HTTPException(422, f"Invalid assessment: {exc}") from exc


def duration(path: Path) -> float:
    try:
        stream = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=codec_type",
                                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                                capture_output=True, text=True, timeout=20, check=True)
        if stream.stdout.strip() != "video":
            raise ValueError("File has no video stream")
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                                capture_output=True, text=True, timeout=20, check=True)
        value = float(result.stdout.strip())
        if value <= 0 or value > 600:
            raise ValueError("Video must be between 0 and 600 seconds")
        return value
    except (subprocess.SubprocessError, ValueError) as exc:
        raise HTTPException(422, f"Invalid or unsupported video: {exc}") from exc


@app.get("/api/health")
def health():
    return {"status": "ok", "research": readiness()}


@app.get("/api/projects")
def projects():
    lessons = store.list_kind("lessons")
    latest = {}
    project_by_lesson = {item["id"]: item["project_id"] for item in lessons}
    run_counts: dict[str, int] = {}
    for run in store.list_kind("runs"):
        project_id = project_by_lesson.get(run["lesson_id"])
        if project_id:
            run_counts[project_id] = run_counts.get(project_id, 0) + 1
    for lesson in lessons:
        if lesson["project_id"] not in latest or lesson["version"] > latest[lesson["project_id"]]["version"]:
            latest[lesson["project_id"]] = lesson
    return [{**public_lesson(item), "trajectory_count": run_counts.get(item["project_id"], 0)}
            for item in latest.values()]


@app.post("/api/lessons")
async def upload_lesson(title: str = Form(...), transcript: str = Form(""),
                        assessment: str = Form(...), video: UploadFile = File(...)):
    title = title.strip()
    if not title or len(title) > 160:
        raise HTTPException(422, "Title must contain 1–160 characters")
    items = parse_assessment(assessment)
    try:
        segments = parse_transcript(transcript)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    project_id, id = uuid4().hex, uuid4().hex
    media_dir = store.DATA_DIR / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in (".mp4", ".m4v", ".mov", ".webm"):
        raise HTTPException(422, "Use an MP4, M4V, MOV or WebM video")
    path = media_dir / f"{id}{suffix}"
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > 250 * 1024 * 1024:
                    raise HTTPException(413, "Video exceeds 250 MB limit")
                digest.update(chunk)
                out.write(chunk)
        if not size:
            raise HTTPException(422, "Video is empty")
        video_duration = duration(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    lesson = LessonVersion(id=id, project_id=project_id, version=1, title=title,
        source_hash=digest.hexdigest(), media_path=str(path), duration_seconds=video_duration,
        transcript=segments, assessment=items)
    lesson.content_hash = content_hash(lesson)
    store.put("lessons", lesson.model_dump())
    job_id = store.create_job("analyze", id)
    return {"lesson": public_lesson(lesson.model_dump()), "job_id": job_id}


@app.get("/api/lessons/{lesson_id}")
def get_lesson(lesson_id: str):
    return public_lesson(require("lessons", lesson_id))


@app.get("/api/lessons/{lesson_id}/assessment")
def get_assessment(lesson_id: str):
    return require("lessons", lesson_id)["assessment"]


@app.put("/api/lessons/{lesson_id}/assessment")
def update_assessment(lesson_id: str, items: list[AssessmentItem]):
    lesson = LessonVersion.model_validate(require("lessons", lesson_id))
    if any(i.correct_index < 0 or i.correct_index >= len(i.options) for i in items):
        raise HTTPException(422, "Answer index is outside the options")
    revised = fork(lesson, next_version(lesson.project_id))
    revised.assessment = items
    revised.content_hash = content_hash(revised)
    revised.analysis_provenance = "assessment edited by educator; source material inherited"
    store.put("lessons", revised.model_dump())
    store.put("reports", report_for(revised).model_dump())
    return public_lesson(revised.model_dump())


@app.put("/api/lessons/{lesson_id}/concepts")
def review_concepts(lesson_id: str, body: dict):
    lesson = LessonVersion.model_validate(require("lessons", lesson_id))
    values = body.get("concepts")
    if not isinstance(values, list) or len(values) > 30 or any(not isinstance(x, str) or not x.strip() or len(x) > 80 for x in values):
        raise HTTPException(422, "Provide up to 30 nonempty concepts")
    revised = fork(lesson, next_version(lesson.project_id))
    revised.concepts = list(dict.fromkeys(x.strip() for x in values))
    revised.concepts_reviewed = True
    revised.analysis_provenance = "concept list reviewed by educator; original transcript retained"
    revised.content_hash = content_hash(revised)
    store.put("lessons", revised.model_dump())
    store.put("reports", report_for(revised).model_dump())
    return public_lesson(revised.model_dump())


@app.get("/api/lessons/{lesson_id}/media")
def media(lesson_id: str):
    path = Path(require("lessons", lesson_id)["media_path"])
    if not path.is_file():
        raise HTTPException(404, "Media unavailable")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.post("/api/lessons/{lesson_id}/transcript")
def supply_transcript(lesson_id: str, body: dict):
    lesson = LessonVersion.model_validate(require("lessons", lesson_id))
    revised = fork(lesson, next_version(lesson.project_id))
    try:
        revised.transcript = parse_transcript(str(body.get("transcript", "")))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    revised.status = "queued"
    revised.sections = []
    revised.concepts = []
    revised.content_hash = content_hash(revised)
    store.put("lessons", revised.model_dump())
    return {"job_id": store.create_job("analyze", revised.id), "lesson": public_lesson(revised.model_dump())}


@app.post("/api/simulations")
def create_simulation(body: SimulationRequest):
    lesson = require("lessons", body.lesson_id)
    if lesson["status"] != "ready":
        raise HTTPException(409, "Analyze a supplied transcript first")
    count = min(body.profile_count, len(body.profiles)) if body.profiles else body.profile_count
    if count * body.seeds_per_profile > 100:
        raise HTTPException(422, "Maximum 100 trajectories")
    if body.profiles:
        store.put("profile_sets", {"id": body.lesson_id, "profiles": [p.model_dump() for p in body.profiles]})
    return {"job_id": store.create_job("simulate", body.lesson_id, body.model_dump())}


@app.get("/api/lessons/{lesson_id}/profiles")
def get_profiles(lesson_id: str):
    lesson = LessonVersion.model_validate(require("lessons", lesson_id))
    saved = store.get("profile_sets", lesson_id)
    return saved["profiles"] if saved else [p.model_dump() for p in default_profiles(lesson.concepts)]


@app.put("/api/lessons/{lesson_id}/profiles")
def update_profiles(lesson_id: str, profiles: list[LearnerProfile]):
    require("lessons", lesson_id)
    if not 1 <= len(profiles) <= 12:
        raise HTTPException(422, "Add between 1 and 12 learner profiles")
    if any(not profile.name.strip() for profile in profiles):
        raise HTTPException(422, "Every learner profile needs a name")
    if len({profile.id for profile in profiles}) != len(profiles):
        raise HTTPException(422, "Learner profile IDs must be unique")
    saved = [profile.model_dump() for profile in profiles]
    store.put("profile_sets", {"id": lesson_id, "profiles": saved})
    return saved


@app.get("/api/simulations/{lesson_id}")
def get_simulations(lesson_id: str):
    return [run for run in store.list_kind("runs") if run["lesson_id"] == lesson_id]


@app.get("/api/findings/{lesson_id}")
def get_findings(lesson_id: str):
    return [item for item in store.list_kind("findings") if item["lesson_id"] == lesson_id]


@app.get("/api/reports/{lesson_id}")
def get_report(lesson_id: str):
    reports = [item for item in store.list_kind("reports") if item["lesson_id"] == lesson_id]
    return reports[0] if reports else {"status": "unavailable"}


@app.get("/api/reports/{lesson_id}/export", response_class=HTMLResponse)
def export_report(lesson_id: str):
    lesson = require("lessons", lesson_id)
    findings = get_findings(lesson_id)
    report = get_report(lesson_id)
    title = html.escape(lesson["title"])
    rows = "".join(f"<article><h2>{html.escape(f['title'])}</h2><p>{html.escape(f['explanation'])}</p>"
                   f"<blockquote>{html.escape(f['source_excerpt'])}</blockquote><small>{html.escape(f['evidence_label'])}</small></article>"
                   for f in findings)
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in report.get("limitations", []))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{title} · Classroom Lab</title>
    <style>body{{font:16px/1.5 system-ui;max-width:760px;margin:60px auto;padding:0 24px;color:#172226}}article{{border-top:1px solid #ccd4d7;padding:24px 0}}small{{color:#6a757a}}blockquote{{border-left:3px solid #b6c9c7;padding-left:16px}}@media print{{body{{margin:20px auto}}}}</style>
    <h1>{title}</h1><p>Lesson version {lesson['version']} · source SHA-256 {html.escape(lesson['source_hash'])} · content SHA-256 {html.escape(lesson['content_hash'])}</p>
    <p><strong>Evidence status:</strong> No behavioral validation for this lesson. Findings are agent-generated hypotheses.</p>
    {rows}<h2>Limitations</h2><ul>{limits}</ul></html>"""


@app.post("/api/revisions")
def create_revision(body: dict):
    finding = Finding.model_validate(require("findings", str(body.get("finding_id", ""))))
    lesson = LessonVersion.model_validate(require("lessons", finding.lesson_id))
    revised = revise(lesson, finding, next_version(lesson.project_id))
    store.put("lessons", revised.model_dump())
    store.put("reports", report_for(revised).model_dump())
    profiles = store.get("profile_sets", lesson.id)
    if profiles:
        store.put("profile_sets", {"id": revised.id, "profiles": profiles["profiles"]})
    return public_lesson(revised.model_dump())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    value = store.job(job_id)
    if not value:
        raise HTTPException(404, "Job not found")
    return value


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    value = store.job(job_id)
    if not value:
        raise HTTPException(404, "Job not found")
    if value["state"] in ("queued", "running"):
        store.update_job(job_id, state="cancelled", stage="cancelled")
    return store.job(job_id)


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str, after: int = 0):
    if not store.job(job_id):
        raise HTTPException(404, "Job not found")
    async def stream():
        cursor = after
        while True:
            for event in store.events(job_id, cursor):
                cursor = event["id"]
                yield f"id: {cursor}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
            state = store.job(job_id)["state"]
            if state in ("complete", "failed", "cancelled"):
                break
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream")


DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    if not (DIST / "index.html").is_file():
        raise HTTPException(503, "Build the frontend first")
    return FileResponse(DIST / "index.html")
