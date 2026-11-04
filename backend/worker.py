from __future__ import annotations

import time
import hashlib
from . import store
from .logic import analyze, content_hash, default_profiles, findings_for, report_for, simulate_one
from .models import LessonVersion, SimulationRequest


def process(job: dict) -> None:
    id = job["id"]
    if job["state"] == "cancelled":
        return
    try:
        store.update_job(id, state="running", stage="loading")
        lesson_data = store.get("lessons", job["target_id"])
        if not lesson_data:
            raise ValueError("Lesson no longer exists")
        lesson = LessonVersion.model_validate(lesson_data)
        if job["kind"] == "analyze":
            store.update_job(id, stage="extracting", progress=25)
            lesson = analyze(lesson)
            lesson.content_hash = content_hash(lesson)
            store.put("lessons", lesson.model_dump())
            store.update_job(id, stage="reporting", progress=75)
            store.put("reports", report_for(lesson).model_dump())
            store.update_job(id, state="complete", stage=lesson.status, progress=100)
        elif job["kind"] == "simulate":
            if lesson.status != "ready":
                raise ValueError("A supplied transcript must be analyzed before simulation")
            config = SimulationRequest.model_validate(job["config"])
            profiles = (config.profiles or default_profiles(lesson.concepts))[:config.profile_count]
            total = len(profiles) * config.seeds_per_profile
            if total > 100:
                raise ValueError("Maximum 100 trajectories")
            # Progress is persisted after each trajectory, so a restarted worker resumes.
            for index in range(job["progress"], total):
                if store.job(id)["state"] == "cancelled":
                    return
                profile = profiles[index // config.seeds_per_profile]
                seed = index % config.seeds_per_profile + 1
                run_id = hashlib.sha256(f"{id}:{index}".encode()).hexdigest()[:32]
                run = simulate_one(lesson, profile, seed, config.max_actions_per_run, run_id)
                store.put("runs", run.model_dump())
                store.update_job(id, stage="simulating", progress=index + 1)
            # Idempotent findings: delete old findings for this lesson before writing.
            existing = [f for f in store.list_kind("findings") if f["lesson_id"] == lesson.id]
            with store.connect() as db:
                for finding in existing:
                    db.execute("DELETE FROM records WHERE kind='findings' AND id=?", (finding["id"],))
            for finding in findings_for(lesson):
                store.put("findings", finding.model_dump())
            store.update_job(id, state="complete", stage="complete", progress=total)
        else:
            raise ValueError("Unknown job type")
    except Exception as exc:
        store.update_job(id, state="failed", stage="failed", error=str(exc))


def run_once() -> bool:
    job = store.next_job()
    if not job:
        return False
    process(job)
    return True


def main() -> None:
    print("Classroom Lab worker ready", flush=True)
    while True:
        if not run_once():
            time.sleep(0.5)


if __name__ == "__main__":
    main()
