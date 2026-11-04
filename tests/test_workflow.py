import json
import hashlib
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from backend import store
from backend.api import app
from backend.worker import run_once
from backend.logic import default_profiles, simulate_one
from backend.models import LessonVersion


def video_file(path: Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=160x90:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-y", str(path)], check=True)


def test_upload_to_revision_keeps_keys_private_and_evidence_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "private")
    path = tmp_path / "fixture.mp4"
    video_file(path)
    client = TestClient(app)
    assessment = [{"id": "q1", "prompt": "Why do folds resist bending?",
                   "options": ["Ridges spread load", "They remove gravity"], "correct_index": 0}]
    with path.open("rb") as video:
        response = client.post("/api/lessons", data={
            "title": "Synthetic bridge lesson",
            "transcript": "00:00 - 00:01 | A flat paper bridge bends.\n00:01 - 00:02 | Folds make the bridge stiffer.",
            "assessment": json.dumps(assessment)},
            files={"video": ("fixture.mp4", video, "video/mp4")})
    assert response.status_code == 200, response.text
    lesson_id = response.json()["lesson"]["id"]
    assert "correct_index" not in response.text
    assert run_once()
    lesson = client.get(f"/api/lessons/{lesson_id}").json()
    assert lesson["status"] == "ready"
    assert lesson["sections"][0]["start"] == 0
    assert "correct_index" not in json.dumps(lesson)
    assert client.get(f"/api/lessons/{lesson_id}/assessment").json()[0]["correct_index"] == 0
    assert client.get(f"/api/reports/{lesson_id}").json()["metrics"]["attention_rating_mae"] is None
    edited_profiles = [profile.model_dump() for profile in default_profiles(lesson["concepts"])[:2]]
    edited_profiles[0]["name"] = "Needs a worked example"
    assert client.put(f"/api/lessons/{lesson_id}/profiles", json=edited_profiles).status_code == 200
    assert client.get(f"/api/lessons/{lesson_id}/profiles").json()[0]["name"] == "Needs a worked example"
    assert client.put(f"/api/lessons/{lesson_id}/profiles", json=edited_profiles[:1] * 2).status_code == 422

    simulation = client.post("/api/simulations", json={"lesson_id": lesson_id,
        "profile_count": 12, "seeds_per_profile": 3, "max_actions_per_run": 3})
    assert simulation.status_code == 200
    # Emulate a worker crash after saving the first run but before its checkpoint.
    first_run_id = hashlib.sha256(f"{simulation.json()['job_id']}:0".encode()).hexdigest()[:32]
    model = LessonVersion.model_validate(store.get("lessons", lesson_id))
    first = simulate_one(model, default_profiles(model.concepts)[0], 1, 3, first_run_id)
    store.put("runs", first.model_dump())
    assert run_once()
    runs = client.get(f"/api/simulations/{lesson_id}").json()
    assert len(runs) == 36
    assert all(len(run["actions"]) <= 3 for run in runs)
    assert all(run["prediction"]["status"] == "unavailable" for run in runs)
    assert "correct_index" not in json.dumps(runs)
    findings = client.get(f"/api/findings/{lesson_id}").json()
    assert findings and findings[0]["source_start"] == 1
    assert all(f["evidence_label"] == "agent-generated hypothesis" for f in findings)

    revision = client.post("/api/revisions", json={"finding_id": findings[0]["id"]}).json()
    assert revision["parent_id"] == lesson_id
    assert client.get(f"/api/lessons/{revision['id']}/profiles").json()[0]["name"] == "Needs a worked example"
    assert revision["source_hash"] == lesson["source_hash"]
    assert revision["content_hash"] != lesson["content_hash"]
    assert client.get(f"/api/lessons/{lesson_id}").json()["sections"] == lesson["sections"]
    branch = client.put(f"/api/lessons/{lesson_id}/concepts", json={"concepts": ["folds", "load"]}).json()
    assert branch["version"] == 3
    assert branch["parent_id"] == lesson_id
    assert branch["concepts_reviewed"] is True
    exported = client.get(f"/api/reports/{revision['id']}/export").text
    assert "No behavioral validation" in exported
    assert client.post("/api/simulations", json={"lesson_id": lesson_id,
        "profile_count": 10, "seeds_per_profile": 10}).status_code == 200
    assert client.post("/api/simulations", json={"lesson_id": lesson_id,
        "profile_count": 11, "seeds_per_profile": 10}).status_code == 422


def test_transcript_validation_and_cache_identity():
    from backend.adapter import feature_cache_key
    from backend.logic import parse_transcript
    import pytest
    with pytest.raises(ValueError):
        parse_transcript("00:02 - 00:01 | backwards")
    with pytest.raises(ValueError):
        parse_transcript("00:00 - 00:02 | timed\nuntimed")
    assert feature_cache_key("source", "prep1", "model1") != feature_cache_key("source", "prep2", "model1")
