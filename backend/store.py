from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
from .models import now


DATA_DIR = Path(os.environ.get("CLASSROOM_LAB_DATA", Path(__file__).resolve().parents[1] / ".local"))


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA_DIR / "classroom.db", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
        PRIMARY KEY(kind,id));
      CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, target_id TEXT NOT NULL,
        state TEXT NOT NULL, stage TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
        config TEXT NOT NULL DEFAULT '{}', error TEXT, updated_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
        type TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
    """)
    return db


def put(kind: str, item: dict) -> None:
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)", (kind, item["id"], json.dumps(item)))


def get(kind: str, id: str) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, id)).fetchone()
    return json.loads(row[0]) if row else None


def list_kind(kind: str) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT data FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def create_job(kind: str, target_id: str, config: dict | None = None) -> str:
    id = uuid4().hex
    with connect() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                   (id, kind, target_id, "queued", "queued", 0, json.dumps(config or {}), None, now()))
    event(id, "queued", {"stage": "queued"})
    return id


def job(id: str) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["config"] = json.loads(result["config"])
    return result


def next_job() -> dict | None:
    with connect() as db:
        row = db.execute("SELECT id FROM jobs WHERE state IN ('queued','running') ORDER BY updated_at LIMIT 1").fetchone()
    return job(row[0]) if row else None


def update_job(id: str, **fields) -> None:
    fields["updated_at"] = now()
    keys = list(fields)
    with connect() as db:
        db.execute("UPDATE jobs SET " + ",".join(f"{key}=?" for key in keys) + " WHERE id=?",
                   [fields[key] for key in keys] + [id])
    event(id, "progress", fields)


def event(job_id: str, type: str, data: dict) -> None:
    with connect() as db:
        db.execute("INSERT INTO events(job_id,type,data,created_at) VALUES(?,?,?,?)",
                   (job_id, type, json.dumps(data), now()))


def events(job_id: str, after: int = 0) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id", (job_id, after)).fetchall()
    return [{**dict(row), "data": json.loads(row["data"])} for row in rows]
