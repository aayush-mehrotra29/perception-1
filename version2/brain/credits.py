"""Persistent conservative reservations, shared by all paid model adapters.

Reservation ceilings must include loading and requested execution time. Provider
billing may lag; this ledger cannot substitute for a provider's billing limit.
"""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import json
import re


class CreditLedger:
    LIMITS = {"modal": 30000000, "meta_api": 150000000}  # integer microdollars

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY, provider TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS settlement_groups (id TEXT PRIMARY KEY, provider TEXT NOT NULL, actual INTEGER NOT NULL, evidence_sha256 TEXT NOT NULL, allocations TEXT NOT NULL)")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, identifier: str, provider: str, maximum_microdollars: int):
        if provider not in self.LIMITS or type(maximum_microdollars) is not int or maximum_microdollars <= 0 or not identifier:
            raise ValueError("valid provider, unique request ID and positive integer reservation required")
        with self._transaction() as db:
            existing = db.execute("SELECT 1 FROM charges WHERE id=?", (identifier,)).fetchone()
            if existing:
                raise ValueError("request already reserved; inspect its status instead of resubmitting")
            committed = db.execute("SELECT COALESCE(SUM(CASE WHEN state='settled' THEN actual ELSE reserved END),0) FROM charges WHERE provider=?", (provider,)).fetchone()[0]
            if committed + maximum_microdollars > self.LIMITS[provider]:
                raise ValueError("credit ceiling exceeded")
            db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (identifier, provider, maximum_microdollars))

    def settle(self, identifier: str, actual_microdollars: int):
        if type(actual_microdollars) is not int or actual_microdollars < 0:
            raise ValueError("nonnegative integer actual cost required")
        with self._transaction() as db:
            row = db.execute("SELECT state FROM charges WHERE id=?", (identifier,)).fetchone()
            if not row or row[0] != "reserved":
                raise ValueError("unknown or already settled request")
            # Retain actual overruns so future reservations see the full liability.
            db.execute("UPDATE charges SET actual=?,state='settled' WHERE id=?", (actual_microdollars, identifier))

    def snapshot(self):
        with self._transaction() as db:
            return {provider: {"limit_microdollars": limit, "committed_microdollars": db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='settled' THEN actual ELSE reserved END),0) FROM charges WHERE provider=?", (provider,)).fetchone()[0]}
                    for provider, limit in self.LIMITS.items()}

    def settle_group(self, group_id, identifiers, actual_microdollars, evidence_sha256):
        """Book an exact stopped-app total using explicitly recorded allocations.

Provider evidence is app-level, not per-request. The allocations below exist
only so older ledger readers retain the same total; they are not individual
provider charges. Caller must verify every request is terminal and keep the
separate billing-lag reserve when the provider report is provisional.
        """
        if (not group_id or not identifiers or len(set(identifiers)) != len(identifiers)
                or type(actual_microdollars) is not int or actual_microdollars < 0
                or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256 or "") is None):
            raise ValueError("unique requests, integer cost and billing evidence required")
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM settlement_groups WHERE id=?", (group_id,)).fetchone():
                raise ValueError("billing group already settled")
            rows = [db.execute("SELECT provider,state,reserved FROM charges WHERE id=?", (name,)).fetchone() for name in identifiers]
            if any(row is None or row[1] != "reserved" for row in rows) or len({row[0] for row in rows}) != 1:
                raise ValueError("all group members must be reserved on the same provider")
            reserved = sum(row[2] for row in rows)
            costs = [actual_microdollars * row[2] // reserved for row in rows]
            costs[-1] += actual_microdollars - sum(costs)
            allocations = dict(zip(identifiers, costs))
            db.execute("INSERT INTO settlement_groups VALUES(?,?,?,?,?)",
                       (group_id, rows[0][0], actual_microdollars, evidence_sha256, json.dumps(allocations)))
            for identifier, amount in allocations.items():
                db.execute("UPDATE charges SET state='settled',actual=? WHERE id=?", (amount, identifier))
