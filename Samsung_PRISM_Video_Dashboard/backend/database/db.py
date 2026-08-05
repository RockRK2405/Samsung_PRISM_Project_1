"""SQLite experiment store. One row per video test.

Stores the full model output (including per-frame scores + temporal
stats as JSON) so old experiments can be reopened with their complete
analysis, and so threshold sweeps can run on stored probabilities with
no model reruns.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config_loader import database_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    ground_truth TEXT,               -- 'Real' | 'Synthetic' | 'Unknown' | NULL
    generator TEXT,                  -- 'Gemini' | 'Veo' | ... | NULL
    prediction TEXT NOT NULL,        -- 'Real' | 'Synthetic'
    confidence REAL,
    synthetic_probability REAL,
    real_probability REAL,
    threshold REAL,
    processing_time REAL,
    num_frames_used INTEGER,
    model_version TEXT,
    mock INTEGER NOT NULL DEFAULT 0,  -- 1 if produced by the mock model
    batch_id TEXT,                   -- groups rows uploaded as one batch
    payload_json TEXT NOT NULL,      -- full model_service payload
    metadata_json TEXT               -- video_service metadata
);
CREATE INDEX IF NOT EXISTS idx_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_batch ON experiments(batch_id);
"""


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def insert_experiment(
    filename: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    ground_truth: str | None = None,
    generator: str | None = None,
    model_version: str | None = None,
    batch_id: str | None = None,
) -> str:
    """Persist one test; returns the new experiment id."""
    exp_id = uuid.uuid4().hex[:12]
    created = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO experiments (
                id, created_at, filename, ground_truth, generator,
                prediction, confidence, synthetic_probability, real_probability,
                threshold, processing_time, num_frames_used, model_version,
                mock, batch_id, payload_json, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                exp_id, created, filename, ground_truth, generator,
                payload.get("prediction"), payload.get("confidence"),
                payload.get("synthetic_probability"), payload.get("real_probability"),
                payload.get("threshold"), payload.get("processing_time"),
                payload.get("num_frames_used"), model_version,
                1 if payload.get("mock") else 0, batch_id,
                json.dumps(payload), json.dumps(metadata or {}),
            ),
        )
    return exp_id


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["mock"] = bool(d.get("mock"))
    d["payload"] = json.loads(d.pop("payload_json") or "{}")
    d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
    return d


def get_experiment(exp_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        return _row_to_dict(row) if row else None


def list_experiments(limit: int | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM experiments ORDER BY created_at DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with _connect() as conn:
        return [_row_to_dict(r) for r in conn.execute(q).fetchall()]


def list_for_evaluation() -> list[dict[str, Any]]:
    """Flat rows with just the fields evaluation_service needs."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ground_truth, generator, prediction,
                      synthetic_probability, real_probability, confidence
               FROM experiments"""
        ).fetchall()
        return [dict(r) for r in rows]


def summary_stats() -> dict[str, Any]:
    """Dashboard summary cards."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ground_truth, prediction, confidence FROM experiments"
        ).fetchall()

    total = len(rows)
    real = sum(1 for r in rows if r["ground_truth"] == "Real")
    synth = sum(1 for r in rows if r["ground_truth"] == "Synthetic")
    labeled = [r for r in rows if r["ground_truth"] in ("Real", "Synthetic")]
    correct = sum(1 for r in labeled if r["ground_truth"] == r["prediction"])
    incorrect = len(labeled) - correct
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None

    return {
        "total_tested": total,
        "real_videos": real,
        "synthetic_videos": synth,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "average_confidence": avg_conf,
    }


def delete_experiment(exp_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        return cur.rowcount > 0
