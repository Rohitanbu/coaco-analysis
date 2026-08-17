"""SQLite persistence for analysis runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "analysis.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    thermal_filename TEXT,
    acoustic_filename TEXT,
    audio_path TEXT,
    features_json TEXT,
    probabilities_json TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def save_run(
    *,
    run_id: str,
    predicted_label: str,
    confidence: float,
    thermal_filename: str | None,
    acoustic_filename: str | None,
    audio_path: str | None,
    features: dict,
    probabilities: list[dict],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                run_id, created_at, predicted_label, confidence,
                thermal_filename, acoustic_filename, audio_path,
                features_json, probabilities_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                predicted_label,
                confidence,
                thermal_filename,
                acoustic_filename,
                audio_path,
                json.dumps(features),
                json.dumps(probabilities),
            ),
        )
        conn.commit()
