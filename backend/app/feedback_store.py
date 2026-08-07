"""
Persists user feedback ("was this verdict correct?") to a local sqlite
file, backend/data/feedback/feedback.db -- the only live-request write path
in this app (contrast app/calibration.py, whose writes only ever happen
offline from scripts/evaluate.py). Unlike every other directory under
backend/data/, this one is NOT regenerable -- it's real feedback history,
not a rebuildable dataset -- see the .gitignore comment above backend/data/.

There is deliberately no "analyses" table to join against: the verdict
being rated is submitted by the client at vote time (it already has it
from the /analyze response), since /analyze itself stays side-effect-free.
analysis_id is the primary key, so a resubmitted vote for the same
analysis overwrites rather than double-counts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import FeedbackStatsOut, VerdictBreakdownOut

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback" / "feedback.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            analysis_id TEXT PRIMARY KEY,
            verdict TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def record_feedback(analysis_id: str, verdict: str, is_correct: bool) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO feedback (analysis_id, verdict, is_correct, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(analysis_id) DO UPDATE SET
                verdict=excluded.verdict,
                is_correct=excluded.is_correct,
                created_at=excluded.created_at
            """,
            (analysis_id, verdict, int(is_correct), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_stats() -> FeedbackStatsOut:
    conn = _connect()
    try:
        total, correct = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(is_correct), 0) FROM feedback"
        ).fetchone()

        by_verdict: dict[str, VerdictBreakdownOut] = {}
        rows = conn.execute(
            "SELECT verdict, COUNT(*), COALESCE(SUM(is_correct), 0) FROM feedback GROUP BY verdict"
        ).fetchall()
        for verdict, v_total, v_correct in rows:
            by_verdict[verdict] = VerdictBreakdownOut(
                total=v_total,
                correct_pct=(v_correct / v_total * 100) if v_total else None,
                incorrect_pct=((v_total - v_correct) / v_total * 100) if v_total else None,
            )

        return FeedbackStatsOut(
            total=total,
            correct_pct=(correct / total * 100) if total else None,
            incorrect_pct=((total - correct) / total * 100) if total else None,
            by_verdict=by_verdict,
        )
    finally:
        conn.close()
