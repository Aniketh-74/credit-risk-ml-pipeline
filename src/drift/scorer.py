"""Drift scorer: reads labeled predictions from PostgreSQL, returns CB-PDD + PSI.

Pure function -- reads data, computes metrics, returns a dict. No writes.
The caller (Phase 5 Airflow task) decides what to do with the result.

Interface:
    compute_drift(db_url: str, window_days: int = None) -> DriftResult

Return shape:
    {
        "drift_score": float,   # CB-PDD last p-value (lower = more drift)
        "psi_score":   float,   # Population Stability Index
        "is_drift":    bool,    # True if 2 consecutive windows exceeded alpha
        "window_end":  date,    # Most recent outcome_received_at in window
    }

Window anchor: max(outcome_received_at) in the table, not wall-clock NOW().
This ensures historical simulation data works without time manipulation.
"""
import os
from datetime import date, datetime
from typing import TypedDict

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.drift.cb_pdd import PerformativeDriftDetector

DRIFT_WINDOW_DAYS = int(os.getenv("DRIFT_WINDOW_DAYS", "30"))


class DriftResult(TypedDict):
    drift_score: float  # CB-PDD p-value — lower means more drift signal
    psi_score: float    # Population Stability Index
    is_drift: bool      # True once 2 consecutive windows exceeded alpha
    window_end: date    # Most recent labeled prediction date in window


def _fetch_labeled_predictions(session, window_days: int, dialect: str) -> list[dict]:
    """Fetch labeled prediction rows for the rolling window.

    Labeled = both predicted_at (on predictions) and outcome_received_at
    (on outcomes) are non-null. Window is anchored to max(outcome_received_at)
    in the table so historical simulation data works correctly.

    Args:
        session: SQLAlchemy session.
        window_days: Number of calendar days to look back.
        dialect: 'postgresql' or 'sqlite' -- controls date arithmetic syntax.

    Returns:
        List of dicts with keys: score, decision, simulation_day,
        actual_default, outcome_received_at. Ordered by outcome_received_at ASC.
    """
    if dialect == "sqlite":
        sql = text("""
            SELECT p.score,
                   p.decision,
                   p.simulation_day,
                   o.actual_default,
                   o.outcome_received_at
            FROM predictions p
            JOIN outcomes o ON o.prediction_id = p.id
            WHERE p.predicted_at IS NOT NULL
              AND o.outcome_received_at IS NOT NULL
              AND o.outcome_received_at >= (
                  SELECT datetime(MAX(o2.outcome_received_at), :neg_days)
                  FROM outcomes o2
                  WHERE o2.outcome_received_at IS NOT NULL
              )
            ORDER BY o.outcome_received_at ASC
        """)
        params = {"neg_days": f"-{window_days} days"}
    else:
        sql = text("""
            SELECT p.score,
                   p.decision,
                   p.simulation_day,
                   o.actual_default,
                   o.outcome_received_at
            FROM predictions p
            JOIN outcomes o ON o.prediction_id = p.id
            WHERE p.predicted_at IS NOT NULL
              AND o.outcome_received_at IS NOT NULL
              AND CAST(o.outcome_received_at AS TIMESTAMPTZ) >= (
                  SELECT MAX(CAST(o2.outcome_received_at AS TIMESTAMPTZ))
                        - INTERVAL '1 day' * :window_days
                  FROM outcomes o2
                  WHERE o2.outcome_received_at IS NOT NULL
              )
            ORDER BY o.outcome_received_at ASC
        """)
        params = {"window_days": window_days}

    result = session.execute(sql, params)
    return [dict(row._mapping) for row in result]


def _compute_psi(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two score distributions.

    PSI < 0.10  -- no significant shift
    PSI 0.10-0.25 -- moderate shift
    PSI > 0.25  -- significant shift

    Uses equal-width bins over [0, 1]. Zeros clipped to 1e-6 to prevent log(0).

    Args:
        reference_scores: Baseline score distribution (first half of window).
        current_scores: Current score distribution (second half of window).
        n_bins: Number of equal-width bins. Default 10.

    Returns:
        PSI scalar (>= 0). Returns 0.0 if either array is empty.
    """
    if len(reference_scores) == 0 or len(current_scores) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ref_hist, _ = np.histogram(reference_scores, bins=bins)
    cur_hist, _ = np.histogram(current_scores, bins=bins)

    ref_pct = ref_hist / (ref_hist.sum() + 1e-9)
    cur_pct = cur_hist / (cur_hist.sum() + 1e-9)

    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift(
    db_url: str,
    window_days: int | None = None,
) -> DriftResult:
    """Compute CB-PDD drift score and PSI over the rolling window.

    Reads labeled predictions from the last window_days days, feeds them
    through PerformativeDriftDetector for CB-PDD, and splits the stream
    in half for PSI (first half = reference, second half = current).

    No side effects. Returns safe defaults on empty window.

    Args:
        db_url: SQLAlchemy database URL.
        window_days: Days to look back. Defaults to DRIFT_WINDOW_DAYS (30).

    Returns:
        DriftResult dict.
    """
    if window_days is None:
        window_days = DRIFT_WINDOW_DAYS

    dialect = "sqlite" if db_url.startswith("sqlite") else "postgresql"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        rows = _fetch_labeled_predictions(session, window_days, dialect)

    engine.dispose()

    if not rows:
        return DriftResult(
            drift_score=1.0,
            psi_score=0.0,
            is_drift=False,
            window_end=date.today(),
        )

    scores = np.array([float(r["score"]) for r in rows])
    y_hats = [1 if r["decision"] == "denied" else 0 for r in rows]
    y_trues = [int(r["actual_default"]) for r in rows]

    raw_end = rows[-1]["outcome_received_at"]
    if isinstance(raw_end, str):
        window_end = datetime.fromisoformat(raw_end).date()
    elif isinstance(raw_end, datetime):
        window_end = raw_end.date()
    else:
        window_end = date.today()

    detector = PerformativeDriftDetector()
    for score, y_hat, y_true in zip(scores, y_hats, y_trues):
        detector.add(score, y_hat, y_true)

    mid = len(scores) // 2
    psi = _compute_psi(scores[:mid], scores[mid:])

    return DriftResult(
        drift_score=detector.last_p_value,
        psi_score=psi,
        is_drift=detector.is_drift,
        window_end=window_end,
    )
