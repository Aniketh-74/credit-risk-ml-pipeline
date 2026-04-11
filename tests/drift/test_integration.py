"""End-to-end integration smoke test for Phase 4.

Exercises the full pipeline:
  1. Insert synthetic denial loop rows into a temp-file SQLite DB
  2. Call compute_drift() — which creates its own engine internally
  3. Verify the output dict shape, types, and window_end

Uses a temp-file (not in-memory) SQLite so compute_drift's engine
connects to the same seeded data.
"""
import json
import os
import tempfile
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import Base
from src.drift.scorer import compute_drift


@pytest.fixture(scope="module")
def denial_loop_db_url():
    """Temp-file SQLite with 3000 denial loop rows across 3 days."""
    tmp = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{tmp}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n_days, n_per_day = 3, 1000

    Session = sessionmaker(bind=engine)
    with Session() as session:
        for day in range(n_days):
            predicted_at = (base + timedelta(days=day)).isoformat()
            outcome_at = (base + timedelta(days=day + 2)).isoformat()

            for i in range(n_per_day):
                score = max(0.1, min(0.99, 0.75 - day * 0.1 - (i / n_per_day) * 0.05))
                decision = "denied" if score > 0.5 else "approved"
                pred_id = str(uuid.uuid4())

                session.execute(
                    text("""
                        INSERT INTO predictions
                            (id, predicted_at, model_version, features, score, decision, path, simulation_day)
                        VALUES
                            (:id, :predicted_at, :model_version, :features, :score, :decision, :path, :sim_day)
                    """),
                    {
                        "id": pred_id,
                        "predicted_at": predicted_at,
                        "model_version": "integration-test",
                        "features": json.dumps({"DebtRatio": 0.35}),
                        "score": score,
                        "decision": decision,
                        "path": "model",
                        "sim_day": day,
                    },
                )
                session.execute(
                    text("""
                        INSERT INTO outcomes
                            (id, prediction_id, actual_default, predicted_at, outcome_received_at)
                        VALUES
                            (:id, :prediction_id, :actual_default, :predicted_at, :outcome_received_at)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "prediction_id": pred_id,
                        "actual_default": score > 0.6,
                        "predicted_at": predicted_at,
                        "outcome_received_at": outcome_at,
                    },
                )
        session.commit()

    engine.dispose()
    yield db_url
    os.unlink(tmp)


class TestFullPipelineIntegration:
    def test_returns_valid_dict(self, denial_loop_db_url):
        result = compute_drift(db_url=denial_loop_db_url, window_days=9999)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"drift_score", "psi_score", "is_drift", "window_end"}

    def test_correct_types(self, denial_loop_db_url):
        result = compute_drift(db_url=denial_loop_db_url, window_days=9999)
        assert isinstance(result["drift_score"], float)
        assert isinstance(result["psi_score"], float)
        assert isinstance(result["is_drift"], bool)
        assert isinstance(result["window_end"], date)

    def test_drift_score_is_p_value(self, denial_loop_db_url):
        result = compute_drift(db_url=denial_loop_db_url, window_days=9999)
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_psi_nonnegative(self, denial_loop_db_url):
        result = compute_drift(db_url=denial_loop_db_url, window_days=9999)
        assert result["psi_score"] >= 0.0

    def test_window_end_is_latest_outcome(self, denial_loop_db_url):
        # outcome_received_at = predicted_at + 2 days; last predicted_at = 2026-01-03
        result = compute_drift(db_url=denial_loop_db_url, window_days=9999)
        assert result["window_end"] == date(2026, 1, 5)

    def test_no_exception_across_window_sizes(self, denial_loop_db_url):
        for window in [1, 7, 30, 365, 9999]:
            result = compute_drift(db_url=denial_loop_db_url, window_days=window)
            assert isinstance(result, dict), f"window_days={window} raised or returned non-dict"
