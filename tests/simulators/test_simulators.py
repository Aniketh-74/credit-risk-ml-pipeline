"""Smoke tests for denial loop and score gaming simulators.

Uses in-memory SQLite (not PostgreSQL) to verify row structure, label delay
modeling, and simulation_day population — without a live API or database.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import Base


@pytest.fixture(scope="module")
def sqlite_engine():
    """In-memory SQLite engine for simulator tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(sqlite_engine):
    session = sessionmaker(bind=sqlite_engine)()
    yield session
    session.rollback()
    session.close()


def _mock_score_response(score: float = 0.75, decision: str = "denied") -> dict:
    """Build a /score API response dict."""
    return {
        "score": score,
        "decision": decision,
        "model_version": "test-mock",
        "path": "model",
    }


class TestDenialLoopSimulator:
    def test_writes_simulation_day_on_every_row(self, sqlite_engine):
        """Every prediction row must have simulation_day populated."""
        from src.simulators.denial_loop import run_denial_loop

        # Mock httpx.Client to return denied responses without hitting real API
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_score_response(score=0.8, decision="denied")
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        db_url = "sqlite:///:memory:"
        # Need to re-create schema in fresh engine
        from sqlalchemy import create_engine as ce
        eng = ce(db_url)
        Base.metadata.create_all(eng)

        with patch("src.simulators.denial_loop.httpx.Client", return_value=mock_client), \
             patch("src.simulators.denial_loop.create_engine", return_value=eng):
            run_denial_loop(n_days=2, n_per_day=10, db_url=db_url, start_day=5)

        Session = sessionmaker(bind=eng)
        with Session() as s:
            rows = s.execute(text("SELECT simulation_day FROM predictions")).fetchall()

        assert len(rows) > 0, "Should have written prediction rows"
        sim_days = {r[0] for r in rows}
        assert 5 in sim_days, "simulation_day=5 must appear (start_day=5)"
        assert 6 in sim_days, "simulation_day=6 must appear (day 2)"
        assert None not in sim_days, "simulation_day must not be NULL on any row"

    def test_outcome_received_at_after_predicted_at(self, sqlite_engine):
        """Label delay: outcome_received_at must be later than predicted_at."""
        from src.simulators.denial_loop import run_denial_loop

        mock_response = MagicMock()
        mock_response.json.return_value = _mock_score_response(score=0.8, decision="denied")
        mock_response.raise_for_status.return_value = None
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        from sqlalchemy import create_engine as ce
        eng = ce("sqlite:///:memory:")
        Base.metadata.create_all(eng)

        with patch("src.simulators.denial_loop.httpx.Client", return_value=mock_client), \
             patch("src.simulators.denial_loop.create_engine", return_value=eng):
            run_denial_loop(n_days=1, n_per_day=5, db_url="sqlite:///:memory:", start_day=0)

        Session = sessionmaker(bind=eng)
        with Session() as s:
            rows = s.execute(text(
                "SELECT p.predicted_at, o.outcome_received_at "
                "FROM predictions p JOIN outcomes o ON o.prediction_id = p.id"
            )).fetchall()

        assert len(rows) > 0
        for predicted_at, outcome_received_at in rows:
            assert outcome_received_at > predicted_at, (
                f"outcome_received_at ({outcome_received_at}) must be after "
                f"predicted_at ({predicted_at})"
            )


class TestScoreGamingSimulator:
    def test_writes_simulation_day_on_every_row(self):
        """Every prediction row written by score gaming must have simulation_day."""
        from src.simulators.score_gaming import run_score_gaming

        # Mocked score function: starts at 0.6, ramps toward 0.3 (threshold) over days
        call_count = [0]
        def mock_score_fn(features):
            score = max(0.3, 0.6 - call_count[0] * 0.0005)
            call_count[0] += 1
            return score

        from sqlalchemy import create_engine as ce
        eng = ce("sqlite:///:memory:")
        Base.metadata.create_all(eng)

        with patch("src.simulators.score_gaming.create_engine", return_value=eng):
            run_score_gaming(
                n_days=3,
                n_per_day=10,
                score_fn=mock_score_fn,
                db_url="sqlite:///:memory:",
                start_day=0,
            )

        Session = sessionmaker(bind=eng)
        with Session() as s:
            rows = s.execute(text("SELECT simulation_day FROM predictions")).fetchall()

        assert len(rows) > 0
        sim_days = {r[0] for r in rows}
        assert {0, 1, 2}.issubset(sim_days), "All 3 simulation days must appear"
        assert None not in sim_days, "simulation_day must not be NULL"

    def test_both_tables_populated(self):
        """Score gaming must write both predictions and outcomes rows."""
        from src.simulators.score_gaming import run_score_gaming

        from sqlalchemy import create_engine as ce
        eng = ce("sqlite:///:memory:")
        Base.metadata.create_all(eng)

        with patch("src.simulators.score_gaming.create_engine", return_value=eng):
            run_score_gaming(
                n_days=1,
                n_per_day=5,
                score_fn=lambda f: 0.7,
                db_url="sqlite:///:memory:",
                start_day=0,
            )

        Session = sessionmaker(bind=eng)
        with Session() as s:
            pred_count = s.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
            outcome_count = s.execute(text("SELECT COUNT(*) FROM outcomes")).scalar()

        assert pred_count == 5, f"Expected 5 prediction rows, got {pred_count}"
        assert outcome_count == 5, f"Expected 5 outcome rows, got {outcome_count}"
