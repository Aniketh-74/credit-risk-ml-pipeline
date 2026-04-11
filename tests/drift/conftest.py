"""Fixtures for drift scorer tests.

Uses a temp-file SQLite database so that compute_drift (which creates its
own engine internally) can connect to the same seeded data.
In-memory SQLite won't work here — each engine.connect() opens a separate
database, so the seeded rows would be invisible to compute_drift's engine.
"""
import json
import os
import tempfile
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import Base


def _iso(d: date) -> str:
    return d.isoformat() + "T00:00:00"


def _seed(session, rows: list[tuple[float, str, date]]) -> None:
    """Insert (score, decision, outcome_date) tuples as prediction+outcome pairs."""
    for score, decision, outcome_date in rows:
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
                "predicted_at": _iso(outcome_date),
                "model_version": "test-v1",
                "features": json.dumps({}),
                "score": score,
                "decision": decision,
                "path": "model",
                "sim_day": None,
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
                "predicted_at": _iso(outcome_date),
                "outcome_received_at": _iso(outcome_date),
            },
        )
    session.commit()


@pytest.fixture(scope="module")
def empty_db_url():
    """Temp-file SQLite with schema but no rows."""
    tmp = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{tmp}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield db_url
    os.unlink(tmp)


@pytest.fixture(scope="module")
def seeded_db_url():
    """Temp-file SQLite seeded with 60 labeled predictions spread over 3 days."""
    tmp = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{tmp}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    anchor = date(2025, 3, 30)  # max outcome_received_at in this dataset
    days = [anchor - timedelta(days=d) for d in (20, 10, 0)]

    # 20 rows per day: alternating low/high scores to populate both halves of PSI
    rows = []
    for d in days:
        for i in range(20):
            score = round(0.2 + (i % 9) * 0.08, 3)  # 0.20 -> 0.84, repeating
            decision = "denied" if score > 0.5 else "approved"
            rows.append((score, decision, d))

    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed(session, rows)

    engine.dispose()
    yield db_url
    os.unlink(tmp)
