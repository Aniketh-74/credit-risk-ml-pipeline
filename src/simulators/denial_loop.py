"""Denial loop simulator.

Calls the real /score API for each applicant each day. Denied applicants
re-enter the next day's pool with features nudged toward approval (lower
DebtRatio, higher MonthlyIncome). Writes both prediction and outcome rows
to PostgreSQL. Used by Phase 5 Airflow tasks.
"""
import os
import uuid
import random
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from db.models import Prediction, Outcome

SCORE_API_URL = os.getenv("SCORE_API_URL", "http://localhost:8001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/creditrisk")

# Baseline approved-profile applicant (non-defaulter, below threshold)
BASELINE_APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5500.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}

# Nudge magnitudes for denied applicants re-applying the next day
DENIAL_NUDGES = {
    "DebtRatio": -0.015,          # reduce debt ratio (0.5-2.5% per day with noise)
    "MonthlyIncome": 150.0,       # inflate income by ~2.7% of median
    "RevolvingUtilizationOfUnsecuredLines": -0.008,  # pay down credit cards
}
NUDGE_NOISE_STD = {
    "DebtRatio": 0.005,
    "MonthlyIncome": 50.0,
    "RevolvingUtilizationOfUnsecuredLines": 0.003,
}


def _random_applicant(base: dict | None = None) -> dict:
    """Generate one synthetic applicant, optionally varying from a base profile."""
    if base is None:
        base = BASELINE_APPLICANT.copy()
    applicant = dict(base)
    # Add realistic variation so each day's pool is not identical
    applicant["age"] = max(18, min(85, base["age"] + random.randint(-5, 5)))
    applicant["MonthlyIncome"] = max(1000.0, base["MonthlyIncome"] + random.gauss(0, 300))
    applicant["DebtRatio"] = max(0.0, min(2.0, base["DebtRatio"] + random.gauss(0, 0.05)))
    applicant["RevolvingUtilizationOfUnsecuredLines"] = max(
        0.0, min(2.0, base["RevolvingUtilizationOfUnsecuredLines"] + random.gauss(0, 0.03))
    )
    return applicant


def _nudge_denied(applicant: dict) -> dict:
    """Return a copy of applicant with features nudged toward approval."""
    nudged = dict(applicant)
    for feature, delta in DENIAL_NUDGES.items():
        noise = random.gauss(0, NUDGE_NOISE_STD[feature])
        nudged[feature] = applicant[feature] + delta + noise
    # Clamp to realistic bounds
    nudged["DebtRatio"] = max(0.0, min(2.0, nudged["DebtRatio"]))
    nudged["MonthlyIncome"] = max(1000.0, nudged["MonthlyIncome"])
    nudged["RevolvingUtilizationOfUnsecuredLines"] = max(
        0.0, min(2.0, nudged["RevolvingUtilizationOfUnsecuredLines"])
    )
    return nudged


def _score_applicant(client: httpx.Client, features: dict) -> dict:
    """POST features to /score endpoint, return response dict."""
    response = client.post(
        f"{SCORE_API_URL}/score",
        json=features,
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def run_denial_loop(
    n_days: int = 30,
    n_per_day: int = 1000,
    db_url: str | None = None,
    start_day: int = 0,
) -> None:
    """Run the denial loop simulation.

    Each day:
      1. Score n_per_day applicants via the /score API.
      2. Write prediction + outcome rows to PostgreSQL with simulation_day set.
      3. Denied applicants form the base of the next day's pool (nudged toward approval).

    Args:
        n_days: Number of simulation days to run. Default 30.
        n_per_day: Applicants per day. Default 1000.
        db_url: SQLAlchemy database URL. Falls back to DATABASE_URL env var.
        start_day: Starting value for simulation_day column. Default 0.
    """
    db_url = db_url or DATABASE_URL
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    # Start with random applicants for day 0
    current_pool = [_random_applicant() for _ in range(n_per_day)]
    base_time = datetime.now(timezone.utc)

    with httpx.Client() as client:
        for day in range(n_days):
            sim_day = start_day + day
            predicted_at = base_time + timedelta(days=day)
            # outcome_received_at = 1-7 days after prediction (label delay)
            label_delay = timedelta(days=random.randint(1, 7))
            outcome_received_at = predicted_at + label_delay

            prediction_rows = []
            outcome_rows = []
            denied_applicants = []

            for applicant in current_pool:
                try:
                    result = _score_applicant(client, applicant)
                except httpx.HTTPError:
                    # Skip failed requests; do not crash the full day
                    continue

                pred_id = str(uuid.uuid4())
                score = result.get("score", 0.5)
                decision = result.get("decision", "denied")
                model_version = result.get("model_version", "unknown")

                prediction_rows.append({
                    "id": pred_id,
                    "model_version": model_version,
                    "features": applicant,
                    "score": score,
                    "decision": decision,
                    "path": result.get("path", "model"),
                    "simulation_day": sim_day,
                    "predicted_at": predicted_at.isoformat(),
                })
                outcome_rows.append({
                    "id": str(uuid.uuid4()),
                    "prediction_id": pred_id,
                    # Simulate ground truth: denied applicants are less likely to default
                    # (they were correctly screened); approved have ~7% default rate
                    "actual_default": 0 if decision == "denied" else (1 if random.random() < 0.07 else 0),
                    "predicted_at": predicted_at.isoformat(),
                    "outcome_received_at": outcome_received_at.isoformat(),
                })

                if decision == "denied":
                    denied_applicants.append(applicant)

            # Bulk write this day's rows
            with Session() as session:
                if prediction_rows:
                    session.execute(insert(Prediction), prediction_rows)
                if outcome_rows:
                    session.execute(insert(Outcome), outcome_rows)
                session.commit()

            print(f"Day {sim_day}: scored {len(prediction_rows)}, "
                  f"denied {len(denied_applicants)}")

            # Build next day's pool: nudged denied applicants + fresh applicants
            nudged = [_nudge_denied(a) for a in denied_applicants]
            fresh_count = n_per_day - len(nudged)
            fresh = [_random_applicant() for _ in range(max(0, fresh_count))]
            current_pool = nudged + fresh
            # Ensure we always have exactly n_per_day applicants
            if len(current_pool) > n_per_day:
                current_pool = current_pool[:n_per_day]


if __name__ == "__main__":
    run_denial_loop()
