"""Score gaming simulator.

Applicants learn to game the model by incrementally adjusting features known
to affect credit scores: reduce DebtRatio, inflate MonthlyIncome, pay down
revolving credit. Unlike the denial loop, score gaming does NOT call the API
— it writes directly to PostgreSQL, using a passed-in score function (mocked
in tests, replaced with real model in Phase 5).

The gradual feature ramp creates a realistic density shift CB-PDD can detect.
Nudge magnitudes ensure detection fires before day 30 (validated against
Phase 2 smoke test parameters: tau=1000, n_per_day=1000).
"""
import os
import uuid
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from db.models import Prediction, Outcome

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/creditrisk")

# Per-day nudge magnitudes (Claude's Discretion — derived from feature scales)
# Goal: gradual ramp over 30 days, NOT instant convergence
SCORE_GAMING_NUDGES = {
    "DebtRatio": -0.01,                          # reduce 1% per day
    "MonthlyIncome": 60.0,                       # add $60/month per day
    "RevolvingUtilizationOfUnsecuredLines": -0.005,  # pay down 0.5% per day
}
NUDGE_NOISE_STD = {
    "DebtRatio": 0.005,
    "MonthlyIncome": 30.0,
    "RevolvingUtilizationOfUnsecuredLines": 0.002,
}

# Starting profile: borderline applicants (score ~0.6 — just above approval threshold)
GAMING_BASELINE = {
    "RevolvingUtilizationOfUnsecuredLines": 0.65,
    "age": 38,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.55,
    "MonthlyIncome": 4200.0,
    "NumberOfOpenCreditLinesAndLoans": 9,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def _make_gaming_applicant(base: dict, rng_seed: int | None = None) -> dict:
    """Create one score-gaming applicant from the baseline profile."""
    rnd = random.Random(rng_seed)
    applicant = dict(base)
    # Spread initial applicants around the baseline (realistic population)
    applicant["DebtRatio"] = max(0.0, min(2.0, base["DebtRatio"] + rnd.gauss(0, 0.08)))
    applicant["MonthlyIncome"] = max(1000.0, base["MonthlyIncome"] + rnd.gauss(0, 400))
    applicant["RevolvingUtilizationOfUnsecuredLines"] = max(
        0.0, min(2.0, base["RevolvingUtilizationOfUnsecuredLines"] + rnd.gauss(0, 0.05))
    )
    return applicant


def _apply_daily_nudge(applicant: dict) -> dict:
    """Apply one day's worth of feature gaming to an applicant."""
    nudged = dict(applicant)
    for feature, delta in SCORE_GAMING_NUDGES.items():
        noise = random.gauss(0, NUDGE_NOISE_STD[feature])
        nudged[feature] = applicant[feature] + delta + noise

    # Clamp to realistic bounds — gaming has limits
    nudged["DebtRatio"] = max(0.0, min(2.0, nudged["DebtRatio"]))
    nudged["MonthlyIncome"] = max(1000.0, min(30000.0, nudged["MonthlyIncome"]))
    nudged["RevolvingUtilizationOfUnsecuredLines"] = max(
        0.0, min(2.0, nudged["RevolvingUtilizationOfUnsecuredLines"])
    )
    return nudged


def run_score_gaming(
    n_days: int = 30,
    n_per_day: int = 1000,
    score_fn=None,
    approval_threshold: float = 0.5,
    db_url: str | None = None,
    start_day: int = 0,
) -> None:
    """Run the score gaming simulation.

    Each day:
      1. All applicants apply their daily feature nudges (game the model).
      2. A score function scores each applicant (default: simple heuristic).
      3. Prediction + outcome rows are written to PostgreSQL with simulation_day.
      4. Applicants who reached approval threshold are replaced with new baseline
         applicants for the next day.

    Args:
        n_days: Number of simulation days. Default 30.
        n_per_day: Applicants per day. Default 1000.
        score_fn: Callable(dict) -> float. If None, uses a linear heuristic.
            Used to inject mocked model in tests.
        approval_threshold: Score below this = approved. Default 0.5 (Youden J).
        db_url: SQLAlchemy database URL. Falls back to DATABASE_URL env var.
        start_day: Starting value for simulation_day. Default 0.
    """
    db_url = db_url or DATABASE_URL
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    if score_fn is None:
        def score_fn(features: dict) -> float:
            # Simple linear heuristic for standalone runs (not used in tests)
            return min(1.0, max(0.0, features["DebtRatio"] * 0.6 +
                                features["RevolvingUtilizationOfUnsecuredLines"] * 0.3 +
                                (1 - min(features["MonthlyIncome"] / 10000, 1.0)) * 0.1))

    # Initialize population: each applicant has a slightly different starting point
    current_pool = [
        _make_gaming_applicant(GAMING_BASELINE, rng_seed=i)
        for i in range(n_per_day)
    ]
    base_time = datetime.now(timezone.utc)

    for day in range(n_days):
        sim_day = start_day + day
        predicted_at = base_time + timedelta(days=day)
        label_delay = timedelta(days=random.randint(1, 7))
        outcome_received_at = predicted_at + label_delay

        prediction_rows = []
        outcome_rows = []
        next_pool = []

        for applicant in current_pool:
            score = float(score_fn(applicant))
            decision = "approved" if score <= approval_threshold else "denied"

            pred_id = str(uuid.uuid4())
            prediction_rows.append({
                "id": pred_id,
                "model_version": "score-gaming-sim",
                "features": applicant,
                "score": score,
                "decision": decision,
                "path": "model",
                "simulation_day": sim_day,
                "predicted_at": predicted_at.isoformat(),
            })
            outcome_rows.append({
                "id": str(uuid.uuid4()),
                "prediction_id": pred_id,
                "actual_default": 0 if decision == "approved" else (
                    1 if random.random() < 0.07 else 0
                ),
                "predicted_at": predicted_at.isoformat(),
                "outcome_received_at": outcome_received_at.isoformat(),
            })

            if decision == "denied":
                # Keep gaming: nudge features and try again tomorrow
                next_pool.append(_apply_daily_nudge(applicant))
            else:
                # Approved — replace with a fresh borderline applicant
                next_pool.append(_make_gaming_applicant(GAMING_BASELINE))

        # Bulk write
        with Session() as session:
            if prediction_rows:
                session.execute(insert(Prediction), prediction_rows)
            if outcome_rows:
                session.execute(insert(Outcome), outcome_rows)
            session.commit()

        approved_count = sum(1 for r in prediction_rows if r["decision"] == "approved")
        print(f"Day {sim_day}: scored {len(prediction_rows)}, approved {approved_count}")

        current_pool = next_pool


if __name__ == "__main__":
    run_score_gaming()
