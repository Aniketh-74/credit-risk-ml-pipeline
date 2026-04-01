"""CB-PDD smoke test — trains champion model locally and validates tau sensitivity.

Runs in isolation with a local SQLite MLflow backend (no docker-compose required).

Usage:
    python scripts/smoke_test_cbpdd.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.training.train import run_training_pipeline
from src.training.promote import register_and_promote, get_champion_auc, load_champion
from src.training.data import load_and_preprocess, build_train_test


def generate_denial_loop_scores(
    model, X_test: pd.DataFrame, n_days: int = 30, n_per_day: int = 100
) -> list[dict]:
    """Simulate denial loop drift over n_days.

    Day 0 uses the full test set as the applicant pool. Each day, a sample is
    scored; denied applicants (score > 0.5) are not returned to the pool, while
    approved applicants re-enter. New low-risk applicants are injected daily
    (RevolvingUtilizationOfUnsecuredLines *= 0.97) to model the gradual downward
    drift in the score distribution that CB-PDD should detect.

    Args:
        model: MLflow pyfunc model with predict() method returning probabilities.
        X_test: Feature DataFrame used as the initial applicant pool.
        n_days: Number of simulation days. Defaults to 30.
        n_per_day: Applicants scored per day. Defaults to 100.

    Returns:
        List of dicts with keys: simulation_day, score, decision.
    """
    rng = np.random.default_rng(42)
    records = []

    pool = X_test.copy().reset_index(drop=True)

    for day in range(n_days):
        n_sample = min(n_per_day, len(pool))
        sample_idx = rng.choice(len(pool), size=n_sample, replace=False)
        sample = pool.iloc[sample_idx]

        # pyfunc model returns a numpy array of probabilities
        raw = model.predict(sample)
        # autolog wraps LGBMClassifier — output is (n, 2) probability array
        if hasattr(raw, "ndim") and raw.ndim == 2:
            scores = raw[:, 1]
        else:
            scores = np.asarray(raw)

        decisions = ["deny" if s > 0.5 else "approve" for s in scores]

        for score, decision in zip(scores, decisions):
            records.append(
                {"simulation_day": day, "score": float(score), "decision": decision}
            )

        # Denial loop: keep approved applicants, inject new low-risk ones
        keep_mask = np.array(decisions) == "approve"
        kept_idx = [sample_idx[i] for i, keep in enumerate(keep_mask) if keep]
        remaining = pool.drop(index=pool.index[sample_idx]).reset_index(drop=True)
        kept = pool.iloc[kept_idx].copy()

        n_inject = min(n_per_day // 2, len(pool))
        new_applicants = pool.sample(n=n_inject, random_state=day).copy()
        if "RevolvingUtilizationOfUnsecuredLines" in new_applicants.columns:
            new_applicants["RevolvingUtilizationOfUnsecuredLines"] *= 0.97

        pool = pd.concat([remaining, kept, new_applicants], ignore_index=True)

        if len(pool) < n_per_day:
            pool = X_test.copy().reset_index(drop=True)

    return records


def simple_cbpdd_smoke(records: list[dict], tau: int, alpha: float = 0.05) -> dict:
    """Minimal CB-PDD simulation for Phase 2 validation — count trials and check for score shift.

    Uses Mann-Whitney U test to detect whether the daily score distribution has shifted
    relative to the first-7-day reference window. Accumulates trials until tau is reached,
    then tests. This is a simplified version; the full CB-PDD implementation is Phase 4.

    Args:
        records: List of prediction dicts with keys: simulation_day, score, decision.
        tau: Number of accumulated trials before running the statistical test.
        alpha: Significance level for drift detection. Defaults to 0.05.

    Returns:
        Dict with keys: tau (int), detections (int), detection_days (list[int]).
    """
    from scipy import stats

    days = sorted(set(r["simulation_day"] for r in records))
    if len(days) < 14:
        return {"tau": tau, "detections": 0, "detection_days": []}

    ref_scores = [r["score"] for r in records if r["simulation_day"] < 7]
    detections = 0
    detection_days = []
    trial_count = 0

    for day in days[7:]:
        day_scores = [r["score"] for r in records if r["simulation_day"] == day]
        trial_count += len(day_scores)
        if trial_count >= tau:
            _, p_value = stats.mannwhitneyu(ref_scores, day_scores, alternative="two-sided")
            if p_value < alpha:
                detections += 1
                detection_days.append(day)
            trial_count = 0

    return {"tau": tau, "detections": detections, "detection_days": detection_days}


def main() -> None:
    """Train, promote, and validate CB-PDD tau sensitivity on a 30-day denial loop."""
    csv_path = "data/raw/cs-training.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow_uri = f"sqlite:///{tmpdir}/smoke_mlflow.db"
        mlflow.set_tracking_uri(mlflow_uri)

        print("Training champion model on real data (this takes ~2-3 min)...")
        run_id = run_training_pipeline(csv_path)
        print(f"Training complete. Run ID: {run_id}")

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        auc_val = run.data.metrics.get("auc_test", 0.0)
        print(f"AUC on test set: {auc_val:.4f}")

        if auc_val < 0.85:
            print(f"WARNING: AUC {auc_val:.4f} < 0.85 target. Check training pipeline.")
        else:
            print(f"AUC target met: {auc_val:.4f} >= 0.85")

        version = register_and_promote(run_id)
        print(f"Champion registered: version={version}")

        champion = load_champion()
        print("Champion model loaded OK via @champion alias")

        print("\nRunning CB-PDD smoke test (30-day denial loop simulation)...")
        X, y = load_and_preprocess(csv_path)
        # SMOTE only affects X_train — X_test is the natural 7% imbalanced split
        X_train, X_test, y_train, y_test, imputer = build_train_test(X, y)

        records = generate_denial_loop_scores(champion, X_test, n_days=30, n_per_day=100)

        print(f"\nCB-PDD Smoke Test Results (30-day denial loop, n_per_day=100):")
        all_detections = []
        for tau in [500, 1000, 2000]:
            result = simple_cbpdd_smoke(records, tau)
            first_day = result["detection_days"][0] if result["detection_days"] else "none"
            print(f"  tau={tau:4d}: detections={result['detections']}, first detection at day {first_day}")
            all_detections.append(result["detections"])

        if all(d == 0 for d in all_detections):
            print(
                "\nWARNING: All tau values produced 0 detections — score distribution may lack spread."
            )
            print("Recommendation: investigate SMOTE ratio and score calibration before Phase 4.")
            sys.exit(1)
        else:
            print("\nSmoke test PASSED: CB-PDD detected drift in at least one tau configuration.")


if __name__ == "__main__":
    main()
