"""Daily credit-risk drift-detect-retrain cycle.

Task order:
    feedback_simulate
        → batch_score
            → drift_check  (BranchPythonOperator)
                ├─ trigger_retrain → promote_if_improved
                └─ skip_retrain

Environment variables (set via docker-compose or Airflow Variables):
    APP_DB_URL          SQLAlchemy URL for the app database (psycopg2 driver)
    MLFLOW_TRACKING_URI MLflow server URL
    SCORE_API_URL       Base URL of the FastAPI scoring service
    DATA_CSV_PATH       Path to cs-training.csv for retraining
    DRIFT_WINDOW_DAYS   Rolling window for drift scorer (default 30)
"""
import os
from datetime import datetime, timedelta

from airflow.sdk import dag, task

_DB_URL = os.environ.get("APP_DB_URL", "")
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_API_URL = os.environ.get("SCORE_API_URL", "http://api:8000")
_CSV_PATH = os.environ.get("DATA_CSV_PATH", "/opt/airflow/data/cs-training.csv")
_DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW_DAYS", "30"))

_DEFAULT_ARGS = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="credit_risk_daily",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["credit-risk", "drift", "mlops"],
)
def credit_risk_daily():
    """Orchestrates the daily feedback-loop simulation and drift-triggered retraining."""

    @task()
    def feedback_simulate():
        """Run one day of denial-loop simulation via the scoring API.

        run_denial_loop reads SCORE_API_URL and DATABASE_URL from the environment.
        The Airflow scheduler service has both set via docker-compose.
        """
        from src.simulators.denial_loop import run_denial_loop
        run_denial_loop(
            db_url=_DB_URL,
            n_days=1,
            n_per_day=int(os.environ.get("SIM_N_PER_DAY", "500")),
        )

    @task()
    def batch_score():
        """Verify the scoring API is healthy and predictions were written.

        Calls GET /health and checks that the predictions table received rows
        from the just-completed simulation. Fails the task (and triggers retry)
        if the API is unreachable or no new predictions exist.
        """
        import httpx
        from sqlalchemy import create_engine, text

        resp = httpx.get(f"{_API_URL}/health", timeout=10)
        resp.raise_for_status()

        # Sync URL for SQLAlchemy: swap asyncpg driver for psycopg2
        sync_url = _DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE predicted_at::date = CURRENT_DATE")
            ).scalar()
        engine.dispose()

        if count == 0:
            raise RuntimeError("No predictions written today — simulation may have failed")

        return int(count)

    @task.branch()
    def drift_check():
        """Compute CB-PDD drift score; route to trigger_retrain or skip_retrain.

        Returns the task_id of the next task to run. Airflow's BranchPythonOperator
        skips all tasks not returned by this function.
        """
        from src.drift.scorer import compute_drift

        sync_url = _DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        result = compute_drift(db_url=sync_url, window_days=_DRIFT_WINDOW)

        print(
            f"drift_score={result['drift_score']:.4f}  "
            f"psi={result['psi_score']:.4f}  "
            f"is_drift={result['is_drift']}  "
            f"window_end={result['window_end']}"
        )

        return "trigger_retrain" if result["is_drift"] else "skip_retrain"

    @task()
    def trigger_retrain():
        """Run a fresh MLflow training run on recent labeled data.

        Returns the new run_id so promote_if_improved can look it up.
        """
        import mlflow
        from src.training.train import run_training_pipeline

        mlflow.set_tracking_uri(_MLFLOW_URI)
        run_id = run_training_pipeline(
            csv_path=_CSV_PATH,
            run_name="daily_retrain",
        )
        print(f"Retrain complete: run_id={run_id}")
        return run_id

    @task()
    def skip_retrain():
        """No-op branch — drift score below threshold, no retraining needed."""
        print("Drift below threshold — skipping retraining for today")

    @task()
    def promote_if_improved(run_id: str):
        """Promote the new model to @champion only if its AUC beats the current champion.

        Args:
            run_id: MLflow run ID returned by trigger_retrain (via XCom).
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.training.promote import get_champion_auc, register_and_promote

        mlflow.set_tracking_uri(_MLFLOW_URI)
        client = MlflowClient()

        champion_auc = get_champion_auc()
        run = client.get_run(run_id)
        new_auc = float(run.data.metrics.get("auc_test", 0.0))

        print(f"champion_auc={champion_auc:.4f}  new_auc={new_auc:.4f}")

        if new_auc > champion_auc:
            version = register_and_promote(run_id)
            print(f"Promoted to @champion: version={version}  auc_delta={new_auc - champion_auc:+.4f}")
        else:
            print(f"New model did not improve ({new_auc:.4f} <= {champion_auc:.4f}) — keeping current champion")

    # --- DAG wiring ---
    sim = feedback_simulate()
    scored = batch_score()
    branch = drift_check()
    retrain = trigger_retrain()
    skip = skip_retrain()
    promote = promote_if_improved(retrain)

    sim >> scored >> branch >> [retrain, skip]
    retrain >> promote


credit_risk_daily()
