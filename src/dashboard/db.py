"""Dashboard data layer — PostgreSQL queries and MLflow REST API calls.

All functions return list[dict] in the exact shape expected by their
corresponding component render() functions, so components need no changes
when switching from mock data to live data.

Functions degrade gracefully: if the database or MLflow server is
unreachable, they return an empty list and let the component display
its "no data" placeholder.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone

import httpx
from sqlalchemy import create_engine, text


def _sync_url(db_url: str) -> str:
    """Swap asyncpg driver for psycopg2 — SQLAlchemy sync engine requirement."""
    return db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


def _engine(db_url: str):
    return create_engine(_sync_url(db_url))


# ---------------------------------------------------------------------------
# drift_scores table
# ---------------------------------------------------------------------------

def get_drift_scores(db_url: str, window_days: int = 30) -> list[dict]:
    """Return the last `window_days` drift score rows, ordered by computed_at.

    Shape matches drift_chart.render() and drift_comparison.render():
        day (int)               — sequential index (1-based)
        score (float)           — CB-PDD drift score (p-value)
        psi_score (float)       — PSI score
        threshold_crossed (bool)— True when score crossed the alert threshold
        computed_at (str)       — ISO datetime string
    """
    try:
        engine = _engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT drift_score, psi_score, threshold_crossed, computed_at
                    FROM drift_scores
                    ORDER BY computed_at DESC
                    LIMIT :limit
                """),
                {"limit": window_days},
            ).fetchall()
        engine.dispose()

        # Reverse so oldest-first for chart rendering
        rows = list(reversed(rows))
        return [
            {
                "day": i + 1,
                "score": float(r.drift_score),
                "psi_score": float(r.psi_score) if r.psi_score is not None else 0.0,
                "threshold_crossed": bool(r.threshold_crossed),
                "computed_at": str(r.computed_at),
            }
            for i, r in enumerate(rows)
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# alerts table
# ---------------------------------------------------------------------------

def get_alerts(db_url: str, limit: int = 50) -> list[dict]:
    """Return the most recent drift alerts.

    Shape matches alert_log.render():
        fired_at (str)      — ISO datetime string
        drift_score (float) — Score that crossed the threshold
        retrain (bool)      — True if a retraining run was triggered
        promoted (bool)     — True if a new model was promoted
    """
    try:
        engine = _engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT fired_at, drift_score, retrain_run_id, promoted
                    FROM alerts
                    ORDER BY fired_at DESC
                    LIMIT :limit
                """),
                {"limit": limit},
            ).fetchall()
        engine.dispose()

        return [
            {
                "fired_at": str(r.fired_at),
                "drift_score": float(r.drift_score),
                "retrain": r.retrain_run_id is not None,
                "promoted": bool(r.promoted) if r.promoted is not None else False,
            }
            for r in rows
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# MLflow REST API — model version history
# ---------------------------------------------------------------------------

def get_model_history(
    mlflow_uri: str,
    model_name: str = "credit-risk-model",
) -> list[dict]:
    """Return all registered versions of the credit-risk model, newest first.

    Uses MLflow REST API directly (no MLflow Python SDK) to avoid import
    weight and env dependencies in the dashboard container.

    Shape matches model_history.render():
        version (str)       — MLflow version number string
        auc (float)         — auc_test metric logged by the training run
        promoted_at (str)   — ISO date string of registration
        status (str)        — 'champion', 'challenger', or 'retired'
    """
    try:
        url = f"{mlflow_uri}/api/2.0/mlflow/model-versions/search"
        params = {"filter": f"name='{model_name}'", "max_results": 50}
        resp = httpx.get(url, params=params, timeout=5)
        resp.raise_for_status()
        versions = resp.json().get("model_versions", [])

        # Find which version has the @champion alias
        champion_version = _get_champion_version(mlflow_uri, model_name)

        result = []
        for v in sorted(versions, key=lambda x: int(x["version"]), reverse=True):
            ver_str = v["version"]
            run_id = v.get("run_id", "")
            created_ts = v.get("creation_timestamp", 0)
            promoted_at = datetime.fromtimestamp(
                created_ts / 1000, tz=timezone.utc
            ).date().isoformat() if created_ts else "unknown"

            auc = _get_run_auc(mlflow_uri, run_id)

            if ver_str == champion_version:
                status = "champion"
            else:
                status = "retired"

            result.append({
                "version": f"v{ver_str}",
                "auc": auc,
                "promoted_at": promoted_at,
                "status": status,
            })

        return result
    except Exception:
        return []


def _get_champion_version(mlflow_uri: str, model_name: str) -> str | None:
    try:
        url = f"{mlflow_uri}/api/2.0/mlflow/registered-models/alias"
        resp = httpx.get(url, params={"name": model_name, "alias": "champion"}, timeout=5)
        resp.raise_for_status()
        return resp.json().get("model_version", {}).get("version")
    except Exception:
        return None


def _get_run_auc(mlflow_uri: str, run_id: str) -> float:
    if not run_id:
        return 0.0
    try:
        url = f"{mlflow_uri}/api/2.0/mlflow/runs/get"
        resp = httpx.get(url, params={"run_id": run_id}, timeout=5)
        resp.raise_for_status()
        metrics = resp.json().get("run", {}).get("data", {}).get("metrics", [])
        for m in metrics:
            if m["key"] == "auc_test":
                return float(m["value"])
        return 0.0
    except Exception:
        return 0.0
