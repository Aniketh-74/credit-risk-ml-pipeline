"""Integration tests for training pipeline — uses synthetic data, no real dataset."""
import numpy as np
import pandas as pd
import pytest
import mlflow
from unittest.mock import patch
from src.training.data import load_and_preprocess, build_train_test
from src.training.train import run_training_pipeline
from src.training.evaluate import compute_auc, log_pr_curve_artifact


def test_training_run_completes(synthetic_credit_df, tmp_path, monkeypatch):
    """Training pipeline runs to completion on synthetic data and returns a run_id."""
    # Use SQLite backend — file:// URIs are not valid on Windows with backslash paths
    db_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)

    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)

    run_id = run_training_pipeline(csv_path=str(csv))
    assert run_id is not None, "run_training_pipeline must return an MLflow run_id"
    assert isinstance(run_id, str) and len(run_id) == 32, (
        f"run_id should be 32-char hex string, got: {run_id!r}"
    )


def test_mlflow_run_logs_auc(synthetic_credit_df, tmp_path, monkeypatch):
    """MLflow run must log auc_test metric."""
    db_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)

    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)

    run_id = run_training_pipeline(csv_path=str(csv))
    client = mlflow.MlflowClient(tracking_uri=db_uri)
    run = client.get_run(run_id)
    assert "auc_test" in run.data.metrics, "auc_test metric not logged to MLflow run"
    auc = run.data.metrics["auc_test"]
    # On synthetic 1000-row data with random labels, AUC > 0.55 confirms metric is logged
    # and model ran — the real 0.85 threshold applies only to the full 150k-row dataset
    assert auc > 0.55, f"AUC too low even for synthetic data: {auc:.4f}"


def test_mlflow_run_logs_pr_curve_artifact(synthetic_credit_df, tmp_path, monkeypatch):
    """MLflow run must include pr_curve.png in artifacts/plots/."""
    db_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)

    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)

    run_id = run_training_pipeline(csv_path=str(csv))
    client = mlflow.MlflowClient(tracking_uri=db_uri)
    artifacts = client.list_artifacts(run_id, "plots")
    artifact_names = [a.path for a in artifacts]
    assert any("pr_curve" in name for name in artifact_names), (
        f"pr_curve artifact not found. Artifacts: {artifact_names}"
    )


def test_compute_auc_returns_float():
    """compute_auc returns a float between 0 and 1."""
    rng = np.random.default_rng(42)
    y_true = (rng.random(200) < 0.07).astype(int)
    y_score = rng.random(200)
    auc = compute_auc(y_true, y_score)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
