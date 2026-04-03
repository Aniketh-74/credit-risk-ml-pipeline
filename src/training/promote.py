"""MLflow Model Registry promotion utilities for the credit-risk-model."""
from __future__ import annotations

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MODEL_NAME: str = "credit-risk-model"


def register_and_promote(run_id: str, model_artifact_path: str = "model") -> str:
    """Register a training run in the MLflow Model Registry and set the @champion alias.

    Uses the MLflow 3.x alias API exclusively. Never calls the deprecated
    transition_model_version_stage API.

    Args:
        run_id: MLflow run ID of the completed training run.
        model_artifact_path: Artifact path under which the model was logged. Defaults to "model".

    Returns:
        Version string of the newly registered model version.
    """
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version

    client = MlflowClient()
    client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=version)

    return version


def get_champion_auc() -> float:
    """Return the auc_test metric logged by the current @champion run.

    Looks up the @champion alias, retrieves the associated run, and returns the
    logged auc_test metric. Returns 0.0 if no champion is registered or if any
    registry error occurs.

    Returns:
        AUC on the held-out test set for the @champion model, or 0.0 on error.
    """
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
        return float(run.data.metrics.get("auc_test", 0.0))
    except Exception:
        return 0.0


def load_champion() -> object:
    """Load the @champion model for inference.

    Returns:
        MLflow pyfunc model backed by the registered @champion version.
    """
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
