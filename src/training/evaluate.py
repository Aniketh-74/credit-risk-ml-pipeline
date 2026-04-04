"""Evaluation utilities for credit risk model."""
from __future__ import annotations

import os
import mlflow
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC score.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability scores for the positive class.

    Returns:
        ROC-AUC score as a float between 0 and 1.
    """
    return float(roc_auc_score(y_true, y_score))


def log_pr_curve_artifact(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc: float,
    output_path: str = "/tmp/pr_curve.png",
) -> None:
    """Compute precision-recall curve and log as MLflow artifact.

    Must be called inside an active mlflow.start_run() context. Saves the plot
    locally then logs it to MLflow under the "plots" artifact path.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability scores for the positive class.
        auc: Pre-computed PR-AUC value to display in the plot title.
        output_path: Local path to save the plot before logging.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, color="steelblue")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve (AUC={auc:.4f})", fontsize=14)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(output_path, artifact_path="plots")


def compute_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Youden's J optimal threshold: argmax(TPR - FPR).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities for the positive class.

    Returns:
        Threshold float that maximises sensitivity + specificity - 1.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    return float(thresholds[optimal_idx])


def log_imputer_artifact(imputer, artifact_subdir: str = "imputer") -> None:
    """Pickle the fitted SimpleImputer and log as MLflow artifact.

    Must be called inside an active mlflow.start_run() context.
    Artifact path: {artifact_subdir}/imputer.pkl
    (API loads it via client.download_artifacts(run_id, 'imputer/imputer.pkl'))

    Args:
        imputer: Fitted sklearn SimpleImputer instance.
        artifact_subdir: MLflow artifact sub-directory. Default 'imputer'.
    """
    import pickle
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imputer.pkl")
        with open(path, "wb") as f:
            pickle.dump(imputer, f)
        mlflow.log_artifact(path, artifact_path=artifact_subdir)
