"""LightGBM training pipeline with MLflow autolog and SMOTE."""
from __future__ import annotations

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import auc as sklearn_auc, precision_recall_curve

from src.training.data import load_and_preprocess, build_train_test
from src.training.evaluate import compute_auc, log_pr_curve_artifact


def run_training_pipeline(
    csv_path: str,
    run_name: str = "lgbm_smote_v1",
) -> str:
    """Run the full training pipeline: load → preprocess → SMOTE → train → evaluate → log.

    Calls mlflow.lightgbm.autolog() before opening the run context so LightGBM's
    native callback logging is active during fit(). The precision-recall curve is
    logged manually because autolog does not produce it for LightGBM classifiers.

    Args:
        csv_path: Path to cs-training.csv.
        run_name: MLflow run name. Defaults to "lgbm_smote_v1".

    Returns:
        MLflow run ID (32-char hex string) of the completed training run.
    """
    mlflow.set_experiment("credit-risk-champion")

    # autolog must be called BEFORE start_run so callbacks are registered
    mlflow.lightgbm.autolog(log_models=True, log_input_examples=False)

    X, y = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test, _imputer = build_train_test(X, y)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "is_unbalance": False,  # SMOTE already balances the training set
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run(run_name=run_name) as run:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        y_proba = model.predict_proba(X_test)[:, 1]

        auc_test = compute_auc(y_test.values, y_proba)
        mlflow.log_metric("auc_test", auc_test)

        # PR-AUC for the artifact title
        precision, recall, _ = precision_recall_curve(y_test.values, y_proba)
        pr_auc = sklearn_auc(recall, precision)
        log_pr_curve_artifact(y_test.values, y_proba, pr_auc)

        run_id = run.info.run_id

    return run_id
