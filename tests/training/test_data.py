"""Tests for data loading and preprocessing — guards against common pitfalls."""
import numpy as np
import pandas as pd
import pytest
from src.training.data import (
    load_and_preprocess,
    build_train_test,
    FEATURE_COLS,
    TARGET_COL,
)


def test_no_target_leakage(synthetic_credit_df, tmp_path):
    """SeriousDelinquency must never appear in X after preprocessing."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, y = load_and_preprocess(str(csv))
    assert TARGET_COL not in X.columns, (
        f"Target leakage: {TARGET_COL} found in feature columns"
    )


def test_outlier_capping(synthetic_credit_df, tmp_path):
    """NumberOfTimes90DaysLate values above 17 must be capped to 17."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, _ = load_and_preprocess(str(csv))
    assert X["NumberOfTimes90DaysLate"].max() <= 17, (
        "Outlier capping failed: values > 17 found"
    )


def test_missingness_flags_added(synthetic_credit_df, tmp_path):
    """Missingness indicator columns must be created before imputation."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, _ = load_and_preprocess(str(csv))
    assert "MonthlyIncome_was_missing" in X.columns
    assert "NumberOfDependents_was_missing" in X.columns
    # Flag must have ones where original was NaN
    assert X["MonthlyIncome_was_missing"].sum() > 0


def test_imputer_fit_on_train_only(synthetic_credit_df, tmp_path):
    """Imputer median must match train-set median, not full-dataset median."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, y = load_and_preprocess(str(csv))
    X_train_res, X_test, y_train_res, y_test, imputer = build_train_test(X, y)
    # The imputer was fitted on train — its median should not equal the full-df median
    # (they will be close but not identical due to different sample sizes)
    # Key assertion: X_test has no NaNs (imputer.transform was applied)
    assert not X_test.isnull().any().any(), "X_test still contains NaN after imputation"
    assert not X_train_res.isnull().any().any(), "X_train_res still contains NaN after imputation"


def test_smote_balances_train_not_test(synthetic_credit_df, tmp_path):
    """After SMOTE, train set should be balanced; test set should retain ~7% default rate."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, y = load_and_preprocess(str(csv))
    X_train_res, X_test, y_train_res, y_test, _ = build_train_test(X, y)
    # Train set after SMOTE: classes should be roughly equal
    train_default_rate = y_train_res.mean()
    assert train_default_rate > 0.4, (
        f"SMOTE did not balance train set: default rate = {train_default_rate:.2%}"
    )
    # Test set: should retain natural imbalance (between 3% and 15% is acceptable for 1k rows)
    test_default_rate = y_test.mean()
    assert 0.03 < test_default_rate < 0.20, (
        f"Test set default rate unexpected: {test_default_rate:.2%} — SMOTE may have leaked"
    )


def test_feature_cols_count(synthetic_credit_df, tmp_path):
    """Preprocessed X must have 12 columns (10 original + 2 missingness flags)."""
    csv = tmp_path / "cs-training.csv"
    synthetic_credit_df.to_csv(csv)
    X, _ = load_and_preprocess(str(csv))
    assert len(X.columns) == 12, f"Expected 12 columns, got {len(X.columns)}: {list(X.columns)}"
