"""Data loading and preprocessing for credit risk model training."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

TARGET_COL = "SeriousDlqin2yrs"
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
LATE_COLS = [
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
]
OUTLIER_CAP = 17


def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load Give Me Credit CSV and apply preprocessing.

    Applies outlier capping for late payment columns (clip at 17), adds missingness
    indicator columns for MonthlyIncome and NumberOfDependents, then returns features
    and target. Does NOT impute — imputation is fitted on train only inside
    build_train_test() to prevent data leakage.

    Args:
        csv_path: Path to cs-training.csv.

    Returns:
        Tuple of (X, y) where X is a feature DataFrame (12 columns: 10 original +
        2 missingness flags) and y is the target Series.

    Raises:
        AssertionError: If TARGET_COL is not in the CSV or ends up in feature columns.
    """
    df = pd.read_csv(csv_path, index_col=0)
    assert TARGET_COL in df.columns, f"Target column {TARGET_COL!r} not found in CSV"
    assert TARGET_COL not in FEATURE_COLS, "Target leakage: target column listed in FEATURE_COLS"

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    # Outlier capping — values like 96/98 are data entry errors, cap at 17
    for col in LATE_COLS:
        X[col] = X[col].clip(upper=OUTLIER_CAP)

    # Add missingness flags BEFORE imputation so model can use missingness as a signal
    X["MonthlyIncome_was_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["NumberOfDependents_was_missing"] = X["NumberOfDependents"].isna().astype(int)

    assert TARGET_COL not in X.columns, "Target leakage: target column found in features after processing"
    return X, y


def build_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SimpleImputer]:
    """Split data, fit imputer on train only, and apply SMOTE to training set.

    SMOTE is applied AFTER the train/test split to prevent test set contamination.
    The SimpleImputer is fitted exclusively on X_train to prevent data leakage from
    the test set's statistics influencing training.

    Args:
        X: Feature DataFrame (output of load_and_preprocess).
        y: Target Series.
        test_size: Fraction of data for test set. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (X_train_resampled, X_test, y_train_resampled, y_test, imputer) where:
        - X_train_resampled: SMOTE-balanced training features (no NaN)
        - X_test: Original test features with imputer applied (no NaN)
        - y_train_resampled: SMOTE-balanced training labels (~50% default rate)
        - y_test: Original test labels (retains natural ~7% imbalance)
        - imputer: Fitted SimpleImputer (for use in inference pipeline)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit imputer on train only, transform both splits — prevents test stats leaking into train
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # SMOTE after split — only resamples the training set
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_arr, y_train_arr = smote.fit_resample(X_train_imputed, y_train)
    X_train_resampled = pd.DataFrame(X_train_arr, columns=X_train.columns)
    y_train_resampled = pd.Series(y_train_arr, name=TARGET_COL)

    return X_train_resampled, X_test_imputed, y_train_resampled, y_test, imputer
