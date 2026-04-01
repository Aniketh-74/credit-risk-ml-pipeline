"""Shared fixtures for training tests."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def synthetic_credit_df() -> pd.DataFrame:
    """1000-row synthetic Give Me Credit dataframe with 7% default rate.

    Replicates the real dataset's structure:
    - 10 feature columns matching FEATURE_COLS
    - 1 target column (SeriousDlqin2yrs)
    - MonthlyIncome has 19.8% NaN
    - NumberOfDependents has 2.6% NaN
    - NumberOfTimes90DaysLate has a handful of values at 96/98 (outlier sentinels)
    """
    rng = np.random.default_rng(42)
    n = 1000
    default_mask = rng.random(n) < 0.07  # 7% default rate

    df = pd.DataFrame({
        "SeriousDlqin2yrs": default_mask.astype(int),
        "RevolvingUtilizationOfUnsecuredLines": rng.beta(1.5, 5, n),
        "age": rng.integers(20, 85, n).astype(float),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 6, n).astype(float),
        "DebtRatio": rng.beta(2, 5, n),
        "MonthlyIncome": rng.lognormal(8.5, 0.8, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n).astype(float),
        "NumberOfTimes90DaysLate": rng.integers(0, 5, n).astype(float),
        "NumberRealEstateLoansOrLines": rng.integers(0, 4, n).astype(float),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 4, n).astype(float),
        "NumberOfDependents": rng.integers(0, 6, n).astype(float),
    })
    # Inject NaNs
    nan_idx_income = rng.choice(n, size=int(0.198 * n), replace=False)
    df.loc[nan_idx_income, "MonthlyIncome"] = np.nan
    nan_idx_dep = rng.choice(n, size=int(0.026 * n), replace=False)
    df.loc[nan_idx_dep, "NumberOfDependents"] = np.nan
    # Inject outlier sentinels in 90DaysLate
    outlier_idx = rng.choice(n, size=5, replace=False)
    df.loc[outlier_idx, "NumberOfTimes90DaysLate"] = 96.0
    return df
