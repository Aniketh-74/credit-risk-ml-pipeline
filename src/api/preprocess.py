"""Inference preprocessing for the scoring API.

Replicates the training-time preprocessing from src/training/data.py so that
the fitted SimpleImputer receives a 12-column array matching the shape it was
fitted on. Column order must be identical to data.py's load_and_preprocess().

The 12 columns are:
  [RevolvingUtilizationOfUnsecuredLines, age,
   NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome,
   NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
   NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse,
   NumberOfDependents, MonthlyIncome_was_missing,
   NumberOfDependents_was_missing]

Omitting the two missingness flags raises:
    ValueError: X has 10 features, but SimpleImputer is expecting 12 features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Must match data.py FEATURE_COLS exactly — column order determines imputer column alignment
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
OUTLIER_CAP = 17
OUTLIER_COLS = [
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
]


def preprocess_for_inference(features: dict, imputer) -> np.ndarray:
    """Convert raw API feature dict to imputed 12-column array for model.predict().

    Steps match data.py load_and_preprocess() order exactly:
      1. Build DataFrame in FEATURE_COLS order (None -> NaN for Optional fields)
      2. Add missingness flags BEFORE imputation (imputer fitted on 12 columns)
      3. Apply outlier clipping (cap late-payment columns at 17)
      4. Call imputer.transform() -> returns (1, 12) float64 array

    Args:
        features: Dict with the 10 raw feature keys from LoanApplicationRequest.
            MonthlyIncome and NumberOfDependents may be None.
        imputer: Fitted SimpleImputer from app.state.imputer.

    Returns:
        (1, 12) numpy float64 array ready for model.predict().
    """
    df = pd.DataFrame([features], columns=FEATURE_COLS)

    # Missingness flags before imputation — imputer was fitted on 12 columns including these
    df["MonthlyIncome_was_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_was_missing"] = df["NumberOfDependents"].isna().astype(int)

    # Outlier clipping — matches training: values above 17 are data entry errors
    for col in OUTLIER_COLS:
        df[col] = df[col].clip(upper=OUTLIER_CAP)

    # Impute: None/NaN -> column median (fitted medians from training set only)
    return imputer.transform(df)
