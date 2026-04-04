"""Pydantic request and response schemas for the scoring API."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class LoanApplicationRequest(BaseModel):
    """Raw Give Me Credit feature payload.

    MonthlyIncome and NumberOfDependents are Optional — these were 19.8% and
    2.6% NaN in training. Callers pass null; the API imputes internally via
    the fitted SimpleImputer. All other 8 fields are required.
    """
    model_config = ConfigDict(populate_by_name=True)

    revolving_utilization: float = Field(alias="RevolvingUtilizationOfUnsecuredLines")
    age: int = Field(alias="age")
    past_due_30_59: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse")
    debt_ratio: float = Field(alias="DebtRatio")
    monthly_income: Optional[float] = Field(default=None, alias="MonthlyIncome")
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans")
    times_90_days_late: int = Field(alias="NumberOfTimes90DaysLate")
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines")
    past_due_60_89: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse")
    dependents: Optional[int] = Field(default=None, alias="NumberOfDependents")


class ScoreResponse(BaseModel):
    """Response from POST /score."""
    score: float          # probability of default in [0, 1]
    decision: str         # "approve" or "deny" (present tense — response contract)
    path: str             # "model" or "checkerboard"
    model_version: str    # @champion version string


class OutcomeRequest(BaseModel):
    """Request body for POST /outcome."""
    prediction_id: str
    actual_default: bool


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str
    model_version: str
