"""FastAPI scoring API for credit-risk-ml-pipeline.

Model, imputer, and Youden threshold are loaded once at startup via the
MLflow @champion alias and stored in app.state. Route handlers read from
app.state — never load artifacts per-request.
"""
from __future__ import annotations
import logging
import pickle
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from sqlalchemy import text

from src.api.db import Session
from src.api.preprocess import FEATURE_COLS as PREPROCESS_FEATURE_COLS, preprocess_for_inference
from src.api.router import checkerboard_score, route_request
from src.api.schemas import HealthResponse, LoanApplicationRequest, ScoreResponse

logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-lgbm"
CHAMPION_ALIAS = "champion"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load @champion model, imputer, and Youden threshold at startup."""
    logger.info("Loading @champion model from MLflow...")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    run = client.get_run(mv.run_id)

    app.state.model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )

    local_imputer_path = client.download_artifacts(mv.run_id, "imputer/imputer.pkl")
    with open(local_imputer_path, "rb") as f:
        app.state.imputer = pickle.load(f)

    app.state.threshold = run.data.metrics["threshold_youdens_j"]
    app.state.model_version = str(mv.version)
    app.state.model_loaded = True

    logger.info(
        "Startup complete: model_version=%s threshold=%.4f",
        app.state.model_version,
        app.state.threshold,
    )

    yield

    app.state.model_loaded = False
    logger.info("Shutdown: model unloaded")


app = FastAPI(title="credit-risk-ml-pipeline", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check — confirms model is loaded and DB is reachable.

    Returns 200 when both pass; 503 when either fails.
    Docker Compose healthcheck uses this endpoint.
    """
    if not getattr(request.app.state, "model_loaded", False):
        return JSONResponse(
            status_code=503,
            content={"status": "model not loaded", "model_version": ""},
        )
    try:
        with Session() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": f"db unreachable: {exc}", "model_version": ""},
        )
    return HealthResponse(
        status="ok",
        model_version=request.app.state.model_version,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: LoanApplicationRequest, req: Request) -> ScoreResponse:
    """Score a loan application.

    Routes to either the LightGBM model or the CheckerBoard stub based on
    CHECKERBOARD_MIX. Returns probability score, approve/deny decision (using
    Youden's J threshold from app.state, not hardcoded 0.5), routing path,
    and current @champion model version.

    Note: Decision written to DB uses past tense ('approved'/'denied') to
    satisfy the predictions CHECK constraint. The response uses present tense
    ('approve'/'deny') per the CONTEXT.md response contract. BackgroundTask
    DB logging is added in Plan 03-03.
    """
    state = req.app.state
    path = route_request()

    if path == "checkerboard":
        raw_score = checkerboard_score()
    else:
        features = {
            "RevolvingUtilizationOfUnsecuredLines": request.revolving_utilization,
            "age": request.age,
            "NumberOfTime30-59DaysPastDueNotWorse": request.past_due_30_59,
            "DebtRatio": request.debt_ratio,
            "MonthlyIncome": request.monthly_income,
            "NumberOfOpenCreditLinesAndLoans": request.open_credit_lines,
            "NumberOfTimes90DaysLate": request.times_90_days_late,
            "NumberRealEstateLoansOrLines": request.real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": request.past_due_60_89,
            "NumberOfDependents": request.dependents,
        }
        X = preprocess_for_inference(features, state.imputer)
        all_cols = PREPROCESS_FEATURE_COLS + [
            "MonthlyIncome_was_missing",
            "NumberOfDependents_was_missing",
        ]
        X_df = pd.DataFrame(X, columns=all_cols)
        proba = state.model.predict(X_df)
        raw_score = float(proba[0])

    # Youden's J threshold — deny when score exceeds threshold
    decision = "deny" if raw_score > state.threshold else "approve"

    return ScoreResponse(
        score=raw_score,
        decision=decision,
        path=path,
        model_version=state.model_version,
    )
