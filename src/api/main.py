"""FastAPI scoring API for credit-risk-ml-pipeline.

Model, imputer, and Youden threshold are loaded once at startup via the
MLflow @champion alias and stored in app.state. Route handlers read from
app.state — never load artifacts per-request.
"""
from __future__ import annotations
import asyncio
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from sqlalchemy import text

from src.api.db import Session
from src.api.preprocess import FEATURE_COLS as PREPROCESS_FEATURE_COLS, preprocess_for_inference
from src.api.router import checkerboard_score, route_request
from src.api.schemas import HealthResponse, LoanApplicationRequest, OutcomeRequest, ScoreResponse
from db.models import Outcome, Prediction

logger = logging.getLogger(__name__)


async def write_prediction_with_retry(payload: dict, Session) -> None:
    """Write prediction to PostgreSQL after the response is returned.

    Retries once after 1 second on any exception. Second failure is logged
    as ERROR and discarded — the caller already received their 200 response.

    IMPORTANT: Creates a fresh Session from the factory — never receives a
    request-scoped session. Request-scoped sessions close when the HTTP
    response completes, before this coroutine runs. Passing a closed session
    causes DetachedInstanceError or PendingRollbackError.

    The payload 'decision' value must be 'approved' or 'denied' (past tense)
    to satisfy the predictions CHECK constraint:
        CHECK (decision IN ('approved', 'denied'))
    The response contract uses present tense ('approve'/'deny') — these are
    mapped explicitly in /score before calling add_task().
    """
    for attempt in range(2):
        try:
            with Session() as session:
                session.add(Prediction(**payload))
                session.commit()
            return
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "Prediction DB write failed (attempt 1), retrying in 1s: %s", exc
                )
                await asyncio.sleep(1)
            else:
                logger.error(
                    "Prediction DB write failed after retry, discarding: %s", exc
                )


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
async def score(
    request: LoanApplicationRequest,
    background_tasks: BackgroundTasks,
    req: Request,
) -> ScoreResponse:
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

    db_decision = "approved" if decision == "approve" else "denied"

    prediction_payload = {
        "model_version": state.model_version,
        "features": request.model_dump(by_alias=True),
        "score": raw_score,
        "decision": db_decision,
        "path": path,
        "simulation_day": None,
    }
    background_tasks.add_task(write_prediction_with_retry, prediction_payload, Session)

    return ScoreResponse(
        score=raw_score,
        decision=decision,
        path=path,
        model_version=state.model_version,
    )


@app.post("/outcome", status_code=201)
async def outcome(request: OutcomeRequest) -> dict:
    """Record the actual default outcome for a prior prediction.

    Writes synchronously — no BackgroundTask. Outcomes are low-volume and
    critical for CB-PDD label availability. A slow synchronous response is
    preferable to a silent discard.

    outcome_received_at is set to the current UTC timestamp — separate from
    predicted_at (when the prediction was made). This label delay is stored
    from day one for Phase 4 SIM-04.

    Returns:
        201 {"outcome_id": str} on success.
        404 if prediction_id does not exist in predictions table.
        422 if request body is missing required fields or types are wrong.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    with Session() as session:
        prediction = session.get(Prediction, request.prediction_id)
        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail=f"prediction_id {request.prediction_id!r} not found",
            )
        outcome_row = Outcome(
            prediction_id=request.prediction_id,
            actual_default=request.actual_default,
            predicted_at=prediction.predicted_at,
            outcome_received_at=now_utc,
        )
        session.add(outcome_row)
        session.commit()
        session.refresh(outcome_row)
        return {"outcome_id": outcome_row.id}
