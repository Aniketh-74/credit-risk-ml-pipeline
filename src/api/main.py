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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from sqlalchemy import text

from src.api.db import Session
from src.api.schemas import HealthResponse

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
