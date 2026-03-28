"""FastAPI application entry point.

Phase 1 stub: health endpoint only. Scoring endpoints added in Phase 3.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Phase 3 will add model loading here via MLflow client.
    """
    # Phase 3: load @champion model here
    yield


app = FastAPI(title="credit-risk-ml-pipeline", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status dict. Phase 3 will add model_version field.
    """
    return {"status": "ok", "version": "stub"}
