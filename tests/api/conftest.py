"""Shared fixtures for API integration tests.

Populates app.state with mocked model, imputer, and threshold so tests
run without a live MLflow server or PostgreSQL instance. Session is patched
at the module level so all route handlers and BackgroundTasks see the mock.
"""
import os
os.environ.setdefault("APP_DB_URL", "postgresql://dummy:dummy@localhost/dummy")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app


@pytest.fixture()
def client():
    """TestClient with synthetic app.state and no-op Session mock."""
    mock_imputer = MagicMock()
    mock_imputer.transform.return_value = np.zeros((1, 12))

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.05])  # below threshold 0.10 -> approve

    app.state.model = mock_model
    app.state.imputer = mock_imputer
    app.state.threshold = 0.10
    app.state.model_version = "5"
    app.state.model_loaded = True

    with patch("src.api.main.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session_cls.return_value = mock_session

        # Bypass the lifespan (which connects to MLflow/PostgreSQL) by
        # replacing it with a no-op. app.state is already populated above.
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        app.router.lifespan_context = _noop_lifespan

        with TestClient(app, raise_server_exceptions=True) as c:
            c._mock_session = mock_session
            yield c


@pytest.fixture()
def valid_payload():
    """Minimal valid LoanApplicationRequest payload with optional fields null."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": None,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": None,
    }
