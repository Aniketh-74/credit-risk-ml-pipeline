"""Tests for POST /score — model path, checkerboard path, decision logic."""
from unittest.mock import patch
import numpy as np


def test_score_model_path_approve(client, valid_payload):
    """Score below threshold returns decision=approve, path=model."""
    with patch("src.api.main.route_request", return_value="model"):
        resp = client.post("/score", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "approve"
    assert data["path"] == "model"
    assert data["model_version"] == "5"
    assert 0.0 <= data["score"] <= 1.0


def test_score_model_path_deny(client, valid_payload):
    """Score above threshold returns decision=deny."""
    from src.api.main import app
    app.state.model.predict.return_value = np.array([0.95])
    try:
        with patch("src.api.main.route_request", return_value="model"):
            resp = client.post("/score", json=valid_payload)
        assert resp.status_code == 200
        assert resp.json()["decision"] == "deny"
    finally:
        app.state.model.predict.return_value = np.array([0.05])


def test_score_checkerboard_path(client, valid_payload):
    """CheckerBoard path returns path=checkerboard."""
    with patch("src.api.main.route_request", return_value="checkerboard"):
        resp = client.post("/score", json=valid_payload)
    assert resp.status_code == 200
    assert resp.json()["path"] == "checkerboard"


def test_score_missing_required_field_returns_422(client, valid_payload):
    """Missing required field raises 422 with field-level errors."""
    del valid_payload["age"]
    resp = client.post("/score", json=valid_payload)
    assert resp.status_code == 422


def test_score_optional_fields_null(client, valid_payload):
    """MonthlyIncome and NumberOfDependents can be null."""
    valid_payload["MonthlyIncome"] = None
    valid_payload["NumberOfDependents"] = None
    with patch("src.api.main.route_request", return_value="model"):
        resp = client.post("/score", json=valid_payload)
    assert resp.status_code == 200
