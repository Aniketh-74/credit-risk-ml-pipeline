"""Tests for POST /outcome."""
from unittest.mock import MagicMock


def test_outcome_404_unknown_prediction(client):
    """Unknown prediction_id returns 404."""
    client._mock_session.get.return_value = None
    resp = client.post("/outcome", json={
        "prediction_id": "00000000-0000-0000-0000-000000000000",
        "actual_default": True,
    })
    assert resp.status_code == 404


def test_outcome_success(client):
    """Valid prediction_id returns 201 with outcome_id."""
    mock_prediction = MagicMock()
    mock_prediction.predicted_at = "2026-04-05T10:00:00+00:00"

    outcome_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    client._mock_session.get.return_value = mock_prediction
    client._mock_session.refresh.side_effect = (
        lambda obj: setattr(obj, "id", outcome_id)
    )

    resp = client.post("/outcome", json={
        "prediction_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "actual_default": False,
    })
    assert resp.status_code == 201
    assert "outcome_id" in resp.json()


def test_outcome_missing_body_returns_422(client):
    """Request missing prediction_id returns 422."""
    resp = client.post("/outcome", json={"actual_default": True})
    assert resp.status_code == 422
