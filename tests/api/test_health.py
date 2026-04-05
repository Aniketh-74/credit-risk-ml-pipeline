"""Tests for GET /health."""


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "5"


def test_health_503_when_model_not_loaded(client):
    from src.api.main import app
    original = app.state.model_loaded
    app.state.model_loaded = False
    try:
        resp = client.get("/health")
        assert resp.status_code == 503
    finally:
        app.state.model_loaded = original
