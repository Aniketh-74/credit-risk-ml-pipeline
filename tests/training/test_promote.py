"""Unit tests for src/training/promote.py — all MlflowClient calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.training.promote import MODEL_NAME, get_champion_auc, register_and_promote


def test_champion_alias_set():
    """register_and_promote must call set_registered_model_alias with alias='champion'."""
    mock_result = MagicMock()
    mock_result.version = "1"
    mock_client = MagicMock()

    with patch("mlflow.register_model", return_value=mock_result) as mock_register, \
         patch("src.training.promote.MlflowClient", return_value=mock_client):
        version = register_and_promote(run_id="abc123")

    mock_register.assert_called_once_with("runs:/abc123/model", MODEL_NAME)
    mock_client.set_registered_model_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="champion",
        version="1",
    )
    assert version == "1"


def test_no_deprecated_stages_api():
    """register_and_promote must NOT call transition_model_version_stage."""
    mock_result = MagicMock()
    mock_result.version = "1"
    mock_client = MagicMock()

    with patch("mlflow.register_model", return_value=mock_result), \
         patch("src.training.promote.MlflowClient", return_value=mock_client):
        register_and_promote(run_id="abc123")

    assert not mock_client.transition_model_version_stage.called, (
        "Deprecated stages API was called — use set_registered_model_alias instead"
    )


def test_get_champion_auc_returns_float():
    """get_champion_auc returns auc_test metric as float."""
    mock_mv = MagicMock()
    mock_mv.run_id = "run_999"
    mock_run = MagicMock()
    mock_run.data.metrics = {"auc_test": 0.872}
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client.get_run.return_value = mock_run

    with patch("src.training.promote.MlflowClient", return_value=mock_client):
        auc = get_champion_auc()

    assert isinstance(auc, float)
    assert auc == pytest.approx(0.872)


def test_get_champion_auc_returns_zero_on_error():
    """get_champion_auc returns 0.0 when no champion exists."""
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("No champion")

    with patch("src.training.promote.MlflowClient", return_value=mock_client):
        auc = get_champion_auc()

    assert auc == 0.0
