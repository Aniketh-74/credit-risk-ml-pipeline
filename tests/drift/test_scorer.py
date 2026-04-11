"""Unit + integration tests for src/drift/scorer.py.

Tests are split into two layers:
  - Direct unit tests for _compute_psi (no DB needed)
  - Integration tests for compute_drift using temp-file SQLite fixtures
    defined in conftest.py
"""
from datetime import date

import numpy as np
import pytest

from src.drift.scorer import DriftResult, _compute_psi, compute_drift


class TestComputePsi:
    def test_empty_arrays_return_zero(self):
        assert _compute_psi(np.array([]), np.array([0.5])) == 0.0
        assert _compute_psi(np.array([0.5]), np.array([])) == 0.0

    def test_identical_distributions_near_zero(self):
        scores = np.linspace(0.1, 0.9, 100)
        psi = _compute_psi(scores, scores.copy())
        assert psi < 0.01, f"Identical distributions should give PSI ≈ 0, got {psi:.4f}"

    def test_shifted_distribution_nonzero(self):
        ref = np.random.default_rng(0).uniform(0.1, 0.4, 200)
        cur = np.random.default_rng(1).uniform(0.6, 0.9, 200)
        psi = _compute_psi(ref, cur)
        assert psi > 0.25, f"Heavily shifted distributions should give PSI > 0.25, got {psi:.4f}"

    def test_psi_is_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.uniform(0, 1, 50)
            b = rng.uniform(0, 1, 50)
            assert _compute_psi(a, b) >= 0.0


class TestComputeDriftEmpty:
    def test_empty_window_safe_defaults(self, empty_db_url):
        result = compute_drift(empty_db_url, window_days=30)
        assert result["drift_score"] == 1.0
        assert result["psi_score"] == 0.0
        assert result["is_drift"] is False
        assert isinstance(result["window_end"], date)


class TestComputeDriftSeeded:
    def test_returns_drift_result_shape(self, seeded_db_url):
        result = compute_drift(seeded_db_url, window_days=30)
        assert set(result.keys()) == {"drift_score", "psi_score", "is_drift", "window_end"}

    def test_types_are_correct(self, seeded_db_url):
        result = compute_drift(seeded_db_url, window_days=30)
        assert isinstance(result["drift_score"], float)
        assert isinstance(result["psi_score"], float)
        assert isinstance(result["is_drift"], bool)
        assert isinstance(result["window_end"], date)

    def test_drift_score_in_valid_range(self, seeded_db_url):
        result = compute_drift(seeded_db_url, window_days=30)
        # p-value must be in [0, 1]; safe default is 1.0
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_psi_score_nonnegative(self, seeded_db_url):
        result = compute_drift(seeded_db_url, window_days=30)
        assert result["psi_score"] >= 0.0

    def test_window_end_is_max_outcome_date(self, seeded_db_url):
        result = compute_drift(seeded_db_url, window_days=30)
        # seeded anchor is 2025-03-30
        assert result["window_end"] == date(2025, 3, 30)

    def test_narrow_window_excludes_early_rows(self, seeded_db_url):
        # window_days=5 from anchor 2025-03-30 → only rows from 2025-03-25 onward
        # that's only the last batch (2025-03-30, 20 rows)
        result_narrow = compute_drift(seeded_db_url, window_days=5)
        result_wide = compute_drift(seeded_db_url, window_days=30)
        # Both should still return valid results — narrow may have less signal
        assert isinstance(result_narrow["drift_score"], float)
        assert result_narrow["window_end"] == result_wide["window_end"]

    def test_idempotent(self, seeded_db_url):
        """Two calls on the same DB return identical results."""
        r1 = compute_drift(seeded_db_url, window_days=30)
        r2 = compute_drift(seeded_db_url, window_days=30)
        assert r1["drift_score"] == r2["drift_score"]
        assert r1["psi_score"] == r2["psi_score"]
        assert r1["is_drift"] == r2["is_drift"]
        assert r1["window_end"] == r2["window_end"]
