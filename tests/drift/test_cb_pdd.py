"""Unit tests for CB-PDD algorithm classes.

Three synthetic dataset tests:
  1. Denial loop data — detector fires (is_drift=True at least once in 30 days)
  2. Score gaming data — detector fires (is_drift=True at least once in 30 days)
  3. Random predictor data — detector does NOT fire (zero false positives)

Mocked model: deterministic scores, no MLflow, no API, no PostgreSQL.
Data is generated purely in Python — no simulators or DB calls.
"""
import random

import pytest

from src.drift.cb_pdd import (
    CheckerBoardPredictor,
    DensityChangeTracker,
    PerformativeDriftDetector,
)

# Fixed seed for reproducible synthetic data
SEED = 42
N_DAYS = 30
N_PER_DAY = 1000


def _generate_denial_loop_data(n_days: int, n_per_day: int, seed: int = SEED):
    """Generate synthetic denial loop stream with within-trial temporal structure.

    Each day (trial of n_per_day instances) is split into two halves:
      - First half:  fresh applicants. Score range shifts from [0.55, 0.95]
                     on day 0 to [0.15, 0.55] on the last day as the pool
                     games toward the 0.5 approval threshold over time.
      - Second half: denied applicants from the first half return in the same
                     trial with nudged-down scores (intra-day feature adjustment).

    y_true is drawn from a score-proportional default rate (score * 0.4) so
    that correction_rate(last_w) < correction_rate(first_w) whenever denied
    returnees have lower scores -- the density-change signal CB-PDD detects.

    Returns list of (score, y_hat, y_true) tuples -- one per instance.
    """
    rng = random.Random(seed)
    half = n_per_day // 2
    stream = []

    for day in range(n_days):
        progress = day / max(n_days - 1, 1)
        low = 0.55 - 0.40 * progress   # 0.55 on day 0 -> 0.15 on last day
        high = 0.95 - 0.40 * progress  # 0.95 on day 0 -> 0.55 on last day

        # First half: fresh applicants
        first_half = []
        returners = []
        for _ in range(half):
            score = rng.uniform(low, high)
            y_hat = 1 if score > 0.5 else 0
            y_true = 1 if rng.random() < score * 0.4 else 0
            first_half.append((score, y_hat, y_true))
            if y_hat == 1:
                nudge = rng.uniform(0.05, 0.15)
                returners.append(max(0.05, score - nudge))

        # Second half: denied returners with nudged (lower) scores
        second_half = []
        for ret in returners:
            y_hat = 1 if ret > 0.5 else 0
            y_true = 1 if rng.random() < ret * 0.4 else 0
            second_half.append((ret, y_hat, y_true))

        # Pad with low-score approved applicants if needed
        while len(second_half) < half:
            score = rng.uniform(0.05, 0.45)
            y_true = 1 if rng.random() < score * 0.4 else 0
            second_half.append((score, 0, y_true))

        stream.extend(first_half)
        stream.extend(second_half[:half])

    return stream


def _generate_score_gaming_data(n_days: int, n_per_day: int, seed: int = SEED):
    """Generate synthetic score gaming stream with within-trial temporal structure.

    Each day (trial) is split into two halves:
      - First half:  gamers at today's base score. Base declines from ~0.90
                     on day 0 to ~0.35 on the last day as gamers suppress
                     features toward the approval threshold.
      - Second half: denied gamers from the first half make an intra-day second
                     attempt with a small additional nudge downward.

    y_true is score-proportional (score * 0.4) so correction_rate drops
    between first and second halves -- the signal CB-PDD detects.

    Returns list of (score, y_hat, y_true) tuples.
    """
    rng = random.Random(seed)
    half = n_per_day // 2
    stream = []

    for day in range(n_days):
        progress = day / max(n_days - 1, 1)
        base = 0.90 - 0.55 * progress  # 0.90 on day 0 -> 0.35 on last day

        # First half: gamers at today's score level
        first_half = []
        returners = []
        for _ in range(half):
            score = rng.uniform(max(0.1, base - 0.10), min(1.0, base + 0.10))
            y_hat = 1 if score > 0.5 else 0
            y_true = 1 if rng.random() < score * 0.4 else 0
            first_half.append((score, y_hat, y_true))
            if y_hat == 1:
                nudge = rng.uniform(0.04, 0.14)
                returners.append(max(0.05, score - nudge))

        # Second half: same gamers after intra-day nudge
        second_half = []
        for ret in returners:
            y_hat = 1 if ret > 0.5 else 0
            y_true = 1 if rng.random() < ret * 0.4 else 0
            second_half.append((ret, y_hat, y_true))

        while len(second_half) < half:
            score = rng.uniform(0.05, 0.45)
            y_true = 1 if rng.random() < score * 0.4 else 0
            second_half.append((score, 0, y_true))

        stream.extend(first_half)
        stream.extend(second_half[:half])

    return stream

def _generate_random_data(n_days: int, n_per_day: int, seed: int = SEED):
    """Generate random predictor stream — null control experiment.

    Scores drawn uniformly from [0, 1] each day. No feedback loop.
    y_true drawn from a fixed 7% default rate (no drift in ground truth).

    Returns list of (score, y_hat, y_true) tuples.
    """
    rng = random.Random(seed)
    stream = []
    for _ in range(n_days * n_per_day):
        score = rng.uniform(0.0, 1.0)
        y_hat = 1 if score > 0.5 else 0
        y_true = 1 if rng.random() < 0.07 else 0
        stream.append((score, y_hat, y_true))
    return stream


class TestCheckBoardPredictor:
    """Unit tests for CheckerBoardPredictor (Algorithm 1)."""

    def test_returns_zero_or_one(self):
        """predict() must return exactly 0 or 1 for any input."""
        cbp = CheckerBoardPredictor(tau=100)
        for x in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.1]:
            result = cbp.predict(x)
            assert result in (0, 1), f"Expected 0 or 1, got {result} for x={x}"

    def test_flips_after_tau_instances(self):
        """Predictions should differ between trial period 0 and trial period 1."""
        tau = 10
        cbp = CheckerBoardPredictor(f=0.5, tau=tau)
        # Sample the same x value in trial 0 and trial 1
        x = 0.3  # maps to feature_id=0 (x/0.5 = 0.6 -> floor=0)

        # Use first instance of trial 0
        pred_trial_0 = cbp.predict(x)

        # Advance to trial 1 (consume tau-1 more instances)
        for _ in range(tau - 1):
            cbp.predict(x)

        # Now in trial 1 — prediction should flip
        pred_trial_1 = cbp.predict(x)
        assert pred_trial_0 != pred_trial_1, (
            f"Prediction should flip after tau={tau} instances. "
            f"Got {pred_trial_0} in trial 0 and {pred_trial_1} in trial 1."
        )

    def test_reset_restarts_count(self):
        """reset() should put the predictor back to its initial state."""
        cbp = CheckerBoardPredictor(tau=10)
        pred_before = cbp.predict(0.3)
        for _ in range(9):
            cbp.predict(0.3)
        cbp.reset()
        pred_after_reset = cbp.predict(0.3)
        assert pred_before == pred_after_reset, "After reset, first prediction should match initial"


class TestDensityChangeTracker:
    """Unit tests for DensityChangeTracker (Algorithm 2)."""

    def test_update_appends_to_groups(self):
        """update() should add exactly one value to group_a or group_b."""
        tracker = DensityChangeTracker(w=5)
        # Create a trial with 10 instances (>= 2*w=10): majority y_hat=1 -> group_a
        trial = [(1, 1)] * 8 + [(0, 0)] * 2  # 8/10 are target_class=1
        tracker.update(trial, target_class=1)
        total = len(tracker.group_a) + len(tracker.group_b)
        assert total == 1, f"Expected 1 value added to groups, got {total}"

    def test_short_trial_is_skipped(self):
        """Trials shorter than 2*w should produce no update."""
        tracker = DensityChangeTracker(w=100)
        short_trial = [(1, 1)] * 50  # < 2*100 = 200
        tracker.update(short_trial, target_class=1)
        assert len(tracker.group_a) == 0 and len(tracker.group_b) == 0, (
            "Short trial (< 2*w instances) should be a no-op"
        )

    def test_reset_clears_groups(self):
        """reset() should empty both groups."""
        tracker = DensityChangeTracker(w=5)
        trial = [(1, 1)] * 12
        tracker.update(trial)
        tracker.reset()
        assert tracker.group_a == [] and tracker.group_b == [], "reset() must clear groups"


class TestPerformativeDriftDetector:
    """Tests for PerformativeDriftDetector — the main orchestrator."""

    def test_init_default_params(self):
        """Default params read from env vars (tau=1000, w=500, alpha=0.05)."""
        detector = PerformativeDriftDetector()
        assert detector.tau == 1000
        assert detector.w == 500
        assert detector.alpha == 0.05
        assert detector.is_drift is False

    def test_invalid_params_raise(self):
        """ValueError must be raised for w >= tau or tau < 100."""
        with pytest.raises(ValueError, match="w.*must be.*tau"):
            PerformativeDriftDetector(tau=1000, w=1000)

        with pytest.raises(ValueError, match="w.*must be.*tau"):
            PerformativeDriftDetector(tau=1000, w=1001)

        with pytest.raises(ValueError, match="tau must be >= 100"):
            PerformativeDriftDetector(tau=50, w=25)

        with pytest.raises(ValueError, match="alpha must be in"):
            PerformativeDriftDetector(tau=1000, w=500, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            PerformativeDriftDetector(tau=1000, w=500, alpha=1.0)

    def test_denial_loop_fires(self):
        """Denial loop data must trigger is_drift=True at least once across 30 days."""
        stream = _generate_denial_loop_data(N_DAYS, N_PER_DAY, seed=SEED)
        detector = PerformativeDriftDetector(tau=1000, w=500, alpha=0.05)

        for score, y_hat, y_true in stream:
            detector.add(score, y_hat, y_true)

        assert detector.is_drift is True, (
            f"Denial loop: expected is_drift=True after {N_DAYS} days x {N_PER_DAY} "
            f"applicants. Got is_drift=False, last_p_value={detector.last_p_value:.4f}, "
            f"trials_completed={detector._trial_count}"
        )

    def test_score_gaming_fires(self):
        """Score gaming data must trigger is_drift=True at least once across 30 days."""
        stream = _generate_score_gaming_data(N_DAYS, N_PER_DAY, seed=SEED)
        detector = PerformativeDriftDetector(tau=1000, w=500, alpha=0.05)

        for score, y_hat, y_true in stream:
            detector.add(score, y_hat, y_true)

        assert detector.is_drift is True, (
            f"Score gaming: expected is_drift=True after {N_DAYS} days x {N_PER_DAY} "
            f"applicants. Got is_drift=False, last_p_value={detector.last_p_value:.4f}, "
            f"trials_completed={detector._trial_count}"
        )

    def test_random_data_does_not_fire(self):
        """Random predictor must produce zero false positives across 30 days."""
        stream = _generate_random_data(N_DAYS, N_PER_DAY, seed=SEED)
        detector = PerformativeDriftDetector(tau=1000, w=500, alpha=0.05)

        for score, y_hat, y_true in stream:
            detector.add(score, y_hat, y_true)

        assert detector.is_drift is False, (
            f"Null control: expected is_drift=False on random data. "
            f"Got is_drift=True, last_p_value={detector.last_p_value:.4f}, "
            f"trials_completed={detector._trial_count}. "
            f"This is a false positive — check consecutive window logic."
        )

    def test_requires_two_consecutive_windows(self):
        """is_drift must remain False after only one detection window."""
        # Use tiny tau/w to control trial boundaries precisely
        detector = PerformativeDriftDetector(tau=100, w=40, alpha=0.99)
        # With alpha=0.99, virtually any data triggers detection
        # But is_drift should be False after exactly one trial
        stream = [(0.7, 1, 0)] * 100  # exactly one trial
        for s, yh, yt in stream:
            detector.add(s, yh, yt)

        # After one trial: _consecutive might be 1, but is_drift must still be False
        assert detector._trial_count == 1
        if detector._consecutive == 1:
            assert detector.is_drift is False, (
                "is_drift should be False after only 1 consecutive detection window"
            )

    def test_reset_clears_state(self):
        """reset() must return detector to initial state."""
        stream = _generate_denial_loop_data(N_DAYS, N_PER_DAY, seed=SEED)
        detector = PerformativeDriftDetector(tau=1000, w=500, alpha=0.05)
        for score, y_hat, y_true in stream:
            detector.add(score, y_hat, y_true)

        detector.reset()
        assert detector.is_drift is False
        assert detector._consecutive == 0
        assert detector._trial_buffer == []
        assert detector.last_p_value == 1.0
