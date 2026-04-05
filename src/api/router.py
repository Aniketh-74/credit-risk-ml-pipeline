"""Split-path prediction router for the scoring API.

route_request() determines per-request whether the CheckerBoard or model
path handles the prediction. Routing is stateless and random — no shared
state, no sticky sessions, no worker coordination required.

Phase 3: checkerboard_score() returns np.random.uniform(0, 1) — a stub
that gives CB-PDD real score contrast between paths.
Phase 4 replaces the body of checkerboard_score() only; the function
signature and the router infrastructure stay unchanged.
"""
from __future__ import annotations
import os
import numpy as np

CHECKERBOARD_MIX = float(os.getenv("CHECKERBOARD_MIX", "0.1"))


def route_request(mix: float = CHECKERBOARD_MIX) -> str:
    """Return routing path for this request.

    Args:
        mix: Fraction of requests routed to CheckerBoard (default 0.1).

    Returns:
        'checkerboard' with probability mix, 'model' with probability 1 - mix.
    """
    return "checkerboard" if np.random.random() < mix else "model"


def checkerboard_score() -> float:
    """Phase 3 stub for CheckerBoard predictor.

    Returns a uniform random score in [0, 1] to give CB-PDD real score
    contrast between the two paths. Phase 4 replaces this body with the
    actual CB-PDD predictor; the function signature stays unchanged.
    """
    return float(np.random.uniform(0, 1))
