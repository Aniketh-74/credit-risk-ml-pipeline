"""Credit Risk ML Pipeline — Monitoring Dashboard.

Entry point for the Streamlit dashboard. Five sections:
  1. KPI header — current drift score, active model, alert count
  2. Drift score timeline — CB-PDD score with alert threshold
  3. CB-PDD vs PSI comparison — why performative drift needs a new detector
  4. Tau sensitivity — detection latency across tau values
  5. Model registry + alert log

Live data is sourced from:
  - APP_DB_URL      → drift_scores and alerts tables (PostgreSQL)
  - MLFLOW_TRACKING_URI → model version history (REST API, no SDK)

Falls back to built-in placeholder data when the database is unreachable
so the dashboard renders in demo mode without a live stack.

Run locally (demo mode, no DB needed):
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os

import streamlit as st

from dashboard.components import (
    alert_log,
    drift_chart,
    drift_comparison,
    metrics_header,
    model_history,
    tau_sensitivity,
)
from dashboard import db

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="credit-risk | ML Monitor",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/Aniketh-74/credit-risk-ml-pipeline",
        "About": "Credit Risk ML Pipeline — performative drift detection dashboard.",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
[data-testid="stSidebar"] { background-color: #0D1B2A; }
[data-testid="stSidebar"] * { color: #A8DADC !important; }
[data-testid="stSidebar"] .stRadio label { color: #A8DADC !important; font-size: 0.95rem; }
[data-testid="stSidebar"] hr { border-color: #1D3557; }
.dash-title { font-size: 1.6rem; font-weight: 700; color: #1D3557; letter-spacing: -0.5px; margin-bottom: 0; }
.dash-subtitle { font-size: 0.85rem; color: #6B7280; margin-top: 0.1rem; margin-bottom: 1.2rem; }
.section-header { font-size: 1.05rem; font-weight: 600; color: #1D3557; border-bottom: 2px solid #457B9D; padding-bottom: 4px; margin-top: 1.2rem; margin-bottom: 0.6rem; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #6B7280 !important; text-transform: uppercase; letter-spacing: 0.06em; }
.dash-footer { margin-top: 2.5rem; padding: 0.6rem 1rem; background-color: #F1F5F9; border-radius: 6px; font-size: 0.78rem; color: #6B7280; text-align: center; }
.dash-divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.4rem 0; }
.demo-banner { background-color: #FFF3CD; border: 1px solid #FFEAA7; border-radius: 6px; padding: 0.5rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: #856404; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get("APP_DB_URL", "")
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# ---------------------------------------------------------------------------
# Placeholder data — used when DB/MLflow are unreachable
# ---------------------------------------------------------------------------

_MOCK_SCORES: list[dict] = [
    {
        "day": i,
        "score": round(max(0.01, 0.95 - i * 0.03 + (0.02 if i % 3 == 0 else 0)), 4),
        "psi_score": round(0.05 + i * 0.009 + (0.04 if i > 18 else 0), 4),
        "threshold_crossed": i >= 24,
        "computed_at": f"2026-03-{i:02d}T02:00:00",
    }
    for i in range(1, 31)
]

_MOCK_MODELS: list[dict] = [
    {"version": "v1", "auc": 0.742, "promoted_at": "2026-02-26", "status": "retired"},
    {"version": "v2", "auc": 0.761, "promoted_at": "2026-03-05", "status": "retired"},
    {"version": "v3", "auc": 0.778, "promoted_at": "2026-03-14", "status": "champion"},
]

_MOCK_ALERTS: list[dict] = [
    {"fired_at": "2026-03-05 02:14", "drift_score": 0.03, "retrain": True, "promoted": True},
    {"fired_at": "2026-03-14 08:31", "drift_score": 0.01, "retrain": True, "promoted": True},
]

# ---------------------------------------------------------------------------
# Live data — cached 60 s, falls back to mock on error
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _load_scores(db_url: str) -> tuple[list[dict], bool]:
    if not db_url:
        return _MOCK_SCORES, False
    rows = db.get_drift_scores(db_url, window_days=30)
    return (rows, True) if rows else (_MOCK_SCORES, False)


@st.cache_data(ttl=60, show_spinner=False)
def _load_alerts(db_url: str) -> tuple[list[dict], bool]:
    if not db_url:
        return _MOCK_ALERTS, False
    rows = db.get_alerts(db_url)
    return (rows, True) if rows else (_MOCK_ALERTS, False)


@st.cache_data(ttl=120, show_spinner=False)
def _load_models(mlflow_uri: str) -> tuple[list[dict], bool]:
    if not mlflow_uri:
        return _MOCK_MODELS, False
    rows = db.get_model_history(mlflow_uri)
    return (rows, True) if rows else (_MOCK_MODELS, False)


scores, _live_scores = _load_scores(_DB_URL)
alerts, _live_alerts = _load_alerts(_DB_URL)
models, _live_models = _load_models(_MLFLOW_URI)

_demo_mode = not (_live_scores and _live_alerts and _live_models)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## credit-risk-ml-pipeline")
    st.markdown("*Performative drift monitor*")
    st.markdown("<hr/>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        options=["Overview", "Drift Analysis", "Model Registry", "Alerts"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    if _demo_mode:
        st.markdown("<small>Demo mode — placeholder data</small>", unsafe_allow_html=True)
        st.markdown("<small>Set APP_DB_URL for live data</small>", unsafe_allow_html=True)
    else:
        st.markdown("<small>Live data — PostgreSQL + MLflow</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown('<p class="dash-title">ML Pipeline Monitor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="dash-subtitle">Credit risk scoring — performative drift detection &amp; automated retraining</p>',
    unsafe_allow_html=True,
)

if _demo_mode:
    st.markdown(
        '<div class="demo-banner">Demo mode — showing placeholder data. '
        "Set <code>APP_DB_URL</code> and <code>MLFLOW_TRACKING_URI</code> "
        "to connect to a live stack.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# KPI header — always visible
# ---------------------------------------------------------------------------

_latest = scores[-1] if scores else {"score": 0.0, "psi_score": 0.0}
_champion = next((m for m in models if m["status"] == "champion"), models[0] if models else {})

st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)
metrics_header.render(
    current_score=_latest.get("psi_score", 0.0),
    model_version=_champion.get("version", "—"),
    alert_count=len(alerts),
    sim_day=len(scores),
)

st.markdown('<hr class="dash-divider"/>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Drift Analysis section
# ---------------------------------------------------------------------------

if nav in ("Overview", "Drift Analysis"):
    st.markdown('<p class="section-header">Drift Score Timeline</p>', unsafe_allow_html=True)
    drift_chart.render(scores=scores)

    st.markdown('<p class="section-header">CB-PDD vs PSI Comparison</p>', unsafe_allow_html=True)
    drift_comparison.render(scores=scores)

    st.markdown('<p class="section-header">Tau (\u03c4) Sensitivity Analysis</p>', unsafe_allow_html=True)
    tau_sensitivity.render()

# ---------------------------------------------------------------------------
# Model Registry + Alerts
# ---------------------------------------------------------------------------

if nav == "Overview":
    st.markdown('<hr class="dash-divider"/>', unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2], gap="large")
    with left_col:
        st.markdown('<p class="section-header">Model Registry</p>', unsafe_allow_html=True)
        model_history.render(models=models)
    with right_col:
        st.markdown('<p class="section-header">Alert Log</p>', unsafe_allow_html=True)
        alert_log.render(alerts=alerts)

elif nav == "Model Registry":
    st.markdown('<p class="section-header">Model Registry</p>', unsafe_allow_html=True)
    model_history.render(models=models)

elif nav == "Alerts":
    st.markdown('<p class="section-header">Alert Log</p>', unsafe_allow_html=True)
    alert_log.render(alerts=alerts)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

_data_source = "live PostgreSQL + MLflow" if not _demo_mode else "placeholder data (demo mode)"
st.markdown(
    f'<div class="dash-footer">credit-risk-ml-pipeline &nbsp;|&nbsp; {_data_source}</div>',
    unsafe_allow_html=True,
)
