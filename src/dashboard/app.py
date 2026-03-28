"""Credit Risk ML Pipeline — Monitoring Dashboard.

Entry point for the Streamlit dashboard. Serves all four monitoring sections:
KPI header, drift score timeline, model version history, and alert log.

Phase 1: Renders with placeholder data so the UI can be evaluated without
         a live database connection.
Phase 6: Replace MOCK_* constants below with database query helpers — the
         component render() interfaces accept the same list[dict] shape.

Run locally:
    streamlit run src/dashboard/app.py
"""

import streamlit as st

from dashboard.components import alert_log, drift_chart, metrics_header, model_history

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script
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
# Custom CSS — dark navy sidebar, clean white main, red alert accents
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0D1B2A;
}
[data-testid="stSidebar"] * {
    color: #A8DADC !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #A8DADC !important;
    font-size: 0.95rem;
}
[data-testid="stSidebar"] hr {
    border-color: #1D3557;
}

/* Page title accent */
.dash-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1D3557;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.dash-subtitle {
    font-size: 0.85rem;
    color: #6B7280;
    margin-top: 0.1rem;
    margin-bottom: 1.2rem;
}

/* Section header underline */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1D3557;
    border-bottom: 2px solid #457B9D;
    padding-bottom: 4px;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}

/* KPI card metric label */
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #6B7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Footer banner */
.dash-footer {
    margin-top: 2.5rem;
    padding: 0.6rem 1rem;
    background-color: #F1F5F9;
    border-radius: 6px;
    font-size: 0.78rem;
    color: #6B7280;
    text-align: center;
}

/* Divider */
.dash-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.4rem 0;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Phase 1 placeholder data
# Replace each constant with a DB query in Phase 6.
# ---------------------------------------------------------------------------

MOCK_DRIFT_SCORES: list[dict] = [
    {
        "day": i,
        "score": round(0.1 + (i * 0.04) + (0.05 if i > 18 else 0), 4),
        "threshold_crossed": i >= 22,
    }
    for i in range(1, 31)
]

MOCK_MODEL_HISTORY: list[dict] = [
    {
        "version": "v1.0.0",
        "auc": 0.742,
        "promoted_at": "2026-02-26",
        "status": "retired",
    },
    {
        "version": "v1.1.0",
        "auc": 0.761,
        "promoted_at": "2026-03-05",
        "status": "retired",
    },
    {
        "version": "v1.2.0",
        "auc": 0.778,
        "promoted_at": "2026-03-14",
        "status": "champion",
    },
]

MOCK_ALERTS: list[dict] = [
    {
        "fired_at": "2026-03-05 02:14",
        "drift_score": 1.04,
        "retrain": True,
        "promoted": True,
    },
    {
        "fired_at": "2026-03-14 08:31",
        "drift_score": 1.12,
        "retrain": True,
        "promoted": True,
    },
]

# ---------------------------------------------------------------------------
# Derived summary values for the KPI header
# ---------------------------------------------------------------------------

_latest_score = MOCK_DRIFT_SCORES[-1]["score"]
_champion = next(m for m in MOCK_MODEL_HISTORY if m["status"] == "champion")
_alert_count = len(MOCK_ALERTS)
_sim_day = len(MOCK_DRIFT_SCORES)

# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## credit-risk-ml-pipeline")
    st.markdown("*Performative drift monitor*")
    st.markdown("<hr/>", unsafe_allow_html=True)

    nav_section = st.radio(
        "Navigate",
        options=["Overview", "Drift Analysis", "Model Registry", "Alerts"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<small>Phase 1 — placeholder data</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small>Phase 6 wires live PostgreSQL</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<p class="dash-title">ML Pipeline Monitor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="dash-subtitle">Credit risk scoring — performative drift detection &amp; automated retraining</p>',
    unsafe_allow_html=True,
)

# KPI header — always visible regardless of nav section
st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)
metrics_header.render(
    current_score=_latest_score,
    model_version=_champion["version"],
    alert_count=_alert_count,
    sim_day=_sim_day,
)

st.markdown('<hr class="dash-divider"/>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section: Drift Analysis
# ---------------------------------------------------------------------------

if nav_section in ("Overview", "Drift Analysis"):
    st.markdown('<p class="section-header">Drift Score Timeline</p>', unsafe_allow_html=True)
    drift_chart.render(scores=MOCK_DRIFT_SCORES)

# ---------------------------------------------------------------------------
# Section: Model Registry + Alerts (side by side on Overview)
# ---------------------------------------------------------------------------

if nav_section == "Overview":
    st.markdown('<hr class="dash-divider"/>', unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown(
            '<p class="section-header">Model Registry</p>', unsafe_allow_html=True
        )
        model_history.render(models=MOCK_MODEL_HISTORY)

    with right_col:
        st.markdown(
            '<p class="section-header">Alert Log</p>', unsafe_allow_html=True
        )
        alert_log.render(alerts=MOCK_ALERTS)

elif nav_section == "Model Registry":
    st.markdown('<p class="section-header">Model Registry</p>', unsafe_allow_html=True)
    model_history.render(models=MOCK_MODEL_HISTORY)

elif nav_section == "Alerts":
    st.markdown('<p class="section-header">Alert Log</p>', unsafe_allow_html=True)
    alert_log.render(alerts=MOCK_ALERTS)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="dash-footer">'
    "Phase 1 — placeholder data &nbsp;|&nbsp; "
    "Phase 6 will connect live PostgreSQL &nbsp;|&nbsp; "
    "credit-risk-ml-pipeline"
    "</div>",
    unsafe_allow_html=True,
)
