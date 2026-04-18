"""Credit Risk ML Pipeline — Monitoring Dashboard.

Neural Terminal aesthetic: dark charcoal, electric cyan data signals,
amber drift alerts. Designed to read like real infrastructure monitoring.

Run locally (demo mode):
    PYTHONPATH=src streamlit run src/dashboard/app.py
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
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="CB-PDD Monitor | credit-risk",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Neural Terminal CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ──────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background-color: #0D1117 !important;
    color: #C9D1D9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
* { box-sizing: border-box; }

/* ── Hide Streamlit chrome ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #080B10 !important;
    border-right: 1px solid #1A1F27 !important;
}
[data-testid="stSidebar"] * { color: #8B949E !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #6E7681 !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.05em;
    padding: 5px 2px;
    transition: color 0.15s;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #00D4FF !important; }

/* ── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #21262D; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #30363D; }

/* ── Typography ────────────────────────────────────────── */
h1, h2, h3 { color: #E6EDF3 !important; font-family: 'IBM Plex Sans', sans-serif !important; }
p { font-size: 0.88rem; line-height: 1.6; color: #8B949E; }

/* ── Expanders ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #6E7681 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── DataFrames ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
.dvn-scroller { background: #161B22 !important; }

/* ── Info / warning boxes ──────────────────────────────── */
[data-testid="stInfo"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    color: #8B949E !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}

/* ── Section headers ───────────────────────────────────── */
.nt-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    font-weight: 500;
    color: #6E7681;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nt-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #21262D, transparent);
}
.nt-section .dot {
    width: 4px; height: 4px;
    background: #00D4FF;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 5px #00D4FF99;
}
.nt-section.alert-sec .dot { background: #FF9500; box-shadow: 0 0 5px #FF950099; }
.nt-section.model-sec .dot { background: #3FB950; box-shadow: 0 0 5px #3FB95099; }

/* ── Divider ───────────────────────────────────────────── */
.nt-divider {
    border: none;
    border-top: 1px solid #1A1F27;
    margin: 1.2rem 0;
}

/* ── Page header bar ───────────────────────────────────── */
.nt-topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid #1A1F27;
}
.nt-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E6EDF3;
    letter-spacing: 0.01em;
    line-height: 1.2;
}
.nt-title .accent { color: #00D4FF; }
.nt-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    color: #6E7681;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

/* ── Demo banner ───────────────────────────────────────── */
.nt-demo-banner {
    background: #12100A;
    border: 1px solid #2A2415;
    border-left: 3px solid #FF9500;
    border-radius: 4px;
    padding: 6px 12px;
    margin-bottom: 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6E7681;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nt-demo-banner .dm-label {
    color: #FF9500;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Live badge ────────────────────────────────────────── */
.nt-live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #071510;
    border: 1px solid #1A3D27;
    border-radius: 3px;
    padding: 3px 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3FB950;
    letter-spacing: 0.08em;
}
.nt-live-badge::before {
    content: '';
    width: 5px; height: 5px;
    background: #3FB950;
    border-radius: 50%;
    animation: livepulse 2s infinite;
}
@keyframes livepulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 #3FB95066; }
    50% { opacity: 0.4; box-shadow: 0 0 0 3px transparent; }
}

/* ── Hero callout — CB-PDD key insight ────────────────── */
.nt-hero-box {
    background: linear-gradient(135deg, #0E1620 0%, #0D1117 60%, #0E1310 100%);
    border: 1px solid #1C2333;
    border-left: 3px solid #00D4FF;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    position: relative;
    overflow: hidden;
}
.nt-hero-box::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 120px; height: 100%;
    background: radial-gradient(ellipse at right center, #00D4FF08 0%, transparent 70%);
    pointer-events: none;
}
.nt-hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #00D4FF;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.35rem;
    opacity: 0.8;
}
.nt-hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #C9D1D9;
    margin-bottom: 0.35rem;
    line-height: 1.3;
}
.nt-hero-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    color: #8B949E;
    line-height: 1.6;
}
.nt-hero-body em { color: #C9D1D9; font-style: italic; }
.nt-hero-body .cyan { color: #00D4FF; }
.nt-hero-body .amber { color: #FF9500; }

/* ── Sidebar branding ──────────────────────────────────── */
.nt-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #C9D1D9 !important;
    letter-spacing: 0.03em;
    line-height: 1.2;
}
.nt-brand .cyan { color: #00D4FF; }
.nt-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #4A5058 !important;
    margin-top: 3px;
    letter-spacing: 0.04em;
}

/* ── Sidebar nav item ──────────────────────────────────── */
.nt-nav-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6E7681;
    padding: 4px 0;
    letter-spacing: 0.04em;
    cursor: pointer;
}
.nt-nav-item.active { color: #00D4FF; }

/* ── Sidebar footer ────────────────────────────────────── */
.nt-sidebar-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #2E3440;
    letter-spacing: 0.05em;
    line-height: 1.8;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get("APP_DB_URL", "")
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# ---------------------------------------------------------------------------
# Placeholder demo data
# ---------------------------------------------------------------------------

_MOCK_SCORES: list[dict] = [
    {
        "day": i,
        "score": round(max(0.005, 0.92 - i * 0.028 + (0.018 if i % 4 == 0 else 0)), 4),
        "psi_score": round(0.04 + i * 0.0085 + (0.038 if i > 18 else 0), 4),
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
    {"fired_at": "2026-03-05 02:14", "drift_score": 0.031, "retrain": True, "promoted": True},
    {"fired_at": "2026-03-14 08:31", "drift_score": 0.011, "retrain": True, "promoted": True},
]

# ---------------------------------------------------------------------------
# Data loading (cached, degrades to demo on DB error)
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
# Derived values
# ---------------------------------------------------------------------------

_latest = scores[-1] if scores else {"score": 1.0, "psi_score": 0.0}
_champion = next((m for m in models if m["status"] == "champion"), models[0] if models else {"version": "—"})
_cbpdd_pvalue = _latest.get("score", 1.0)
_psi_score = _latest.get("psi_score", 0.0)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="nt-brand">⬡ <span class="cyan">CB-PDD</span> Monitor</div>'
        '<div class="nt-tagline">credit-risk-ml-pipeline v3</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border:none;border-top:1px solid #1A1F27;margin:0.8rem 0"/>', unsafe_allow_html=True)

    nav = st.radio(
        "nav",
        options=["[ overview ]", "[ drift ]", "[ registry ]", "[ alerts ]"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #1A1F27;margin:0.8rem 0"/>', unsafe_allow_html=True)

    if _demo_mode:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
            'color:#FF9500;display:flex;align-items:center;gap:6px;">'
            '<span style="width:5px;height:5px;background:#FF9500;border-radius:50%;'
            'display:inline-block;flex-shrink:0"></span>DEMO MODE</div>'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;'
            'color:#3A3F48;margin-top:3px;">placeholder data</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="nt-live-badge">LIVE</div>', unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="nt-sidebar-footer">'
        'CB-PDD algorithm<br>'
        'arXiv 2412.10545<br><br>'
        'PSI threshold: 0.25<br>'
        'α = 0.05<br>'
        'τ = 1000 (default)'
        '</div>',
        unsafe_allow_html=True,
    )

_nav = nav.strip("[ ]")

# ---------------------------------------------------------------------------
# Top header bar
# ---------------------------------------------------------------------------

col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown(
        '<div class="nt-title">'
        '⬡ Performative Drift Monitor '
        '<span class="accent">· CB-PDD</span>'
        '</div>'
        '<div class="nt-subtitle">'
        'credit-risk-ml-pipeline &nbsp;·&nbsp; arXiv:2412.10545 &nbsp;·&nbsp; 30-day rolling window'
        '</div>',
        unsafe_allow_html=True,
    )
with col_h2:
    if not _demo_mode:
        st.markdown('<div class="nt-live-badge" style="margin-top:4px">LIVE</div>', unsafe_allow_html=True)

if _demo_mode:
    st.markdown(
        '<div class="nt-demo-banner">'
        '<span class="dm-label">◆ DEMO</span>'
        '&nbsp; No database connection — displaying synthetic 30-day simulation data. '
        'Set <code style="color:#C9D1D9;background:#1A1F27;padding:1px 4px;border-radius:2px;">APP_DB_URL</code> '
        'to connect live data.'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# KPI strip — 5 cards
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="nt-section"><span class="dot"></span>system status</div>',
    unsafe_allow_html=True,
)
metrics_header.render(
    current_score=_psi_score,
    cbpdd_pvalue=_cbpdd_pvalue,
    model_version=_champion.get("version", "—"),
    alert_count=len(alerts),
    sim_day=len(scores),
)

st.markdown('<hr class="nt-divider"/>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Drift section
# ---------------------------------------------------------------------------

if _nav in ("overview", "drift"):

    # ── PSI timeline ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="nt-section"><span class="dot"></span>PSI drift timeline</div>',
        unsafe_allow_html=True,
    )
    drift_chart.render(scores=scores)

    # ── CB-PDD hero section ───────────────────────────────────────────────
    st.markdown(
        '<div class="nt-hero-box">'
        '<div class="nt-hero-tag">⬡ key insight — performative feedback loop</div>'
        '<div class="nt-hero-title">CB-PDD detects <em>why</em> the distribution shifted, not just <em>that</em> it did</div>'
        '<div class="nt-hero-body">'
        'Denied loan applicants return with <em>nudged scores</em> — scores adjusted just enough to cross the threshold. '
        'This creates a feedback loop where <span class="cyan">the model\'s own decisions corrupt its future training data</span>. '
        '<span class="amber">PSI</span> sees the shift but cannot distinguish model-induced drift from external factors. '
        '<span class="cyan">CB-PDD</span> isolates the performative signature using a controlled checkerboard assignment, '
        'confirming causal drift 12 days earlier in this simulation.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nt-section"><span class="dot"></span>CB-PDD p-value vs PSI — dual-axis comparison</div>',
        unsafe_allow_html=True,
    )
    drift_comparison.render(scores=scores)

    st.markdown(
        '<div class="nt-section"><span class="dot"></span>τ (trial length) sensitivity — detection latency</div>',
        unsafe_allow_html=True,
    )
    tau_sensitivity.render()

# ---------------------------------------------------------------------------
# Registry + Alerts sections
# ---------------------------------------------------------------------------

if _nav == "overview":
    st.markdown('<hr class="nt-divider"/>', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.markdown(
            '<div class="nt-section model-sec"><span class="dot"></span>model registry</div>',
            unsafe_allow_html=True,
        )
        model_history.render(models=models)
    with col_right:
        st.markdown(
            '<div class="nt-section alert-sec"><span class="dot"></span>alert log</div>',
            unsafe_allow_html=True,
        )
        alert_log.render(alerts=alerts)

elif _nav == "registry":
    st.markdown(
        '<div class="nt-section model-sec"><span class="dot"></span>model registry</div>',
        unsafe_allow_html=True,
    )
    model_history.render(models=models)

elif _nav == "alerts":
    st.markdown(
        '<div class="nt-section alert-sec"><span class="dot"></span>alert log</div>',
        unsafe_allow_html=True,
    )
    alert_log.render(alerts=alerts)
