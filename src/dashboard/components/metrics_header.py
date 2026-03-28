"""Top-level KPI metric cards component.

Renders four metric tiles across the top of the dashboard:
current drift score, active model version, alert count (30 days),
and the current simulation day.
"""

import streamlit as st


def render(
    current_score: float,
    model_version: str,
    alert_count: int,
    sim_day: int,
) -> None:
    """Render top-level KPI metric cards.

    Args:
        current_score: Latest PSI drift score from the detector.
        model_version: Semantic version string of the champion model (e.g. "v1.2.0").
        alert_count: Number of drift alerts fired in the last 30 days.
        sim_day: Current simulation day index (1-based).
    """
    col1, col2, col3, col4 = st.columns(4)

    threshold = 1.0
    drift_delta = round(current_score - threshold, 3)
    drift_delta_str = f"{drift_delta:+.3f} vs threshold"

    with col1:
        st.metric(
            label="Current Drift Score",
            value=f"{current_score:.3f}",
            delta=drift_delta_str,
            delta_color="inverse",  # red when above threshold
            help="Population Stability Index (PSI). Values above 1.0 trigger retraining.",
        )

    with col2:
        st.metric(
            label="Active Model",
            value=model_version,
            delta="champion",
            delta_color="off",
            help="MLflow model alias currently serving predictions.",
        )

    with col3:
        alert_delta = f"{alert_count} fired" if alert_count > 0 else "none"
        st.metric(
            label="Alerts (30d)",
            value=str(alert_count),
            delta=alert_delta,
            delta_color="inverse" if alert_count > 0 else "off",
            help="Drift threshold crossings that triggered the retraining pipeline.",
        )

    with col4:
        st.metric(
            label="Simulation Day",
            value=f"Day {sim_day}",
            delta="Phase 1 — mock data",
            delta_color="off",
            help="Current day in the 30-day performative drift simulation.",
        )
