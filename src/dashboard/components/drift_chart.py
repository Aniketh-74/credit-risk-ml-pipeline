"""Drift score timeline chart component.

Renders a Plotly line chart of PSI drift scores over simulated days,
overlaid with a red threshold line at 1.0 and alert markers at crossing
points. Designed to be wired to the `drift_scores` PostgreSQL table in
Phase 6 with zero UI changes.
"""

import plotly.graph_objects as go
import streamlit as st


_THRESHOLD = 1.0
_NAVY = "#0D1B2A"
_RED = "#E63946"
_AMBER = "#F4A261"
_ACCENT_BLUE = "#457B9D"
_LIGHT_BLUE = "#A8DADC"


def render(scores: list[dict]) -> None:
    """Render drift score timeline chart.

    Args:
        scores: List of dicts with keys:
            - day (int): Simulation day index (1-based).
            - score (float): PSI drift score for that day.
            - threshold_crossed (bool): True if score >= threshold on this day.
    """
    if not scores:
        st.info("No drift scores available.")
        return

    days = [r["day"] for r in scores]
    values = [r["score"] for r in scores]
    alert_days = [r["day"] for r in scores if r.get("threshold_crossed")]
    alert_scores = [r["score"] for r in scores if r.get("threshold_crossed")]

    fig = go.Figure()

    # Main drift score line
    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines+markers",
            name="Drift Score (PSI)",
            line=dict(color=_ACCENT_BLUE, width=2.5),
            marker=dict(size=5, color=_ACCENT_BLUE),
            hovertemplate="Day %{x}<br>PSI: %{y:.3f}<extra></extra>",
        )
    )

    # Red threshold line
    fig.add_hline(
        y=_THRESHOLD,
        line_dash="dash",
        line_color=_RED,
        line_width=1.8,
        annotation_text="Retraining threshold (1.0)",
        annotation_position="top right",
        annotation_font_color=_RED,
    )

    # Alert markers where threshold was crossed
    if alert_days:
        fig.add_trace(
            go.Scatter(
                x=alert_days,
                y=alert_scores,
                mode="markers",
                name="Alert fired",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=_RED,
                    line=dict(color="white", width=1),
                ),
                hovertemplate="Alert — Day %{x}<br>PSI: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Performative Drift Score — 30-Day Simulation",
            font=dict(size=16, color="#1D3557"),
            x=0,
        ),
        xaxis=dict(
            title="Simulation Day",
            gridcolor="#E8EDF2",
            zeroline=False,
        ),
        yaxis=dict(
            title="PSI Score",
            gridcolor="#E8EDF2",
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=360,
    )

    st.plotly_chart(fig, use_container_width=True)
