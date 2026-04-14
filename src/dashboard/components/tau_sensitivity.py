"""CB-PDD tau sensitivity chart component.

Shows drift detection latency (first detection day) across three tau values:
500, 1000, and 2000. Lower tau = faster detection but higher false-positive risk.
Higher tau = more statistical power but slower response.

Data source: Phase 2 smoke test results (validated empirically on denial loop
simulation with n_per_day=1000 over 30 days).
"""

import plotly.graph_objects as go
import streamlit as st

_NAVY = "#1D3557"
_BLUE = "#457B9D"
_LIGHT_BLUE = "#A8DADC"
_AMBER = "#F4A261"
_RED = "#E63946"

# Phase 2 smoke test results — first detection day per tau value
# Source: 02-03 CB-PDD smoke test, n_per_day=1000, denial loop simulation
_TAU_RESULTS = [
    {"tau": 500,  "first_detection_day": 6,  "detection_rate": 1.0, "risk": "higher FP risk"},
    {"tau": 1000, "first_detection_day": 14, "detection_rate": 1.0, "risk": "balanced (default)"},
    {"tau": 2000, "first_detection_day": 28, "detection_rate": 1.0, "risk": "lowest FP risk"},
]


def render() -> None:
    """Render tau sensitivity bar chart with detection latency annotations."""
    taus = [str(r["tau"]) for r in _TAU_RESULTS]
    days = [r["first_detection_day"] for r in _TAU_RESULTS]
    colors = [_AMBER, _BLUE, _LIGHT_BLUE]
    risk_labels = [r["risk"] for r in _TAU_RESULTS]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=taus,
            y=days,
            marker_color=colors,
            text=[f"Day {d}" for d in days],
            textposition="outside",
            hovertemplate=(
                "τ = %{x}<br>"
                "First detection: Day %{y}<br>"
                "<extra></extra>"
            ),
            width=0.45,
        )
    )

    # Annotate risk level beneath each bar
    for i, (tau, risk) in enumerate(zip(taus, risk_labels)):
        fig.add_annotation(
            x=tau,
            y=-2.5,
            text=f"<i>{risk}</i>",
            showarrow=False,
            font=dict(size=10, color="#6B7280"),
            yref="y",
        )

    fig.update_layout(
        title=dict(
            text="CB-PDD Tau (τ) Sensitivity — Detection Latency vs Trial Length",
            font=dict(size=15, color=_NAVY),
            x=0,
        ),
        xaxis=dict(
            title="Trial length τ (instances per window)",
            tickfont=dict(size=13),
            gridcolor="#E8EDF2",
        ),
        yaxis=dict(
            title="First detection (simulation day)",
            gridcolor="#E8EDF2",
            zeroline=False,
            range=[0, 35],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=40),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown(
            "Each bar shows how many simulation days elapsed before CB-PDD first "
            "confirmed drift at that τ setting. All three τ values eventually detect "
            "the denial-loop feedback pattern — the trade-off is speed vs robustness:\n\n"
            "- **τ=500**: detects by Day 6 but has higher false-positive exposure "
            "(2 consecutive windows of 500 = only 1000 samples before alarm)\n"
            "- **τ=1000** *(default)*: balanced — Day 14 detection, 2000 samples for "
            "confirmation, aligns with the paper's recommended setting\n"
            "- **τ=2000**: most conservative — Day 28 detection, 4000 samples required; "
            "best for low-volume models where false alarms are costly\n\n"
            "Detection rate was 100% across all τ values on 30-day denial loop simulation."
        )
