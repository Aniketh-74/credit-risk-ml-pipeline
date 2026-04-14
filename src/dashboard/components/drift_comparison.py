"""CB-PDD vs PSI side-by-side comparison chart component.

Renders both drift metrics on the same time axis so a viewer can see:
  - PSI detects distribution shift but ignores feedback-loop structure
  - CB-PDD detects performative drift that PSI misses or detects later

Designed to make the portfolio argument visually obvious: CB-PDD adds
signal that a standard population metric cannot capture.
"""

import plotly.graph_objects as go
import streamlit as st

_THRESHOLD = 0.25   # PSI "significant shift" line
_CB_PDD_THRESHOLD = 0.05   # CB-PDD significance level (alpha)
_RED = "#E63946"
_BLUE = "#457B9D"
_AMBER = "#F4A261"
_NAVY = "#1D3557"


def render(scores: list[dict]) -> None:
    """Render overlaid CB-PDD p-value and PSI score time series.

    Args:
        scores: List of dicts with keys:
            - day (int): Simulation day index (1-based).
            - score (float): CB-PDD drift score (p-value, lower = more drift).
            - psi_score (float): PSI score (higher = more drift).
            - threshold_crossed (bool): True when CB-PDD alert fired.
    """
    if not scores:
        st.info("No drift comparison data available.")
        return

    days = [r["day"] for r in scores]
    cbpdd_values = [r["score"] for r in scores]
    psi_values = [r.get("psi_score", 0.0) for r in scores]

    fig = go.Figure()

    # CB-PDD p-value (left y-axis) — inverted: low p-value = high drift
    fig.add_trace(
        go.Scatter(
            x=days,
            y=cbpdd_values,
            mode="lines+markers",
            name="CB-PDD p-value",
            line=dict(color=_BLUE, width=2.5),
            marker=dict(size=4),
            yaxis="y1",
            hovertemplate="Day %{x}<br>CB-PDD p: %{y:.4f}<extra></extra>",
        )
    )

    # PSI score (right y-axis) — higher = more drift
    fig.add_trace(
        go.Scatter(
            x=days,
            y=psi_values,
            mode="lines+markers",
            name="PSI score",
            line=dict(color=_AMBER, width=2.5, dash="dot"),
            marker=dict(size=4),
            yaxis="y2",
            hovertemplate="Day %{x}<br>PSI: %{y:.3f}<extra></extra>",
        )
    )

    # CB-PDD detection threshold (alpha = 0.05)
    fig.add_hline(
        y=_CB_PDD_THRESHOLD,
        line_dash="dash",
        line_color=_BLUE,
        line_width=1.2,
        annotation_text="CB-PDD α=0.05",
        annotation_position="top left",
        annotation_font_color=_BLUE,
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(
            text="CB-PDD vs PSI — Drift Detection Comparison",
            font=dict(size=15, color=_NAVY),
            x=0,
        ),
        xaxis=dict(title="Simulation Day", gridcolor="#E8EDF2", zeroline=False),
        yaxis=dict(
            title="CB-PDD p-value (lower = more drift)",
            titlefont=dict(color=_BLUE),
            tickfont=dict(color=_BLUE),
            gridcolor="#E8EDF2",
            zeroline=False,
            range=[0, 1.05],
        ),
        yaxis2=dict(
            title="PSI score (higher = more drift)",
            titlefont=dict(color=_AMBER),
            tickfont=dict(color=_AMBER),
            overlaying="y",
            side="right",
            zeroline=False,
            showgrid=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=60, t=60, b=0),
        height=360,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why two metrics?"):
        st.markdown(
            "**PSI** (Population Stability Index) measures raw score distribution shift. "
            "It cannot distinguish *why* the distribution changed — random drift and "
            "performative feedback loops look similar.\n\n"
            "**CB-PDD** (CheckerBoard Performative Drift Detector, arxiv 2412.10545) "
            "uses a controlled checkerboard prediction assignment to detect the "
            "specific density-change signature of denied applicants returning with "
            "nudged scores. It fires on the feedback-loop pattern that PSI treats as "
            "ordinary distribution shift."
        )
