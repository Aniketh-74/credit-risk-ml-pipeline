"""Drift score timeline — Neural Terminal dark theme."""

import plotly.graph_objects as go
import streamlit as st

_BG = "#0D1117"
_CARD = "#161B22"
_BORDER = "#21262D"
_CYAN = "#00D4FF"
_AMBER = "#FF9500"
_MUTED = "#6E7681"
_THRESHOLD = 0.25


def render(scores: list[dict]) -> None:
    if not scores:
        st.info("No drift scores available.")
        return

    days = [r["day"] for r in scores]
    values = [r.get("psi_score", r.get("score", 0)) for r in scores]
    alert_days = [r["day"] for r in scores if r.get("threshold_crossed")]
    alert_vals = [r.get("psi_score", r.get("score", 0)) for r in scores if r.get("threshold_crossed")]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days, y=values,
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=values,
        mode="lines+markers",
        name="PSI score",
        line=dict(color=_CYAN, width=2),
        marker=dict(size=4, color=_CYAN),
        hovertemplate="Day %{x} &nbsp; PSI: <b>%{y:.3f}</b><extra></extra>",
    ))
    if alert_days:
        fig.add_trace(go.Scatter(
            x=alert_days, y=alert_vals,
            mode="markers", name="Alert fired",
            marker=dict(symbol="triangle-up", size=11, color=_AMBER,
                        line=dict(color=_BG, width=1.5)),
            hovertemplate="◆ Alert — Day %{x}<extra></extra>",
        ))

    fig.add_hline(
        y=_THRESHOLD, line_dash="dot", line_color=_AMBER, line_width=1,
        annotation_text="threshold 0.25",
        annotation_position="top right",
        annotation_font_color=_AMBER,
        annotation_font_size=10,
    )

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        font=dict(family="IBM Plex Mono, monospace", color=_MUTED, size=11),
        margin=dict(l=8, r=8, t=12, b=8), height=260,
        xaxis=dict(title=dict(text="simulation day", font=dict(size=10)),
                   gridcolor=_BORDER, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="PSI score", font=dict(size=10)),
                   gridcolor=_BORDER, zeroline=False, tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig, use_container_width=True)
