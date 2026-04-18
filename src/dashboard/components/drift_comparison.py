"""CB-PDD vs PSI comparison chart — hero visual, Neural Terminal dark theme."""

import plotly.graph_objects as go
import streamlit as st

_BG = "#0D1117"
_CARD = "#161B22"
_BORDER = "#21262D"
_CYAN = "#00D4FF"
_AMBER = "#FF9500"
_MUTED = "#6E7681"
_TEXT = "#C9D1D9"


def render(scores: list[dict]) -> None:
    if not scores:
        st.info("No comparison data available.")
        return

    days = [r["day"] for r in scores]
    cbpdd = [r.get("score", 0.5) for r in scores]
    psi = [r.get("psi_score", 0.0) for r in scores]

    fig = go.Figure()

    # PSI area (right axis)
    fig.add_trace(go.Scatter(
        x=days, y=psi,
        fill="tozeroy",
        fillcolor="rgba(255,149,0,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        yaxis="y2", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=psi,
        mode="lines",
        name="PSI score",
        line=dict(color=_AMBER, width=1.8, dash="dot"),
        yaxis="y2",
        hovertemplate="Day %{x} &nbsp; PSI: <b>%{y:.3f}</b><extra></extra>",
    ))

    # CB-PDD area (left axis, inverted — low p = more drift)
    fig.add_trace(go.Scatter(
        x=days, y=cbpdd,
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cbpdd,
        mode="lines+markers",
        name="CB-PDD p-value",
        line=dict(color=_CYAN, width=2.2),
        marker=dict(size=3.5, color=_CYAN),
        hovertemplate="Day %{x} &nbsp; p-value: <b>%{y:.4f}</b><extra></extra>",
    ))

    # Detection threshold α=0.05
    fig.add_hline(
        y=0.05, line_dash="dash", line_color=_CYAN, line_width=1,
        annotation_text="α = 0.05",
        annotation_position="top left",
        annotation_font_color=_CYAN,
        annotation_font_size=10,
    )

    # Annotate the first detection point
    detect_day = next((r["day"] for r in scores if r.get("score", 1) < 0.05), None)
    if detect_day:
        detect_val = next(r.get("score", 0) for r in scores if r["day"] == detect_day)
        fig.add_annotation(
            x=detect_day, y=detect_val,
            text="first detection",
            showarrow=True, arrowhead=2,
            arrowcolor=_CYAN, arrowsize=1, arrowwidth=1.5,
            ax=40, ay=-30,
            font=dict(color=_CYAN, size=10, family="IBM Plex Mono, monospace"),
            bgcolor="#161B22",
            bordercolor=_CYAN,
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        font=dict(family="IBM Plex Mono, monospace", color=_MUTED, size=11),
        margin=dict(l=8, r=60, t=12, b=8), height=300,
        xaxis=dict(
            title=dict(text="simulation day", font=dict(size=10)),
            gridcolor=_BORDER, zeroline=False,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(text="CB-PDD p-value ↓ more drift", font=dict(size=10, color=_CYAN)),
            gridcolor=_BORDER, zeroline=False,
            tickfont=dict(size=10, color=_CYAN),
            range=[0, 1.05],
        ),
        yaxis2=dict(
            title=dict(text="PSI ↑ more drift", font=dict(size=10, color=_AMBER)),
            overlaying="y", side="right",
            zeroline=False, showgrid=False,
            tickfont=dict(size=10, color=_AMBER),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("why two metrics?", expanded=False):
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
            'color:#8B949E;line-height:1.7;">'
            '<span style="color:#00D4FF">CB-PDD</span> uses a controlled checkerboard '
            'prediction assignment to detect the specific density-change signature of '
            'denied applicants returning with nudged scores — the <em>performative feedback loop</em>.<br><br>'
            '<span style="color:#FF9500">PSI</span> measures raw distribution shift. '
            'It sees the same signal but cannot tell whether the model caused it.'
            '</div>',
            unsafe_allow_html=True,
        )
