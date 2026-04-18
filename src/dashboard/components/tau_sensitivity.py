"""CB-PDD tau sensitivity chart — Neural Terminal dark theme."""

import plotly.graph_objects as go
import streamlit as st

_CARD = "#161B22"
_BORDER = "#21262D"
_CYAN = "#00D4FF"
_MUTED = "#6E7681"
_BG = "#0D1117"

_TAU_RESULTS = [
    {"tau": "500",  "day": 6,  "risk": "higher FP risk",   "color": "#F85149"},
    {"tau": "1000", "day": 14, "risk": "balanced · default", "color": "#00D4FF"},
    {"tau": "2000", "day": 28, "risk": "lowest FP risk",   "color": "#3FB950"},
]


def render() -> None:
    taus = [r["tau"] for r in _TAU_RESULTS]
    days = [r["day"] for r in _TAU_RESULTS]
    colors = [r["color"] for r in _TAU_RESULTS]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=taus, y=days,
        marker_color=colors,
        marker_line_color=_BORDER,
        marker_line_width=1,
        text=[f"Day {d}" for d in days],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#C9D1D9"),
        hovertemplate="τ = %{x}<br>First detection: <b>Day %{y}</b><extra></extra>",
        width=0.4,
    ))

    for i, r in enumerate(_TAU_RESULTS):
        fig.add_annotation(
            x=r["tau"], y=-3,
            text=f'<span style="color:{r["color"]}">{r["risk"]}</span>',
            showarrow=False,
            font=dict(family="IBM Plex Mono, monospace", size=9, color=r["color"]),
            yref="y",
        )

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        font=dict(family="IBM Plex Mono, monospace", color=_MUTED, size=11),
        margin=dict(l=8, r=8, t=12, b=40), height=260,
        xaxis=dict(
            title=dict(text="trial length τ", font=dict(size=10)),
            gridcolor=_BORDER, zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="first detection (day)", font=dict(size=10)),
            gridcolor=_BORDER, zeroline=False,
            range=[0, 36],
            tickfont=dict(size=10),
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("how to read this", expanded=False):
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
            'color:#8B949E;line-height:1.7;">'
            'Each bar = days until CB-PDD first confirmed drift at that τ setting '
            '(n_per_day=1000, 30-day denial loop simulation, Phase 2 smoke test).<br><br>'
            '<span style="color:#F85149">τ=500</span> — fast but 2×500 samples before alarm<br>'
            '<span style="color:#00D4FF">τ=1000</span> — default: Day 14, 2000 samples confirmed<br>'
            '<span style="color:#3FB950">τ=2000</span> — conservative: Day 28, lowest false-positive risk'
            '</div>',
            unsafe_allow_html=True,
        )
