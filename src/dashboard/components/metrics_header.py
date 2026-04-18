"""Top-level KPI metric cards — custom HTML cards, Neural Terminal theme."""

import streamlit as st


def render(
    current_score: float,
    cbpdd_pvalue: float,
    model_version: str,
    alert_count: int,
    sim_day: int,
) -> None:
    psi_threshold = 0.25
    drift_status = current_score >= psi_threshold
    cbpdd_alert = cbpdd_pvalue < 0.05

    cards = [
        {
            "label": "PSI Score",
            "value": f"{current_score:.3f}",
            "sub": f"threshold 0.25",
            "badge": "DRIFT" if drift_status else "STABLE",
            "badge_color": "#FF9500" if drift_status else "#3FB950",
            "badge_bg": "#1A0F00" if drift_status else "#0D2018",
            "accent": "#FF9500" if drift_status else "#3FB950",
            "help": "Population Stability Index. >0.25 = significant distribution shift.",
        },
        {
            "label": "CB-PDD p-value",
            "value": f"{cbpdd_pvalue:.4f}",
            "sub": "α = 0.05 threshold",
            "badge": "DRIFT" if cbpdd_alert else "NULL",
            "badge_color": "#FF9500" if cbpdd_alert else "#00D4FF",
            "badge_bg": "#1A0F00" if cbpdd_alert else "#001A22",
            "accent": "#FF9500" if cbpdd_alert else "#00D4FF",
            "help": "Performative drift p-value. <0.05 = model-induced distribution shift confirmed.",
        },
        {
            "label": "Champion Model",
            "value": model_version,
            "sub": "MLflow registry",
            "badge": "LIVE",
            "badge_color": "#3FB950",
            "badge_bg": "#0D2018",
            "accent": "#3FB950",
            "help": "Active champion model alias in MLflow.",
        },
        {
            "label": "Drift Alerts",
            "value": str(alert_count),
            "sub": f"{'triggered retraining' if alert_count else 'no alerts fired'}",
            "badge": "ALERTS" if alert_count else "CLEAR",
            "badge_color": "#FF9500" if alert_count else "#6E7681",
            "badge_bg": "#1A0F00" if alert_count else "#161B22",
            "accent": "#FF9500" if alert_count else "#6E7681",
            "help": "CB-PDD threshold crossings triggering retraining.",
        },
        {
            "label": "Window Days",
            "value": f"{sim_day}",
            "sub": "30-day rolling",
            "badge": "DAY",
            "badge_color": "#6E7681",
            "badge_bg": "#161B22",
            "accent": "#6E7681",
            "help": "Days covered in the current drift detection window.",
        },
    ]

    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#161B22;
                    border:1px solid #21262D;
                    border-top:2px solid {card['accent']};
                    border-radius:6px;
                    padding:0.9rem 1rem 0.75rem;
                    position:relative;
                    transition:border-color 0.2s;
                ">
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:flex-start;
                        margin-bottom:0.5rem;
                    ">
                        <span style="
                            font-family:'IBM Plex Mono',monospace;
                            font-size:0.62rem;
                            color:#6E7681;
                            text-transform:uppercase;
                            letter-spacing:0.1em;
                        ">{card['label']}</span>
                        <span style="
                            font-family:'IBM Plex Mono',monospace;
                            font-size:0.58rem;
                            color:{card['badge_color']};
                            background:{card['badge_bg']};
                            border:1px solid {card['badge_color']}33;
                            border-radius:2px;
                            padding:1px 5px;
                            letter-spacing:0.06em;
                        ">{card['badge']}</span>
                    </div>
                    <div style="
                        font-family:'IBM Plex Mono',monospace;
                        font-size:1.65rem;
                        font-weight:600;
                        color:#E6EDF3;
                        line-height:1.1;
                        margin-bottom:0.25rem;
                    ">{card['value']}</div>
                    <div style="
                        font-family:'IBM Plex Mono',monospace;
                        font-size:0.62rem;
                        color:#6E7681;
                        letter-spacing:0.03em;
                    ">{card['sub']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
