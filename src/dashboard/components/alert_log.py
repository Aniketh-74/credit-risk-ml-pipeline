"""Alert log table component.

Renders a table of all drift alerts that have fired, showing the timestamp,
drift score, whether retraining was triggered, and whether a new model was
promoted. Wired to the `retraining_alerts` PostgreSQL table in Phase 6.
"""

import pandas as pd
import streamlit as st


def render(alerts: list[dict]) -> None:
    """Render alert log table.

    Args:
        alerts: List of dicts with keys:
            - fired_at (str): ISO datetime string when the alert fired.
            - drift_score (float): PSI score that crossed the threshold.
            - retrain (bool): True if a retraining run was triggered.
            - promoted (bool): True if a new model was promoted after retraining.
    """
    if not alerts:
        st.info("No alerts have fired yet.")
        return

    df = pd.DataFrame(alerts)

    df = df.rename(
        columns={
            "fired_at": "Fired At",
            "drift_score": "Drift Score",
            "retrain": "Retrain Triggered",
            "promoted": "Model Promoted",
        }
    )

    # Convert booleans to readable labels
    df["Retrain Triggered"] = df["Retrain Triggered"].map(
        {True: "yes", False: "no"}
    )
    df["Model Promoted"] = df["Model Promoted"].map(
        {True: "yes", False: "no"}
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fired At": st.column_config.TextColumn("Fired At", width="medium"),
            "Drift Score": st.column_config.NumberColumn(
                "Drift Score", format="%.3f", width="small"
            ),
            "Retrain Triggered": st.column_config.TextColumn(
                "Retrain Triggered", width="small"
            ),
            "Model Promoted": st.column_config.TextColumn(
                "Model Promoted", width="small"
            ),
        },
    )
