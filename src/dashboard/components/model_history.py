"""Model version history table component.

Renders a styled dataframe showing every model version that has been promoted,
its AUC score, promotion date, and current status (champion / challenger /
retired). Wired to the `model_versions` MLflow registry table in Phase 6.
"""

import pandas as pd
import streamlit as st


_STATUS_ICONS = {
    "champion": "champion",
    "challenger": "challenger",
    "retired": "retired",
}


def render(models: list[dict]) -> None:
    """Render model version history table.

    Args:
        models: List of dicts with keys:
            - version (str): Semantic version string (e.g. "v1.2.0").
            - auc (float): Area under ROC curve on hold-out test set.
            - promoted_at (str): ISO date string when model was promoted.
            - status (str): One of 'champion', 'challenger', 'retired'.
    """
    if not models:
        st.info("No model versions available.")
        return

    df = pd.DataFrame(models)

    # Friendly column labels
    df = df.rename(
        columns={
            "version": "Version",
            "auc": "AUC",
            "promoted_at": "Promoted",
            "status": "Status",
        }
    )

    # Highlight the champion row
    def _highlight_champion(row):
        if str(row["Status"]).lower() == "champion":
            return ["background-color: #e8f4f8; font-weight: 600"] * len(row)
        elif str(row["Status"]).lower() == "retired":
            return ["color: #999999"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_champion, axis=1).format({"AUC": "{:.3f}"})

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version": st.column_config.TextColumn("Version", width="small"),
            "AUC": st.column_config.NumberColumn("AUC", format="%.3f", width="small"),
            "Promoted": st.column_config.TextColumn("Promoted", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )
