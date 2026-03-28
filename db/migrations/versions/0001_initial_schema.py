"""initial_schema

Creates the four core tables for the credit-risk-ml-pipeline:
predictions, outcomes, drift_scores, alerts.

This migration is the schema contract for all downstream phases.
Column names and types must not be changed without a follow-up migration.

Revision ID: 0001
Revises:
Create Date: 2026-03-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all four tables in dependency order.

    predictions is created first because outcomes has a FK into it.
    drift_scores and alerts have no FK dependencies.
    UUID primary keys use gen_random_uuid() which requires the
    pgcrypto extension (available by default in PostgreSQL 13+).
    """
    # --- predictions ---------------------------------------------------------
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "predicted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(10), nullable=False),
        sa.Column("path", sa.String(20), nullable=False),
        sa.Column("simulation_day", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "decision IN ('approved', 'denied')",
            name="ck_predictions_decision",
        ),
        sa.CheckConstraint(
            "path IN ('model', 'checkerboard')",
            name="ck_predictions_path",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- outcomes ------------------------------------------------------------
    op.create_table(
        "outcomes",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("prediction_id", sa.UUID(), nullable=False),
        sa.Column("actual_default", sa.Boolean(), nullable=False),
        sa.Column("predicted_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("outcome_received_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_outcomes_prediction_id",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id", name="uq_outcomes_prediction_id"),
    )

    # --- drift_scores --------------------------------------------------------
    op.create_table(
        "drift_scores",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "computed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("psi_score", sa.Float(), nullable=True),
        sa.Column("threshold_crossed", sa.Boolean(), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False),
        sa.Column("trial_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- alerts --------------------------------------------------------------
    op.create_table(
        "alerts",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "fired_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("retrain_run_id", sa.String(100), nullable=True),
        sa.Column("promoted", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop all four tables in reverse FK dependency order."""
    op.drop_table("alerts")
    op.drop_table("drift_scores")
    op.drop_table("outcomes")
    op.drop_table("predictions")
