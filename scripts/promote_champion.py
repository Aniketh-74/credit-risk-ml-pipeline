"""CLI entrypoint for promoting a training run to @champion in MLflow Registry.

Usage:
    python scripts/promote_champion.py --run-id <run_id>
    python scripts/promote_champion.py --run-id <run_id> --mlflow-uri http://localhost:5001
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.training.promote import register_and_promote, get_champion_auc, load_champion


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Promote a training run to @champion alias in MLflow Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/promote_champion.py --run-id abc123def456
  python scripts/promote_champion.py --run-id abc123def456 --mlflow-uri http://localhost:5001
        """,
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID to promote")
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5001",
        help="MLflow tracking server URI (default: http://localhost:5001)",
    )
    return parser.parse_args()


def main() -> None:
    """Register and promote a run to @champion, then verify the alias resolves."""
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)

    version = register_and_promote(args.run_id)
    auc = get_champion_auc()

    try:
        load_champion()
        print(f"Champion promoted: version={version}, AUC={auc:.4f}")
        print(f"Verify in MLflow UI: {args.mlflow_uri}/#/models/credit-risk-model")
    except Exception as e:
        print(f"ERROR: Champion alias did not resolve: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
