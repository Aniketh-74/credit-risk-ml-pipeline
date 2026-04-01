"""CLI entrypoint for training the champion credit risk model.

Trains a LightGBM classifier on the Give Me Credit dataset using SMOTE for class
imbalance handling and logs all metrics, parameters, and artifacts to MLflow.

Usage:
    python scripts/train_champion.py --data data/raw/cs-training.csv
    python scripts/train_champion.py --data data/raw/cs-training.csv --mlflow-uri http://localhost:5001
"""
from __future__ import annotations

import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.training.train import run_training_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with data and mlflow_uri attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train the champion credit risk LightGBM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_champion.py
  python scripts/train_champion.py --data data/raw/cs-training.csv
  python scripts/train_champion.py --data data/raw/cs-training.csv --mlflow-uri http://localhost:5001
        """,
    )
    parser.add_argument(
        "--data",
        default="data/raw/cs-training.csv",
        help="Path to cs-training.csv (default: data/raw/cs-training.csv)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5001",
        help="MLflow tracking server URI (default: http://localhost:5001)",
    )
    return parser.parse_args()


def main() -> None:
    """Run training pipeline and print the resulting MLflow run ID."""
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}", file=sys.stderr)
        print("Download it with: kaggle competitions download -c GiveMeSomeCredit", file=sys.stderr)
        sys.exit(1)

    mlflow.set_tracking_uri(args.mlflow_uri)

    print(f"Training champion model on {args.data} ...")
    print(f"MLflow tracking URI: {args.mlflow_uri}")

    run_id = run_training_pipeline(csv_path=args.data, run_name="lgbm_smote_v1")

    print(f"\nTraining complete.")
    print(f"MLflow run ID: {run_id}")
    print(f"View run: {args.mlflow_uri}/#/experiments/1/runs/{run_id}")


if __name__ == "__main__":
    main()
