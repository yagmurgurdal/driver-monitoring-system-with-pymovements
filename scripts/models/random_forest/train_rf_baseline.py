import argparse
from pathlib import Path

from scripts.models.random_forest.train_random_forest import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description="Train the baseline Random Forest model.")
    parser.add_argument(
        "--window-root",
        default=str(PROJECT_ROOT / "window_dataset"),
        help="Root folder containing baseline window datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "random_forest_baseline"),
        help="Directory for training outputs.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=0, help="Maximum tree depth. 0 means None.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Random Forest.")
    args = parser.parse_args()

    run_training(
        window_root=args.window_root,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        feature_set="baseline",
        use_high_confidence=False,
    )


if __name__ == "__main__":
    main()
