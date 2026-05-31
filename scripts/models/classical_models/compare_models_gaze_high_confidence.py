import argparse
from pathlib import Path

from scripts.models.classical_models.compare_models import run_model_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on the gaze-supported high-confidence subset.")
    parser.add_argument(
        "--window-root",
        default=str(PROJECT_ROOT / "window_dataset_with_gaze"),
        help="Root folder containing merged baseline and gaze window datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "model_comparison_gaze_high_confidence"),
        help="Directory for model comparison outputs.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for tree-based models.")
    parser.add_argument("--models", default="all", help="Comma-separated model keys to run, or 'all'.")
    args = parser.parse_args()

    run_model_comparison(
        window_root=args.window_root,
        output_dir=args.output_dir,
        feature_set="gaze",
        use_high_confidence=True,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        models=args.models,
    )


if __name__ == "__main__":
    main()
