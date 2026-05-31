import argparse
from pathlib import Path

from scripts.models.classical_models.compare_models import run_model_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_named_model_cli(
    *,
    description: str,
    model_key: str,
    feature_set: str,
    use_high_confidence: bool,
    default_window_root: str,
    default_output_dir: str,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-root",
        default=default_window_root,
        help="Root folder containing the relevant window datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory for model outputs.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for supported models.")
    args = parser.parse_args()

    summary_df = run_model_comparison(
        window_root=args.window_root,
        output_dir=args.output_dir,
        feature_set=feature_set,
        use_high_confidence=use_high_confidence,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        models=model_key,
    )
    print("\nComparison Summary:")
    print(summary_df.to_string(index=False))
