import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOW_ROOT = Path(r"D:\window") if Path(r"D:\window").exists() else PROJECT_ROOT / "window_dataset"
DEFAULT_GAZE_WINDOW_ROOT = PROJECT_ROOT / "window_dataset_with_gaze"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_DATASET_REPORTS_ROOT = PROJECT_ROOT / "reports" / "dataset"


def run_step(title: str, command: list[str], cwd: Path):
    print(f"\n=== {title} ===")
    print("Command:", " ".join(str(part) for part in command))
    sys.stdout.flush()
    subprocess.run(command, cwd=str(cwd), check=True)


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_comparison_summary(results_root: Path):
    rows = []
    for model_name, metrics_path in [
        ("baseline", results_root / "random_forest_baseline" / "metrics.json"),
        ("high_confidence", results_root / "random_forest_high_confidence" / "metrics.json"),
        ("gaze_baseline", results_root / "random_forest_gaze_baseline" / "metrics.json"),
        ("gaze_high_confidence", results_root / "random_forest_gaze_high_confidence" / "metrics.json"),
    ]:
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "train_rows": metrics.get("train_rows"),
                "test_rows": metrics.get("test_rows"),
                "train_groups": metrics.get("train_groups"),
                "test_groups": metrics.get("test_groups"),
            }
        )

    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_path = results_root / "random_forest_comparison.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"\nComparison summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the current bitirme projesi baseline pipeline end-to-end."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess steps.",
    )
    parser.add_argument(
        "--window-root",
        default=str(DEFAULT_WINDOW_ROOT),
        help="Root folder for per-video window outputs.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root folder for model results.",
    )
    parser.add_argument(
        "--gaze-window-root",
        default=str(DEFAULT_GAZE_WINDOW_ROOT),
        help="Root folder for merged window datasets with gaze features.",
    )
    parser.add_argument(
        "--skip-window-build",
        action="store_true",
        help="Skip rebuilding drowsiness/distraction window files.",
    )
    parser.add_argument(
        "--skip-window-validate",
        action="store_true",
        help="Skip validating drowsiness/distraction window files.",
    )
    parser.add_argument(
        "--skip-normal-build",
        action="store_true",
        help="Skip rebuilding normal window files.",
    )
    parser.add_argument(
        "--skip-normal-validate",
        action="store_true",
        help="Skip validating normal window files.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Random Forest training runs.",
    )
    parser.add_argument(
        "--skip-gaze-train",
        action="store_true",
        help="Skip Random Forest training runs on merged gaze features.",
    )
    parser.add_argument(
        "--run-model-comparison",
        action="store_true",
        help="Run extended multi-model comparison experiments after Random Forest training.",
    )
    args = parser.parse_args()

    python_exe = args.python
    window_root = Path(args.window_root)
    gaze_window_root = Path(args.gaze_window_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_window_build:
        run_step(
            "Build Drowsiness/Distraction Windows",
            [
                python_exe,
                "-m",
                "scripts.dataset.build_window_dataset",
                "--output-root",
                str(window_root),
                "--summary-path",
                str(DEFAULT_DATASET_REPORTS_ROOT / "window_dataset_summary.xlsx"),
            ],
            PROJECT_ROOT,
        )

    if not args.skip_window_validate:
        run_step(
            "Validate Drowsiness/Distraction Windows",
            [
                python_exe,
                "-m",
                "scripts.dataset.validate_window_outputs",
                "--window-root",
                str(window_root),
                "--report-path",
                str(DEFAULT_DATASET_REPORTS_ROOT / "window_validation_report.xlsx"),
            ],
            PROJECT_ROOT,
        )

    if not args.skip_normal_build:
        run_step(
            "Build Normal Windows",
            [
                python_exe,
                "-m",
                "scripts.dataset.build_normal_window_dataset",
                "--output-root",
                str(window_root / "normal"),
                "--summary-path",
                str(DEFAULT_DATASET_REPORTS_ROOT / "normal_window_dataset_summary.xlsx"),
            ],
            PROJECT_ROOT,
        )

    if not args.skip_normal_validate:
        run_step(
            "Validate Normal Windows",
            [
                python_exe,
                "-m",
                "scripts.dataset.validate_normal_window_outputs",
                "--window-root",
                str(window_root / "normal"),
                "--report-path",
                str(DEFAULT_DATASET_REPORTS_ROOT / "normal_window_validation_report.xlsx"),
            ],
            PROJECT_ROOT,
        )

    if not args.skip_train:
        run_step(
            "Train Random Forest Baseline",
            [
                python_exe,
                "-m",
                "scripts.models.random_forest.train_random_forest",
                "--window-root",
                str(window_root),
                "--output-dir",
                str(results_root / "random_forest_baseline"),
                "--n-jobs",
                "1",
            ],
            PROJECT_ROOT,
        )

        run_step(
            "Train Random Forest High Confidence",
            [
                python_exe,
                "-m",
                "scripts.models.random_forest.train_random_forest",
                "--window-root",
                str(window_root),
                "--output-dir",
                str(results_root / "random_forest_high_confidence"),
                "--n-jobs",
                "1",
                "--use-high-confidence",
            ],
            PROJECT_ROOT,
        )

        if not args.skip_gaze_train:
            run_step(
                "Train Random Forest Gaze Baseline",
                [
                    python_exe,
                    "-m",
                    "scripts.models.random_forest.train_random_forest",
                    "--window-root",
                    str(gaze_window_root),
                    "--output-dir",
                    str(results_root / "random_forest_gaze_baseline"),
                    "--n-jobs",
                    "1",
                    "--feature-set",
                    "gaze",
                ],
                PROJECT_ROOT,
            )

            run_step(
                "Train Random Forest Gaze High Confidence",
                [
                    python_exe,
                    "-m",
                    "scripts.models.random_forest.train_random_forest",
                    "--window-root",
                    str(gaze_window_root),
                    "--output-dir",
                    str(results_root / "random_forest_gaze_high_confidence"),
                    "--n-jobs",
                    "1",
                    "--feature-set",
                    "gaze",
                    "--use-high-confidence",
                ],
                PROJECT_ROOT,
            )

        if args.run_model_comparison:
            run_step(
                "Compare Baseline Models",
                [
                    python_exe,
                    "-m",
                    "scripts.models.classical_models.compare_models_baseline",
                ],
                PROJECT_ROOT,
            )

            run_step(
                "Compare High-Confidence Models",
                [
                    python_exe,
                    "-m",
                    "scripts.models.classical_models.compare_models_high_confidence",
                ],
                PROJECT_ROOT,
            )

            if not args.skip_gaze_train:
                run_step(
                    "Compare Gaze Baseline Models",
                    [
                        python_exe,
                        "-m",
                        "scripts.models.classical_models.compare_models_gaze_baseline",
                    ],
                    PROJECT_ROOT,
                )

                run_step(
                    "Compare Gaze High-Confidence Models",
                    [
                        python_exe,
                        "-m",
                        "scripts.models.classical_models.compare_models_gaze_high_confidence",
                    ],
                    PROJECT_ROOT,
                )

        write_comparison_summary(results_root)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
