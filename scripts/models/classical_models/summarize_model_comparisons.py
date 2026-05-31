import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "results"
OUTPUT_PATH_DEFAULT = RESULTS_ROOT_DEFAULT / "model_comparison_master_summary.xlsx"

COMPARISON_DIRS: Dict[str, str] = {
    "baseline": "model_comparison_baseline",
    "high_confidence": "model_comparison_high_confidence",
    "gaze_baseline": "model_comparison_gaze_baseline",
    "gaze_high_confidence": "model_comparison_gaze_high_confidence",
}


def load_summary(results_root: Path, scenario_key: str, directory_name: str) -> pd.DataFrame | None:
    path = results_root / directory_name / "model_comparison_summary.xlsx"
    if not path.exists():
        return None

    df = pd.read_excel(path)
    df.insert(0, "scenario", scenario_key)
    return df


def build_master_summary(results_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for scenario_key, directory_name in COMPARISON_DIRS.items():
        df = load_summary(results_root, scenario_key, directory_name)
        if df is not None:
            rows.append(df)

    if not rows:
        raise FileNotFoundError("No model comparison summary files were found.")

    master_df = pd.concat(rows, ignore_index=True)
    return master_df.sort_values(
        ["scenario", "accuracy", "macro_f1", "weighted_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Combine all model comparison outputs into one master summary workbook.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT_DEFAULT,
        help="Root directory containing model comparison result folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH_DEFAULT,
        help="Excel output path for the combined summary.",
    )
    args = parser.parse_args()

    master_df = build_master_summary(args.results_root)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_excel(args.output_path, index=False)
    print(f"Master summary written to: {args.output_path}")
    print(master_df.to_string(index=False))


if __name__ == "__main__":
    main()
