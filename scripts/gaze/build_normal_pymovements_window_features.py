import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.gaze.build_pymovements_window_features import (
    WindowConfig,
    compute_window_features,
    estimate_expected_frame_step,
    estimate_sample_period_sec,
    load_input_csv,
)


INPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_input", "normal")
OUTPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_dataset_normal")
SUMMARY_PATH_DEFAULT = os.path.join(os.getcwd(), "normal_pymovements_window_summary_aligned.xlsx")
WINDOW_SIZE = 90


@dataclass(frozen=True)
class NormalGazeWindowConfig:
    output_window_suffix: str = "_normal_gaze_windows.xlsx"
    window_size: int = WINDOW_SIZE
    max_files: int = 0
    feature_config: WindowConfig = WindowConfig()


def infer_source_group(file_stem: str) -> str:
    lower_name = file_stem.lower()
    if lower_name.startswith("distraction"):
        return "distraction"
    if lower_name.startswith("drowsiness"):
        return "drowsiness"
    return "unknown"


def discover_input_files(input_root: str, max_files: int) -> List[str]:
    root = Path(input_root)
    if not root.exists():
        return []

    files = sorted(str(path) for path in root.rglob("*.csv"))
    if max_files > 0:
        files = files[:max_files]
    return files


def parse_metadata(path: str, input_root: str) -> Dict[str, str]:
    relative_path = os.path.relpath(path, input_root)
    parts = Path(relative_path).parts

    modality = parts[0] if len(parts) >= 1 else "unknown"
    file_name = os.path.basename(path)
    raw_stem = Path(file_name).stem
    file_stem = raw_stem.replace("_normal_pymovements_input", "")

    return {
        "relative_path": relative_path,
        "label": "normal",
        "modality": modality,
        "file_name": file_name,
        "file_stem": file_stem,
        "source_stem": file_stem,
        "source_group": infer_source_group(file_stem),
    }


def build_output_path(metadata: Dict[str, str], output_root: str, suffix: str) -> str:
    output_dir = os.path.join(output_root, metadata["modality"])
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{metadata['file_stem']}{suffix}")


def build_windows_for_file(path: str, input_root: str, output_root: str, config: NormalGazeWindowConfig) -> Dict:
    metadata = parse_metadata(path, input_root)
    df = load_input_csv(path)

    if df.empty:
        raise ValueError("Input file has no rows.")

    remainder = len(df) % config.window_size
    if remainder != 0:
        raise ValueError(
            f"{metadata['file_name']} row count {len(df)} is not divisible by normal window size {config.window_size}."
        )

    sample_period_sec = estimate_sample_period_sec(df)
    expected_frame_step = estimate_expected_frame_step(df)

    rows: List[Dict] = []
    for idx in range(0, len(df), config.window_size):
        window_df = df.iloc[idx : idx + config.window_size].copy().reset_index(drop=True)
        row = compute_window_features(
            window_df=window_df,
            window_id=(idx // config.window_size) + 1,
            start_time=float(window_df["time_sec"].iloc[0]),
            end_time=float(window_df["time_sec"].iloc[-1]),
            metadata=metadata,
            config=config.feature_config,
            sample_period_sec=sample_period_sec,
            expected_frame_step=expected_frame_step,
        )
        row["file_stem"] = metadata["file_stem"]
        row["source_group"] = metadata["source_group"]
        rows.append(row)

    output_path = build_output_path(metadata, output_root, config.output_window_suffix)
    per_file_df = pd.DataFrame(rows)
    per_file_df.to_excel(output_path, index=False)

    return {
        "label": "normal",
        "modality": metadata["modality"],
        "source_group": metadata["source_group"],
        "source_file": metadata["file_name"],
        "file_stem": metadata["file_stem"],
        "output_file": output_path,
        "window_count": len(per_file_df),
        "usable_window_count": int(per_file_df["usable_window"].fillna(0).sum()),
        "mean_valid_ratio": round(float(per_file_df["gaze_valid_ratio"].mean()), 4),
    }


def build_dataset(input_root: str, output_root: str, summary_path: str, config: NormalGazeWindowConfig):
    input_files = discover_input_files(input_root, config.max_files)
    if not input_files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    os.makedirs(output_root, exist_ok=True)

    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for path in input_files:
        metadata = parse_metadata(path, input_root)
        print(f"Processing normal/{metadata['modality']}: {metadata['file_name']}")
        try:
            summary_rows.append(build_windows_for_file(path, input_root, output_root, config))
        except Exception as exc:
            error_rows.append(
                {
                    "label": "normal",
                    "modality": metadata["modality"],
                    "source_file": metadata["file_name"],
                    "error": str(exc),
                }
            )
            print(f"  [ERROR] {exc}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
        print(f"Summary written to: {summary_path}")

    if error_rows:
        error_path = os.path.join(output_root, "normal_gaze_window_errors.xlsx")
        pd.DataFrame(error_rows).to_excel(error_path, index=False)
        print(f"Errors written to: {error_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build normal-class PyMovements gaze windows aligned with the baseline normal windowing logic."
    )
    parser.add_argument("--input-root", default=INPUT_ROOT_DEFAULT, help="Root folder containing normal PyMovements input CSV files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT, help="Output root for aligned normal gaze windows.")
    parser.add_argument("--summary-path", default=SUMMARY_PATH_DEFAULT, help="Excel path for summary output.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit the number of files for quick testing. 0 means no limit.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Frames per normal window. Default is 90.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.50, help="Minimum valid gaze ratio for a usable window.")
    parser.add_argument("--idt-min-duration-ms", type=int, default=100, help="Minimum I-DT fixation duration in milliseconds.")
    parser.add_argument("--idt-dispersion-threshold", type=float, default=1.0, help="I-DT dispersion threshold.")
    parser.add_argument("--ivt-min-duration-ms", type=int, default=100, help="Minimum I-VT fixation duration in milliseconds.")
    parser.add_argument("--ivt-velocity-threshold", type=float, default=20.0, help="I-VT low-velocity fixation threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_config = WindowConfig(
        min_valid_ratio=float(args.min_valid_ratio),
        idt_min_duration_ms=int(args.idt_min_duration_ms),
        idt_dispersion_threshold=float(args.idt_dispersion_threshold),
        ivt_min_duration_ms=int(args.ivt_min_duration_ms),
        ivt_velocity_threshold=float(args.ivt_velocity_threshold),
    )
    config = NormalGazeWindowConfig(
        window_size=max(1, int(args.window_size)),
        max_files=max(0, int(args.max_files)),
        feature_config=feature_config,
    )
    build_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        summary_path=args.summary_path,
        config=config,
    )


if __name__ == "__main__":
    main()
