import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


NORMAL_SOURCE_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned", "normal")
WINDOW_ROOT_DEFAULT = (
    r"D:\window\normal" if os.path.exists(r"D:\window") else os.path.join(os.getcwd(), "window_dataset", "normal")
)
REPORT_PATH_DEFAULT = os.path.join(os.getcwd(), "reports", "dataset", "normal_window_validation_report.xlsx")

WINDOW_SIZE = 90

NUMERIC_COLUMNS = [
    "frame",
    "time_sec",
    "face_detected",
    "yaw",
    "pitch",
    "roll",
    "pose_valid",
    "left_ear",
    "right_ear",
    "avg_ear",
    "ear_valid",
    "ear_suspicious",
]

CHECK_COLUMNS = [
    "window_start_frame",
    "window_end_frame",
    "window_start_time",
    "window_end_time",
    "closed_eye_frames",
    "total_frames_perclos",
    "frame_count_source",
    "perclos",
    "perclos_percent",
    "face_detect_ratio",
    "pose_valid_ratio",
    "ear_valid_ratio",
    "ear_suspicious_ratio",
    "mean_ear",
    "std_ear",
    "min_ear",
    "max_ear",
    "mean_abs_yaw",
    "std_yaw",
    "max_abs_yaw",
    "mean_abs_pitch",
    "std_pitch",
    "max_abs_pitch",
    "mean_abs_roll",
    "std_roll",
    "max_abs_roll",
    "is_usable",
]


@dataclass(frozen=True)
class ValidationConfig:
    window_size: int = WINDOW_SIZE
    min_face_ratio: float = 0.80
    min_pose_ratio: float = 0.80
    min_ear_ratio: float = 0.80
    tolerance: float = 1e-4


def normalize_value(value):
    if pd.isna(value):
        return None
    return float(value)


def values_match(a, b, tolerance: float) -> bool:
    a_val = normalize_value(a)
    b_val = normalize_value(b)
    if a_val is None and b_val is None:
        return True
    if a_val is None or b_val is None:
        return False
    return abs(a_val - b_val) <= tolerance


def safe_std(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(clean.std(ddof=1))


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)


def load_window_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in CHECK_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def recompute_window(window_df: pd.DataFrame, config: ValidationConfig) -> Dict:
    frame_count = len(window_df)
    face_detect_ratio = float((window_df["face_detected"] == 1).mean())
    pose_valid_ratio = float((window_df["pose_valid"] == 1).mean())
    ear_valid_ratio = float((window_df["ear_valid"] == 1).mean())
    if "ear_suspicious" in window_df.columns:
        ear_suspicious_ratio = float((window_df["ear_suspicious"] == 1).mean())
    else:
        ear_suspicious_ratio = None

    closed_eye_frames = int((window_df["avg_ear"] < 0.21).sum())
    perclos = float(closed_eye_frames / frame_count) if frame_count > 0 else None

    return {
        "window_start_frame": int(window_df["frame"].iloc[0]),
        "window_end_frame": int(window_df["frame"].iloc[-1]),
        "window_start_time": float(window_df["time_sec"].iloc[0]),
        "window_end_time": float(window_df["time_sec"].iloc[-1]),
        "closed_eye_frames": closed_eye_frames,
        "total_frames_perclos": frame_count,
        "frame_count_source": frame_count,
        "perclos": round(perclos, 4) if perclos is not None else None,
        "perclos_percent": round(perclos * 100.0, 2) if perclos is not None else None,
        "face_detect_ratio": round(face_detect_ratio, 4),
        "pose_valid_ratio": round(pose_valid_ratio, 4),
        "ear_valid_ratio": round(ear_valid_ratio, 4),
        "ear_suspicious_ratio": round(ear_suspicious_ratio, 4) if ear_suspicious_ratio is not None else None,
        "mean_ear": float(window_df["avg_ear"].mean()),
        "std_ear": safe_std(window_df["avg_ear"]),
        "min_ear": float(window_df["avg_ear"].min()),
        "max_ear": float(window_df["avg_ear"].max()),
        "mean_abs_yaw": float(window_df["yaw"].abs().mean()),
        "std_yaw": safe_std(window_df["yaw"]),
        "max_abs_yaw": float(window_df["yaw"].abs().max()),
        "mean_abs_pitch": float(window_df["pitch"].abs().mean()),
        "std_pitch": safe_std(window_df["pitch"]),
        "max_abs_pitch": float(window_df["pitch"].abs().max()),
        "mean_abs_roll": float(window_df["roll"].abs().mean()),
        "std_roll": safe_std(window_df["roll"]),
        "max_abs_roll": float(window_df["roll"].abs().max()),
        "is_usable": int(
            face_detect_ratio >= config.min_face_ratio
            and pose_valid_ratio >= config.min_pose_ratio
            and ear_valid_ratio >= config.min_ear_ratio
        ),
    }


def source_path_from_window(window_path: str, window_root: str, normal_source_root: str) -> str:
    relative_path = os.path.relpath(window_path, window_root)
    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        raise ValueError(f"Unexpected window path structure: {window_path}")
    modality = parts[0]
    window_name = parts[-1]
    file_stem = window_name.replace("_windows.xlsx", "")

    for source_group in ("drowsiness", "distraction"):
        candidate = os.path.join(
            normal_source_root,
            modality,
            source_group,
            f"{file_stem}_normal_frames.xlsx",
        )
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Matching normal source file not found for {window_path}")


def collect_window_files(window_root: str) -> List[str]:
    files: List[str] = []
    for root, _, file_names in os.walk(window_root):
        for file_name in file_names:
            if file_name.startswith("~$"):
                continue
            if file_name.lower().endswith("_windows.xlsx"):
                files.append(os.path.join(root, file_name))
    return sorted(files)


def validate_file(window_path: str, window_root: str, normal_source_root: str, config: ValidationConfig) -> Dict:
    source_path = source_path_from_window(window_path, window_root, normal_source_root)
    result = {
        "window_file": window_path,
        "source_file": source_path,
        "status": "exact_match",
        "window_row_count": "",
        "expected_row_count": "",
        "mismatch_count": 0,
        "first_problem": "",
    }

    window_df = load_window_table(window_path)
    source_df = load_table(source_path)

    expected_rows = len(source_df) // config.window_size if len(source_df) % config.window_size == 0 else None
    result["window_row_count"] = len(window_df)
    result["expected_row_count"] = expected_rows if expected_rows is not None else ""

    if expected_rows is None:
        result["status"] = "source_not_divisible"
        result["first_problem"] = f"Source row count {len(source_df)} is not divisible by {config.window_size}."
        return result

    if len(window_df) != expected_rows:
        result["status"] = "row_count_mismatch"
        result["first_problem"] = "Window row count does not match expected row count."
        return result

    mismatch_count = 0
    first_problem = ""

    for idx in range(expected_rows):
        source_window = source_df.iloc[idx * config.window_size : (idx + 1) * config.window_size].copy().reset_index(drop=True)
        expected = recompute_window(source_window, config)
        actual = window_df.iloc[idx]

        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if not values_match(actual_value, expected_value, config.tolerance):
                mismatch_count += 1
                if not first_problem:
                    first_problem = f"Window {idx + 1}, column '{key}': actual={actual_value}, expected={expected_value}"
                break

    result["mismatch_count"] = mismatch_count
    result["first_problem"] = first_problem
    if mismatch_count > 0:
        result["status"] = "value_mismatch"
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Validate normal-class window Excel files.")
    parser.add_argument("--window-root", default=WINDOW_ROOT_DEFAULT, help="Root folder containing normal *_windows.xlsx files.")
    parser.add_argument("--normal-source-root", default=NORMAL_SOURCE_ROOT_DEFAULT, help="Root folder containing *_normal_frames.xlsx files.")
    parser.add_argument("--report-path", default=REPORT_PATH_DEFAULT, help="Excel path for validation report.")
    parser.add_argument("--tolerance", type=float, default=ValidationConfig.tolerance, help="Numeric comparison tolerance.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ValidationConfig(tolerance=args.tolerance)
    window_files = collect_window_files(args.window_root)
    print(f"Normal window files found: {len(window_files)}")

    rows: List[Dict] = []
    for window_path in window_files:
        print(f"Validating: {os.path.basename(window_path)}")
        rows.append(
            validate_file(
                window_path=window_path,
                window_root=args.window_root,
                normal_source_root=args.normal_source_root,
                config=config,
            )
        )

    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    pd.DataFrame(rows).to_excel(args.report_path, index=False)
    print(f"Validation report: {args.report_path}")


if __name__ == "__main__":
    main()
