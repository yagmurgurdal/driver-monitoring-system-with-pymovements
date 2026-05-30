import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


CSV_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned")
PERCLOS_ROOT_DEFAULT = os.path.join(os.getcwd(), "perclos")
WINDOW_ROOT_DEFAULT = (
    r"D:\window" if os.path.exists(r"D:\window") else os.path.join(os.getcwd(), "window_dataset")
)
REPORT_PATH_DEFAULT = os.path.join(os.getcwd(), "window_validation_report.xlsx")

LABELS = ("drowsiness", "distraction")
MODALITIES = ("IR", "RGB")

WINDOW_COLUMNS = [
    "window_id",
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

SOURCE_NUMERIC_COLUMNS = [
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

PERCLOS_COLUMNS = [
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "closed_eye_frames",
    "total_frames",
    "perclos",
    "perclos_percent",
]


@dataclass(frozen=True)
class ValidationConfig:
    min_face_ratio: float = 0.80
    min_pose_ratio: float = 0.80
    min_ear_ratio: float = 0.80
    tolerance: float = 1e-4


def infer_metadata_from_window_path(window_path: str, window_root: str) -> Dict[str, str]:
    relative_path = os.path.relpath(window_path, window_root)
    parts = relative_path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Unexpected window path structure: {window_path}")

    label = parts[0]
    modality = parts[1]
    file_name = parts[-1]

    if label not in LABELS:
        raise ValueError(f"Unknown label: {window_path}")
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {window_path}")

    stem = file_name.replace("_windows.xlsx", "")
    return {
        "label": label,
        "modality": modality,
        "file_stem": stem,
        "window_file": file_name,
    }


def load_source_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in SOURCE_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)


def load_perclos_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in PERCLOS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_window_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in WINDOW_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def safe_std(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(clean.std(ddof=1))


def normalize_value(value):
    if pd.isna(value):
        return None
    return float(value)


def values_match(a, b, tol: float) -> bool:
    a_norm = normalize_value(a)
    b_norm = normalize_value(b)
    if a_norm is None and b_norm is None:
        return True
    if a_norm is None or b_norm is None:
        return False
    return abs(a_norm - b_norm) <= tol


def recompute_window_features(window_df: pd.DataFrame, perclos_row: pd.Series, config: ValidationConfig) -> Dict:
    face_detect_ratio = float((window_df["face_detected"] == 1).mean())
    pose_valid_ratio = float((window_df["pose_valid"] == 1).mean())
    ear_valid_ratio = float((window_df["ear_valid"] == 1).mean())
    if "ear_suspicious" in window_df.columns:
        ear_suspicious_ratio = float((window_df["ear_suspicious"] == 1).mean())
    else:
        ear_suspicious_ratio = None

    return {
        "window_start_frame": int(perclos_row["start_frame"]),
        "window_end_frame": int(perclos_row["end_frame"]),
        "window_start_time": float(perclos_row["start_time"]),
        "window_end_time": float(perclos_row["end_time"]),
        "closed_eye_frames": int(perclos_row["closed_eye_frames"]),
        "total_frames_perclos": int(perclos_row["total_frames"]),
        "frame_count_source": int(len(window_df)),
        "perclos": float(perclos_row["perclos"]),
        "perclos_percent": float(perclos_row["perclos_percent"]),
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


def validate_file(
    window_path: str,
    window_root: str,
    csv_root: str,
    perclos_root: str,
    config: ValidationConfig,
) -> Dict:
    metadata = infer_metadata_from_window_path(window_path, window_root)
    source_path = os.path.join(csv_root, metadata["label"], metadata["modality"], f"{metadata['file_stem']}.xlsx")
    perclos_path = os.path.join(perclos_root, metadata["label"], metadata["modality"], f"{metadata['file_stem']}_perclos.xlsx")

    result = {
        "window_file": window_path,
        "source_file": source_path,
        "perclos_file": perclos_path,
        "status": "exact_match",
        "window_row_count": "",
        "perclos_row_count": "",
        "mismatch_count": 0,
        "first_problem": "",
    }

    if not os.path.exists(source_path):
        result["status"] = "source_missing"
        return result
    if not os.path.exists(perclos_path):
        result["status"] = "perclos_missing"
        return result

    window_df = load_window_table(window_path)
    source_df = load_source_table(source_path)
    perclos_df = load_perclos_table(perclos_path)

    result["window_row_count"] = len(window_df)
    result["perclos_row_count"] = len(perclos_df)

    if len(window_df) != len(perclos_df):
        result["status"] = "row_count_mismatch"
        result["first_problem"] = "Window row count does not match PERCLOS row count."
        return result

    mismatch_count = 0
    first_problem = ""

    for idx, (_, window_row) in enumerate(window_df.iterrows()):
        perclos_row = perclos_df.iloc[idx]
        start_frame = int(perclos_row["start_frame"])
        end_frame = int(perclos_row["end_frame"])
        source_window_df = source_df[
            (source_df["frame"] >= start_frame) & (source_df["frame"] <= end_frame)
        ].copy()

        if source_window_df.empty:
            mismatch_count += 1
            if not first_problem:
                first_problem = f"Window {idx + 1}: source frame range returned no rows."
            continue

        expected = recompute_window_features(source_window_df, perclos_row, config)
        for key, expected_value in expected.items():
            actual_value = window_row.get(key)
            if not values_match(actual_value, expected_value, config.tolerance):
                mismatch_count += 1
                if not first_problem:
                    first_problem = (
                        f"Window {idx + 1}, column '{key}': actual={actual_value}, "
                        f"expected={expected_value}"
                    )
                break

    result["mismatch_count"] = mismatch_count
    result["first_problem"] = first_problem
    if mismatch_count > 0:
        result["status"] = "value_mismatch"

    return result


def collect_window_files(window_root: str) -> List[str]:
    files: List[str] = []
    for label in LABELS:
        for modality in MODALITIES:
            folder = os.path.join(window_root, label, modality)
            if not os.path.exists(folder):
                continue
            for file_name in os.listdir(folder):
                if file_name.startswith("~$"):
                    continue
                if file_name.lower().endswith("_windows.xlsx"):
                    files.append(os.path.join(folder, file_name))
    return sorted(files)


def write_report(rows: List[Dict], report_path: str):
    pd.DataFrame(rows).to_excel(report_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate per-video window output Excel files.")
    parser.add_argument("--window-root", default=WINDOW_ROOT_DEFAULT, help="Root folder containing *_windows.xlsx files.")
    parser.add_argument("--csv-root", default=CSV_ROOT_DEFAULT, help="Root folder of csvcleaned files.")
    parser.add_argument("--perclos-root", default=PERCLOS_ROOT_DEFAULT, help="Root folder of perclos files.")
    parser.add_argument("--report-path", default=REPORT_PATH_DEFAULT, help="Excel path for validation report.")
    parser.add_argument("--tolerance", type=float, default=ValidationConfig.tolerance, help="Numeric comparison tolerance.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ValidationConfig(tolerance=args.tolerance)
    window_files = collect_window_files(args.window_root)
    print(f"Window files found: {len(window_files)}")

    rows: List[Dict] = []
    for window_path in window_files:
        print(f"Validating: {os.path.basename(window_path)}")
        rows.append(
            validate_file(
                window_path=window_path,
                window_root=args.window_root,
                csv_root=args.csv_root,
                perclos_root=args.perclos_root,
                config=config,
            )
        )

    write_report(rows, args.report_path)
    print(f"Validation report: {args.report_path}")


if __name__ == "__main__":
    main()
