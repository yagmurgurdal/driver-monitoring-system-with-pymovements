import argparse
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


CSV_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned")
PERCLOS_ROOT_DEFAULT = os.path.join(os.getcwd(), "perclos")
OUTPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "window_dataset")
SUMMARY_PATH_DEFAULT = os.path.join(os.getcwd(), "reports", "dataset", "window_dataset_summary.xlsx")

LABELS = ("drowsiness", "distraction")
MODALITIES = ("IR", "RGB")

SOURCE_REQUIRED_COLUMNS = [
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

PERCLOS_REQUIRED_COLUMNS = [
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
class WindowDatasetConfig:
    min_face_ratio: float = 0.80
    min_pose_ratio: float = 0.80
    min_ear_ratio: float = 0.80
    ignore_temp_excel_files: bool = True


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


def infer_modality_and_label(source_path: str, csv_root: str) -> Dict[str, str]:
    relative_path = os.path.relpath(source_path, csv_root)
    parts = relative_path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Unexpected source path structure: {source_path}")

    label = parts[0]
    modality = parts[1]

    if label not in LABELS:
        raise ValueError(f"Unknown label in path: {source_path}")
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality in path: {source_path}")

    return {
        "label": label,
        "modality": modality,
        "relative_path": relative_path,
        "file_name": os.path.basename(source_path),
        "file_stem": os.path.splitext(os.path.basename(source_path))[0],
    }


def build_perclos_path(source_path: str, csv_root: str, perclos_root: str) -> str:
    metadata = infer_modality_and_label(source_path, csv_root)
    perclos_name = f"{metadata['file_stem']}_perclos.xlsx"
    return os.path.join(
        perclos_root,
        metadata["label"],
        metadata["modality"],
        perclos_name,
    )


def build_output_path(source_path: str, csv_root: str, output_root: str) -> str:
    metadata = infer_modality_and_label(source_path, csv_root)
    output_dir = os.path.join(output_root, metadata["label"], metadata["modality"])
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{metadata['file_stem']}_windows.xlsx")


def load_source_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    missing_columns = [col for col in SOURCE_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{os.path.basename(path)} missing source columns: {missing_text}")

    for col in SOURCE_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)
    return df


def load_perclos_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    missing_columns = [col for col in PERCLOS_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{os.path.basename(path)} missing perclos columns: {missing_text}")

    for col in PERCLOS_REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["start_frame", "end_frame"]).reset_index(drop=True)
    return df


def series_std(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(clean.std(ddof=1))


def compute_window_features(
    window_df: pd.DataFrame,
    perclos_row: pd.Series,
    metadata: Dict[str, str],
    window_id: int,
    config: WindowDatasetConfig,
) -> Dict:
    frame_count = len(window_df)
    if frame_count == 0:
        raise ValueError("Window has no source frames.")

    face_detect_ratio = float((window_df["face_detected"] == 1).mean())
    pose_valid_ratio = float((window_df["pose_valid"] == 1).mean())
    ear_valid_ratio = float((window_df["ear_valid"] == 1).mean())

    if "ear_suspicious" in window_df.columns:
        ear_suspicious_ratio = float((window_df["ear_suspicious"] == 1).mean())
    else:
        ear_suspicious_ratio = None

    return {
        "source_file": metadata["relative_path"],
        "file_name": metadata["file_name"],
        "file_stem": metadata["file_stem"],
        "label": metadata["label"],
        "modality": metadata["modality"],
        "window_id": window_id,
        "window_start_frame": safe_int(perclos_row["start_frame"]),
        "window_end_frame": safe_int(perclos_row["end_frame"]),
        "window_start_time": safe_float(perclos_row["start_time"]),
        "window_end_time": safe_float(perclos_row["end_time"]),
        "closed_eye_frames": safe_int(perclos_row["closed_eye_frames"]),
        "total_frames_perclos": safe_int(perclos_row["total_frames"]),
        "frame_count_source": frame_count,
        "perclos": safe_float(perclos_row["perclos"]),
        "perclos_percent": safe_float(perclos_row["perclos_percent"]),
        "face_detect_ratio": round(face_detect_ratio, 4),
        "pose_valid_ratio": round(pose_valid_ratio, 4),
        "ear_valid_ratio": round(ear_valid_ratio, 4),
        "ear_suspicious_ratio": round(ear_suspicious_ratio, 4) if ear_suspicious_ratio is not None else None,
        "mean_ear": safe_float(window_df["avg_ear"].mean()),
        "std_ear": safe_float(series_std(window_df["avg_ear"])),
        "min_ear": safe_float(window_df["avg_ear"].min()),
        "max_ear": safe_float(window_df["avg_ear"].max()),
        "mean_abs_yaw": safe_float(window_df["yaw"].abs().mean()),
        "std_yaw": safe_float(series_std(window_df["yaw"])),
        "max_abs_yaw": safe_float(window_df["yaw"].abs().max()),
        "mean_abs_pitch": safe_float(window_df["pitch"].abs().mean()),
        "std_pitch": safe_float(series_std(window_df["pitch"])),
        "max_abs_pitch": safe_float(window_df["pitch"].abs().max()),
        "mean_abs_roll": safe_float(window_df["roll"].abs().mean()),
        "std_roll": safe_float(series_std(window_df["roll"])),
        "max_abs_roll": safe_float(window_df["roll"].abs().max()),
        "is_usable": int(
            face_detect_ratio >= config.min_face_ratio
            and pose_valid_ratio >= config.min_pose_ratio
            and ear_valid_ratio >= config.min_ear_ratio
        ),
    }


def process_source_file(
    source_path: str,
    csv_root: str,
    perclos_root: str,
    config: WindowDatasetConfig,
) -> List[Dict]:
    metadata = infer_modality_and_label(source_path, csv_root)
    perclos_path = build_perclos_path(source_path, csv_root, perclos_root)

    if not os.path.exists(perclos_path):
        raise FileNotFoundError(f"Matching perclos file not found: {perclos_path}")

    source_df = load_source_table(source_path)
    perclos_df = load_perclos_table(perclos_path)

    rows: List[Dict] = []
    for window_id, (_, perclos_row) in enumerate(perclos_df.iterrows(), start=1):
        start_frame = int(perclos_row["start_frame"])
        end_frame = int(perclos_row["end_frame"])

        window_df = source_df[
            (source_df["frame"] >= start_frame) & (source_df["frame"] <= end_frame)
        ].copy()

        if window_df.empty:
            continue

        rows.append(
            compute_window_features(
                window_df=window_df,
                perclos_row=perclos_row,
                metadata=metadata,
                window_id=window_id,
                config=config,
            )
        )

    return rows


def collect_source_files(csv_root: str, config: WindowDatasetConfig) -> List[str]:
    files: List[str] = []
    for label in LABELS:
        for modality in MODALITIES:
            folder = os.path.join(csv_root, label, modality)
            if not os.path.exists(folder):
                continue

            for file_name in os.listdir(folder):
                if not file_name.lower().endswith(".xlsx"):
                    continue
                if config.ignore_temp_excel_files and file_name.startswith("~$"):
                    continue
                files.append(os.path.join(folder, file_name))

    return sorted(files)


def build_summary(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "modality",
                "window_count",
                "usable_window_count",
                "mean_perclos",
                "mean_ear",
                "mean_abs_yaw",
                "mean_abs_pitch",
            ]
        )

    summary_df = (
        dataset_df.groupby(["label", "modality"], dropna=False)
        .agg(
            window_count=("window_id", "count"),
            usable_window_count=("is_usable", "sum"),
            mean_perclos=("perclos", "mean"),
            mean_ear=("mean_ear", "mean"),
            mean_abs_yaw=("mean_abs_yaw", "mean"),
            mean_abs_pitch=("mean_abs_pitch", "mean"),
        )
        .reset_index()
    )

    for col in ["mean_perclos", "mean_ear", "mean_abs_yaw", "mean_abs_pitch"]:
        summary_df[col] = summary_df[col].round(4)

    return summary_df


def build_window_dataset(
    csv_root: str,
    perclos_root: str,
    output_root: str,
    summary_path: str,
    config: WindowDatasetConfig,
) -> pd.DataFrame:
    source_files = collect_source_files(csv_root, config)
    print(f"Source files found: {len(source_files)}")

    all_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for source_path in source_files:
        print(f"Processing: {os.path.basename(source_path)}")
        try:
            rows = process_source_file(
                source_path=source_path,
                csv_root=csv_root,
                perclos_root=perclos_root,
                config=config,
            )
            all_rows.extend(rows)
            per_file_df = pd.DataFrame(rows)
            output_path = build_output_path(source_path, csv_root, output_root)
            per_file_df.to_excel(output_path, index=False)
            print(f"  Windows added: {len(rows)}")
            print(f"  Saved: {output_path}")
        except Exception as exc:
            error_rows.append(
                {
                    "source_file": source_path,
                    "error": str(exc),
                }
            )
            print(f"  Error: {exc}")

    dataset_df = pd.DataFrame(all_rows)
    summary_df = build_summary(dataset_df)

    os.makedirs(output_root, exist_ok=True)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    summary_df.to_excel(summary_path, index=False)

    if error_rows:
        error_path = os.path.join(output_root, "window_dataset_errors.xlsx")
        pd.DataFrame(error_rows).to_excel(error_path, index=False)
        print(f"Error report: {error_path}")

    print(f"Per-video window datasets saved under: {output_root}")
    print(f"Summary file: {summary_path}")
    return dataset_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a window-level dataset for drowsiness and distraction."
    )
    parser.add_argument(
        "--csv-root",
        default=CSV_ROOT_DEFAULT,
        help="Root folder of csvcleaned data.",
    )
    parser.add_argument(
        "--perclos-root",
        default=PERCLOS_ROOT_DEFAULT,
        help="Root folder of perclos data.",
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT_DEFAULT,
        help="Root folder for per-video window datasets.",
    )
    parser.add_argument(
        "--summary-path",
        default=SUMMARY_PATH_DEFAULT,
        help="Excel path for the summary table.",
    )
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=WindowDatasetConfig.min_face_ratio,
        help="Minimum face_detected ratio for a usable window.",
    )
    parser.add_argument(
        "--min-pose-ratio",
        type=float,
        default=WindowDatasetConfig.min_pose_ratio,
        help="Minimum pose_valid ratio for a usable window.",
    )
    parser.add_argument(
        "--min-ear-ratio",
        type=float,
        default=WindowDatasetConfig.min_ear_ratio,
        help="Minimum ear_valid ratio for a usable window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = WindowDatasetConfig(
        min_face_ratio=args.min_face_ratio,
        min_pose_ratio=args.min_pose_ratio,
        min_ear_ratio=args.min_ear_ratio,
    )

    print("Window dataset config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    build_window_dataset(
        csv_root=args.csv_root,
        perclos_root=args.perclos_root,
        output_root=args.output_root,
        summary_path=args.summary_path,
        config=config,
    )


if __name__ == "__main__":
    main()
