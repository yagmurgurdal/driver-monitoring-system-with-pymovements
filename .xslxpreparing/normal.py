import argparse
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd

from gaze_event_features import IRIS_SOURCE_COLUMNS


# =========================================================
# CONFIGURATION
# =========================================================
FPS = 30
WINDOW_SEC = 3
STEP_SEC = 3

# If you run this file directly from an editor, these folders are used
# automatically when command-line arguments are not provided.
DEFAULT_INPUT_FOLDER = r"D:\csvcleaned\drowsiness\RGB"
DEFAULT_OUTPUT_FOLDER = r"D:\csvcleaned\normal\RGB\drowsiness"

EAR_CLOSED_THRESHOLD = 0.21
PERCLOS_THRESHOLD = 0.15
MEAN_EAR_THRESHOLD = 0.22
EAR_STD_THRESHOLD = 0.03
MEAN_ABS_YAW_THRESHOLD = 15.0
MEAN_ABS_PITCH_THRESHOLD = 10.0
MAX_ABS_YAW_THRESHOLD = 20.0
MAX_ABS_PITCH_THRESHOLD = 15.0

RECURSIVE_SEARCH = False
IGNORE_TEMP_EXCEL_FILES = True


# =========================================================
# SCHEMA
# =========================================================
CORE_REQUIRED_COLUMNS = [
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

OPTIONAL_COLUMNS = ["ear_suspicious"]

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
    *IRIS_SOURCE_COLUMNS,
]

OUTPUT_WINDOW_COLUMNS = [
    "window_id",
    "window_start_frame",
    "window_end_frame",
    "window_start_time",
    "window_end_time",
    "window_perclos",
    "window_mean_ear",
    "window_std_ear",
    "window_mean_abs_yaw",
    "window_mean_abs_pitch",
    "window_max_abs_yaw",
    "window_max_abs_pitch",
    "label",
]


# =========================================================
# CONFIG DATA CLASS
# =========================================================
@dataclass(frozen=True)
class NormalExtractionConfig:
    fps: int = FPS
    window_sec: int = WINDOW_SEC
    step_sec: int = STEP_SEC
    ear_closed_threshold: float = EAR_CLOSED_THRESHOLD
    perclos_threshold: float = PERCLOS_THRESHOLD
    mean_ear_threshold: float = MEAN_EAR_THRESHOLD
    ear_std_threshold: float = EAR_STD_THRESHOLD
    mean_abs_yaw_threshold: float = MEAN_ABS_YAW_THRESHOLD
    mean_abs_pitch_threshold: float = MEAN_ABS_PITCH_THRESHOLD
    max_abs_yaw_threshold: float = MAX_ABS_YAW_THRESHOLD
    max_abs_pitch_threshold: float = MAX_ABS_PITCH_THRESHOLD

    @property
    def window_size(self) -> int:
        return int(self.fps * self.window_sec)

    @property
    def step_size(self) -> int:
        return int(self.fps * self.step_sec)


# =========================================================
# HELPERS
# =========================================================
def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _join_unique(values: List) -> str:
    seen = set()
    ordered = []
    for value in values:
        key = _format_value(value)
        if key == "" or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return "; ".join(ordered)


def _first_usable_index(df: pd.DataFrame) -> Optional[int]:
    usable_candidates = [
        "frame",
        "time_sec",
        "face_detected",
        "pose_valid",
        "ear_valid",
        "avg_ear",
        "yaw",
        "pitch",
        "roll",
    ]
    usable_columns = [col for col in usable_candidates if col in df.columns]
    if not usable_columns:
        return None

    usable_rows = df[usable_columns].dropna(how="all")
    if usable_rows.empty:
        return None

    return int(usable_rows.index.min())


def _build_output_path(input_xlsx: str, input_folder: str, output_folder: str) -> str:
    relative_path = os.path.relpath(input_xlsx, input_folder)
    relative_dir = os.path.dirname(relative_path)
    base_name = os.path.splitext(os.path.basename(input_xlsx))[0]
    output_dir = os.path.join(output_folder, relative_dir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}_normal_frames.xlsx")


# =========================================================
# 1) LOAD AND CLEAN
# =========================================================
def load_and_clean_xlsx(input_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(input_xlsx)

    missing_columns = [col for col in CORE_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"{os.path.basename(input_xlsx)} is missing required columns: {missing_text}"
        )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    first_valid_idx = _first_usable_index(df)
    if first_valid_idx is None:
        raise ValueError(f"{os.path.basename(input_xlsx)} has no usable data.")

    df = df.loc[first_valid_idx:].copy().reset_index(drop=True)

    valid_mask = (
        (df["face_detected"] == 1)
        & (df["pose_valid"] == 1)
        & (df["ear_valid"] == 1)
    )

    if "ear_suspicious" in df.columns:
        valid_mask &= (df["ear_suspicious"] != 1) | (df["ear_suspicious"].isna())

    df = df.loc[valid_mask].copy()
    if df.empty:
        return df

    df = df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        mean_value = df[col].mean()
        if pd.notna(mean_value):
            df[col] = df[col].fillna(mean_value)

    return df


# =========================================================
# 2) COMPUTE WINDOW FEATURES
# =========================================================
def compute_window_features(
    window_df: pd.DataFrame,
    window_id: int,
    config: NormalExtractionConfig,
) -> Dict:
    eye_closed = (window_df["avg_ear"] < config.ear_closed_threshold).astype(int)
    perclos = float(eye_closed.sum() / len(window_df))

    mean_ear = float(window_df["avg_ear"].mean())
    std_ear = float(window_df["avg_ear"].std(ddof=1))
    mean_abs_yaw = float(window_df["yaw"].abs().mean())
    mean_abs_pitch = float(window_df["pitch"].abs().mean())
    max_abs_yaw = float(window_df["yaw"].abs().max())
    max_abs_pitch = float(window_df["pitch"].abs().max())

    is_normal = all(
        [
            perclos < config.perclos_threshold,
            mean_ear > config.mean_ear_threshold,
            std_ear < config.ear_std_threshold,
            mean_abs_yaw < config.mean_abs_yaw_threshold,
            mean_abs_pitch < config.mean_abs_pitch_threshold,
            max_abs_yaw < config.max_abs_yaw_threshold,
            max_abs_pitch < config.max_abs_pitch_threshold,
        ]
    )

    return {
        "window_id": window_id,
        "window_start_frame": int(window_df["frame"].iloc[0]),
        "window_end_frame": int(window_df["frame"].iloc[-1]),
        "window_start_time": float(window_df["time_sec"].iloc[0]),
        "window_end_time": float(window_df["time_sec"].iloc[-1]),
        "window_perclos": round(perclos, 4),
        "window_mean_ear": round(mean_ear, 4),
        "window_std_ear": round(std_ear, 4),
        "window_mean_abs_yaw": round(mean_abs_yaw, 4),
        "window_mean_abs_pitch": round(mean_abs_pitch, 4),
        "window_max_abs_yaw": round(max_abs_yaw, 4),
        "window_max_abs_pitch": round(max_abs_pitch, 4),
        "label": "normal" if is_normal else "non_normal",
        "is_normal": is_normal,
    }


# =========================================================
# 3) EXTRACT NORMAL FRAMES FROM FILE
# =========================================================
def extract_normal_frames_from_file(
    input_xlsx: str,
    output_folder: str,
    config: NormalExtractionConfig,
    input_folder: Optional[str] = None,
) -> Dict:
    cleaned_df = load_and_clean_xlsx(input_xlsx)

    summary = {
        "file_name": os.path.basename(input_xlsx),
        "total_valid_frames": int(len(cleaned_df)),
        "normal_frame_count": 0,
        "normal_window_count": 0,
        "normal_ratio": 0.0,
        "status": "no_normal_frames",
    }

    if cleaned_df.empty:
        summary["status"] = "no_valid_frames_after_cleaning"
        return summary

    if len(cleaned_df) < config.window_size:
        summary["status"] = "not_enough_valid_frames"
        return summary

    frame_to_windows: Dict[int, List[Dict]] = {}
    normal_window_count = 0

    for start in range(0, len(cleaned_df) - config.window_size + 1, config.step_size):
        end = start + config.window_size
        window_df = cleaned_df.iloc[start:end].copy()
        window_features = compute_window_features(
            window_df=window_df,
            window_id=normal_window_count + 1,
            config=config,
        )

        if not window_features["is_normal"]:
            continue

        normal_window_count += 1
        window_features["window_id"] = normal_window_count

        for _, row in window_df.iterrows():
            frame_key = int(row["frame"])
            frame_to_windows.setdefault(frame_key, []).append(window_features)

    if not frame_to_windows:
        summary["status"] = "no_normal_frames"
        return summary

    selected_frames = sorted(frame_to_windows.keys())
    output_df = cleaned_df[cleaned_df["frame"].isin(selected_frames)].copy()
    output_df = output_df.sort_values("frame").reset_index(drop=True)

    for col in OUTPUT_WINDOW_COLUMNS:
        output_df[col] = ""

    for row_index, row in output_df.iterrows():
        frame_key = int(row["frame"])
        window_records = frame_to_windows[frame_key]

        output_df.at[row_index, "window_id"] = _join_unique(
            [item["window_id"] for item in window_records]
        )
        output_df.at[row_index, "window_start_frame"] = _join_unique(
            [item["window_start_frame"] for item in window_records]
        )
        output_df.at[row_index, "window_end_frame"] = _join_unique(
            [item["window_end_frame"] for item in window_records]
        )
        output_df.at[row_index, "window_start_time"] = _join_unique(
            [item["window_start_time"] for item in window_records]
        )
        output_df.at[row_index, "window_end_time"] = _join_unique(
            [item["window_end_time"] for item in window_records]
        )
        output_df.at[row_index, "window_perclos"] = _join_unique(
            [item["window_perclos"] for item in window_records]
        )
        output_df.at[row_index, "window_mean_ear"] = _join_unique(
            [item["window_mean_ear"] for item in window_records]
        )
        output_df.at[row_index, "window_std_ear"] = _join_unique(
            [item["window_std_ear"] for item in window_records]
        )
        output_df.at[row_index, "window_mean_abs_yaw"] = _join_unique(
            [item["window_mean_abs_yaw"] for item in window_records]
        )
        output_df.at[row_index, "window_mean_abs_pitch"] = _join_unique(
            [item["window_mean_abs_pitch"] for item in window_records]
        )
        output_df.at[row_index, "window_max_abs_yaw"] = _join_unique(
            [item["window_max_abs_yaw"] for item in window_records]
        )
        output_df.at[row_index, "window_max_abs_pitch"] = _join_unique(
            [item["window_max_abs_pitch"] for item in window_records]
        )
        output_df.at[row_index, "label"] = "normal"

    output_path = _build_output_path(
        input_xlsx=input_xlsx,
        input_folder=input_folder or os.path.dirname(input_xlsx),
        output_folder=output_folder,
    )
    output_df.to_excel(output_path, index=False)

    summary["normal_frame_count"] = int(len(output_df))
    summary["normal_window_count"] = int(normal_window_count)
    summary["normal_ratio"] = round(len(output_df) / len(cleaned_df), 4)
    summary["status"] = "success"
    return summary


# =========================================================
# 4) PROCESS FOLDER
# =========================================================
def process_folder(
    input_folder: str,
    output_folder: str,
    config: Optional[NormalExtractionConfig] = None,
    recursive: bool = RECURSIVE_SEARCH,
) -> pd.DataFrame:
    config = config or NormalExtractionConfig()

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    if recursive:
        xlsx_files = []
        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if not file_name.lower().endswith(".xlsx"):
                    continue
                if IGNORE_TEMP_EXCEL_FILES and file_name.startswith("~$"):
                    continue
                xlsx_files.append(os.path.join(root, file_name))
    else:
        xlsx_files = [
            os.path.join(input_folder, file_name)
            for file_name in os.listdir(input_folder)
            if file_name.lower().endswith(".xlsx")
            and not (IGNORE_TEMP_EXCEL_FILES and file_name.startswith("~$"))
        ]

    xlsx_files = sorted(xlsx_files)

    summary_path = os.path.join(output_folder, "normal_extraction_summary.xlsx")

    print(f"Total files found: {len(xlsx_files)}")
    if not xlsx_files:
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv")
        video_files = [
            file_name
            for file_name in os.listdir(input_folder)
            if file_name.lower().endswith(video_extensions)
        ]

        summary_df = pd.DataFrame(
            columns=[
                "file_name",
                "total_valid_frames",
                "normal_frame_count",
                "normal_window_count",
                "normal_ratio",
                "status",
            ]
        )
        summary_df.to_excel(summary_path, index=False)

        print("No .xlsx files were found in the input folder.")
        if video_files:
            print(
                "This folder seems to contain videos, but this script expects "
                "frame-level feature .xlsx files."
            )
        print(f"Empty summary file: {summary_path}")
        return summary_df

    summary_rows: List[Dict] = []
    success_count = 0
    no_normal_count = 0
    error_count = 0

    for input_xlsx in xlsx_files:
        print(f"\nProcessing: {os.path.basename(input_xlsx)}")
        try:
            summary = extract_normal_frames_from_file(
                input_xlsx=input_xlsx,
                output_folder=output_folder,
                config=config,
                input_folder=input_folder,
            )
            summary_rows.append(summary)

            if summary["status"] == "success":
                success_count += 1
                print(
                    f"  Saved normal frames: {summary['normal_frame_count']} | "
                    f"normal windows: {summary['normal_window_count']}"
                )
            else:
                no_normal_count += 1
                print(f"  Skipped: {summary['status']}")

        except Exception as exc:
            error_count += 1
            error_message = str(exc)
            summary_rows.append(
                {
                    "file_name": os.path.basename(input_xlsx),
                    "total_valid_frames": 0,
                    "normal_frame_count": 0,
                    "normal_window_count": 0,
                    "normal_ratio": 0.0,
                    "status": f"error: {error_message}",
                }
            )
            print(f"  Error: {error_message}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(summary_path, index=False)

    print("\nProcess completed.")
    print(f"Files successfully processed: {success_count}")
    print(f"Files with no normal frames: {no_normal_count}")
    print(f"Files with errors: {error_count}")
    print(f"Summary file: {summary_path}")

    return summary_df


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract normal windows/frames from frame-level driver feature Excel files."
    )
    parser.add_argument(
        "--input-folder",
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing input .xlsx files.",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder for normal output .xlsx files.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Video FPS.")
    parser.add_argument("--window-sec", type=int, default=WINDOW_SEC, help="Window length in seconds.")
    parser.add_argument("--step-sec", type=int, default=STEP_SEC, help="Sliding step in seconds.")
    parser.add_argument(
        "--ear-closed-threshold",
        type=float,
        default=EAR_CLOSED_THRESHOLD,
        help="avg_ear threshold used to define closed-eye frames.",
    )
    parser.add_argument(
        "--perclos-threshold",
        type=float,
        default=PERCLOS_THRESHOLD,
        help="Maximum allowed PERCLOS for a normal window.",
    )
    parser.add_argument(
        "--mean-ear-threshold",
        type=float,
        default=MEAN_EAR_THRESHOLD,
        help="Minimum mean avg_ear for a normal window.",
    )
    parser.add_argument(
        "--ear-std-threshold",
        type=float,
        default=EAR_STD_THRESHOLD,
        help="Maximum std(avg_ear) for a normal window.",
    )
    parser.add_argument(
        "--mean-abs-yaw-threshold",
        type=float,
        default=MEAN_ABS_YAW_THRESHOLD,
        help="Maximum mean(abs(yaw)) for a normal window.",
    )
    parser.add_argument(
        "--mean-abs-pitch-threshold",
        type=float,
        default=MEAN_ABS_PITCH_THRESHOLD,
        help="Maximum mean(abs(pitch)) for a normal window.",
    )
    parser.add_argument(
        "--max-abs-yaw-threshold",
        type=float,
        default=MAX_ABS_YAW_THRESHOLD,
        help="Maximum max(abs(yaw)) for a normal window.",
    )
    parser.add_argument(
        "--max-abs-pitch-threshold",
        type=float,
        default=MAX_ABS_PITCH_THRESHOLD,
        help="Maximum max(abs(pitch)) for a normal window.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process .xlsx files recursively under the input folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_folder or not args.output_folder:
        raise ValueError(
            "Input/output folder is empty. Set DEFAULT_INPUT_FOLDER and "
            "DEFAULT_OUTPUT_FOLDER at the top of normal.py or pass them as arguments."
        )

    config = NormalExtractionConfig(
        fps=args.fps,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        ear_closed_threshold=args.ear_closed_threshold,
        perclos_threshold=args.perclos_threshold,
        mean_ear_threshold=args.mean_ear_threshold,
        ear_std_threshold=args.ear_std_threshold,
        mean_abs_yaw_threshold=args.mean_abs_yaw_threshold,
        mean_abs_pitch_threshold=args.mean_abs_pitch_threshold,
        max_abs_yaw_threshold=args.max_abs_yaw_threshold,
        max_abs_pitch_threshold=args.max_abs_pitch_threshold,
    )

    print("Normal extraction config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"  window_size: {config.window_size}")
    print(f"  step_size: {config.step_size}")
    print(f"  input_folder: {args.input_folder}")
    print(f"  output_folder: {args.output_folder}")

    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        config=config,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
