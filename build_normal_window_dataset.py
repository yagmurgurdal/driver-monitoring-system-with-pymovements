import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


NORMAL_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned", "normal")
OUTPUT_ROOT_DEFAULT = r"D:\window\normal"
SUMMARY_PATH_DEFAULT = os.path.join(os.getcwd(), "normal_window_dataset_summary.xlsx")

WINDOW_SIZE = 90
LABEL = "normal"
MODALITIES = ("IR", "RGB")
SOURCE_GROUPS = ("drowsiness", "distraction")

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


@dataclass(frozen=True)
class NormalWindowConfig:
    window_size: int = WINDOW_SIZE
    min_face_ratio: float = 0.80
    min_pose_ratio: float = 0.80
    min_ear_ratio: float = 0.80


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def safe_std(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(clean.std(ddof=1))


def load_normal_frame_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)
    return df


def compute_window_features(
    window_df: pd.DataFrame,
    file_name: str,
    file_stem: str,
    modality: str,
    source_group: str,
    window_id: int,
    config: NormalWindowConfig,
) -> Dict:
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
        "source_file": os.path.join(modality, source_group, file_name),
        "file_name": file_name,
        "file_stem": file_stem,
        "label": LABEL,
        "modality": modality,
        "source_group": source_group,
        "window_id": window_id,
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
        "mean_ear": safe_float(window_df["avg_ear"].mean()),
        "std_ear": safe_std(window_df["avg_ear"]),
        "min_ear": safe_float(window_df["avg_ear"].min()),
        "max_ear": safe_float(window_df["avg_ear"].max()),
        "mean_abs_yaw": safe_float(window_df["yaw"].abs().mean()),
        "std_yaw": safe_std(window_df["yaw"]),
        "max_abs_yaw": safe_float(window_df["yaw"].abs().max()),
        "mean_abs_pitch": safe_float(window_df["pitch"].abs().mean()),
        "std_pitch": safe_std(window_df["pitch"]),
        "max_abs_pitch": safe_float(window_df["pitch"].abs().max()),
        "mean_abs_roll": safe_float(window_df["roll"].abs().mean()),
        "std_roll": safe_std(window_df["roll"]),
        "max_abs_roll": safe_float(window_df["roll"].abs().max()),
        "is_usable": int(
            face_detect_ratio >= config.min_face_ratio
            and pose_valid_ratio >= config.min_pose_ratio
            and ear_valid_ratio >= config.min_ear_ratio
        ),
    }


def process_file(
    path: str,
    modality: str,
    source_group: str,
    config: NormalWindowConfig,
) -> List[Dict]:
    df = load_normal_frame_file(path)
    if df.empty:
        return []

    remainder = len(df) % config.window_size
    if remainder != 0:
        raise ValueError(
            f"{os.path.basename(path)} row count {len(df)} is not divisible by window size {config.window_size}."
        )

    file_name = os.path.basename(path)
    file_stem = os.path.splitext(file_name)[0].replace("_normal_frames", "")

    rows: List[Dict] = []
    for idx in range(0, len(df), config.window_size):
        window_df = df.iloc[idx : idx + config.window_size].copy().reset_index(drop=True)
        rows.append(
            compute_window_features(
                window_df=window_df,
                file_name=file_name,
                file_stem=file_stem,
                modality=modality,
                source_group=source_group,
                window_id=(idx // config.window_size) + 1,
                config=config,
            )
        )
    return rows


def save_per_file(rows: List[Dict], output_root: str, modality: str, file_stem: str):
    output_dir = os.path.join(output_root, modality)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_stem}_windows.xlsx")
    pd.DataFrame(rows).to_excel(output_path, index=False)
    return output_path


def build_summary(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "modality",
                "source_group",
                "window_count",
                "usable_window_count",
                "mean_perclos",
                "mean_ear",
                "mean_abs_yaw",
                "mean_abs_pitch",
            ]
        )

    summary = (
        dataset_df.groupby(["label", "modality", "source_group"], dropna=False)
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
    for col in ("mean_perclos", "mean_ear", "mean_abs_yaw", "mean_abs_pitch"):
        summary[col] = summary[col].round(4)
    return summary


def build_normal_windows(normal_root: str, output_root: str, summary_path: str, config: NormalWindowConfig):
    all_rows: List[Dict] = []
    errors: List[Dict] = []

    for modality in MODALITIES:
        for source_group in SOURCE_GROUPS:
            folder = os.path.join(normal_root, modality, source_group)
            if not os.path.exists(folder):
                continue

            files = sorted(
                os.path.join(folder, file_name)
                for file_name in os.listdir(folder)
                if file_name.lower().endswith(".xlsx") and not file_name.startswith("~$")
            )

            print(f"{modality}/{source_group}: {len(files)} file(s)")
            for path in files:
                print(f"Processing: {os.path.basename(path)}")
                try:
                    rows = process_file(path, modality, source_group, config)
                    all_rows.extend(rows)
                    file_stem = os.path.splitext(os.path.basename(path))[0].replace("_normal_frames", "")
                    output_path = save_per_file(rows, output_root, modality, file_stem)
                    print(f"  Windows added: {len(rows)}")
                    print(f"  Saved: {output_path}")
                except Exception as exc:
                    errors.append({"source_file": path, "error": str(exc)})
                    print(f"  Error: {exc}")

    dataset_df = pd.DataFrame(all_rows)
    summary_df = build_summary(dataset_df)
    summary_df.to_excel(summary_path, index=False)

    if errors:
        error_path = os.path.join(output_root, "normal_window_errors.xlsx")
        pd.DataFrame(errors).to_excel(error_path, index=False)
        print(f"Error report: {error_path}")

    print(f"Summary file: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build normal-class per-video window Excel files.")
    parser.add_argument("--normal-root", default=NORMAL_ROOT_DEFAULT, help="Root folder of *_normal_frames.xlsx files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT, help="Output folder for normal window Excel files.")
    parser.add_argument("--summary-path", default=SUMMARY_PATH_DEFAULT, help="Excel path for summary file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = NormalWindowConfig()
    build_normal_windows(
        normal_root=args.normal_root,
        output_root=args.output_root,
        summary_path=args.summary_path,
        config=config,
    )


if __name__ == "__main__":
    main()
