import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pymovements as pm


INPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_input")
OUTPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_dataset")
SUMMARY_PATH_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_summary.xlsx")

REQUIRED_COLUMNS = [
    "frame",
    "time_sec",
    "face_detected",
    "iris_x_norm",
    "iris_y_norm",
]


@dataclass(frozen=True)
class WindowConfig:
    window_sec: float = 3.0
    step_sec: float = 3.0
    min_valid_ratio: float = 0.50
    idt_min_duration_ms: int = 100
    idt_dispersion_threshold: float = 1.0
    ivt_min_duration_ms: int = 100
    ivt_velocity_threshold: float = 20.0
    max_files: int = 0


def safe_float(value, digits: int = 6):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return round(float(value), digits)


def safe_int(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return int(value)


def series_std(values: pd.Series) -> Optional[float]:
    if values.empty:
        return None
    std_value = values.std(ddof=0)
    if pd.isna(std_value):
        return None
    return float(std_value)


def estimate_sample_period_sec(df: pd.DataFrame) -> float:
    diffs = pd.to_numeric(df["time_sec"], errors="coerce").diff()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0 / 30.0
    return float(diffs.median())


def estimate_expected_frame_step(df: pd.DataFrame) -> int:
    frame_diffs = pd.to_numeric(df["frame"], errors="coerce").diff()
    frame_diffs = frame_diffs[frame_diffs > 0]
    if frame_diffs.empty:
        return 1
    return max(1, int(round(float(frame_diffs.median()))))


def load_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing_text}")

    numeric_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("frame").reset_index(drop=True)
    return df


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

    label = parts[0] if len(parts) >= 1 else "unknown"
    modality = parts[1] if len(parts) >= 2 else "unknown"
    file_name = os.path.basename(path)
    stem = Path(file_name).stem
    source_stem = stem.replace("_pymovements_input", "")

    return {
        "relative_path": relative_path,
        "label": label,
        "modality": modality,
        "file_name": file_name,
        "file_stem": stem,
        "source_stem": source_stem,
    }


def build_output_path(path: str, input_root: str, output_root: str) -> str:
    metadata = parse_metadata(path, input_root)
    output_dir = os.path.join(output_root, metadata["label"], metadata["modality"])
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{metadata['source_stem']}_gaze_windows.xlsx")


def iter_window_ranges(last_time_sec: float, window_sec: float, step_sec: float) -> Iterable[tuple[int, float, float]]:
    window_id = 1
    start_time = 0.0
    tolerance = 1e-9

    while start_time + window_sec <= last_time_sec + tolerance:
        end_time = start_time + window_sec
        yield window_id, start_time, end_time
        window_id += 1
        start_time += step_sec


def split_contiguous_segments(valid_df: pd.DataFrame, expected_frame_step: int) -> List[pd.DataFrame]:
    if valid_df.empty:
        return []

    frame_diff = valid_df["frame"].diff().fillna(expected_frame_step)
    segment_ids = (frame_diff > expected_frame_step).cumsum()
    return [segment.copy() for _, segment in valid_df.groupby(segment_ids, sort=False)]


def compute_segment_velocity(segment_df: pd.DataFrame) -> np.ndarray:
    positions = segment_df[["iris_x_norm", "iris_y_norm"]].to_numpy(dtype=np.float64)
    velocities = np.zeros_like(positions)
    if len(segment_df) < 2:
        return velocities

    time_diff = segment_df["time_sec"].diff().to_numpy(dtype=np.float64)
    pos_diff = positions[1:] - positions[:-1]

    valid_dt = np.isfinite(time_diff[1:]) & (time_diff[1:] > 0)
    velocities[1:][valid_dt] = pos_diff[valid_dt] / time_diff[1:][valid_dt, None]
    return velocities


def events_to_sample_counts(events) -> List[int]:
    if events is None:
        return []
    rows = events.frame.to_dicts()
    counts: List[int] = []
    for row in rows:
        onset = int(row["onset"])
        offset = int(row["offset"])
        counts.append(max(0, offset - onset + 1))
    return counts


def summarize_durations_ms(sample_counts: List[int], sample_period_sec: float) -> Dict[str, Optional[float]]:
    if not sample_counts:
        return {
            "count": 0,
            "mean_ms": None,
            "max_ms": None,
            "total_ms": 0.0,
            "total_samples": 0,
        }

    durations_ms = [count * sample_period_sec * 1000.0 for count in sample_counts]
    total_samples = int(sum(sample_counts))
    return {
        "count": len(sample_counts),
        "mean_ms": float(np.mean(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
        "total_ms": float(np.sum(durations_ms)),
        "total_samples": total_samples,
    }


def duration_or_zero(summary: Dict[str, Optional[float]], key: str) -> float:
    value = summary.get(key)
    if value is None and summary.get("count", 0) == 0:
        return 0.0
    return float(value) if value is not None else 0.0


def boolean_run_sample_counts(mask: np.ndarray, min_samples: int) -> List[int]:
    if mask.size == 0:
        return []
    counts: List[int] = []
    current = 0
    for flag in mask.astype(bool):
        if flag:
            current += 1
            continue
        if current >= min_samples:
            counts.append(current)
        current = 0
    if current >= min_samples:
        counts.append(current)
    return counts



def compute_window_features(
    window_df: pd.DataFrame,
    window_id: int,
    start_time: float,
    end_time: float,
    metadata: Dict[str, str],
    config: WindowConfig,
    sample_period_sec: float,
    expected_frame_step: int,
) -> Dict:
    total_samples = len(window_df)
    valid_df = window_df[
        (window_df["face_detected"] == 1)
        & window_df["iris_x_norm"].notna()
        & window_df["iris_y_norm"].notna()
    ].copy()

    valid_samples = len(valid_df)
    valid_ratio = float(valid_samples / total_samples) if total_samples > 0 else 0.0
    usable_window = int(valid_ratio >= config.min_valid_ratio and valid_samples >= 3)

    base_row = {
        "label": metadata["label"],
        "modality": metadata["modality"],
        "source_file": metadata["file_name"],
        "source_stem": metadata["source_stem"],
        "window_id": window_id,
        "window_start_time": safe_float(start_time, 3),
        "window_end_time": safe_float(end_time, 3),
        "window_start_frame": safe_int(window_df["frame"].min()) if total_samples else None,
        "window_end_frame": safe_int(window_df["frame"].max()) if total_samples else None,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "gaze_valid_ratio": safe_float(valid_ratio, 4),
        "usable_window": usable_window,
    }

    if valid_df.empty:
        return base_row

    x_series = valid_df["iris_x_norm"]
    y_series = valid_df["iris_y_norm"]

    segments = split_contiguous_segments(valid_df, expected_frame_step)
    min_idt_samples = max(2, int(round(config.idt_min_duration_ms / (sample_period_sec * 1000.0))))
    min_ivt_samples = max(2, int(round(config.ivt_min_duration_ms / (sample_period_sec * 1000.0))))

    all_velocity_norms: List[np.ndarray] = []
    idt_counts: List[int] = []
    ivt_counts: List[int] = []
    rapid_counts: List[int] = []
    valid_step_count = 0
    path_length = 0.0

    for segment in segments:
        if len(segment) < 2:
            continue

        positions = segment[["iris_x_norm", "iris_y_norm"]].to_numpy(dtype=np.float64)
        velocities = compute_segment_velocity(segment)
        velocity_norm = np.linalg.norm(velocities[1:], axis=1)
        step_distance = np.linalg.norm(positions[1:] - positions[:-1], axis=1)

        if velocity_norm.size:
            all_velocity_norms.append(velocity_norm)
        if step_distance.size:
            path_length += float(np.nansum(step_distance))
            valid_step_count += int(np.isfinite(step_distance).sum())

        try:
            idt_events = pm.events.detection.idt(
                positions,
                minimum_duration=min_idt_samples,
                dispersion_threshold=config.idt_dispersion_threshold,
            )
            idt_counts.extend(events_to_sample_counts(idt_events))
        except Exception:
            pass

        try:
            ivt_events = pm.events.detection.ivt(
                velocities,
                minimum_duration=min_ivt_samples,
                velocity_threshold=config.ivt_velocity_threshold,
            )
            ivt_counts.extend(events_to_sample_counts(ivt_events))
        except Exception:
            pass

        rapid_mask = velocity_norm > config.ivt_velocity_threshold
        rapid_counts.extend(boolean_run_sample_counts(rapid_mask, min_ivt_samples))

    all_velocity = np.concatenate(all_velocity_norms) if all_velocity_norms else np.array([], dtype=np.float64)
    idt_summary = summarize_durations_ms(idt_counts, sample_period_sec)
    ivt_summary = summarize_durations_ms(ivt_counts, sample_period_sec)
    rapid_summary = summarize_durations_ms(rapid_counts, sample_period_sec)

    dispersion_x = float(x_series.max() - x_series.min()) if valid_samples else None
    dispersion_y = float(y_series.max() - y_series.min()) if valid_samples else None
    dispersion_xy = None
    if dispersion_x is not None and dispersion_y is not None:
        dispersion_xy = dispersion_x + dispersion_y

    feature_row = {
        **base_row,
        "mean_iris_x_norm": safe_float(x_series.mean()),
        "std_iris_x_norm": safe_float(series_std(x_series)),
        "min_iris_x_norm": safe_float(x_series.min()),
        "max_iris_x_norm": safe_float(x_series.max()),
        "mean_iris_y_norm": safe_float(y_series.mean()),
        "std_iris_y_norm": safe_float(series_std(y_series)),
        "min_iris_y_norm": safe_float(y_series.min()),
        "max_iris_y_norm": safe_float(y_series.max()),
        "gaze_dispersion_x": safe_float(dispersion_x),
        "gaze_dispersion_y": safe_float(dispersion_y),
        "gaze_dispersion_xy": safe_float(dispersion_xy),
        "gaze_path_length": safe_float(path_length),
        "mean_step_distance": safe_float(path_length / valid_step_count) if valid_step_count > 0 else None,
        "mean_velocity_norm": safe_float(float(np.mean(all_velocity))) if all_velocity.size else None,
        "std_velocity_norm": safe_float(float(np.std(all_velocity))) if all_velocity.size else None,
        "max_velocity_norm": safe_float(float(np.max(all_velocity))) if all_velocity.size else None,
        "idt_fixation_count": idt_summary["count"],
        "idt_fixation_mean_duration_ms": safe_float(duration_or_zero(idt_summary, "mean_ms"), 3),
        "idt_fixation_max_duration_ms": safe_float(duration_or_zero(idt_summary, "max_ms"), 3),
        "idt_fixation_ratio": safe_float(idt_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
        "ivt_fixation_count": ivt_summary["count"],
        "ivt_fixation_mean_duration_ms": safe_float(duration_or_zero(ivt_summary, "mean_ms"), 3),
        "ivt_fixation_max_duration_ms": safe_float(duration_or_zero(ivt_summary, "max_ms"), 3),
        "ivt_fixation_ratio": safe_float(ivt_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
        "rapid_shift_count": rapid_summary["count"],
        "rapid_shift_mean_duration_ms": safe_float(duration_or_zero(rapid_summary, "mean_ms"), 3),
        "rapid_shift_max_duration_ms": safe_float(duration_or_zero(rapid_summary, "max_ms"), 3),
        "rapid_shift_ratio": safe_float(rapid_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
    }
    return feature_row


def build_windows_for_file(path: str, input_root: str, output_root: str, config: WindowConfig) -> Dict:
    metadata = parse_metadata(path, input_root)
    df = load_input_csv(path)

    if df.empty:
        raise ValueError("Input file has no rows.")

    sample_period_sec = estimate_sample_period_sec(df)
    expected_frame_step = estimate_expected_frame_step(df)
    last_time_sec = float(df["time_sec"].max())

    rows: List[Dict] = []
    for window_id, start_time, end_time in iter_window_ranges(last_time_sec, config.window_sec, config.step_sec):
        if window_id == 1:
            mask = (df["time_sec"] >= start_time) & (df["time_sec"] <= end_time)
        else:
            mask = (df["time_sec"] > start_time) & (df["time_sec"] <= end_time)
        window_df = df.loc[mask].copy()
        if window_df.empty:
            continue

        rows.append(
            compute_window_features(
                window_df=window_df,
                window_id=window_id,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata,
                config=config,
                sample_period_sec=sample_period_sec,
                expected_frame_step=expected_frame_step,
            )
        )

    if not rows:
        raise ValueError("No full windows could be created from file.")

    output_path = build_output_path(path, input_root, output_root)
    per_file_df = pd.DataFrame(rows)
    per_file_df.to_excel(output_path, index=False)

    summary_row = {
        "label": metadata["label"],
        "modality": metadata["modality"],
        "source_file": metadata["file_name"],
        "output_file": output_path,
        "window_count": len(per_file_df),
        "usable_window_count": int(per_file_df["usable_window"].fillna(0).sum()),
        "mean_valid_ratio": safe_float(per_file_df["gaze_valid_ratio"].mean(), 4),
        "mean_idt_fixation_count": safe_float(per_file_df["idt_fixation_count"].mean(), 4),
        "mean_ivt_fixation_count": safe_float(per_file_df["ivt_fixation_count"].mean(), 4),
        "mean_rapid_shift_count": safe_float(per_file_df["rapid_shift_count"].mean(), 4),
    }
    return summary_row


def build_dataset(input_root: str, output_root: str, summary_path: str, config: WindowConfig):
    input_files = discover_input_files(input_root, config.max_files)
    if not input_files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    os.makedirs(output_root, exist_ok=True)

    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for path in input_files:
        metadata = parse_metadata(path, input_root)
        print(f"Processing {metadata['label']}/{metadata['modality']}: {metadata['file_name']}")
        try:
            summary_rows.append(build_windows_for_file(path, input_root, output_root, config))
        except Exception as exc:
            error_rows.append(
                {
                    "label": metadata["label"],
                    "modality": metadata["modality"],
                    "source_file": metadata["file_name"],
                    "error": str(exc),
                }
            )
            print(f"  [ERROR] {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(summary_path, index=False)
        print(f"Summary written to: {summary_path}")

    if error_rows:
        error_path = os.path.join(output_root, "pymovements_window_errors.xlsx")
        pd.DataFrame(error_rows).to_excel(error_path, index=False)
        print(f"Errors written to: {error_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 3-second PyMovements-based gaze window features from extracted iris time-series CSV files."
    )
    parser.add_argument("--input-root", default=INPUT_ROOT_DEFAULT, help="Root folder containing pymovements input CSV files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT, help="Root folder for per-file gaze window outputs.")
    parser.add_argument("--summary-path", default=SUMMARY_PATH_DEFAULT, help="Path for the summary Excel report.")
    parser.add_argument("--window-sec", type=float, default=3.0, help="Window duration in seconds.")
    parser.add_argument("--step-sec", type=float, default=3.0, help="Window step in seconds.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.50, help="Minimum valid gaze ratio for a usable window.")
    parser.add_argument("--idt-min-duration-ms", type=int, default=100, help="Minimum I-DT fixation duration in milliseconds.")
    parser.add_argument("--idt-dispersion-threshold", type=float, default=1.0, help="I-DT dispersion threshold on normalized iris positions.")
    parser.add_argument("--ivt-min-duration-ms", type=int, default=100, help="Minimum I-VT fixation duration in milliseconds.")
    parser.add_argument("--ivt-velocity-threshold", type=float, default=20.0, help="I-VT low-velocity fixation threshold on normalized iris velocities.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit the number of files for quick testing. 0 means no limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = WindowConfig(
        window_sec=float(args.window_sec),
        step_sec=float(args.step_sec),
        min_valid_ratio=float(args.min_valid_ratio),
        idt_min_duration_ms=int(args.idt_min_duration_ms),
        idt_dispersion_threshold=float(args.idt_dispersion_threshold),
        ivt_min_duration_ms=int(args.ivt_min_duration_ms),
        ivt_velocity_threshold=float(args.ivt_velocity_threshold),
        max_files=max(0, int(args.max_files)),
    )
    build_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        summary_path=args.summary_path,
        config=config,
    )


if __name__ == "__main__":
    main()
