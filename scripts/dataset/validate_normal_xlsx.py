import argparse
import csv
import os

import pandas as pd


REQUIRED_COLS = [
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

NUMERIC_COLS = [
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


DEFAULT_CONFIG = {
    "fps": 30,
    "window_sec": 3,
    "step_sec": 3,
    "ear_closed_threshold": 0.21,
    "perclos_threshold": 0.15,
    "mean_ear_threshold": 0.22,
    "ear_std_threshold": 0.03,
    "mean_abs_yaw_threshold": 15,
    "mean_abs_pitch_threshold": 10,
    "max_abs_yaw_threshold": 20,
    "max_abs_pitch_threshold": 15,
}


def load_table(path):
    df = pd.read_excel(path)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def recompute_expected_normal_frames(source_path, config):
    df = load_table(source_path)

    first_valid_idx = df[["avg_ear", "yaw", "pitch"]].dropna(how="all").index.min()
    if pd.isna(first_valid_idx):
        raise ValueError("No usable data in source file.")

    df = df.loc[first_valid_idx:].copy().reset_index(drop=True)
    valid_df = df[(df["pose_valid"] == 1) & (df["ear_valid"] == 1)].copy()
    if valid_df.empty:
        raise ValueError("No valid frames in source file.")

    for col in ["avg_ear", "yaw", "pitch", "roll", "left_ear", "right_ear"]:
        valid_df[col] = valid_df[col].fillna(valid_df[col].mean())

    valid_df["abs_yaw"] = valid_df["yaw"].abs()
    valid_df["abs_pitch"] = valid_df["pitch"].abs()
    valid_df["eye_closed"] = (valid_df["avg_ear"] < config["ear_closed_threshold"]).astype(int)

    window_size = int(config["fps"] * config["window_sec"])
    step_size = int(config["fps"] * config["step_sec"])

    if len(valid_df) < window_size:
        return valid_df, []

    normal_indices = []

    for start in range(0, len(valid_df) - window_size + 1, step_size):
        end = start + window_size
        window = valid_df.iloc[start:end]

        perclos = window["eye_closed"].sum() / len(window)
        mean_ear = window["avg_ear"].mean()
        std_ear = window["avg_ear"].std()
        mean_abs_yaw = window["abs_yaw"].mean()
        mean_abs_pitch = window["abs_pitch"].mean()
        max_abs_yaw = window["abs_yaw"].max()
        max_abs_pitch = window["abs_pitch"].max()

        is_normal = (
            perclos < config["perclos_threshold"]
            and mean_ear > config["mean_ear_threshold"]
            and std_ear < config["ear_std_threshold"]
            and mean_abs_yaw < config["mean_abs_yaw_threshold"]
            and mean_abs_pitch < config["mean_abs_pitch_threshold"]
            and max_abs_yaw < config["max_abs_yaw_threshold"]
            and max_abs_pitch < config["max_abs_pitch_threshold"]
        )

        if is_normal:
            normal_indices.extend(window.index.tolist())

    normal_indices = sorted(set(normal_indices))
    return valid_df, normal_indices


def infer_source_path(normal_path, csvcleaned_root):
    relative_path = os.path.relpath(normal_path, csvcleaned_root)
    parts = relative_path.split(os.sep)

    if len(parts) < 4 or parts[0].lower() != "normal":
        raise ValueError("Unexpected normal file path structure.")

    modality = parts[1]
    label = parts[2]
    file_name = parts[-1]
    source_name = file_name.replace("_normal_frames.xlsx", ".xlsx")

    return os.path.join(csvcleaned_root, label, modality, source_name)


def summarize_segments(frames):
    if not frames:
        return 0, 0, 0

    segment_count = 1
    max_gap = 0

    for prev_frame, current_frame in zip(frames, frames[1:]):
        gap = int(current_frame) - int(prev_frame)
        if gap > 1:
            segment_count += 1
            max_gap = max(max_gap, gap)

    longest_segment = 1
    current_segment = 1

    for prev_frame, current_frame in zip(frames, frames[1:]):
        if int(current_frame) - int(prev_frame) == 1:
            current_segment += 1
        else:
            longest_segment = max(longest_segment, current_segment)
            current_segment = 1

    longest_segment = max(longest_segment, current_segment)
    return segment_count, longest_segment, max_gap


def compare_frames(expected_frames, actual_frames):
    expected_set = set(expected_frames)
    actual_set = set(actual_frames)

    missing_frames = sorted(expected_set - actual_set)
    extra_frames = sorted(actual_set - expected_set)

    exact_match = expected_frames == actual_frames
    return exact_match, missing_frames, extra_frames


def validate_file(normal_path, csvcleaned_root, config):
    source_path = infer_source_path(normal_path, csvcleaned_root)
    if not os.path.exists(source_path):
        return {
            "normal_file": normal_path,
            "source_file": source_path,
            "status": "source_missing",
        }

    normal_df = load_table(normal_path)
    valid_df, normal_indices = recompute_expected_normal_frames(source_path, config)
    expected_df = valid_df.loc[normal_indices, REQUIRED_COLS].copy() if normal_indices else valid_df.iloc[0:0][REQUIRED_COLS].copy()

    actual_frames = normal_df["frame"].dropna().astype(int).tolist()
    expected_frames = expected_df["frame"].dropna().astype(int).tolist()

    exact_match, missing_frames, extra_frames = compare_frames(expected_frames, actual_frames)
    segment_count, longest_segment, max_gap = summarize_segments(actual_frames)

    status = "exact_match" if exact_match else "frame_mismatch"

    return {
        "normal_file": normal_path,
        "source_file": source_path,
        "status": status,
        "actual_rows": len(actual_frames),
        "expected_rows": len(expected_frames),
        "missing_frame_count": len(missing_frames),
        "extra_frame_count": len(extra_frames),
        "first_missing_frame": missing_frames[0] if missing_frames else "",
        "first_extra_frame": extra_frames[0] if extra_frames else "",
        "segment_count": segment_count,
        "longest_segment_frames": longest_segment,
        "max_frame_gap": max_gap,
    }


def write_csv_report(rows, output_path):
    if not rows:
        return

    fieldnames = [
        "normal_file",
        "source_file",
        "status",
        "actual_rows",
        "expected_rows",
        "missing_frame_count",
        "extra_frame_count",
        "first_missing_frame",
        "first_extra_frame",
        "segment_count",
        "longest_segment_frames",
        "max_frame_gap",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate *_normal_frames.xlsx outputs.")
    parser.add_argument(
        "--csvcleaned-root",
        default=os.path.join(os.getcwd(), "csvcleaned"),
        help="Root csvcleaned directory.",
    )
    parser.add_argument(
        "--report-path",
        default=os.path.join(os.getcwd(), "normal_xlsx_validation_report.csv"),
        help="CSV path for the validation report.",
    )
    args = parser.parse_args()

    normal_root = os.path.join(args.csvcleaned_root, "normal")
    rows = []

    for root, _, files in os.walk(normal_root):
        for file_name in files:
            if not file_name.lower().endswith(".xlsx"):
                continue

            normal_path = os.path.join(root, file_name)
            try:
                rows.append(validate_file(normal_path, args.csvcleaned_root, DEFAULT_CONFIG))
            except Exception as exc:
                rows.append(
                    {
                        "normal_file": normal_path,
                        "source_file": "",
                        "status": f"error: {exc}",
                    }
                )

    rows = sorted(rows, key=lambda row: (row.get("status", ""), row.get("normal_file", "")))
    write_csv_report(rows, args.report_path)

    total = len(rows)
    exact = sum(row.get("status") == "exact_match" for row in rows)
    mismatch = sum(row.get("status") == "frame_mismatch" for row in rows)
    missing_source = sum(row.get("status") == "source_missing" for row in rows)
    errors = total - exact - mismatch - missing_source

    print(f"Total normal xlsx files: {total}")
    print(f"Exact matches: {exact}")
    print(f"Frame mismatches: {mismatch}")
    print(f"Missing source files: {missing_source}")
    print(f"Errors: {errors}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
