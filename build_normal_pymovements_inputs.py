import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


NORMAL_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned", "normal")
CSVCLEANED_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned")
PYMOVEMENTS_INPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_input")
OUTPUT_ROOT_DEFAULT = os.path.join(PYMOVEMENTS_INPUT_ROOT_DEFAULT, "normal")
REPORT_PATH_DEFAULT = os.path.join(os.getcwd(), "normal_pymovements_input_summary.xlsx")


PYMOVEMENTS_COLUMNS = [
    "frame",
    "time_sec",
    "face_detected",
    "left_iris_center_x_px",
    "left_iris_center_y_px",
    "right_iris_center_x_px",
    "right_iris_center_y_px",
    "iris_center_x_px",
    "iris_center_y_px",
    "left_iris_x_norm",
    "left_iris_y_norm",
    "right_iris_x_norm",
    "right_iris_y_norm",
    "iris_x_norm",
    "iris_y_norm",
]


@dataclass(frozen=True)
class BuildConfig:
    max_files: int = 0
    duration_tolerance_sec: float = 3.0


def natural_numeric_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def gaze_name_key(name: str) -> Tuple[int, int, int, int, str]:
    panel_flag = 1 if "panel" in name.lower() else 0
    match = re.match(r"g([A-Z])_(\d+)_s(\d+)", name)
    if not match:
        return (10**9, 10**9, 10**9, panel_flag, name)
    return (
        ord(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        panel_flag,
        name,
    )


def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def infer_source_path(normal_path: str, csvcleaned_root: str) -> str:
    relative_path = os.path.relpath(normal_path, os.path.join(csvcleaned_root, "normal"))
    parts = relative_path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Unexpected normal path structure: {normal_path}")

    modality = parts[0]
    label = parts[1]
    file_name = parts[-1]
    source_name = file_name.replace("_normal_frames.xlsx", ".xlsx")
    return os.path.join(csvcleaned_root, label, modality, source_name)


def discover_normal_files(normal_root: str, max_files: int) -> List[str]:
    root = Path(normal_root)
    if not root.exists():
        return []

    files = [
        str(path)
        for path in root.rglob("*_normal_frames.xlsx")
        if not path.name.startswith("~$")
    ]
    files = sorted(files)
    if max_files > 0:
        files = files[:max_files]
    return files


def collect_source_files(csvcleaned_root: str, label: str, modality: str) -> List[str]:
    folder = Path(csvcleaned_root) / label / modality
    if not folder.exists():
        return []

    files: List[str] = []
    for path in folder.glob("*.xlsx"):
        if path.name.startswith("~$"):
            continue
        try:
            pd.read_excel(path, engine="openpyxl", nrows=1)
        except Exception as exc:
            print(f"Skipping unreadable source file: {path.name} ({exc})")
            continue
        files.append(str(path))

    return sorted(files, key=lambda path: natural_numeric_key(Path(path).stem))


def collect_gaze_files(pymovements_input_root: str, label: str, modality: str) -> List[str]:
    folder = Path(pymovements_input_root) / label / modality
    if not folder.exists():
        return []

    files = [
        str(path)
        for path in folder.glob("*.csv")
        if not path.name.startswith("~$")
    ]
    return sorted(files, key=lambda path: gaze_name_key(Path(path).stem))


def estimate_duration_xlsx(path: str) -> float:
    df = load_excel(path)
    return float(pd.to_numeric(df["time_sec"], errors="coerce").max())


def estimate_duration_csv(path: str) -> float:
    df = pd.read_csv(path, usecols=["time_sec"])
    return float(pd.to_numeric(df["time_sec"], errors="coerce").max())


def estimate_row_count_xlsx(path: str) -> int:
    df = load_excel(path)
    return len(df)


def estimate_row_count_csv(path: str) -> int:
    df = pd.read_csv(path, usecols=["frame"])
    return len(df)


def build_source_to_gaze_mapping(
    csvcleaned_root: str,
    pymovements_input_root: str,
    label: str,
    modality: str,
    duration_tolerance_sec: float,
    required_source_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    source_files = collect_source_files(csvcleaned_root, label, modality)
    if required_source_paths is not None:
        required_set = {os.path.abspath(path) for path in required_source_paths}
        source_files = [
            path for path in source_files
            if os.path.abspath(path) in required_set
        ]
    gaze_files = collect_gaze_files(pymovements_input_root, label, modality)

    mapping: Dict[str, str] = {}
    mismatch_messages: List[str] = []
    gaze_stats = {
        path: {
            "duration": estimate_duration_csv(path),
            "rows": estimate_row_count_csv(path),
        }
        for path in gaze_files
    }

    for source_path in source_files:
        source_duration = estimate_duration_xlsx(source_path)
        source_rows = estimate_row_count_xlsx(source_path)
        candidate_path = None
        candidate_score = None

        for gaze_path in gaze_files:
            gaze_duration = gaze_stats[gaze_path]["duration"]
            gaze_rows = gaze_stats[gaze_path]["rows"]
            duration_diff = abs(source_duration - gaze_duration)
            row_diff = abs(source_rows - gaze_rows)
            score = (round(duration_diff, 6), row_diff, gaze_name_key(Path(gaze_path).stem))

            if candidate_score is None or score < candidate_score:
                candidate_path = gaze_path
                candidate_score = score

        if candidate_path is None:
            mismatch_messages.append(
                f"{os.path.basename(source_path)} -> no gaze candidate"
            )
            continue

        if candidate_score is None or candidate_score[0] > duration_tolerance_sec:
            mismatch_messages.append(
                f"{os.path.basename(source_path)} -> {os.path.basename(candidate_path)} "
                f"(source={source_duration:.3f}s, gaze={gaze_stats[candidate_path]['duration']:.3f}s)"
            )
            continue

        mapping[os.path.abspath(source_path)] = os.path.abspath(candidate_path)

    if mismatch_messages:
        mismatch_text = "\n".join(mismatch_messages[:10])
        print(
            f"Warning while mapping {label}/{modality}: "
            f"{len(mismatch_messages)} source file(s) had no safe gaze match.\n{mismatch_text}"
        )

    return mapping


def load_normal_frames(path: str) -> pd.DataFrame:
    df = load_excel(path)
    if "frame" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing frame column.")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df.dropna(subset=["frame"]).copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)
    return df


def load_gaze_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = [col for col in PYMOVEMENTS_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing_text}")

    for col in PYMOVEMENTS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("frame").drop_duplicates(subset=["frame"]).reset_index(drop=True)
    return df


def build_output_path(normal_path: str, normal_root: str, output_root: str) -> str:
    relative_path = os.path.relpath(normal_path, normal_root)
    parts = relative_path.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Unexpected normal path structure: {normal_path}")

    modality = parts[0]
    file_name = parts[-1]
    base_name = file_name.replace("_normal_frames.xlsx", "_normal_pymovements_input.csv")

    target_dir = os.path.join(output_root, modality)
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, base_name)


def build_normal_pymovements_file(
    normal_path: str,
    normal_root: str,
    source_to_gaze_map: Dict[str, str],
    csvcleaned_root: str,
    output_root: str,
) -> Dict:
    source_path = os.path.abspath(infer_source_path(normal_path, csvcleaned_root))
    if source_path not in source_to_gaze_map:
        raise KeyError(f"No PyMovements source mapping found for {source_path}")

    gaze_path = source_to_gaze_map[source_path]

    normal_df = load_normal_frames(normal_path)
    gaze_df = load_gaze_input(gaze_path)

    selected_frames = set(normal_df["frame"].tolist())
    output_df = gaze_df[gaze_df["frame"].isin(selected_frames)].copy()
    output_df = output_df.sort_values("frame").reset_index(drop=True)

    missing_frames = sorted(selected_frames - set(output_df["frame"].tolist()))
    if missing_frames:
        missing_preview = ", ".join(str(frame) for frame in missing_frames[:10])
        raise ValueError(
            f"{os.path.basename(normal_path)} missing {len(missing_frames)} frame(s) "
            f"in gaze input. First missing: {missing_preview}"
        )

    output_path = build_output_path(normal_path, normal_root, output_root)
    output_df.to_csv(output_path, index=False)

    return {
        "normal_file": os.path.basename(normal_path),
        "source_file": os.path.basename(source_path),
        "gaze_input_file": os.path.basename(gaze_path),
        "output_file": output_path,
        "normal_frame_count": len(normal_df),
        "output_row_count": len(output_df),
        "window_label": "normal",
        "status": "ok",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PyMovements input CSV files for the derived normal class."
    )
    parser.add_argument("--normal-root", default=NORMAL_ROOT_DEFAULT, help="Root folder of csvcleaned normal frames.")
    parser.add_argument("--csvcleaned-root", default=CSVCLEANED_ROOT_DEFAULT, help="Root folder of csvcleaned source files.")
    parser.add_argument("--pymovements-input-root", default=PYMOVEMENTS_INPUT_ROOT_DEFAULT, help="Root folder of PyMovements input CSV files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT, help="Root folder for normal PyMovements input CSV files.")
    parser.add_argument("--report-path", default=REPORT_PATH_DEFAULT, help="Path for the build summary report.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit file count for quick testing. 0 means no limit.")
    parser.add_argument("--duration-tolerance-sec", type=float, default=3.0, help="Maximum allowed duration difference while mapping source files.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BuildConfig(
        max_files=max(0, int(args.max_files)),
        duration_tolerance_sec=float(args.duration_tolerance_sec),
    )

    normal_files = discover_normal_files(args.normal_root, config.max_files)
    if not normal_files:
        raise FileNotFoundError(f"No normal frame files found under {args.normal_root}")

    source_maps: Dict[Tuple[str, str], Dict[str, str]] = {}
    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for normal_path in normal_files:
        relative_path = os.path.relpath(normal_path, args.normal_root)
        parts = relative_path.split(os.sep)
        modality = parts[0]
        label = parts[1]
        key = (label, modality)

        if key not in source_maps:
            required_source_paths = [
                os.path.abspath(infer_source_path(path, args.csvcleaned_root))
                for path in normal_files
                if os.path.relpath(path, args.normal_root).split(os.sep)[:2] == [modality, label]
            ]
            print(f"Building source mapping for {label}/{modality} ...")
            source_maps[key] = build_source_to_gaze_mapping(
                csvcleaned_root=args.csvcleaned_root,
                pymovements_input_root=args.pymovements_input_root,
                label=label,
                modality=modality,
                duration_tolerance_sec=config.duration_tolerance_sec,
                required_source_paths=required_source_paths,
            )

        print(f"Processing normal gaze input: {os.path.basename(normal_path)}")
        try:
            summary_rows.append(
                build_normal_pymovements_file(
                    normal_path=normal_path,
                    normal_root=args.normal_root,
                    source_to_gaze_map=source_maps[key],
                    csvcleaned_root=args.csvcleaned_root,
                    output_root=args.output_root,
                )
            )
        except Exception as exc:
            error_rows.append(
                {
                    "normal_file": os.path.basename(normal_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"  [ERROR] {exc}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_excel(args.report_path, index=False)
        print(f"Summary written to: {args.report_path}")

    if error_rows:
        error_path = os.path.join(args.output_root, "normal_pymovements_input_errors.xlsx")
        pd.DataFrame(error_rows).to_excel(error_path, index=False)
        print(f"Errors written to: {error_path}")


if __name__ == "__main__":
    main()
