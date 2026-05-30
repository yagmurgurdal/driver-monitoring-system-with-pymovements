import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scripts.gaze.build_normal_pymovements_inputs import build_source_to_gaze_mapping


CSV_ROOT_DEFAULT = os.path.join(os.getcwd(), "csvcleaned")
PYMOVEMENTS_INPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_input")
GAZE_WINDOW_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_dataset")
NORMAL_GAZE_WINDOW_ROOT_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_dataset_normal")
BASELINE_WINDOW_ROOT_DEFAULT = r"D:\window" if os.path.exists(r"D:\window") else os.path.join(os.getcwd(), "window_dataset")
OUTPUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "window_dataset_with_gaze")
SUMMARY_PATH_DEFAULT = os.path.join(os.getcwd(), "window_dataset_with_gaze_summary.xlsx")

LABELS = ("distraction", "drowsiness")
MODALITIES = ("IR", "RGB")


def discover_baseline_files(root: str, label: str, modality: str) -> List[str]:
    folder = Path(root) / label / modality
    if not folder.exists():
        return []
    return sorted(str(path) for path in folder.glob("*_windows.xlsx") if not path.name.startswith("~$"))


def discover_normal_baseline_files(root: str, modality: str) -> List[str]:
    folder = Path(root) / "normal" / modality
    if not folder.exists():
        return []
    return sorted(str(path) for path in folder.glob("*_windows.xlsx") if not path.name.startswith("~$"))


def prefix_gaze_columns(gaze_df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in gaze_df.columns:
        if col == "window_id":
            continue
        renamed[col] = f"gaze_{col}"
    return gaze_df.rename(columns=renamed)


def save_merged_file(df: pd.DataFrame, output_root: str, label: str, modality: str, file_stem: str) -> str:
    output_dir = os.path.join(output_root, label, modality)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_stem}_windows_with_gaze.xlsx")
    df.to_excel(output_path, index=False)
    return output_path


def merge_one_file(
    baseline_path: str,
    gaze_path: str,
    output_root: str,
    label: str,
    modality: str,
) -> Dict:
    baseline_df = pd.read_excel(baseline_path)
    gaze_df = pd.read_excel(gaze_path)
    gaze_df = prefix_gaze_columns(gaze_df)

    merged_df = baseline_df.merge(gaze_df, on="window_id", how="left", validate="one_to_one")
    file_stem = Path(baseline_path).stem.replace("_windows", "")
    output_path = save_merged_file(merged_df, output_root, label, modality, file_stem)

    matched_rows = int(merged_df["gaze_source_file"].notna().sum()) if "gaze_source_file" in merged_df.columns else 0
    return {
        "label": label,
        "modality": modality,
        "file_stem": file_stem,
        "baseline_file": baseline_path,
        "gaze_file": gaze_path,
        "output_file": output_path,
        "baseline_rows": len(baseline_df),
        "gaze_rows": len(gaze_df),
        "matched_rows": matched_rows,
        "missing_rows": int(len(baseline_df) - matched_rows),
        "status": "ok" if matched_rows == len(baseline_df) else "partial",
    }


def build_label_mappings(csvcleaned_root: str, pymovements_input_root: str, baseline_root: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    mappings: Dict[Tuple[str, str], Dict[str, str]] = {}
    for label in LABELS:
        for modality in MODALITIES:
            baseline_files = discover_baseline_files(baseline_root, label, modality)
            required_sources = [
                os.path.abspath(os.path.join(csvcleaned_root, label, modality, f"{Path(path).stem.replace('_windows', '')}.xlsx"))
                for path in baseline_files
            ]
            mappings[(label, modality)] = build_source_to_gaze_mapping(
                csvcleaned_root=csvcleaned_root,
                pymovements_input_root=pymovements_input_root,
                label=label,
                modality=modality,
                duration_tolerance_sec=3.0,
                required_source_paths=required_sources,
            )
    return mappings


def resolve_gaze_window_for_label(
    baseline_path: str,
    csvcleaned_root: str,
    gaze_window_root: str,
    mapping: Dict[str, str],
    label: str,
    modality: str,
) -> str:
    baseline_stem = Path(baseline_path).stem.replace("_windows", "")
    source_path = os.path.abspath(os.path.join(csvcleaned_root, label, modality, f"{baseline_stem}.xlsx"))
    if source_path not in mapping:
        raise FileNotFoundError(f"No gaze source mapping found for {source_path}")

    gaze_csv_path = mapping[source_path]
    gaze_source_stem = Path(gaze_csv_path).stem.replace("_pymovements_input", "")
    gaze_window_path = os.path.join(gaze_window_root, label, modality, f"{gaze_source_stem}_gaze_windows.xlsx")
    if not os.path.exists(gaze_window_path):
        raise FileNotFoundError(f"Gaze window file not found: {gaze_window_path}")
    return gaze_window_path


def resolve_normal_gaze_window(baseline_path: str, normal_gaze_window_root: str, modality: str) -> str:
    baseline_stem = Path(baseline_path).stem.replace("_windows", "")
    candidates = [
        os.path.join(normal_gaze_window_root, modality, f"{baseline_stem}_normal_gaze_windows.xlsx"),
        os.path.join(normal_gaze_window_root, "normal", modality, f"{baseline_stem}_normal_gaze_windows.xlsx"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Normal gaze window file not found for {baseline_stem}")


def build_merged_dataset(
    baseline_root: str,
    gaze_window_root: str,
    normal_gaze_window_root: str,
    csvcleaned_root: str,
    pymovements_input_root: str,
    output_root: str,
    summary_path: str,
    max_files: int,
):
    os.makedirs(output_root, exist_ok=True)

    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []
    mappings = build_label_mappings(csvcleaned_root, pymovements_input_root, baseline_root)

    for label in LABELS:
        for modality in MODALITIES:
            baseline_files = discover_baseline_files(baseline_root, label, modality)
            if max_files > 0:
                baseline_files = baseline_files[:max_files]

            for baseline_path in baseline_files:
                print(f"Merging {label}/{modality}: {os.path.basename(baseline_path)}")
                try:
                    gaze_path = resolve_gaze_window_for_label(
                        baseline_path=baseline_path,
                        csvcleaned_root=csvcleaned_root,
                        gaze_window_root=gaze_window_root,
                        mapping=mappings[(label, modality)],
                        label=label,
                        modality=modality,
                    )
                    summary_rows.append(merge_one_file(baseline_path, gaze_path, output_root, label, modality))
                except Exception as exc:
                    error_rows.append(
                        {
                            "label": label,
                            "modality": modality,
                            "baseline_file": baseline_path,
                            "error": str(exc),
                        }
                    )
                    print(f"  [ERROR] {exc}")

    for modality in MODALITIES:
        baseline_files = discover_normal_baseline_files(baseline_root, modality)
        if max_files > 0:
            baseline_files = baseline_files[:max_files]

        for baseline_path in baseline_files:
            print(f"Merging normal/{modality}: {os.path.basename(baseline_path)}")
            try:
                gaze_path = resolve_normal_gaze_window(baseline_path, normal_gaze_window_root, modality)
                summary_rows.append(merge_one_file(baseline_path, gaze_path, output_root, "normal", modality))
            except Exception as exc:
                error_rows.append(
                    {
                        "label": "normal",
                        "modality": modality,
                        "baseline_file": baseline_path,
                        "error": str(exc),
                    }
                )
                print(f"  [ERROR] {exc}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
        print(f"Summary written to: {summary_path}")

    if error_rows:
        error_path = os.path.join(output_root, "window_with_gaze_errors.xlsx")
        pd.DataFrame(error_rows).to_excel(error_path, index=False)
        print(f"Errors written to: {error_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge baseline window datasets with PyMovements gaze window features.")
    parser.add_argument("--baseline-window-root", default=BASELINE_WINDOW_ROOT_DEFAULT, help="Root folder of baseline window datasets.")
    parser.add_argument("--gaze-window-root", default=GAZE_WINDOW_ROOT_DEFAULT, help="Root folder of distraction/drowsiness gaze window datasets.")
    parser.add_argument("--normal-gaze-window-root", default=NORMAL_GAZE_WINDOW_ROOT_DEFAULT, help="Root folder of aligned normal gaze window datasets.")
    parser.add_argument("--csvcleaned-root", default=CSV_ROOT_DEFAULT, help="Root folder of csvcleaned source files.")
    parser.add_argument("--pymovements-input-root", default=PYMOVEMENTS_INPUT_ROOT_DEFAULT, help="Root folder of PyMovements input CSV files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT, help="Output root for merged per-file window datasets.")
    parser.add_argument("--summary-path", default=SUMMARY_PATH_DEFAULT, help="Excel path for summary output.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files per label/modality for quick testing. 0 means no limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_merged_dataset(
        baseline_root=args.baseline_window_root,
        gaze_window_root=args.gaze_window_root,
        normal_gaze_window_root=args.normal_gaze_window_root,
        csvcleaned_root=args.csvcleaned_root,
        pymovements_input_root=args.pymovements_input_root,
        output_root=args.output_root,
        summary_path=args.summary_path,
        max_files=max(0, int(args.max_files)),
    )


if __name__ == "__main__":
    main()
