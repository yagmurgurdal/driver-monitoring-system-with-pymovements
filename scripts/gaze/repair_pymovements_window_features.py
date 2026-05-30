import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from scripts.gaze import build_pymovements_window_features as window_builder


REPORT_PATH_DEFAULT = os.path.join(os.getcwd(), "pymovements_window_repair_report.xlsx")


def expected_window_count_from_csv(csv_path: str, config: window_builder.WindowConfig) -> int:
    df = window_builder.load_input_csv(csv_path)
    if df.empty:
        return 0

    last_time_sec = float(df["time_sec"].max())
    count = 0
    for _window_id, _start_time, _end_time in window_builder.iter_window_ranges(
        last_time_sec, config.window_sec, config.step_sec
    ):
        count += 1
    return count


def classify_window_output(
    csv_path: str,
    output_path: str,
    config: window_builder.WindowConfig,
) -> Tuple[str, int, int]:
    expected_rows = expected_window_count_from_csv(csv_path, config)

    if not os.path.exists(output_path):
        return "missing", expected_rows, 0

    try:
        df = pd.read_excel(output_path)
    except Exception:
        return "unreadable", expected_rows, 0

    actual_rows = len(df)
    required_columns = {
        "window_id",
        "window_start_time",
        "window_end_time",
        "gaze_valid_ratio",
        "usable_window",
    }
    if not required_columns.issubset(df.columns):
        return "schema_mismatch", expected_rows, actual_rows

    if expected_rows <= 0:
        return "unknown_expected", expected_rows, actual_rows

    if actual_rows == 0:
        return "empty", expected_rows, actual_rows

    if actual_rows < expected_rows:
        return "incomplete", expected_rows, actual_rows

    if actual_rows > expected_rows:
        return "row_mismatch", expected_rows, actual_rows

    return "complete", expected_rows, actual_rows


def audit_outputs(
    input_root: str,
    output_root: str,
    config: window_builder.WindowConfig,
) -> Tuple[List[Dict], List[Dict]]:
    audit_rows: List[Dict] = []
    faulty_rows: List[Dict] = []

    input_files = window_builder.discover_input_files(input_root, config.max_files)
    for csv_path in input_files:
        metadata = window_builder.parse_metadata(csv_path, input_root)
        output_path = window_builder.build_output_path(csv_path, input_root, output_root)
        status, expected_rows, actual_rows = classify_window_output(csv_path, output_path, config)

        row = {
            "label": metadata["label"],
            "modality": metadata["modality"],
            "source_file": metadata["file_name"],
            "source_stem": metadata["source_stem"],
            "csv_path": csv_path,
            "output_path": output_path,
            "expected_window_rows": expected_rows,
            "actual_window_rows": actual_rows,
            "status": status,
            "row_delta": actual_rows - expected_rows if expected_rows > 0 else None,
        }
        audit_rows.append(row)
        if status != "complete":
            faulty_rows.append(row)

    return audit_rows, faulty_rows


def repair_outputs(
    faulty_rows: List[Dict],
    input_root: str,
    output_root: str,
    config: window_builder.WindowConfig,
) -> List[Dict]:
    repaired_rows: List[Dict] = []
    for row in faulty_rows:
        print(
            f"Repairing {row['label']}/{row['modality']}: "
            f"{row['source_file']} ({row['status']})"
        )
        summary = window_builder.build_windows_for_file(
            row["csv_path"],
            input_root,
            output_root,
            config,
        )
        status_after, expected_after, actual_after = classify_window_output(
            row["csv_path"],
            row["output_path"],
            config,
        )
        repaired_rows.append(
            {
                **row,
                **summary,
                "status_after_repair": status_after,
                "expected_window_rows_after_repair": expected_after,
                "actual_window_rows_after_repair": actual_after,
            }
        )
        print(f"  Status after repair: {status_after}")
        print(f"  Windows after repair: {actual_after}")

    return repaired_rows


def write_report(
    report_path: str,
    audit_rows: List[Dict],
    repaired_rows: List[Dict],
    config: window_builder.WindowConfig,
):
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="audit", index=False)
        if repaired_rows:
            pd.DataFrame(repaired_rows).to_excel(writer, sheet_name="repaired", index=False)
        pd.DataFrame([asdict(config)]).to_excel(writer, sheet_name="config", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit gaze window XLSX files and repair only missing or incomplete outputs."
    )
    parser.add_argument(
        "--input-root",
        default=window_builder.INPUT_ROOT_DEFAULT,
        help="Root folder containing PyMovements input CSV files.",
    )
    parser.add_argument(
        "--output-root",
        default=window_builder.OUTPUT_ROOT_DEFAULT,
        help="Root folder containing gaze window XLSX files.",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_PATH_DEFAULT,
        help="Path to the audit/repair report.",
    )
    parser.add_argument("--window-sec", type=float, default=3.0, help="Window duration in seconds.")
    parser.add_argument("--step-sec", type=float, default=3.0, help="Window step in seconds.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.50, help="Minimum valid gaze ratio for a usable window.")
    parser.add_argument("--idt-min-duration-ms", type=int, default=100, help="Minimum I-DT fixation duration in milliseconds.")
    parser.add_argument("--idt-dispersion-threshold", type=float, default=1.0, help="I-DT dispersion threshold.")
    parser.add_argument("--ivt-min-duration-ms", type=int, default=100, help="Minimum I-VT fixation duration in milliseconds.")
    parser.add_argument("--ivt-velocity-threshold", type=float, default=20.0, help="I-VT velocity threshold.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit the number of files for quick testing. 0 means no limit.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only write the audit report; do not repair faulty files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = window_builder.WindowConfig(
        window_sec=float(args.window_sec),
        step_sec=float(args.step_sec),
        min_valid_ratio=float(args.min_valid_ratio),
        idt_min_duration_ms=int(args.idt_min_duration_ms),
        idt_dispersion_threshold=float(args.idt_dispersion_threshold),
        ivt_min_duration_ms=int(args.ivt_min_duration_ms),
        ivt_velocity_threshold=float(args.ivt_velocity_threshold),
        max_files=max(0, int(args.max_files)),
    )

    audit_rows, faulty_rows = audit_outputs(args.input_root, args.output_root, config)
    print(f"Audit complete. Faulty window files found: {len(faulty_rows)}")

    repaired_rows: List[Dict] = []
    if not args.report_only:
        repaired_rows = repair_outputs(faulty_rows, args.input_root, args.output_root, config)

    write_report(args.report_path, audit_rows, repaired_rows, config)
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
