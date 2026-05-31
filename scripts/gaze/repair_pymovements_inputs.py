import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from scripts.gaze.extract_pymovements_inputs import (
    DEFAULT_DISTRACTION_ROOT,
    DEFAULT_DROWSINESS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    CSV_COLUMNS,
    ExtractionConfig,
    build_output_path,
    collect_videos,
    count_output_rows,
    expected_output_rows,
    get_face_mesh,
    is_completed_output,
    process_video,
)


REPORT_PATH_DEFAULT = os.path.join(os.getcwd(), "reports", "gaze", "repair", "pymovements_input_repair_report.xlsx")


def classify_output(output_path: str, expected_rows: int) -> Tuple[str, int]:
    if not os.path.exists(output_path):
        return "missing", 0

    try:
        with open(output_path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip()
        actual_rows = count_output_rows(output_path)
    except OSError:
        return "unreadable", 0

    expected_header = ",".join(CSV_COLUMNS)
    if header != expected_header:
        return "header_mismatch", actual_rows

    if expected_rows <= 0:
        return "unknown_expected", actual_rows

    if actual_rows == 0:
        return "empty", actual_rows

    if actual_rows == expected_rows:
        return "complete", actual_rows

    if actual_rows < expected_rows:
        return "incomplete", actual_rows

    return "row_mismatch", actual_rows


def audit_group(
    label: str,
    root: str,
    output_root: str,
    config: ExtractionConfig,
) -> Tuple[List[Dict], List[Dict]]:
    audit_rows: List[Dict] = []
    faulty_rows: List[Dict] = []

    for modality in ("IR", "RGB"):
        videos = collect_videos(root, modality, config)
        for video_path in videos:
            output_path = build_output_path(output_root, label, modality, video_path)
            expected_rows = expected_output_rows(video_path, config.frame_stride)
            status, actual_rows = classify_output(output_path, expected_rows)

            row = {
                "label": label,
                "modality": modality,
                "video_name": video_path.name,
                "video_path": str(video_path),
                "output_path": output_path,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "status": status,
                "row_delta": actual_rows - expected_rows if expected_rows > 0 else None,
            }
            audit_rows.append(row)

            if status != "complete":
                faulty_rows.append(row)

    return audit_rows, faulty_rows


def repair_group(
    label: str,
    modality: str,
    rows: List[Dict],
    config: ExtractionConfig,
) -> List[Dict]:
    repaired_rows: List[Dict] = []
    if not rows:
        return repaired_rows

    face_mesh = get_face_mesh()
    try:
        for row in rows:
            video_path = row["video_path"]
            output_path = row["output_path"]
            print(f"Repairing {label}/{modality}: {row['video_name']} ({row['status']})")
            written_rows = process_video(video_path=video_path, output_path=output_path, face_mesh=face_mesh, config=config)
            expected_rows = row["expected_rows"]
            new_status, actual_rows = classify_output(output_path, expected_rows)
            repaired_rows.append(
                {
                    **row,
                    "repaired_rows_written": written_rows,
                    "status_after_repair": new_status,
                    "actual_rows_after_repair": actual_rows,
                }
            )
            print(f"  Saved: {output_path}")
            print(f"  Rows written: {written_rows}")
            print(f"  Status after repair: {new_status}")
    finally:
        face_mesh.close()

    return repaired_rows


def write_report(report_path: str, audit_rows: List[Dict], repaired_rows: List[Dict], config: ExtractionConfig):
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="audit", index=False)
        if repaired_rows:
            pd.DataFrame(repaired_rows).to_excel(writer, sheet_name="repaired", index=False)
        pd.DataFrame([asdict(config)]).to_excel(writer, sheet_name="config", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit PyMovements input CSV files and repair only missing or incomplete outputs."
    )
    parser.add_argument("--distraction-root", default=DEFAULT_DISTRACTION_ROOT, help="Root folder for distraction videos.")
    parser.add_argument("--drowsiness-root", default=DEFAULT_DROWSINESS_ROOT, help="Root folder for drowsiness videos.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output folder for PyMovements input CSV files.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Expected frame stride used during extraction.")
    parser.add_argument("--max-videos-per-group", type=int, default=0, help="Limit videos per label/modality for testing. 0 means no limit.")
    parser.add_argument("--report-path", default=REPORT_PATH_DEFAULT, help="Path to the audit/repair report.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only write the audit report; do not repair faulty files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExtractionConfig(
        frame_stride=max(1, args.frame_stride),
        max_videos_per_group=max(0, args.max_videos_per_group),
        overwrite_existing=False,
    )

    all_audit_rows: List[Dict] = []
    faulty_rows_by_group: Dict[Tuple[str, str], List[Dict]] = {}

    for label, root in (
        ("distraction", args.distraction_root),
        ("drowsiness", args.drowsiness_root),
    ):
        audit_rows, faulty_rows = audit_group(label, root, args.output_root, config)
        all_audit_rows.extend(audit_rows)
        for row in faulty_rows:
            key = (row["label"], row["modality"])
            faulty_rows_by_group.setdefault(key, []).append(row)

    total_faulty = sum(len(rows) for rows in faulty_rows_by_group.values())
    print(f"Audit complete. Faulty files found: {total_faulty}")

    repaired_rows: List[Dict] = []
    if not args.report_only:
        for (label, modality), rows in sorted(faulty_rows_by_group.items()):
            repaired_rows.extend(repair_group(label, modality, rows, config))

    write_report(args.report_path, all_audit_rows, repaired_rows, config)
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
