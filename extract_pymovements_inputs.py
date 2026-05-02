import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv")

LEFT_EYE_IDS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDS = (362, 385, 387, 263, 373, 380)

# MediaPipe FaceMesh iris landmarks, available when refine_landmarks=True
LEFT_IRIS_IDS = tuple(sorted({i for pair in mp.solutions.face_mesh.FACEMESH_LEFT_IRIS for i in pair}))
RIGHT_IRIS_IDS = tuple(sorted({i for pair in mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS for i in pair}))

DEFAULT_DISTRACTION_ROOT = r"D:\DMD Dataset-pymovements\distractionrgb\dmd"
DEFAULT_DROWSINESS_ROOT = r"D:\DMD Dataset-pymovements\drowsiness"
DEFAULT_OUTPUT_ROOT = os.path.join(os.getcwd(), "pymovements_input")


CSV_COLUMNS = [
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
class ExtractionConfig:
    frame_stride: int = 1
    max_videos_per_group: int = 0
    prefer_path_keyword: str = "dmd"
    overwrite_existing: bool = False


def safe_round(value: Optional[float], digits: int = 6):
    if value is None:
        return ""
    return round(float(value), digits)


def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_points_2d(landmarks, ids: Sequence[int], width: int, height: int) -> np.ndarray:
    points = np.zeros((len(ids), 2), dtype=np.float64)
    for idx, landmark_id in enumerate(ids):
        lm = landmarks[landmark_id]
        points[idx, 0] = lm.x * width
        points[idx, 1] = lm.y * height
    return points


def average_point(points: np.ndarray) -> Tuple[float, float]:
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def normalize_iris_position(
    iris_center: np.ndarray,
    eye_points: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    if eye_points.shape != (6, 2):
        return None, None

    p1, p2, p3, p4, p5, p6 = eye_points

    # Use the two horizontal eye corners and upper/lower eyelid centers to get
    # head-motion-reduced gaze-like coordinates in the eye-local frame.
    if p1[0] <= p4[0]:
        left_corner, right_corner = p1, p4
    else:
        left_corner, right_corner = p4, p1

    horizontal_axis = right_corner - left_corner
    horizontal_norm_sq = float(np.dot(horizontal_axis, horizontal_axis))
    if horizontal_norm_sq < 1e-6:
        return None, None

    upper_center = (p2 + p3) / 2.0
    lower_center = (p5 + p6) / 2.0
    vertical_axis = lower_center - upper_center
    vertical_norm_sq = float(np.dot(vertical_axis, vertical_axis))
    if vertical_norm_sq < 1e-6:
        return None, None

    x_norm = float(np.dot(iris_center - left_corner, horizontal_axis) / horizontal_norm_sq)
    y_norm = float(np.dot(iris_center - upper_center, vertical_axis) / vertical_norm_sq)
    return x_norm, y_norm


def choose_unique_videos(paths: Iterable[Path], prefer_keyword: str) -> List[Path]:
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        grouped.setdefault(path.name.lower(), []).append(path)

    unique_paths: List[Path] = []
    for _, candidates in sorted(grouped.items()):
        candidates = sorted(candidates, key=lambda p: (prefer_keyword.lower() not in str(p).lower(), len(str(p)), str(p)))
        unique_paths.append(candidates[0])
    return unique_paths


def collect_videos(root: str, modality: str, config: ExtractionConfig) -> List[Path]:
    base = Path(root)
    if not base.exists():
        return []

    modality_token = f"_{modality.lower()}_face_std"
    candidates = [
        path
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and modality_token in path.name.lower()
    ]

    unique_paths = choose_unique_videos(candidates, config.prefer_path_keyword)
    if config.max_videos_per_group > 0:
        unique_paths = unique_paths[: config.max_videos_per_group]
    return unique_paths


def build_output_path(output_root: str, label: str, modality: str, video_path: Path) -> str:
    output_dir = os.path.join(output_root, label, modality)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{video_path.stem}_pymovements_input.csv")


def expected_output_rows(video_path: Path, frame_stride: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not frame_count or frame_count <= 0 or np.isnan(frame_count):
            return 0
        total_frames = int(round(frame_count))
        stride = max(1, int(frame_stride))
        return (total_frames + stride - 1) // stride
    finally:
        cap.release()


def count_output_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        return sum(1 for line in handle if line.strip())


def is_completed_output(path: str, expected_rows: int) -> bool:
    if not os.path.exists(path):
        return False

    try:
        if os.path.getsize(path) <= 0:
            return False

        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip()
        row_count = count_output_rows(path)
    except OSError:
        return False

    expected_header = ",".join(CSV_COLUMNS)
    return header == expected_header and row_count == expected_rows and expected_rows > 0


def process_video(video_path: Path, output_path: str, face_mesh, config: ExtractionConfig):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    temp_output_path = f"{output_path}.tmp"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    try:
        with open(temp_output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)

            frame_index = 0
            written_rows = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None:
                    continue

                frame_index += 1
                if config.frame_stride > 1 and frame_index % config.frame_stride != 0:
                    continue

                time_sec = frame_index / fps
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                face_detected = 0
                left_center = (None, None)
                right_center = (None, None)
                iris_center = (None, None)
                left_norm = (None, None)
                right_norm = (None, None)
                center_norm = (None, None)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    face_detected = 1

                    left_eye = get_points_2d(landmarks, LEFT_EYE_IDS, width, height)
                    right_eye = get_points_2d(landmarks, RIGHT_EYE_IDS, width, height)
                    left_iris = get_points_2d(landmarks, LEFT_IRIS_IDS, width, height)
                    right_iris = get_points_2d(landmarks, RIGHT_IRIS_IDS, width, height)

                    left_center = average_point(left_iris)
                    right_center = average_point(right_iris)

                    left_center_np = np.array(left_center, dtype=np.float64)
                    right_center_np = np.array(right_center, dtype=np.float64)
                    iris_center_np = (left_center_np + right_center_np) / 2.0
                    iris_center = (float(iris_center_np[0]), float(iris_center_np[1]))

                    left_norm = normalize_iris_position(left_center_np, left_eye)
                    right_norm = normalize_iris_position(right_center_np, right_eye)

                    if left_norm[0] is not None and right_norm[0] is not None:
                        center_norm = (
                            float((left_norm[0] + right_norm[0]) / 2.0),
                            float((left_norm[1] + right_norm[1]) / 2.0),
                        )

                writer.writerow(
                    [
                        frame_index,
                        safe_round(time_sec, 6),
                        face_detected,
                        safe_round(left_center[0], 6),
                        safe_round(left_center[1], 6),
                        safe_round(right_center[0], 6),
                        safe_round(right_center[1], 6),
                        safe_round(iris_center[0], 6),
                        safe_round(iris_center[1], 6),
                        safe_round(left_norm[0], 6),
                        safe_round(left_norm[1], 6),
                        safe_round(right_norm[0], 6),
                        safe_round(right_norm[1], 6),
                        safe_round(center_norm[0], 6),
                        safe_round(center_norm[1], 6),
                    ]
                )
                written_rows += 1

        os.replace(temp_output_path, output_path)
        return written_rows
    finally:
        cap.release()
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def extract_group(label: str, root: str, output_root: str, config: ExtractionConfig):
    for modality in ("IR", "RGB"):
        videos = collect_videos(root, modality, config)
        print(f"{label}/{modality}: {len(videos)} video(s) found")
        if not videos:
            continue

        face_mesh = get_face_mesh()
        processed_count = 0
        skipped_count = 0
        try:
            for video_path in videos:
                output_path = build_output_path(output_root, label, modality, video_path)
                expected_rows = expected_output_rows(video_path, config.frame_stride)

                if not config.overwrite_existing and is_completed_output(output_path, expected_rows):
                    skipped_count += 1
                    print(f"Skipping existing: {video_path.name}")
                    continue

                print(f"Processing: {video_path.name}")
                rows = process_video(video_path, output_path, face_mesh, config)
                print(f"  Saved: {output_path}")
                print(f"  Rows written: {rows}")
                processed_count += 1
        finally:
            face_mesh.close()
        print(f"{label}/{modality}: processed={processed_count}, skipped_existing={skipped_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract PyMovements-ready iris x/y time series from source videos."
    )
    parser.add_argument("--distraction-root", default=DEFAULT_DISTRACTION_ROOT, help="Root folder for distraction videos.")
    parser.add_argument("--drowsiness-root", default=DEFAULT_DROWSINESS_ROOT, help="Root folder for drowsiness videos.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output folder for PyMovements input CSV files.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-videos-per-group", type=int, default=0, help="Limit videos per label/modality for testing. 0 means no limit.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Reprocess files even if a completed output CSV already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExtractionConfig(
        frame_stride=max(1, args.frame_stride),
        max_videos_per_group=max(0, args.max_videos_per_group),
        overwrite_existing=bool(args.overwrite_existing),
    )

    extract_group("distraction", args.distraction_root, args.output_root, config)
    extract_group("drowsiness", args.drowsiness_root, args.output_root, config)


if __name__ == "__main__":
    main()
