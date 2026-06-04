from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.gaze.extract_pymovements_inputs import (
    LEFT_IRIS_IDS,
    RIGHT_IRIS_IDS,
    average_point,
    get_points_2d,
    normalize_iris_position,
)
from scripts.utils.driver_monitoring_features_fixed import (
    EAR_SUSPICIOUS_YAW_DEG,
    FAST_MODE,
    HEAD_POSE_IDS,
    LEFT_EYE_IDS,
    RIGHT_EYE_IDS,
    build_camera_matrix,
    calculate_ear_robust,
    extract_pose_angles,
    fill_points_2d,
    get_face_mesh,
    is_pose_valid,
    resize_for_facemesh,
    solve_head_pose,
)


@dataclass(frozen=True)
class UnifiedExtractionResult:
    frame_features: pd.DataFrame
    gaze_samples: pd.DataFrame


def _safe_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def extract_video_features(
    video_path: str | Path,
    fast_mode: bool = FAST_MODE,
) -> UnifiedExtractionResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    face_mesh = get_face_mesh(fast_mode=fast_mode)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        frame_rows: list[dict[str, float | int | None]] = []
        gaze_rows: list[dict[str, float | int | None]] = []

        cam_matrix = None
        min_h_px = None
        last_w = None
        last_h = None

        face_2d = np.empty((len(HEAD_POSE_IDS), 2), dtype=np.float64)
        left_pts = np.empty((len(LEFT_EYE_IDS), 2), dtype=np.float64)
        right_pts = np.empty((len(RIGHT_EYE_IDS), 2), dtype=np.float64)

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue

            frame_index += 1
            time_sec = frame_index / fps

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            h_img, w_img = frame.shape[:2]
            if w_img != last_w or h_img != last_h:
                cam_matrix = build_camera_matrix(w_img, h_img)
                min_h_px = max(2.0, w_img * 0.005)
                last_w, last_h = w_img, h_img

            frame_for_mesh = resize_for_facemesh(frame, fast_mode=fast_mode)
            frame_rgb = cv2.cvtColor(frame_for_mesh, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = face_mesh.process(frame_rgb)

            face_detected = 0
            pose_valid = 0
            ear_valid = 0
            ear_suspicious = 0

            yaw = None
            pitch = None
            roll = None
            left_ear = None
            right_ear = None
            avg_ear = None

            left_center = (None, None)
            right_center = (None, None)
            iris_center = (None, None)
            left_norm = (None, None)
            right_norm = (None, None)
            center_norm = (None, None)

            if results.multi_face_landmarks:
                face_detected = 1
                landmarks = results.multi_face_landmarks[0].landmark

                fill_points_2d(landmarks, HEAD_POSE_IDS, w_img, h_img, face_2d)
                pose_solution = solve_head_pose(face_2d, cam_matrix)
                if pose_solution is not None:
                    pitch, yaw, roll = extract_pose_angles(pose_solution, frame_index)
                    pose_valid = is_pose_valid(
                        pitch,
                        yaw,
                        roll,
                        pose_solution["reproj_error_px"],
                    )
                    if not pose_valid:
                        pitch = None
                        yaw = None
                        roll = None

                fill_points_2d(landmarks, LEFT_EYE_IDS, w_img, h_img, left_pts)
                fill_points_2d(landmarks, RIGHT_EYE_IDS, w_img, h_img, right_pts)
                left_ear, l_valid = calculate_ear_robust(left_pts, min_horizontal_px=min_h_px)
                right_ear, r_valid = calculate_ear_robust(right_pts, min_horizontal_px=min_h_px)

                if l_valid and r_valid:
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_valid = 1
                    pose_unknown = yaw is None
                    yaw_extreme = yaw is not None and abs(yaw) > EAR_SUSPICIOUS_YAW_DEG
                    if pose_unknown or yaw_extreme:
                        ear_suspicious = 1

                left_eye = get_points_2d(landmarks, LEFT_EYE_IDS, w_img, h_img)
                right_eye = get_points_2d(landmarks, RIGHT_EYE_IDS, w_img, h_img)
                left_iris = get_points_2d(landmarks, LEFT_IRIS_IDS, w_img, h_img)
                right_iris = get_points_2d(landmarks, RIGHT_IRIS_IDS, w_img, h_img)

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

            frame_rows.append(
                {
                    "frame": frame_index,
                    "time_sec": _safe_float(time_sec, 6),
                    "face_detected": face_detected,
                    "yaw": _safe_float(yaw, 4),
                    "pitch": _safe_float(pitch, 4),
                    "roll": _safe_float(roll, 4),
                    "pose_valid": pose_valid,
                    "left_ear": _safe_float(left_ear, 4),
                    "right_ear": _safe_float(right_ear, 4),
                    "avg_ear": _safe_float(avg_ear, 4),
                    "ear_valid": ear_valid,
                    "ear_suspicious": ear_suspicious,
                }
            )

            gaze_rows.append(
                {
                    "frame": frame_index,
                    "time_sec": _safe_float(time_sec, 6),
                    "face_detected": face_detected,
                    "left_iris_center_x_px": _safe_float(left_center[0], 6),
                    "left_iris_center_y_px": _safe_float(left_center[1], 6),
                    "right_iris_center_x_px": _safe_float(right_center[0], 6),
                    "right_iris_center_y_px": _safe_float(right_center[1], 6),
                    "iris_center_x_px": _safe_float(iris_center[0], 6),
                    "iris_center_y_px": _safe_float(iris_center[1], 6),
                    "left_iris_x_norm": _safe_float(left_norm[0], 6),
                    "left_iris_y_norm": _safe_float(left_norm[1], 6),
                    "right_iris_x_norm": _safe_float(right_norm[0], 6),
                    "right_iris_y_norm": _safe_float(right_norm[1], 6),
                    "iris_x_norm": _safe_float(center_norm[0], 6),
                    "iris_y_norm": _safe_float(center_norm[1], 6),
                }
            )
    finally:
        cap.release()
        face_mesh.close()

    return UnifiedExtractionResult(
        frame_features=pd.DataFrame(frame_rows),
        gaze_samples=pd.DataFrame(gaze_rows),
    )
