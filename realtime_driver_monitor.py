import argparse
import pickle
import sqlite3
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import pymovements as pm

try:
    import winsound
except ImportError:
    winsound = None

from build_pymovements_window_features import (
    boolean_run_sample_counts,
    compute_segment_velocity,
    duration_or_zero,
    estimate_expected_frame_step,
    estimate_sample_period_sec,
    events_to_sample_counts,
    split_contiguous_segments,
    summarize_durations_ms,
)
from driver_monitoring_features_fixed import (
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
from extract_pymovements_inputs import (
    LEFT_IRIS_IDS,
    RIGHT_IRIS_IDS,
    average_point,
    normalize_iris_position,
)
from train_random_forest import BASELINE_FEATURE_COLUMNS


EAR_CLOSED_THRESHOLD = 0.21
DEFAULT_MODEL_BUNDLE = Path("results") / "random_forest_gaze_high_confidence" / "model_bundle.pkl"
DEFAULT_DB_PATH = Path("results") / "realtime_monitor.db"
FRAME_DB_FLUSH_SIZE = 30

COLOR_BY_LABEL = {
    "normal": (0, 200, 0),
    "drowsiness": (0, 165, 255),
    "distraction": (0, 0, 255),
    "unknown": (180, 180, 180),
}

CAMERA_BACKENDS = (
    ("default", None),
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
)


@dataclass(frozen=True)
class RuntimeConfig:
    window_sec: float = 3.0
    predict_every_frames: int = 10
    min_face_ratio: float = 0.50
    min_pose_ratio: float = 0.40
    min_ear_ratio: float = 0.40
    ear_closed_threshold: float = EAR_CLOSED_THRESHOLD
    fast_mode: bool = FAST_MODE
    smoothing_windows: int = 3
    display_max_width: int = 960
    display_max_height: int = 540
    min_confidence: float = 0.55
    min_confidence_margin: float = 0.08
    switch_confirmations: int = 2
    unknown_confirmations: int = 2
    drowsiness_alert_seconds: float = 2.0
    drowsiness_alert_cooldown_sec: float = 4.0
    drowsiness_alert_enabled: bool = True


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def series_std(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    return float(clean.std(ddof=1))


def safe_float(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def load_model_bundle(path: Path) -> Dict:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    return bundle


def play_drowsiness_alert_sound():
    if winsound is None:
        print("\a", end="", flush=True)
        return

    try:
        winsound.PlaySound(
            "SystemExclamation",
            winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except RuntimeError:
        print("\a", end="", flush=True)


def open_video_source(video_path: str, camera_index: int):
    if video_path:
        cap = cv2.VideoCapture(video_path)
        return cap, f"video:{video_path}"

    tried = []
    for backend_name, backend_flag in CAMERA_BACKENDS:
        if backend_flag is None:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, backend_flag)

        if cap.isOpened():
            return cap, f"camera:{camera_index} backend:{backend_name}"

        tried.append(backend_name)
        cap.release()

    tried_text = ", ".join(tried)
    raise RuntimeError(
        f"Could not open camera index {camera_index}. Tried backends: {tried_text}. "
        "Check Windows camera permission, whether another app is using the webcam, "
        "or try --camera-index 1 / 2. You can also test with --video-path."
    )


class RealtimeDatabaseLogger:
    def __init__(
        self,
        db_path: Path,
        model_bundle_path: Path,
        source_description: str,
        config: RuntimeConfig,
        fps: float,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.frame_buffer: List[tuple] = []
        self._init_schema()
        self.session_id = self._create_session(
            model_bundle_path=model_bundle_path,
            source_description=source_description,
            config=config,
            fps=fps,
        )

    def _init_schema(self):
        cursor = self.connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                source TEXT NOT NULL,
                model_bundle_path TEXT NOT NULL,
                window_sec REAL NOT NULL,
                predict_every_frames INTEGER NOT NULL,
                min_face_ratio REAL NOT NULL,
                min_pose_ratio REAL NOT NULL,
                min_ear_ratio REAL NOT NULL,
                fps REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS frame_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                time_sec REAL NOT NULL,
                face_detected INTEGER NOT NULL,
                yaw REAL,
                pitch REAL,
                roll REAL,
                pose_valid INTEGER NOT NULL,
                left_ear REAL,
                right_ear REAL,
                avg_ear REAL,
                ear_valid INTEGER NOT NULL,
                ear_suspicious INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS window_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL,
                quality_usable INTEGER NOT NULL,
                perclos REAL,
                perclos_percent REAL,
                mean_ear REAL,
                std_ear REAL,
                min_ear REAL,
                max_ear REAL,
                mean_abs_yaw REAL,
                std_yaw REAL,
                max_abs_yaw REAL,
                mean_abs_pitch REAL,
                std_pitch REAL,
                max_abs_pitch REAL,
                mean_abs_roll REAL,
                std_roll REAL,
                max_abs_roll REAL,
                face_detect_ratio REAL,
                pose_valid_ratio REAL,
                ear_valid_ratio REAL,
                is_usable INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        self.connection.commit()

    def _create_session(
        self,
        model_bundle_path: Path,
        source_description: str,
        config: RuntimeConfig,
        fps: float,
    ) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (
                started_at, source, model_bundle_path, window_sec, predict_every_frames,
                min_face_ratio, min_pose_ratio, min_ear_ratio, fps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                source_description,
                str(model_bundle_path),
                float(config.window_sec),
                int(config.predict_every_frames),
                float(config.min_face_ratio),
                float(config.min_pose_ratio),
                float(config.min_ear_ratio),
                float(fps),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def log_frame(self, live_row: Dict):
        self.frame_buffer.append(
            (
                self.session_id,
                utc_now_iso(),
                int(live_row["frame"]),
                float(live_row["time_sec"]),
                int(live_row["face_detected"]),
                live_row.get("yaw"),
                live_row.get("pitch"),
                live_row.get("roll"),
                int(live_row["pose_valid"]),
                live_row.get("left_ear"),
                live_row.get("right_ear"),
                live_row.get("avg_ear"),
                int(live_row["ear_valid"]),
                int(live_row["ear_suspicious"]),
            )
        )
        if len(self.frame_buffer) >= FRAME_DB_FLUSH_SIZE:
            self.flush_frames()

    def flush_frames(self):
        if not self.frame_buffer:
            return
        self.connection.executemany(
            """
            INSERT INTO frame_measurements (
                session_id, created_at, frame_index, time_sec, face_detected, yaw, pitch, roll,
                pose_valid, left_ear, right_ear, avg_ear, ear_valid, ear_suspicious
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self.frame_buffer,
        )
        self.connection.commit()
        self.frame_buffer.clear()

    def log_prediction(
        self,
        frame_index: int,
        predicted_label: str,
        confidence: float,
        quality_usable: bool,
        window_features: Optional[Dict],
    ):
        feature_values = window_features or {}
        self.connection.execute(
            """
            INSERT INTO window_predictions (
                session_id, created_at, frame_index, predicted_label, confidence, quality_usable,
                perclos, perclos_percent, mean_ear, std_ear, min_ear, max_ear,
                mean_abs_yaw, std_yaw, max_abs_yaw, mean_abs_pitch, std_pitch, max_abs_pitch,
                mean_abs_roll, std_roll, max_abs_roll, face_detect_ratio, pose_valid_ratio,
                ear_valid_ratio, is_usable
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                utc_now_iso(),
                int(frame_index),
                predicted_label,
                None if confidence < 0 else float(confidence),
                int(bool(quality_usable)),
                feature_values.get("perclos"),
                feature_values.get("perclos_percent"),
                feature_values.get("mean_ear"),
                feature_values.get("std_ear"),
                feature_values.get("min_ear"),
                feature_values.get("max_ear"),
                feature_values.get("mean_abs_yaw"),
                feature_values.get("std_yaw"),
                feature_values.get("max_abs_yaw"),
                feature_values.get("mean_abs_pitch"),
                feature_values.get("std_pitch"),
                feature_values.get("max_abs_pitch"),
                feature_values.get("mean_abs_roll"),
                feature_values.get("std_roll"),
                feature_values.get("max_abs_roll"),
                feature_values.get("face_detect_ratio"),
                feature_values.get("pose_valid_ratio"),
                feature_values.get("ear_valid_ratio"),
                feature_values.get("is_usable"),
            ),
        )
        self.connection.commit()

    def close(self):
        self.flush_frames()
        self.connection.close()


class DrowsinessAlertManager:
    def __init__(self, config: RuntimeConfig):
        self.enabled = bool(config.drowsiness_alert_enabled)
        self.trigger_after_sec = max(0.0, float(config.drowsiness_alert_seconds))
        self.cooldown_sec = max(0.0, float(config.drowsiness_alert_cooldown_sec))
        self.current_start_time: Optional[float] = None
        self.last_alert_time: Optional[float] = None

    def update(self, label: str, time_sec: float) -> bool:
        if not self.enabled:
            return False

        if label != "drowsiness":
            self.current_start_time = None
            return False

        if self.current_start_time is None:
            self.current_start_time = float(time_sec)
            return False

        elapsed = float(time_sec) - self.current_start_time
        if elapsed < self.trigger_after_sec:
            return False

        if self.last_alert_time is not None:
            if (float(time_sec) - self.last_alert_time) < self.cooldown_sec:
                return False

        play_drowsiness_alert_sound()
        self.last_alert_time = float(time_sec)
        return True


class LiveFeatureExtractor:
    def __init__(self, fast_mode: bool):
        self.fast_mode = fast_mode
        self.cam_matrix = None
        self.min_h_px = None
        self.last_w = None
        self.last_h = None
        self.face_2d = np.empty((len(HEAD_POSE_IDS), 2), dtype=np.float64)
        self.left_pts = np.empty((len(LEFT_EYE_IDS), 2), dtype=np.float64)
        self.right_pts = np.empty((len(RIGHT_EYE_IDS), 2), dtype=np.float64)
        self.left_iris_pts = np.empty((len(LEFT_IRIS_IDS), 2), dtype=np.float64)
        self.right_iris_pts = np.empty((len(RIGHT_IRIS_IDS), 2), dtype=np.float64)

    def extract(self, frame: np.ndarray, frame_index: int, fps: float, face_mesh) -> Dict:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h_img, w_img = frame.shape[:2]
        if w_img != self.last_w or h_img != self.last_h:
            self.cam_matrix = build_camera_matrix(w_img, h_img)
            self.min_h_px = max(2.0, w_img * 0.005)
            self.last_w, self.last_h = w_img, h_img

        frame_for_mesh = resize_for_facemesh(frame, fast_mode=self.fast_mode)
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
        iris_x_norm = None
        iris_y_norm = None

        if results.multi_face_landmarks:
            face_detected = 1
            landmarks = results.multi_face_landmarks[0].landmark

            fill_points_2d(landmarks, HEAD_POSE_IDS, w_img, h_img, self.face_2d)
            pose_solution = solve_head_pose(self.face_2d, self.cam_matrix)

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

            fill_points_2d(landmarks, LEFT_EYE_IDS, w_img, h_img, self.left_pts)
            fill_points_2d(landmarks, RIGHT_EYE_IDS, w_img, h_img, self.right_pts)

            left_ear, l_valid = calculate_ear_robust(self.left_pts, min_horizontal_px=self.min_h_px)
            right_ear, r_valid = calculate_ear_robust(self.right_pts, min_horizontal_px=self.min_h_px)

            if l_valid and r_valid:
                avg_ear = (left_ear + right_ear) / 2.0
                ear_valid = 1
                pose_unknown = yaw is None
                yaw_extreme = yaw is not None and abs(yaw) > EAR_SUSPICIOUS_YAW_DEG
                if pose_unknown or yaw_extreme:
                    ear_suspicious = 1

            fill_points_2d(landmarks, LEFT_IRIS_IDS, w_img, h_img, self.left_iris_pts)
            fill_points_2d(landmarks, RIGHT_IRIS_IDS, w_img, h_img, self.right_iris_pts)
            left_center = np.array(average_point(self.left_iris_pts), dtype=np.float64)
            right_center = np.array(average_point(self.right_iris_pts), dtype=np.float64)
            left_norm = normalize_iris_position(left_center, self.left_pts)
            right_norm = normalize_iris_position(right_center, self.right_pts)
            if left_norm[0] is not None and right_norm[0] is not None:
                iris_x_norm = float((left_norm[0] + right_norm[0]) / 2.0)
                iris_y_norm = float((left_norm[1] + right_norm[1]) / 2.0)

        return {
            "frame": frame_index,
            "time_sec": float(frame_index / fps),
            "face_detected": face_detected,
            "yaw": safe_float(yaw),
            "pitch": safe_float(pitch),
            "roll": safe_float(roll),
            "pose_valid": pose_valid,
            "left_ear": safe_float(left_ear),
            "right_ear": safe_float(right_ear),
            "avg_ear": safe_float(avg_ear),
            "ear_valid": ear_valid,
            "ear_suspicious": ear_suspicious,
            "iris_x_norm": safe_float(iris_x_norm),
            "iris_y_norm": safe_float(iris_y_norm),
        }


def compute_live_window_features(frame_rows: List[Dict], config: RuntimeConfig) -> Dict:
    df = pd.DataFrame(frame_rows)
    if df.empty:
        raise ValueError("Cannot compute window features from an empty buffer.")

    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_perclos_df = df[(df["face_detected"] == 1) & (df["avg_ear"].notna())].copy()
    if valid_perclos_df.empty:
        perclos = None
        perclos_percent = None
        closed_eye_frames = 0
        total_frames = 0
    else:
        valid_perclos_df["eye_closed"] = (
            valid_perclos_df["avg_ear"] < config.ear_closed_threshold
        ).astype(int)
        closed_eye_frames = int(valid_perclos_df["eye_closed"].sum())
        total_frames = int(len(valid_perclos_df))
        perclos = float(closed_eye_frames / total_frames)
        perclos_percent = float(perclos * 100.0)

    face_detect_ratio = float((df["face_detected"] == 1).mean())
    pose_valid_ratio = float((df["pose_valid"] == 1).mean())
    ear_valid_ratio = float((df["ear_valid"] == 1).mean())

    features = {
        "perclos": safe_float(perclos),
        "perclos_percent": safe_float(perclos_percent),
        "mean_ear": safe_float(df["avg_ear"].mean()),
        "std_ear": safe_float(series_std(df["avg_ear"])),
        "min_ear": safe_float(df["avg_ear"].min()),
        "max_ear": safe_float(df["avg_ear"].max()),
        "mean_abs_yaw": safe_float(df["yaw"].abs().mean()),
        "std_yaw": safe_float(series_std(df["yaw"])),
        "max_abs_yaw": safe_float(df["yaw"].abs().max()),
        "mean_abs_pitch": safe_float(df["pitch"].abs().mean()),
        "std_pitch": safe_float(series_std(df["pitch"])),
        "max_abs_pitch": safe_float(df["pitch"].abs().max()),
        "mean_abs_roll": safe_float(df["roll"].abs().mean()),
        "std_roll": safe_float(series_std(df["roll"])),
        "max_abs_roll": safe_float(df["roll"].abs().max()),
        "face_detect_ratio": round(face_detect_ratio, 4),
        "pose_valid_ratio": round(pose_valid_ratio, 4),
        "ear_valid_ratio": round(ear_valid_ratio, 4),
        "closed_eye_frames": closed_eye_frames,
        "total_frames_perclos": total_frames,
    }
    features["is_usable"] = int(
        features["face_detect_ratio"] >= config.min_face_ratio
        and features["pose_valid_ratio"] >= config.min_pose_ratio
        and features["ear_valid_ratio"] >= config.min_ear_ratio
        and features["perclos"] is not None
    )
    return features


def compute_live_gaze_features(frame_rows: List[Dict], config: RuntimeConfig) -> Dict:
    df = pd.DataFrame(frame_rows)
    if df.empty:
        raise ValueError("Cannot compute gaze features from an empty buffer.")

    for col in ["frame", "time_sec", "face_detected", "iris_x_norm", "iris_y_norm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sample_period_sec = estimate_sample_period_sec(df)
    expected_frame_step = estimate_expected_frame_step(df)
    total_samples = len(df)
    valid_df = df[
        (df["face_detected"] == 1)
        & df["iris_x_norm"].notna()
        & df["iris_y_norm"].notna()
    ].copy()

    valid_samples = len(valid_df)
    valid_ratio = float(valid_samples / total_samples) if total_samples > 0 else 0.0
    usable_window = int(valid_ratio >= 0.50 and valid_samples >= 3)

    base_row = {
        "gaze_gaze_valid_ratio": safe_float(valid_ratio),
        "gaze_usable_window": usable_window,
    }

    if valid_df.empty:
        return base_row

    x_series = valid_df["iris_x_norm"]
    y_series = valid_df["iris_y_norm"]
    segments = split_contiguous_segments(valid_df, expected_frame_step)
    min_idt_samples = max(2, int(round(100 / (sample_period_sec * 1000.0))))
    min_ivt_samples = max(2, int(round(100 / (sample_period_sec * 1000.0))))

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
                dispersion_threshold=1.0,
            )
            idt_counts.extend(events_to_sample_counts(idt_events))
        except Exception:
            pass

        try:
            ivt_events = pm.events.detection.ivt(
                velocities,
                minimum_duration=min_ivt_samples,
                velocity_threshold=20.0,
            )
            ivt_counts.extend(events_to_sample_counts(ivt_events))
        except Exception:
            pass

        rapid_mask = velocity_norm > 20.0
        rapid_counts.extend(boolean_run_sample_counts(rapid_mask, min_ivt_samples))

    all_velocity = np.concatenate(all_velocity_norms) if all_velocity_norms else np.array([], dtype=np.float64)
    idt_summary = summarize_durations_ms(idt_counts, sample_period_sec)
    ivt_summary = summarize_durations_ms(ivt_counts, sample_period_sec)
    rapid_summary = summarize_durations_ms(rapid_counts, sample_period_sec)

    dispersion_x = float(x_series.max() - x_series.min()) if valid_samples else None
    dispersion_y = float(y_series.max() - y_series.min()) if valid_samples else None
    dispersion_xy = (dispersion_x + dispersion_y) if dispersion_x is not None and dispersion_y is not None else None

    return {
        **base_row,
        "gaze_mean_iris_x_norm": safe_float(x_series.mean()),
        "gaze_std_iris_x_norm": safe_float(series_std(x_series)),
        "gaze_min_iris_x_norm": safe_float(x_series.min()),
        "gaze_max_iris_x_norm": safe_float(x_series.max()),
        "gaze_mean_iris_y_norm": safe_float(y_series.mean()),
        "gaze_std_iris_y_norm": safe_float(series_std(y_series)),
        "gaze_min_iris_y_norm": safe_float(y_series.min()),
        "gaze_max_iris_y_norm": safe_float(y_series.max()),
        "gaze_gaze_dispersion_x": safe_float(dispersion_x),
        "gaze_gaze_dispersion_y": safe_float(dispersion_y),
        "gaze_gaze_dispersion_xy": safe_float(dispersion_xy),
        "gaze_gaze_path_length": safe_float(path_length),
        "gaze_mean_step_distance": safe_float(path_length / valid_step_count) if valid_step_count > 0 else None,
        "gaze_mean_velocity_norm": safe_float(float(np.mean(all_velocity))) if all_velocity.size else None,
        "gaze_std_velocity_norm": safe_float(float(np.std(all_velocity))) if all_velocity.size else None,
        "gaze_max_velocity_norm": safe_float(float(np.max(all_velocity))) if all_velocity.size else None,
        "gaze_idt_fixation_count": idt_summary["count"],
        "gaze_idt_fixation_mean_duration_ms": safe_float(duration_or_zero(idt_summary, "mean_ms"), 3),
        "gaze_idt_fixation_max_duration_ms": safe_float(duration_or_zero(idt_summary, "max_ms"), 3),
        "gaze_idt_fixation_ratio": safe_float(idt_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
        "gaze_ivt_fixation_count": ivt_summary["count"],
        "gaze_ivt_fixation_mean_duration_ms": safe_float(duration_or_zero(ivt_summary, "mean_ms"), 3),
        "gaze_ivt_fixation_max_duration_ms": safe_float(duration_or_zero(ivt_summary, "max_ms"), 3),
        "gaze_ivt_fixation_ratio": safe_float(ivt_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
        "gaze_rapid_shift_count": rapid_summary["count"],
        "gaze_rapid_shift_mean_duration_ms": safe_float(duration_or_zero(rapid_summary, "mean_ms"), 3),
        "gaze_rapid_shift_max_duration_ms": safe_float(duration_or_zero(rapid_summary, "max_ms"), 3),
        "gaze_rapid_shift_ratio": safe_float(rapid_summary["total_samples"] / valid_samples, 4) if valid_samples else None,
    }


def build_feature_frame(window_features: Dict, feature_columns: List[str]) -> pd.DataFrame:
    feature_row = {}
    missing = []
    for col in feature_columns:
        value = window_features.get(col)
        if value is None or pd.isna(value):
            missing.append(col)
        feature_row[col] = value

    if missing:
        missing_text = ", ".join(missing[:5])
        raise ValueError(f"Current live window is missing required features: {missing_text}")

    return pd.DataFrame([feature_row], columns=feature_columns)


def can_predict(window_features: Dict, feature_columns: List[str]) -> bool:
    for col in feature_columns:
        value = window_features.get(col)
        if value is None or pd.isna(value):
            return False
    return True


def matching_high_confidence_labels(window_features: Dict) -> List[str]:
    perclos = window_features.get("perclos")
    mean_ear = window_features.get("mean_ear")
    std_ear = window_features.get("std_ear")
    min_ear = window_features.get("min_ear")
    mean_abs_yaw = window_features.get("mean_abs_yaw")
    max_abs_yaw = window_features.get("max_abs_yaw")
    mean_abs_pitch = window_features.get("mean_abs_pitch")
    max_abs_pitch = window_features.get("max_abs_pitch")

    required_values = [
        perclos,
        mean_ear,
        std_ear,
        min_ear,
        mean_abs_yaw,
        max_abs_yaw,
        mean_abs_pitch,
        max_abs_pitch,
    ]
    if any(value is None or pd.isna(value) for value in required_values):
        return []

    labels: List[str] = []

    normal_match = (
        perclos <= 0.05
        and mean_ear >= 0.24
        and std_ear <= 0.025
        and mean_abs_yaw <= 12
        and max_abs_yaw <= 18
        and mean_abs_pitch <= 7
        and max_abs_pitch <= 12
    )
    if normal_match:
        labels.append("normal")

    drowsiness_match = (
        perclos >= 0.20
        and mean_ear <= 0.20
        and min_ear <= 0.08
        and mean_abs_yaw <= 10
        and max_abs_yaw <= 18
    )
    if drowsiness_match:
        labels.append("drowsiness")

    distraction_match = (
        mean_abs_yaw >= 18
        and max_abs_yaw >= 30
        and perclos <= 0.12
    )
    if distraction_match:
        labels.append("distraction")

    return labels


def is_relaxed_normal_window(window_features: Dict) -> bool:
    required_keys = [
        "perclos",
        "mean_ear",
        "std_ear",
        "mean_abs_yaw",
        "max_abs_yaw",
        "mean_abs_pitch",
        "max_abs_pitch",
        "face_detect_ratio",
        "pose_valid_ratio",
        "ear_valid_ratio",
    ]
    values = [window_features.get(key) for key in required_keys]
    if any(value is None or pd.isna(value) for value in values):
        return False

    return bool(
        window_features["perclos"] <= 0.10
        and window_features["mean_ear"] >= 0.22
        and window_features["std_ear"] <= 0.045
        and window_features["mean_abs_yaw"] <= 15
        and window_features["max_abs_yaw"] <= 24
        and window_features["mean_abs_pitch"] <= 10
        and window_features["max_abs_pitch"] <= 16
        and window_features["face_detect_ratio"] >= 0.75
        and window_features["pose_valid_ratio"] >= 0.45
        and window_features["ear_valid_ratio"] >= 0.45
    )


def is_relaxed_drowsiness_window(window_features: Dict) -> bool:
    required_keys = [
        "perclos",
        "mean_ear",
        "min_ear",
        "mean_abs_yaw",
        "max_abs_yaw",
    ]
    values = [window_features.get(key) for key in required_keys]
    if any(value is None or pd.isna(value) for value in values):
        return False

    return bool(
        window_features["perclos"] >= 0.16
        and window_features["mean_ear"] <= 0.22
        and window_features["min_ear"] <= 0.11
        and window_features["mean_abs_yaw"] <= 14
        and window_features["max_abs_yaw"] <= 24
    )


def is_relaxed_distraction_window(window_features: Dict) -> bool:
    required_keys = [
        "mean_abs_yaw",
        "max_abs_yaw",
        "perclos",
    ]
    values = [window_features.get(key) for key in required_keys]
    if any(value is None or pd.isna(value) for value in values):
        return False

    return bool(
        window_features["mean_abs_yaw"] >= 16
        and window_features["max_abs_yaw"] >= 24
        and window_features["perclos"] <= 0.18
    )


def is_safe_normal_window(window_features: Dict) -> bool:
    required_keys = [
        "perclos",
        "mean_ear",
        "mean_abs_yaw",
        "max_abs_yaw",
        "mean_abs_pitch",
        "face_detect_ratio",
        "pose_valid_ratio",
        "ear_valid_ratio",
        "is_usable",
    ]
    values = [window_features.get(key) for key in required_keys]
    if any(value is None or pd.isna(value) for value in values):
        return False

    return bool(
        window_features["is_usable"] == 1
        and window_features["perclos"] <= 0.18
        and window_features["mean_ear"] >= 0.20
        and window_features["mean_abs_yaw"] <= 14
        and window_features["max_abs_yaw"] <= 22
        and window_features["mean_abs_pitch"] <= 10
        and window_features["face_detect_ratio"] >= 0.65
        and window_features["pose_valid_ratio"] >= 0.35
        and window_features["ear_valid_ratio"] >= 0.35
    )


def decide_prediction(
    probs: np.ndarray,
    classes: np.ndarray,
    window_features: Dict,
    config: RuntimeConfig,
    enforce_high_confidence_rules: bool,
) -> tuple[str, float]:
    best_idx = int(np.argmax(probs))
    best_label = str(classes[best_idx])
    best_confidence = float(probs[best_idx])
    prob_by_label = {str(label): float(prob) for label, prob in zip(classes, probs)}

    sorted_probs = np.sort(probs)
    second_confidence = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    confidence_margin = best_confidence - second_confidence

    drowsiness_prob = prob_by_label.get("drowsiness", 0.0)
    distraction_prob = prob_by_label.get("distraction", 0.0)
    normal_prob = prob_by_label.get("normal", 0.0)

    if is_relaxed_drowsiness_window(window_features) and drowsiness_prob >= 0.45:
        return "drowsiness", drowsiness_prob

    if is_relaxed_distraction_window(window_features) and distraction_prob >= 0.45:
        return "distraction", distraction_prob

    matched_labels = matching_high_confidence_labels(window_features)
    if (
        is_relaxed_normal_window(window_features)
        and "drowsiness" not in matched_labels
        and "distraction" not in matched_labels
    ):
        other_prob = max(drowsiness_prob, distraction_prob)
        if normal_prob >= 0.22 and normal_prob + 0.04 >= other_prob:
            return "normal", normal_prob

    if (
        is_safe_normal_window(window_features)
        and not is_relaxed_drowsiness_window(window_features)
        and not is_relaxed_distraction_window(window_features)
        and drowsiness_prob < 0.55
        and distraction_prob < 0.55
    ):
        return "normal", max(normal_prob, 0.51)

    if best_confidence < config.min_confidence:
        return "unknown", best_confidence
    if confidence_margin < config.min_confidence_margin:
        return "unknown", best_confidence

    if enforce_high_confidence_rules:
        if best_label not in matched_labels:
            return "unknown", best_confidence

    return best_label, best_confidence


def average_probabilities(probability_history: Deque[np.ndarray]) -> np.ndarray:
    if not probability_history:
        raise ValueError("Cannot smooth probabilities without history.")
    return np.mean(np.stack(probability_history, axis=0), axis=0)


def update_stable_label(
    displayed_label: str,
    displayed_confidence: float,
    candidate_label: str,
    candidate_confidence: float,
    pending_label: Optional[str],
    pending_count: int,
    unknown_streak: int,
    config: RuntimeConfig,
) -> tuple[str, float, Optional[str], int, int]:
    if candidate_label == "unknown":
        unknown_streak += 1
        pending_label = None
        pending_count = 0
        if unknown_streak >= config.unknown_confirmations:
            return "unknown", -1.0, None, 0, unknown_streak
        return displayed_label, displayed_confidence, None, 0, unknown_streak

    unknown_streak = 0
    if candidate_label == displayed_label:
        return displayed_label, candidate_confidence, None, 0, unknown_streak

    if pending_label == candidate_label:
        pending_count += 1
    else:
        pending_label = candidate_label
        pending_count = 1

    if displayed_label == "unknown":
        required_count = max(1, config.switch_confirmations)
    elif candidate_label == "normal" and displayed_label in {"drowsiness", "distraction"}:
        required_count = 1
    else:
        required_count = max(2, config.switch_confirmations)

    if pending_count >= required_count:
        return candidate_label, candidate_confidence, None, 0, unknown_streak

    return displayed_label, displayed_confidence, pending_label, pending_count, unknown_streak


def resolve_live_window_features(frame_rows: List[Dict], feature_set: str, config: RuntimeConfig) -> Dict:
    baseline_features = compute_live_window_features(frame_rows, config)
    if feature_set != "gaze":
        return baseline_features

    gaze_features = compute_live_gaze_features(frame_rows, config)
    merged = {**baseline_features, **gaze_features}
    merged["is_usable"] = int(
        bool(baseline_features.get("is_usable"))
        and bool(gaze_features.get("gaze_usable_window"))
    )
    return merged


def smooth_label(prediction_history: Deque[str]) -> str:
    if not prediction_history:
        return "unknown"
    counter = Counter(prediction_history)
    return counter.most_common(1)[0][0]


def resize_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return frame

    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 1.0:
        return frame

    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


def show_status_screen(title: str, lines: List[str], width: int = 960, height: int = 540):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (width - 20, height - 20), (80, 80, 220), 2)
    cv2.putText(
        canvas,
        title,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    for idx, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (40, 120 + idx * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
        )

    cv2.namedWindow("Realtime Driver Monitor", cv2.WINDOW_NORMAL)
    cv2.imshow("Realtime Driver Monitor", canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q"), 13):
            break
    cv2.destroyAllWindows()


def draw_overlay(
    frame: np.ndarray,
    live_row: Dict,
    window_features: Optional[Dict],
    label: str,
    confidence: float,
    usable: bool,
):
    color = COLOR_BY_LABEL.get(label, COLOR_BY_LABEL["unknown"])
    lines = [
        f"{label.upper()}",
        f"Conf {confidence:.0%}" if confidence >= 0 else "Collecting...",
        (
            f"P:{window_features['perclos']:.3f} F:{window_features['face_detect_ratio']:.2f} "
            f"Po:{window_features['pose_valid_ratio']:.2f} E:{window_features['ear_valid_ratio']:.2f}"
            if window_features and window_features.get("perclos") is not None
            else "Waiting for full window"
        ),
        f"Yaw {live_row['yaw']:.1f}" if live_row.get("yaw") is not None else "Yaw None",
        f"EAR {live_row['avg_ear']:.3f}" if live_row.get("avg_ear") is not None else "EAR None",
        "usable" if usable else "low quality",
    ]

    panel_w = 360
    panel_h = 150
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), color, 2)
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (26, 40 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 if idx < 2 else 0.5,
            color if idx < 2 else (230, 230, 230),
            2,
        )


def main():
    parser = argparse.ArgumentParser(description="Run realtime driver monitoring on webcam or video.")
    parser.add_argument(
        "--model-bundle",
        default=str(DEFAULT_MODEL_BUNDLE),
        help="Path to model_bundle.pkl produced by train_random_forest.py",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to open when --video-path is not provided.",
    )
    parser.add_argument(
        "--video-path",
        default="",
        help="Optional video file path instead of a live camera.",
    )
    parser.add_argument("--window-sec", type=float, default=3.0, help="Sliding window length in seconds.")
    parser.add_argument(
        "--predict-every-frames",
        type=int,
        default=10,
        help="Run inference every N frames once the window is full.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Resize frames before MediaPipe to reduce latency.",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=960,
        help="Maximum preview width for the OpenCV window.",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=540,
        help="Maximum preview height for the OpenCV window.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database file path for logging live frames and predictions.",
    )
    parser.add_argument(
        "--disable-db",
        action="store_true",
        help="Disable database logging.",
    )
    parser.add_argument("--min-face-ratio", type=float, default=0.50, help="Minimum face detection ratio for a high-quality live window.")
    parser.add_argument("--min-pose-ratio", type=float, default=0.40, help="Minimum valid head-pose ratio for a high-quality live window.")
    parser.add_argument("--min-ear-ratio", type=float, default=0.40, help="Minimum valid EAR ratio for a high-quality live window.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum top-class probability required before showing a prediction.")
    parser.add_argument("--min-confidence-margin", type=float, default=0.08, help="Minimum probability gap between the top two classes.")
    parser.add_argument("--switch-confirmations", type=int, default=2, help="How many consecutive candidate windows are required before switching labels.")
    parser.add_argument("--unknown-confirmations", type=int, default=2, help="How many consecutive uncertain windows are required before falling back to unknown.")
    parser.add_argument(
        "--drowsiness-alert-seconds",
        type=float,
        default=2.0,
        help="Play a sound if the stable label stays drowsiness for this many consecutive seconds.",
    )
    parser.add_argument(
        "--drowsiness-alert-cooldown-sec",
        type=float,
        default=4.0,
        help="Minimum time between repeated drowsiness alerts while the driver remains drowsy.",
    )
    parser.add_argument(
        "--disable-drowsiness-alert",
        action="store_true",
        help="Disable the drowsiness sound alert.",
    )
    parser.add_argument(
        "--enable-high-confidence-rule-gate",
        action="store_true",
        help="Enforce the handcrafted high-confidence rules when a high-confidence model bundle is loaded.",
    )
    args = parser.parse_args()

    model_bundle_path = Path(args.model_bundle)
    if not model_bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {model_bundle_path}. "
            "Run train_random_forest.py first to create model_bundle.pkl."
        )

    bundle = load_model_bundle(model_bundle_path)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    feature_set = str(bundle.get("feature_set", "baseline"))
    use_high_confidence_model = bool(bundle.get("use_high_confidence")) or ("high_confidence" in str(model_bundle_path).lower())

    config = RuntimeConfig(
        window_sec=args.window_sec,
        predict_every_frames=max(1, args.predict_every_frames),
        fast_mode=args.fast_mode,
        min_face_ratio=args.min_face_ratio,
        min_pose_ratio=args.min_pose_ratio,
        min_ear_ratio=args.min_ear_ratio,
        display_max_width=args.display_max_width,
        display_max_height=args.display_max_height,
        min_confidence=args.min_confidence,
        min_confidence_margin=args.min_confidence_margin,
        switch_confirmations=max(1, args.switch_confirmations),
        unknown_confirmations=max(1, args.unknown_confirmations),
        drowsiness_alert_seconds=max(0.0, float(args.drowsiness_alert_seconds)),
        drowsiness_alert_cooldown_sec=max(0.0, float(args.drowsiness_alert_cooldown_sec)),
        drowsiness_alert_enabled=not bool(args.disable_drowsiness_alert),
    )

    cap, source_description = open_video_source(args.video_path, args.camera_index)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    window_frames = max(1, int(round(fps * config.window_sec)))
    frame_buffer: Deque[Dict] = deque(maxlen=window_frames)
    probability_history: Deque[np.ndarray] = deque(maxlen=config.smoothing_windows)
    extractor = LiveFeatureExtractor(fast_mode=config.fast_mode)
    face_mesh = get_face_mesh(fast_mode=config.fast_mode)
    alert_manager = DrowsinessAlertManager(config)
    db_logger = None
    if not args.disable_db:
        db_logger = RealtimeDatabaseLogger(
            db_path=Path(args.db_path),
            model_bundle_path=model_bundle_path,
            source_description=source_description,
            config=config,
            fps=fps,
        )

    frame_index = 0
    current_label = "unknown"
    current_confidence = -1.0
    current_window_features = None
    current_usable = False
    read_failures = 0
    pending_label: Optional[str] = None
    pending_count = 0
    unknown_streak = 0

    try:
        print(f"[INFO] Opened source: {source_description}")
        if db_logger is not None:
            print(f"[INFO] Logging to database: {Path(args.db_path)}")
        cv2.namedWindow("Realtime Driver Monitor", cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if frame_index == 0:
                    message = [
                        f"Source opened but no frame could be read: {source_description}",
                        "This usually means the webcam is busy, blocked by privacy settings,",
                        "or the backend can open the device but cannot stream frames.",
                        "Close Zoom/Teams/Camera, check Windows camera permission,",
                        "or try --camera-index 1 / 2.",
                        "Press Q, ESC, or Enter to close.",
                    ]
                    print("[ERROR] No frames received from source.")
                    show_status_screen("No Frames Received", message)
                    return

                if read_failures >= 5:
                    print("[WARN] Frame stream ended or stalled.")
                    break
                continue

            try:
                frame_index += 1
                read_failures = 0
                live_row = extractor.extract(frame, frame_index, fps, face_mesh)
                frame_buffer.append(live_row)
                if db_logger is not None:
                    db_logger.log_frame(live_row)

                if len(frame_buffer) >= window_frames and frame_index % config.predict_every_frames == 0:
                    current_window_features = resolve_live_window_features(list(frame_buffer), feature_set, config)
                    current_usable = bool(current_window_features["is_usable"])

                    if can_predict(current_window_features, feature_columns):
                        feature_df = build_feature_frame(current_window_features, feature_columns)
                        probability_history.append(model.predict_proba(feature_df)[0])
                        smoothed_probs = average_probabilities(probability_history)
                        raw_label, raw_confidence = decide_prediction(
                            probs=smoothed_probs,
                            classes=model.classes_,
                            window_features=current_window_features,
                            config=config,
                            enforce_high_confidence_rules=(
                                use_high_confidence_model and args.enable_high_confidence_rule_gate
                            ),
                        )
                        (
                            current_label,
                            current_confidence,
                            pending_label,
                            pending_count,
                            unknown_streak,
                        ) = update_stable_label(
                            displayed_label=current_label,
                            displayed_confidence=current_confidence,
                            candidate_label=raw_label,
                            candidate_confidence=raw_confidence,
                            pending_label=pending_label,
                            pending_count=pending_count,
                            unknown_streak=unknown_streak,
                            config=config,
                        )
                    else:
                        probability_history.clear()
                        (
                            current_label,
                            current_confidence,
                            pending_label,
                            pending_count,
                            unknown_streak,
                        ) = update_stable_label(
                            displayed_label=current_label,
                            displayed_confidence=current_confidence,
                            candidate_label="unknown",
                            candidate_confidence=-1.0,
                            pending_label=pending_label,
                            pending_count=pending_count,
                            unknown_streak=unknown_streak,
                            config=config,
                        )

                    if db_logger is not None:
                        db_logger.log_prediction(
                            frame_index=frame_index,
                            predicted_label=current_label,
                            confidence=current_confidence,
                            quality_usable=current_usable,
                            window_features=current_window_features,
                        )

                    if alert_manager.update(current_label, live_row["time_sec"]):
                        print(
                            f"[ALERT] Drowsiness warning triggered at "
                            f"{live_row['time_sec']:.2f}s after {config.drowsiness_alert_seconds:.1f}s."
                        )

                draw_overlay(
                    frame=frame,
                    live_row=live_row,
                    window_features=current_window_features,
                    label=current_label,
                    confidence=current_confidence,
                    usable=current_usable,
                )

                display_frame = resize_for_display(
                    frame,
                    max_width=config.display_max_width,
                    max_height=config.display_max_height,
                )
                cv2.imshow("Realtime Driver Monitor", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            except Exception as exc:
                error_text = traceback.format_exc()
                print(error_text)
                show_status_screen(
                    "Runtime Error",
                    [
                        f"{type(exc).__name__}: {exc}",
                        "The monitor hit an exception after startup.",
                        "Terminal traceback has been printed.",
                        "Press Q, ESC, or Enter to close.",
                    ],
                )
                return
    finally:
        if db_logger is not None:
            db_logger.close()
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
