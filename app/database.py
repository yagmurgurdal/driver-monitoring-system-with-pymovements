from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "driver_state_app.sqlite3"


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_slug TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                source_video_name TEXT NOT NULL,
                source_video_path TEXT NOT NULL,
                source_label TEXT NOT NULL,
                source_modality TEXT NOT NULL,
                model_key TEXT NOT NULL,
                feature_set TEXT NOT NULL,
                fast_mode INTEGER NOT NULL,
                overall_label TEXT NOT NULL,
                overall_confidence REAL NOT NULL,
                risk_score REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                detected_face_ratio REAL NOT NULL,
                valid_pose_ratio REAL NOT NULL,
                valid_ear_ratio REAL NOT NULL,
                suspicious_ear_ratio REAL NOT NULL,
                valid_frame_count INTEGER NOT NULL,
                mean_probabilities_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS window_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                window_id INTEGER NOT NULL,
                window_start_time REAL,
                window_end_time REAL,
                predicted_label TEXT NOT NULL,
                prob_normal REAL NOT NULL,
                prob_drowsiness REAL NOT NULL,
                prob_distraction REAL NOT NULL,
                window_risk_score REAL NOT NULL,
                perclos_percent REAL,
                mean_ear REAL,
                max_abs_yaw REAL,
                is_usable INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
            ON analysis_runs(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_window_predictions_analysis_id
            ON window_predictions(analysis_id);
            """
        )


def _to_native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def save_analysis(
    *,
    analysis_slug: str,
    created_at: str,
    source_video_name: str,
    source_video_path: str,
    source_label: str,
    source_modality: str,
    model_key: str,
    feature_set: str,
    fast_mode: bool,
    summary: dict[str, Any],
    quality: dict[str, Any],
    artifacts: dict[str, str],
    predictions_df: pd.DataFrame,
) -> int:
    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_runs (
                analysis_slug,
                created_at,
                source_video_name,
                source_video_path,
                source_label,
                source_modality,
                model_key,
                feature_set,
                fast_mode,
                overall_label,
                overall_confidence,
                risk_score,
                total_frames,
                detected_face_ratio,
                valid_pose_ratio,
                valid_ear_ratio,
                suspicious_ear_ratio,
                valid_frame_count,
                mean_probabilities_json,
                artifacts_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_slug,
                created_at,
                source_video_name,
                source_video_path,
                source_label,
                source_modality,
                model_key,
                feature_set,
                int(bool(fast_mode)),
                summary["overall_label"],
                float(summary["overall_confidence"]),
                float(summary["risk_score"]),
                int(quality["total_frames"]),
                float(quality["detected_face_ratio"]),
                float(quality["valid_pose_ratio"]),
                float(quality["valid_ear_ratio"]),
                float(quality["suspicious_ear_ratio"]),
                int(quality["valid_frame_count"]),
                json.dumps(summary["mean_probabilities"], ensure_ascii=False),
                json.dumps(artifacts, ensure_ascii=False),
            ),
        )
        analysis_id = int(cursor.lastrowid)

        rows = []
        for record in predictions_df.to_dict(orient="records"):
            rows.append(
                (
                    analysis_id,
                    int(_to_native(record["window_id"])),
                    _to_native(record.get("window_start_time")),
                    _to_native(record.get("window_end_time")),
                    str(record["predicted_label"]),
                    float(_to_native(record.get("prob_normal", 0.0)) or 0.0),
                    float(_to_native(record.get("prob_drowsiness", 0.0)) or 0.0),
                    float(_to_native(record.get("prob_distraction", 0.0)) or 0.0),
                    float(_to_native(record.get("window_risk_score", 0.0)) or 0.0),
                    _to_native(record.get("perclos_percent")),
                    _to_native(record.get("mean_ear")),
                    _to_native(record.get("max_abs_yaw")),
                    int(_to_native(record.get("is_usable", 0)) or 0),
                )
            )

        connection.executemany(
            """
            INSERT INTO window_predictions (
                analysis_id,
                window_id,
                window_start_time,
                window_end_time,
                predicted_label,
                prob_normal,
                prob_drowsiness,
                prob_distraction,
                window_risk_score,
                perclos_percent,
                mean_ear,
                max_abs_yaw,
                is_usable
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return analysis_id


def list_recent_analyses(limit: int = 5) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                source_video_name,
                source_label,
                source_modality,
                model_key,
                overall_label,
                overall_confidence,
                risk_score
            FROM analysis_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]
