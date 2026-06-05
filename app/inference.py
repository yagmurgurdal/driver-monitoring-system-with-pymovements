from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from app.database import DATABASE_PATH, save_analysis
from app.unified_extractor import extract_video_features
from scripts.dataset.build_window_dataset import WindowDatasetConfig, compute_window_features as compute_baseline_window_features
from scripts.gaze.build_pymovements_window_features import (
    WindowConfig as GazeWindowConfig,
    compute_window_features as compute_gaze_window_features,
    estimate_expected_frame_step,
    estimate_sample_period_sec,
    iter_window_ranges,
)
from scripts.gaze.merge_window_with_gaze_features import prefix_gaze_columns
from scripts.models.random_forest.train_random_forest import fill_missing_gaze_duration_features
from scripts.utils.driver_monitoring_features_fixed import FAST_MODE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_RUNTIME_ROOT = PROJECT_ROOT / "app_runtime"
UPLOADS_ROOT = APP_RUNTIME_ROOT / "uploads"
ANALYSES_ROOT = APP_RUNTIME_ROOT / "analyses"

EAR_THRESHOLD = 0.21
WINDOW_SEC = 3
REFERENCE_FPS = 30
WINDOW_SIZE = WINDOW_SEC * REFERENCE_FPS
DEFAULT_MODALITY = "RGB"
KNOWN_LABELS = ("normal", "drowsiness", "distraction")
KNOWN_MODALITIES = ("RGB", "IR")

MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "xgboost_baseline": {
        "label": "XGBoost Baseline",
        "path": str(PROJECT_ROOT / "results" / "xgboost_baseline" / "xgboost" / "model_bundle.pkl"),
        "description": "Gaze kullanmadan hizli analiz icin guclu baseline model.",
    },
    "xgboost_gaze_high_confidence": {
        "label": "XGBoost Gaze High Confidence",
        "path": str(PROJECT_ROOT / "results" / "xgboost_gaze_high_confidence" / "xgboost" / "model_bundle.pkl"),
        "description": "Repodaki en iyi dogrulanmis model. Gaze destekli oldugu icin daha yavas ama daha guclu.",
    },
    "extra_trees_gaze_high_confidence": {
        "label": "Extra Trees Gaze High Confidence",
        "path": str(PROJECT_ROOT / "results" / "extra_trees_gaze_high_confidence" / "extra_trees" / "model_bundle.pkl"),
        "description": "En iyi ikinci gaze destekli alternatif model.",
    },
    "random_forest_baseline": {
        "label": "Random Forest Baseline",
        "path": str(PROJECT_ROOT / "results" / "random_forest_baseline" / "model_bundle.pkl"),
        "description": "Daha genel kullanım için temel model.",
    },
    "random_forest_high_confidence": {
        "label": "Random Forest High Confidence",
        "path": str(PROJECT_ROOT / "results" / "random_forest_high_confidence" / "model_bundle.pkl"),
        "description": "Daha temiz pencereler için eğitilmiş alternatif model.",
    },
}


@dataclass(frozen=True)
class QualitySummary:
    total_frames: int
    detected_face_ratio: float
    valid_pose_ratio: float
    valid_ear_ratio: float
    suspicious_ear_ratio: float
    valid_frame_count: int


@dataclass(frozen=True)
class AnalysisPreset:
    mode: str
    label: str
    model_key: str
    clip_seconds: int | None
    use_fast_mode: bool


ANALYSIS_PRESETS: dict[str, AnalysisPreset] = {
    "quick": AnalysisPreset(
        mode="quick",
        label="Hizli Analiz",
        model_key="xgboost_baseline",
        clip_seconds=30,
        use_fast_mode=True,
    ),
    "full": AnalysisPreset(
        mode="full",
        label="Tam Analiz",
        model_key="xgboost_gaze_high_confidence",
        clip_seconds=None,
        use_fast_mode=True,
    ),
}


def ensure_runtime_dirs() -> None:
    for path in (UPLOADS_ROOT, ANALYSES_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def create_short_clip(source_path: Path, target_path: Path, max_duration_sec: int) -> Path:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {source_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        max_frames = max(1, int(round(float(max_duration_sec) * float(fps))))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Kisa klip yazilamadi: {target_path}")

        try:
            frame_counter = 0
            while frame_counter < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                frame_counter += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return target_path


def infer_label_from_name(name: str) -> str:
    lowered = name.lower()
    for label in KNOWN_LABELS:
        if label in lowered:
            return label
    if "sleep" in lowered or "uyku" in lowered:
        return "drowsiness"
    if "dikkat" in lowered:
        return "distraction"
    return "uploaded"


def infer_modality_from_name(name: str) -> str:
    lowered = name.lower()
    if "ir" in lowered:
        return "IR"
    return "RGB"


def load_model_bundle(model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_OPTIONS:
        raise KeyError(f"Unknown model key: {model_key}")

    model_path = Path(MODEL_OPTIONS[model_key]["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    with model_path.open("rb") as handle:
        return pickle.load(handle)


def read_feature_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_perclos_windows(source_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = source_df[(source_df["face_detected"] == 1) & (source_df["avg_ear"].notna())].copy()
    if len(valid_df) < WINDOW_SIZE:
        return pd.DataFrame(
            columns=[
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "closed_eye_frames",
                "total_frames",
                "perclos",
                "perclos_percent",
            ]
        )

    valid_df["eye_closed"] = (valid_df["avg_ear"] < EAR_THRESHOLD).astype(int)
    perclos_rows: list[dict[str, Any]] = []

    for start in range(0, len(valid_df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = valid_df.iloc[start : start + WINDOW_SIZE].copy()
        perclos_value = float(window["eye_closed"].sum() / len(window))
        perclos_rows.append(
            {
                "start_frame": int(window["frame"].iloc[0]),
                "end_frame": int(window["frame"].iloc[-1]),
                "start_time": float(window["time_sec"].iloc[0]),
                "end_time": float(window["time_sec"].iloc[-1]),
                "closed_eye_frames": int(window["eye_closed"].sum()),
                "total_frames": int(len(window)),
                "perclos": round(perclos_value, 4),
                "perclos_percent": round(perclos_value * 100.0, 2),
            }
        )

    return pd.DataFrame(perclos_rows)


def summarize_quality(source_df: pd.DataFrame) -> QualitySummary:
    total_frames = int(len(source_df))
    valid_frame_mask = (source_df["face_detected"] == 1) & source_df["avg_ear"].notna()
    return QualitySummary(
        total_frames=total_frames,
        detected_face_ratio=float((source_df["face_detected"] == 1).mean()) if total_frames else 0.0,
        valid_pose_ratio=float((source_df["pose_valid"] == 1).mean()) if total_frames else 0.0,
        valid_ear_ratio=float((source_df["ear_valid"] == 1).mean()) if total_frames else 0.0,
        suspicious_ear_ratio=float((source_df["ear_suspicious"] == 1).mean()) if total_frames else 0.0,
        valid_frame_count=int(valid_frame_mask.sum()),
    )


def build_window_feature_table(
    source_df: pd.DataFrame,
    perclos_df: pd.DataFrame,
    video_name: str,
    source_label: str,
    modality: str,
    config: WindowDatasetConfig | None = None,
) -> pd.DataFrame:
    config = config or WindowDatasetConfig()
    metadata = {
        "relative_path": video_name,
        "file_name": video_name,
        "file_stem": Path(video_name).stem,
        "label": source_label,
        "modality": modality,
    }

    rows: list[dict[str, Any]] = []
    for window_id, (_, perclos_row) in enumerate(perclos_df.iterrows(), start=1):
        start_frame = int(perclos_row["start_frame"])
        end_frame = int(perclos_row["end_frame"])
        window_df = source_df[
            (source_df["frame"] >= start_frame) & (source_df["frame"] <= end_frame)
        ].copy()
        if window_df.empty:
            continue
        rows.append(
            compute_baseline_window_features(
                window_df=window_df,
                perclos_row=perclos_row,
                metadata=metadata,
                window_id=window_id,
                config=config,
            )
        )

    return pd.DataFrame(rows)


def build_gaze_window_table(
    gaze_source_df: pd.DataFrame,
    video_name: str,
    source_label: str,
    modality: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    config = GazeWindowConfig()
    metadata = {
        "label": source_label,
        "modality": modality,
        "file_name": video_name,
        "source_stem": Path(video_name).stem,
    }
    sample_period_sec = estimate_sample_period_sec(gaze_source_df)
    expected_frame_step = estimate_expected_frame_step(gaze_source_df)
    last_time_sec = float(gaze_source_df["time_sec"].max())

    rows: list[dict[str, Any]] = []
    for window_id, start_time, end_time in iter_window_ranges(last_time_sec, config.window_sec, config.step_sec):
        if window_id == 1:
            mask = (gaze_source_df["time_sec"] >= start_time) & (gaze_source_df["time_sec"] <= end_time)
        else:
            mask = (gaze_source_df["time_sec"] > start_time) & (gaze_source_df["time_sec"] <= end_time)
        window_df = gaze_source_df.loc[mask].copy()
        if window_df.empty:
            continue
        rows.append(
            compute_gaze_window_features(
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
    return pd.DataFrame(rows), {}


def merge_baseline_with_gaze(baseline_df: pd.DataFrame, gaze_df: pd.DataFrame) -> pd.DataFrame:
    prefixed_gaze_df = prefix_gaze_columns(gaze_df)
    merged_df = baseline_df.merge(prefixed_gaze_df, on="window_id", how="left", validate="one_to_one")
    return fill_missing_gaze_duration_features(merged_df)


def decode_prediction_labels(raw_predictions: np.ndarray, model_bundle: dict[str, Any]) -> list[str]:
    encoded_classes = list(model_bundle.get("encoded_classes") or [])
    if encoded_classes and np.issubdtype(np.asarray(raw_predictions).dtype, np.number):
        return [str(encoded_classes[int(value)]) for value in raw_predictions]
    return [str(value) for value in raw_predictions]


def probability_class_names(model: Any, model_bundle: dict[str, Any]) -> list[str]:
    encoded_classes = list(model_bundle.get("encoded_classes") or [])
    classes = np.asarray(getattr(model, "classes_", []))
    if encoded_classes and classes.size and np.issubdtype(classes.dtype, np.number):
        return [str(encoded_classes[int(value)]) for value in classes.tolist()]
    return [str(value) for value in classes.tolist()]


def build_prediction_table(windows_df: pd.DataFrame, model_bundle: dict[str, Any]) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_columns: list[str] = list(model_bundle["feature_columns"])
    feature_set = str(model_bundle.get("feature_set", "baseline"))

    usable_df = windows_df[windows_df["is_usable"] == 1].copy()
    if feature_set == "gaze" and "gaze_usable_window" in usable_df.columns:
        usable_df = usable_df[usable_df["gaze_usable_window"] == 1].copy()
    usable_df = usable_df.dropna(subset=feature_columns).reset_index(drop=True)
    if usable_df.empty:
        return usable_df

    raw_predictions = np.asarray(model.predict(usable_df[feature_columns]))
    usable_df["predicted_label"] = decode_prediction_labels(raw_predictions, model_bundle)

    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(usable_df[feature_columns])
        class_names = probability_class_names(model, model_bundle)
        prob_df = pd.DataFrame(prob_matrix, columns=[f"prob_{name}" for name in class_names])
        usable_df = pd.concat([usable_df, prob_df], axis=1)
    else:
        for label in ("normal", "drowsiness", "distraction"):
            usable_df[f"prob_{label}"] = np.where(usable_df["predicted_label"] == label, 1.0, 0.0)

    for label in ("normal", "drowsiness", "distraction"):
        prob_col = f"prob_{label}"
        if prob_col not in usable_df.columns:
            usable_df[prob_col] = 0.0

    usable_df["window_risk_score"] = (
        (usable_df["prob_drowsiness"] + usable_df["prob_distraction"]) * 100.0
    ).round(1)

    return usable_df


def summarize_predictions(predictions_df: pd.DataFrame) -> dict[str, Any]:
    probability_columns = ["prob_normal", "prob_drowsiness", "prob_distraction"]
    mean_probabilities = predictions_df[probability_columns].mean().to_dict()
    label_map = {
        "prob_normal": "normal",
        "prob_drowsiness": "drowsiness",
        "prob_distraction": "distraction",
    }
    top_prob_key = max(mean_probabilities, key=mean_probabilities.get)
    overall_label = label_map[top_prob_key]
    overall_confidence = float(mean_probabilities[top_prob_key])
    risk_score = float((mean_probabilities["prob_drowsiness"] + mean_probabilities["prob_distraction"]) * 100.0)
    predicted_counts = {
        label: int((predictions_df["predicted_label"] == label).sum())
        for label in ("normal", "drowsiness", "distraction")
    }
    usable_window_count = int(len(predictions_df))
    dominant_window_count = int(predicted_counts.get(overall_label, 0))
    dominant_window_ratio = (
        float(dominant_window_count / usable_window_count) if usable_window_count else 0.0
    )
    mean_winner_confidence = float(predictions_df[probability_columns].max(axis=1).mean())

    return {
        "overall_label": overall_label,
        "overall_confidence": round(overall_confidence, 4),
        "risk_score": round(risk_score, 1),
        "mean_winner_confidence": round(mean_winner_confidence, 4),
        "usable_window_count": usable_window_count,
        "dominant_window_count": dominant_window_count,
        "dominant_window_ratio": round(dominant_window_ratio, 4),
        "predicted_window_counts": predicted_counts,
        "mean_probabilities": {
            label_map[key]: round(float(value), 4) for key, value in mean_probabilities.items()
        },
    }


def save_analysis_artifacts(
    analysis_dir: Path,
    quality: QualitySummary,
    perclos_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    perclos_df.to_csv(analysis_dir / "perclos_windows.csv", index=False)
    windows_df.to_csv(analysis_dir / "window_features.csv", index=False)
    predictions_df.to_csv(analysis_dir / "window_predictions.csv", index=False)

    payload = {
        "quality": asdict(quality),
        "summary": summary,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def analyze_video(
    video_path: str | Path,
    model_key: str = "xgboost_gaze_high_confidence",
    fast_mode: bool = FAST_MODE,
    label_override: str | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    source_video_path = Path(video_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_slug = f"{source_video_path.stem}_{timestamp}"
    analysis_dir = ANALYSES_ROOT / analysis_slug
    analysis_dir.mkdir(parents=True, exist_ok=True)

    saved_video_path = analysis_dir / source_video_path.name
    if source_video_path.resolve() != saved_video_path.resolve():
        saved_video_path.write_bytes(source_video_path.read_bytes())

    model_bundle = load_model_bundle(model_key)
    feature_set = str(model_bundle.get("feature_set", "baseline"))
    source_label = (label_override or infer_label_from_name(source_video_path.name)).strip().lower()
    modality = infer_modality_from_name(source_video_path.name)
    extraction_result = extract_video_features(saved_video_path, fast_mode=fast_mode)
    source_df = extraction_result.frame_features
    gaze_source_df = extraction_result.gaze_samples
    frame_csv_path = analysis_dir / "frame_features.csv"
    source_df.to_csv(frame_csv_path, sep=";", index=False)
    quality = summarize_quality(source_df)
    perclos_df = build_perclos_windows(source_df)
    windows_df = build_window_feature_table(
        source_df,
        perclos_df,
        saved_video_path.name,
        source_label=source_label,
        modality=modality,
    )
    artifacts: dict[str, str] = {"frame_csv": str(frame_csv_path)}

    if windows_df.empty:
        return {
            "status": "error",
            "message": "Video yeterli sayıda 3 saniyelik pencere üretmedi. Daha uzun veya daha net bir video deneyin.",
            "analysis_dir": str(analysis_dir),
            "quality": asdict(quality),
            "artifacts": artifacts,
        }

    if feature_set == "gaze":
        gaze_df, gaze_artifacts = build_gaze_window_table(
            gaze_source_df=gaze_source_df,
            video_name=saved_video_path.name,
            source_label=source_label,
            modality=modality,
        )
        windows_df = merge_baseline_with_gaze(windows_df, gaze_df)
        gaze_input_csv_path = analysis_dir / "gaze_input.csv"
        gaze_window_csv_path = analysis_dir / "gaze_windows.csv"
        gaze_source_df.to_csv(gaze_input_csv_path, index=False)
        gaze_df.to_csv(gaze_window_csv_path, index=False)
        artifacts.update(
            {
                "gaze_input_csv": str(gaze_input_csv_path),
                "gaze_window_csv": str(gaze_window_csv_path),
            }
        )
        artifacts.update(gaze_artifacts)

    predictions_df = build_prediction_table(windows_df, model_bundle)
    if predictions_df.empty:
        return {
            "status": "error",
            "message": "Kullanılabilir pencere bulunamadı. Yüz tespiti veya göz/baş pozu kalitesi düşük olabilir.",
            "analysis_dir": str(analysis_dir),
            "quality": asdict(quality),
            "artifacts": artifacts,
        }

    summary = summarize_predictions(predictions_df)
    save_analysis_artifacts(analysis_dir, quality, perclos_df, windows_df, predictions_df, summary)
    artifacts.update(
        {
            "perclos_csv": str(analysis_dir / "perclos_windows.csv"),
            "window_features_csv": str(analysis_dir / "window_features.csv"),
            "window_predictions_csv": str(analysis_dir / "window_predictions.csv"),
            "summary_json": str(analysis_dir / "summary.json"),
        }
    )
    analysis_id = save_analysis(
        analysis_slug=analysis_slug,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_video_name=saved_video_path.name,
        source_video_path=str(saved_video_path),
        source_label=source_label,
        source_modality=modality,
        model_key=model_key,
        feature_set=feature_set,
        fast_mode=fast_mode,
        summary=summary,
        quality=asdict(quality),
        artifacts=artifacts,
        predictions_df=predictions_df,
    )

    return {
        "status": "ok",
        "message": "Analiz tamamlandı.",
        "analysis_id": analysis_id,
        "analysis_dir": str(analysis_dir),
        "database_path": str(DATABASE_PATH),
        "video_path": str(saved_video_path),
        "model_key": model_key,
        "feature_set": feature_set,
        "source_label": source_label,
        "source_modality": modality,
        "quality": asdict(quality),
        "summary": summary,
        "timeline": predictions_df.to_dict(orient="records"),
        "artifacts": artifacts,
    }
