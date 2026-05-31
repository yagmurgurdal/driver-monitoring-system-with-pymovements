import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOW_ROOT_DEFAULT = str(
    Path(r"D:\window") if os.path.exists(r"D:\window") else PROJECT_ROOT / "window_dataset"
)
OUTPUT_DIR_DEFAULT = str(PROJECT_ROOT / "results" / "random_forest_baseline")

BASELINE_FEATURE_COLUMNS = [
    "perclos",
    "perclos_percent",
    "mean_ear",
    "std_ear",
    "min_ear",
    "max_ear",
    "mean_abs_yaw",
    "std_yaw",
    "max_abs_yaw",
    "mean_abs_pitch",
    "std_pitch",
    "max_abs_pitch",
    "mean_abs_roll",
    "std_roll",
    "max_abs_roll",
    "face_detect_ratio",
    "pose_valid_ratio",
    "ear_valid_ratio",
]

GAZE_FEATURE_COLUMNS = [
    "gaze_gaze_valid_ratio",
    "gaze_usable_window",
    "gaze_mean_iris_x_norm",
    "gaze_std_iris_x_norm",
    "gaze_min_iris_x_norm",
    "gaze_max_iris_x_norm",
    "gaze_mean_iris_y_norm",
    "gaze_std_iris_y_norm",
    "gaze_min_iris_y_norm",
    "gaze_max_iris_y_norm",
    "gaze_gaze_dispersion_x",
    "gaze_gaze_dispersion_y",
    "gaze_gaze_dispersion_xy",
    "gaze_gaze_path_length",
    "gaze_mean_step_distance",
    "gaze_mean_velocity_norm",
    "gaze_std_velocity_norm",
    "gaze_max_velocity_norm",
    "gaze_idt_fixation_count",
    "gaze_idt_fixation_mean_duration_ms",
    "gaze_idt_fixation_max_duration_ms",
    "gaze_idt_fixation_ratio",
    "gaze_ivt_fixation_count",
    "gaze_ivt_fixation_mean_duration_ms",
    "gaze_ivt_fixation_max_duration_ms",
    "gaze_ivt_fixation_ratio",
    "gaze_rapid_shift_count",
    "gaze_rapid_shift_mean_duration_ms",
    "gaze_rapid_shift_max_duration_ms",
    "gaze_rapid_shift_ratio",
]

LABELS = ("normal", "drowsiness", "distraction")
FEATURE_SET_SUFFIXES = {
    "baseline": "*_windows.xlsx",
    "gaze": "*_windows_with_gaze.xlsx",
}


def fill_missing_gaze_duration_features(dataset: pd.DataFrame) -> pd.DataFrame:
    zero_fill_specs = {
        "gaze_idt_fixation_count": [
            "gaze_idt_fixation_mean_duration_ms",
            "gaze_idt_fixation_max_duration_ms",
        ],
        "gaze_ivt_fixation_count": [
            "gaze_ivt_fixation_mean_duration_ms",
            "gaze_ivt_fixation_max_duration_ms",
        ],
        "gaze_rapid_shift_count": [
            "gaze_rapid_shift_mean_duration_ms",
            "gaze_rapid_shift_max_duration_ms",
        ],
    }

    for count_col, duration_cols in zero_fill_specs.items():
        if count_col not in dataset.columns:
            continue
        count_series = pd.to_numeric(dataset[count_col], errors="coerce")
        zero_count_mask = count_series == 0
        for col in duration_cols:
            if col not in dataset.columns:
                continue
            duration_series = pd.to_numeric(dataset[col], errors="coerce")
            dataset[col] = duration_series.mask(zero_count_mask & duration_series.isna(), 0.0)

    return dataset


def resolve_feature_columns(feature_set: str) -> List[str]:
    if feature_set == "baseline":
        return list(BASELINE_FEATURE_COLUMNS)
    if feature_set == "gaze":
        return list(BASELINE_FEATURE_COLUMNS) + list(GAZE_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def collect_window_files(window_root: str, feature_set: str) -> List[str]:
    files: List[str] = []
    pattern = FEATURE_SET_SUFFIXES[feature_set]
    for label in LABELS:
        folder = Path(window_root) / label
        if not folder.exists():
            continue
        for path in folder.rglob(pattern):
            if path.name.startswith("~$"):
                continue
            files.append(str(path))
    return sorted(files)


def load_dataset(window_root: str, feature_set: str, feature_columns: List[str]) -> pd.DataFrame:
    rows = []
    for path in collect_window_files(window_root, feature_set):
        df = pd.read_excel(path)
        df["window_file"] = path
        rows.append(df)

    if not rows:
        raise ValueError(f"No *_windows.xlsx files found under: {window_root}")

    dataset = pd.concat(rows, ignore_index=True)
    dataset["label"] = dataset["label"].astype(str)
    dataset["modality"] = dataset["modality"].astype(str)
    dataset["file_stem"] = dataset["file_stem"].astype(str)
    dataset["group_id"] = dataset["modality"] + "::" + dataset["file_stem"]

    for col in feature_columns + ["is_usable"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    if feature_set == "gaze":
        dataset = fill_missing_gaze_duration_features(dataset)

    return dataset


def filter_dataset(dataset: pd.DataFrame, feature_columns: List[str], feature_set: str) -> pd.DataFrame:
    filtered = dataset[dataset["is_usable"] == 1].copy()
    if feature_set == "gaze" and "gaze_usable_window" in filtered.columns:
        filtered = filtered[filtered["gaze_usable_window"] == 1].copy()
    filtered = filtered.dropna(subset=feature_columns + ["label", "group_id"]).reset_index(drop=True)
    return filtered


def apply_high_confidence_rules(dataset: pd.DataFrame) -> pd.DataFrame:
    normal_mask = (
        (dataset["label"] == "normal")
        & (dataset["perclos"] <= 0.05)
        & (dataset["mean_ear"] >= 0.24)
        & (dataset["std_ear"] <= 0.025)
        & (dataset["mean_abs_yaw"] <= 12)
        & (dataset["max_abs_yaw"] <= 18)
        & (dataset["mean_abs_pitch"] <= 7)
        & (dataset["max_abs_pitch"] <= 12)
    )

    drowsiness_mask = (
        (dataset["perclos"] >= 0.20)
        & (dataset["mean_ear"] <= 0.20)
        & (dataset["min_ear"] <= 0.08)
        & (dataset["mean_abs_yaw"] <= 10)
        & (dataset["max_abs_yaw"] <= 18)
    )

    distraction_mask = (
        (dataset["mean_abs_yaw"] >= 18)
        & (dataset["max_abs_yaw"] >= 30)
        & (dataset["perclos"] <= 0.12)
    )

    hc_df = dataset[normal_mask | drowsiness_mask | distraction_mask].copy()
    hc_df = hc_df.reset_index(drop=True)
    return hc_df


def build_split(dataset: pd.DataFrame, feature_columns: List[str], random_state: int) -> Dict[str, pd.Index]:
    per_label_group_counts = dataset.groupby("label")["group_id"].nunique()
    min_group_count = int(per_label_group_counts.min()) if not per_label_group_counts.empty else 0
    n_splits = min(5, min_group_count)

    X = dataset[feature_columns]
    y = dataset["label"]
    groups = dataset["group_id"]

    if n_splits >= 2:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return {"train": train_idx, "test": test_idx}

    rng = np.random.default_rng(random_state)
    test_group_ids = set()

    for label, label_df in dataset.groupby("label", sort=True):
        label_groups = label_df["group_id"].drop_duplicates().to_numpy()
        if label_groups.size <= 1:
            continue

        shuffled_groups = rng.permutation(label_groups)
        desired_test_groups = max(1, int(round(label_groups.size * 0.2)))
        desired_test_groups = min(desired_test_groups, label_groups.size - 1)
        test_group_ids.update(shuffled_groups[:desired_test_groups].tolist())

    if not test_group_ids:
        raise ValueError(
            "Could not create a group-aware train/test split. "
            "Need at least one label with two or more distinct groups."
        )

    test_mask = dataset["group_id"].isin(test_group_ids)
    train_idx = dataset.index[~test_mask]
    test_idx = dataset.index[test_mask]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Group-aware fallback split produced an empty train or test set.")

    return {"train": train_idx, "test": test_idx}


def save_outputs(
    output_dir: str,
    metrics: Dict,
    report_dict: Dict,
    confusion_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_bundle: Dict,
):
    os.makedirs(output_dir, exist_ok=True)

    def writable_path(base_name: str) -> str:
        base_path = Path(output_dir) / base_name
        if not base_path.exists():
            return str(base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(base_path.with_stem(f"{base_path.stem}_{timestamp}"))

    def dump_json(payload: Dict, file_name: str):
        target = writable_path(file_name)
        try:
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except PermissionError:
            fallback = writable_path(file_name)
            with open(fallback, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

    def dump_excel(df: pd.DataFrame, file_name: str, index: bool = True):
        target = writable_path(file_name)
        try:
            df.to_excel(target, index=index)
        except PermissionError:
            fallback = writable_path(file_name)
            df.to_excel(fallback, index=index)

    def dump_pickle(payload: Dict, file_name: str):
        target = writable_path(file_name)
        try:
            with open(target, "wb") as handle:
                pickle.dump(payload, handle)
        except PermissionError:
            fallback = writable_path(file_name)
            with open(fallback, "wb") as handle:
                pickle.dump(payload, handle)

    dump_json(metrics, "metrics.json")
    dump_json(report_dict, "classification_report.json")
    dump_excel(confusion_df, "confusion_matrix.xlsx", index=True)
    dump_excel(feature_importance_df, "feature_importance.xlsx", index=False)
    dump_excel(train_df, "train_split.xlsx", index=False)
    dump_excel(test_df, "test_predictions.xlsx", index=False)
    dump_pickle(model_bundle, "model_bundle.pkl")


def run_training(
    window_root: str,
    output_dir: str,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 0,
    n_jobs: int = 1,
    feature_set: str = "baseline",
    use_high_confidence: bool = False,
) -> Dict:
    feature_columns = resolve_feature_columns(feature_set)
    dataset = load_dataset(window_root, feature_set, feature_columns)
    filtered = filter_dataset(dataset, feature_columns, feature_set)
    if use_high_confidence:
        filtered = apply_high_confidence_rules(filtered)

    print(f"All windows: {len(dataset)}")
    print(f"Usable windows: {len(filtered)}")
    print(f"High confidence mode: {use_high_confidence}")
    print("Label counts:")
    print(filtered["label"].value_counts().to_string())

    split = build_split(filtered, feature_columns, random_state)
    filtered = filtered.reset_index(drop=True)
    train_df = filtered.iloc[split["train"]].copy().reset_index(drop=True)
    test_df = filtered.iloc[split["test"]].copy().reset_index(drop=True)

    print("\nTrain label counts:")
    print(train_df["label"].value_counts().to_string())
    print("\nTest label counts:")
    print(test_df["label"].value_counts().to_string())

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth <= 0 else max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )

    clf.fit(train_df[feature_columns], train_df["label"])
    predictions = clf.predict(test_df[feature_columns])

    accuracy = accuracy_score(test_df["label"], predictions)
    macro_f1 = f1_score(test_df["label"], predictions, average="macro")
    weighted_f1 = f1_score(test_df["label"], predictions, average="weighted")

    report_dict = classification_report(test_df["label"], predictions, output_dict=True, digits=4)
    confusion = confusion_matrix(test_df["label"], predictions, labels=list(LABELS))
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["group_id"].nunique()),
        "test_groups": int(test_df["group_id"].nunique()),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
    }

    test_df["predicted_label"] = predictions
    save_outputs(
        output_dir=output_dir,
        metrics=metrics,
        report_dict=report_dict,
        confusion_df=confusion_df,
        feature_importance_df=feature_importance_df,
        train_df=train_df,
        test_df=test_df,
        model_bundle={
            "model": clf,
            "feature_columns": feature_columns,
            "feature_set": feature_set,
            "use_high_confidence": bool(use_high_confidence),
            "labels": list(LABELS),
            "metrics": metrics,
        },
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a 3-class Random Forest baseline.")
    parser.add_argument("--window-root", default=WINDOW_ROOT_DEFAULT, help="Root folder containing window datasets.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Directory for training outputs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=0, help="Maximum tree depth. 0 means None.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Random Forest.")
    parser.add_argument(
        "--feature-set",
        choices=("baseline", "gaze"),
        default="baseline",
        help="Feature set to train on. Use 'gaze' for merged baseline+PyMovements gaze features.",
    )
    parser.add_argument(
        "--use-high-confidence",
        action="store_true",
        help="Train only on high-confidence normal/drowsiness/distraction windows.",
    )
    args = parser.parse_args()
    run_training(
        window_root=args.window_root,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        feature_set=args.feature_set,
        use_high_confidence=args.use_high_confidence,
    )


if __name__ == "__main__":
    main()
