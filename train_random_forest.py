import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


WINDOW_ROOT_DEFAULT = r"D:\window"
OUTPUT_DIR_DEFAULT = os.path.join(os.getcwd(), "results", "random_forest_baseline")

FEATURE_COLUMNS = [
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

LABELS = ("normal", "drowsiness", "distraction")


def collect_window_files(window_root: str) -> List[str]:
    files: List[str] = []
    for label in LABELS:
        folder = Path(window_root) / label
        if not folder.exists():
            continue
        for path in folder.rglob("*_windows.xlsx"):
            if path.name.startswith("~$"):
                continue
            files.append(str(path))
    return sorted(files)


def load_dataset(window_root: str) -> pd.DataFrame:
    rows = []
    for path in collect_window_files(window_root):
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

    for col in FEATURE_COLUMNS + ["is_usable"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    return dataset


def filter_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    filtered = dataset[dataset["is_usable"] == 1].copy()
    filtered = filtered.dropna(subset=FEATURE_COLUMNS + ["label", "group_id"]).reset_index(drop=True)
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


def build_split(dataset: pd.DataFrame, random_state: int) -> Dict[str, pd.Index]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]
    groups = dataset["group_id"]

    train_idx, test_idx = next(splitter.split(X, y, groups))
    return {"train": train_idx, "test": test_idx}


def save_outputs(
    output_dir: str,
    metrics: Dict,
    report_dict: Dict,
    confusion_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
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

    dump_json(metrics, "metrics.json")
    dump_json(report_dict, "classification_report.json")
    dump_excel(confusion_df, "confusion_matrix.xlsx", index=True)
    dump_excel(feature_importance_df, "feature_importance.xlsx", index=False)
    dump_excel(train_df, "train_split.xlsx", index=False)
    dump_excel(test_df, "test_predictions.xlsx", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train a 3-class Random Forest baseline.")
    parser.add_argument("--window-root", default=WINDOW_ROOT_DEFAULT, help="Root folder containing window datasets.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Directory for training outputs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=0, help="Maximum tree depth. 0 means None.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Random Forest.")
    parser.add_argument(
        "--use-high-confidence",
        action="store_true",
        help="Train only on high-confidence normal/drowsiness/distraction windows.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.window_root)
    filtered = filter_dataset(dataset)
    if args.use_high_confidence:
        filtered = apply_high_confidence_rules(filtered)

    print(f"All windows: {len(dataset)}")
    print(f"Usable windows: {len(filtered)}")
    print(f"High confidence mode: {args.use_high_confidence}")
    print("Label counts:")
    print(filtered["label"].value_counts().to_string())

    split = build_split(filtered, args.random_state)
    train_df = filtered.iloc[split["train"]].copy().reset_index(drop=True)
    test_df = filtered.iloc[split["test"]].copy().reset_index(drop=True)

    print("\nTrain label counts:")
    print(train_df["label"].value_counts().to_string())
    print("\nTest label counts:")
    print(test_df["label"].value_counts().to_string())

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )

    clf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    predictions = clf.predict(test_df[FEATURE_COLUMNS])

    accuracy = accuracy_score(test_df["label"], predictions)
    macro_f1 = f1_score(test_df["label"], predictions, average="macro")
    weighted_f1 = f1_score(test_df["label"], predictions, average="weighted")

    report_dict = classification_report(test_df["label"], predictions, output_dict=True, digits=4)
    confusion = confusion_matrix(test_df["label"], predictions, labels=list(LABELS))
    confusion_df = pd.DataFrame(confusion, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])

    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
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
        "feature_columns": FEATURE_COLUMNS,
    }

    test_df["predicted_label"] = predictions
    save_outputs(
        output_dir=args.output_dir,
        metrics=metrics,
        report_dict=report_dict,
        confusion_df=confusion_df,
        feature_importance_df=feature_importance_df,
        train_df=train_df,
        test_df=test_df,
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nConfusion Matrix:")
    print(confusion_df.to_string())
    print("\nTop Feature Importances:")
    print(feature_importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
