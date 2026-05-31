import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from scripts.models.classical_models.adaboost import MODEL_SPEC as ADABOOST_MODEL_SPEC
from scripts.models.classical_models.common import ModelSpec
from scripts.models.classical_models.decision_tree import MODEL_SPEC as DECISION_TREE_MODEL_SPEC
from scripts.models.classical_models.extra_trees import MODEL_SPEC as EXTRA_TREES_MODEL_SPEC
from scripts.models.classical_models.gradient_boosting import MODEL_SPEC as GRADIENT_BOOSTING_MODEL_SPEC
from scripts.models.classical_models.knn import MODEL_SPEC as KNN_MODEL_SPEC
from scripts.models.classical_models.linear_svm import MODEL_SPEC as LINEAR_SVM_MODEL_SPEC
from scripts.models.classical_models.logistic_regression import MODEL_SPEC as LOGISTIC_REGRESSION_MODEL_SPEC
from scripts.models.classical_models.rbf_svm import MODEL_SPEC as RBF_SVM_MODEL_SPEC
from scripts.models.classical_models.xgboost import MODEL_SPEC as XGBOOST_MODEL_SPEC
from scripts.models.random_forest.train_random_forest import (
    LABELS,
    PROJECT_ROOT,
    apply_high_confidence_rules,
    build_split,
    filter_dataset,
    load_dataset,
    resolve_feature_columns,
)


WINDOW_ROOT_DEFAULT = str(
    Path(r"D:\window") if os.path.exists(r"D:\window") else PROJECT_ROOT / "window_dataset"
)
GAZE_WINDOW_ROOT_DEFAULT = str(PROJECT_ROOT / "window_dataset_with_gaze")
OUTPUT_DIR_DEFAULT = str(PROJECT_ROOT / "results" / "model_comparison_baseline")


MODEL_SPECS: Dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random Forest",
        builder=lambda random_state, n_jobs: RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        ),
    ),
    ADABOOST_MODEL_SPEC.key: ADABOOST_MODEL_SPEC,
    DECISION_TREE_MODEL_SPEC.key: DECISION_TREE_MODEL_SPEC,
    EXTRA_TREES_MODEL_SPEC.key: EXTRA_TREES_MODEL_SPEC,
    GRADIENT_BOOSTING_MODEL_SPEC.key: GRADIENT_BOOSTING_MODEL_SPEC,
    KNN_MODEL_SPEC.key: KNN_MODEL_SPEC,
    LOGISTIC_REGRESSION_MODEL_SPEC.key: LOGISTIC_REGRESSION_MODEL_SPEC,
    LINEAR_SVM_MODEL_SPEC.key: LINEAR_SVM_MODEL_SPEC,
    RBF_SVM_MODEL_SPEC.key: RBF_SVM_MODEL_SPEC,
    XGBOOST_MODEL_SPEC.key: XGBOOST_MODEL_SPEC,
}


def writable_path(output_dir: str, base_name: str) -> str:
    base_path = Path(output_dir) / base_name
    if not base_path.exists():
        return str(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(base_path.with_stem(f"{base_path.stem}_{timestamp}"))


def dump_json(output_dir: str, payload: Dict, file_name: str):
    target = writable_path(output_dir, file_name)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dump_excel(output_dir: str, df: pd.DataFrame, file_name: str, index: bool = True):
    target = writable_path(output_dir, file_name)
    df.to_excel(target, index=index)


def dump_pickle(output_dir: str, payload: Dict, file_name: str):
    target = writable_path(output_dir, file_name)
    with open(target, "wb") as handle:
        pickle.dump(payload, handle)


def resolve_model_keys(models_arg: str) -> List[str]:
    if not models_arg or models_arg.strip().lower() == "all":
        return list(MODEL_SPECS.keys())

    requested = [item.strip() for item in models_arg.split(",") if item.strip()]
    invalid = [item for item in requested if item not in MODEL_SPECS]
    if invalid:
        raise ValueError(
            f"Unsupported model keys: {', '.join(invalid)}. "
            f"Supported keys: {', '.join(MODEL_SPECS)}"
        )
    return requested


def prepare_split(
    window_root: str,
    feature_set: str,
    use_high_confidence: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
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
    return train_df, test_df, feature_columns, filtered


def compute_importance_df(model: ClassifierMixin, feature_columns: List[str]) -> Optional[pd.DataFrame]:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["classifier"]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0)
    else:
        return None

    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)


def train_one_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    output_root: str,
    random_state: int,
    n_jobs: int,
    feature_set: str,
    use_high_confidence: bool,
) -> Dict:
    model = spec.builder(random_state, n_jobs)
    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(str)
    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(str)

    fit_kwargs = {}
    if spec.use_balanced_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)

    label_encoder = None
    y_train_fit = y_train
    y_test_eval = y_test

    if spec.key == "xgboost":
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train)
        y_test_eval = label_encoder.transform(y_test)
        if "sample_weight" in fit_kwargs:
            fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train_fit)

    model.fit(X_train, y_train_fit, **fit_kwargs)
    predictions_raw = model.predict(X_test)
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(np.asarray(predictions_raw, dtype=int))
    else:
        predictions = predictions_raw

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    report_dict = classification_report(y_test, predictions, output_dict=True, digits=4)
    confusion = confusion_matrix(y_test, predictions, labels=list(LABELS))
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    )

    metrics = {
        "model_key": spec.key,
        "model_label": spec.label,
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["group_id"].nunique()),
        "test_groups": int(test_df["group_id"].nunique()),
        "feature_set": feature_set,
        "use_high_confidence": bool(use_high_confidence),
    }

    model_output_dir = Path(output_root) / spec.key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = test_df.copy()
    prediction_df["predicted_label"] = predictions
    importance_df = compute_importance_df(model, feature_columns)

    dump_json(str(model_output_dir), metrics, "metrics.json")
    dump_json(str(model_output_dir), report_dict, "classification_report.json")
    dump_excel(str(model_output_dir), confusion_df, "confusion_matrix.xlsx", index=True)
    dump_excel(str(model_output_dir), prediction_df, "test_predictions.xlsx", index=False)
    if importance_df is not None:
        dump_excel(str(model_output_dir), importance_df, "feature_importance.xlsx", index=False)
    dump_pickle(
        str(model_output_dir),
        {
            "model": model,
            "feature_columns": feature_columns,
            "feature_set": feature_set,
            "use_high_confidence": bool(use_high_confidence),
            "labels": list(LABELS),
            "encoded_classes": list(label_encoder.classes_) if label_encoder is not None else None,
            "metrics": metrics,
        },
        "model_bundle.pkl",
    )

    print(f"\n[{spec.label}]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_model_comparison(
    window_root: str,
    output_dir: str,
    feature_set: str = "baseline",
    use_high_confidence: bool = False,
    random_state: int = 42,
    n_jobs: int = 1,
    models: str = "all",
) -> pd.DataFrame:
    model_keys = resolve_model_keys(models)
    train_df, test_df, feature_columns, _filtered = prepare_split(
        window_root=window_root,
        feature_set=feature_set,
        use_high_confidence=use_high_confidence,
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        metrics = train_one_model(
            spec=spec,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            output_root=output_dir,
            random_state=random_state,
            n_jobs=n_jobs,
            feature_set=feature_set,
            use_high_confidence=use_high_confidence,
        )
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=False,
    )
    dump_excel(output_dir, summary_df, "model_comparison_summary.xlsx", index=False)
    dump_json(
        output_dir,
        {
            "feature_set": feature_set,
            "use_high_confidence": bool(use_high_confidence),
            "models": model_keys,
            "summary_rows": summary_rows,
        },
        "model_comparison_summary.json",
    )
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple classical models on the same split.")
    parser.add_argument(
        "--window-root",
        default=WINDOW_ROOT_DEFAULT,
        help="Root folder containing baseline or gaze window datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for model-comparison outputs.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("baseline", "gaze"),
        default="baseline",
        help="Feature set to compare on.",
    )
    parser.add_argument(
        "--use-high-confidence",
        action="store_true",
        help="Restrict training/evaluation to the high-confidence subset.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for tree-based models.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys to run, or 'all'.",
    )
    args = parser.parse_args()

    summary_df = run_model_comparison(
        window_root=args.window_root,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        use_high_confidence=bool(args.use_high_confidence),
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        models=args.models,
    )
    print("\nComparison Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
