import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def accuracy_from_confusion_matrix(path: Path) -> dict:
    df = pd.read_excel(path, index_col=0)
    total = int(df.to_numpy().sum())
    correct = int(sum(df.iloc[i, i] for i in range(min(df.shape))))
    accuracy = (correct / total) if total else 0.0
    return {
        "source": str(path),
        "method": "confusion_matrix",
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def accuracy_from_predictions(path: Path) -> dict:
    df = pd.read_excel(path)
    required = {"label", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} missing required columns: {missing_text}")

    correct = int((df["label"].astype(str) == df["predicted_label"].astype(str)).sum())
    total = int(len(df))
    accuracy = (correct / total) if total else 0.0
    return {
        "source": str(path),
        "method": "test_predictions",
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def load_reported_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("accuracy")
    return float(value) if value is not None else None


def resolve_latest_file(results_dir: Path, stem: str, suffix: str) -> Path:
    direct = results_dir / f"{stem}{suffix}"
    if direct.exists():
        return direct

    matches = sorted(results_dir.glob(f"{stem}*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {stem}*{suffix} in {results_dir}")
    return matches[0]


def print_result(title: str, result: dict, reported: float | None):
    print(f"\n{title}")
    print(f"  Source:   {result['source']}")
    print(f"  Method:   {result['method']}")
    print(f"  Correct:  {result['correct']}")
    print(f"  Total:    {result['total']}")
    print(f"  Accuracy: {result['accuracy']:.6f} ({result['accuracy'] * 100:.2f}%)")
    if reported is not None:
        diff = abs(result["accuracy"] - reported)
        print(f"  Reported: {reported:.6f}")
        print(f"  Diff:     {diff:.10f}")


def main():
    parser = argparse.ArgumentParser(description="Recalculate and verify model accuracy.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "random_forest_high_confidence",
        help="Results directory containing confusion_matrix.xlsx, test_predictions.xlsx, and metrics.json",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    confusion_path = resolve_latest_file(results_dir, "confusion_matrix", ".xlsx")
    predictions_path = resolve_latest_file(results_dir, "test_predictions", ".xlsx")
    metrics_path = resolve_latest_file(results_dir, "metrics", ".json")

    reported = load_reported_accuracy(metrics_path)
    confusion_result = accuracy_from_confusion_matrix(confusion_path)
    prediction_result = accuracy_from_predictions(predictions_path)

    print_result("Accuracy From Confusion Matrix", confusion_result, reported)
    print_result("Accuracy From Test Predictions", prediction_result, reported)


if __name__ == "__main__":
    main()
