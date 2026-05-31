from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "report_assets" / "model_comparison_20260531"

MODELS = OrderedDict(
    [
        ("random_forest", "Random Forest"),
        ("extra_trees", "Extra Trees"),
        ("gradient_boosting", "Gradient Boosting"),
        ("decision_tree", "Decision Tree"),
        ("adaboost", "AdaBoost"),
        ("logistic_regression", "Logistic Regression"),
        ("linear_svm", "Linear SVM"),
        ("rbf_svm", "RBF SVM"),
        ("knn", "K-Nearest Neighbors"),
        ("xgboost", "XGBoost"),
    ]
)

SCENARIOS = OrderedDict(
    [
        ("baseline", "Baseline"),
        ("high_confidence", "High-Confidence"),
        ("gaze_baseline", "Gaze Baseline"),
        ("gaze_high_confidence", "Gaze High-Confidence"),
    ]
)

SCENARIO_COLORS = {
    "Baseline": "#94a3b8",
    "High-Confidence": "#2563eb",
    "Gaze Baseline": "#f59e0b",
    "Gaze High-Confidence": "#16a34a",
}


def resolve_result_file(model_key: str, scenario_key: str, filename: str) -> Path:
    result_dir = RESULTS_ROOT / f"{model_key}_{scenario_key}"
    direct_path = result_dir / filename
    nested_path = result_dir / model_key / filename

    if direct_path.exists():
        return direct_path
    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(f"Could not find {filename} for {model_key}_{scenario_key}")


def load_metrics_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_key, model_label in MODELS.items():
        for scenario_key, scenario_label in SCENARIOS.items():
            metrics_path = resolve_result_file(model_key, scenario_key, "metrics.json")
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label,
                    "accuracy": payload["accuracy"],
                    "macro_f1": payload["macro_f1"],
                    "weighted_f1": payload["weighted_f1"],
                }
            )

    return pd.DataFrame(rows)


def build_table_one(metrics_df: pd.DataFrame) -> pd.DataFrame:
    accuracy_wide = (
        metrics_df.pivot(index="model_label", columns="scenario_label", values="accuracy")
        .reindex(index=list(MODELS.values()), columns=list(SCENARIOS.values()))
        .reset_index()
    )

    best_scores = (
        metrics_df.groupby("model_label")[["macro_f1", "weighted_f1"]]
        .max()
        .rename(columns={"macro_f1": "Best Macro F1", "weighted_f1": "Best Weighted F1"})
    )

    table = accuracy_wide.merge(best_scores, left_on="model_label", right_index=True)
    table = table.rename(columns={"model_label": "Model"})
    return table


def build_table_two(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        metrics_df[metrics_df["scenario_key"] == "gaze_high_confidence"]
        .sort_values(["accuracy", "macro_f1", "weighted_f1"], ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    ranked = ranked.rename(
        columns={
            "model_label": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "weighted_f1": "Weighted F1",
        }
    )
    return ranked[["Rank", "Model", "Accuracy", "Macro F1", "Weighted F1"]]


def save_table_files(table: pd.DataFrame, stem: str, title: str) -> None:
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    xlsx_path = OUTPUT_DIR / f"{stem}.xlsx"
    png_path = OUTPUT_DIR / f"{stem}.png"

    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)

    display_table = table.copy()
    for column in display_table.columns:
        if display_table[column].dtype.kind in {"f", "c"}:
            display_table[column] = display_table[column].map(lambda value: f"{value:.6f}")

    fig_height = 0.55 * (len(display_table) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)

    table_artist = ax.table(
        cellText=display_table.values,
        colLabels=list(display_table.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10)
    table_artist.scale(1, 1.45)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#eef2ff")

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_grouped_accuracy_chart(metrics_df: pd.DataFrame) -> None:
    pivot = (
        metrics_df.pivot(index="model_label", columns="scenario_label", values="accuracy")
        .reindex(index=list(MODELS.values()), columns=list(SCENARIOS.values()))
    )

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(15, 7))

    for idx, scenario in enumerate(pivot.columns):
        offsets = x + (idx - 1.5) * width
        bars = ax.bar(
            offsets,
            pivot[scenario].values,
            width=width,
            label=scenario,
            color=SCENARIO_COLORS[scenario],
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Experimental Settings", fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_1_grouped_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_model_comparison_charts(metrics_df: pd.DataFrame) -> None:
    within_dir = OUTPUT_DIR / "within_model_figures"
    within_dir.mkdir(parents=True, exist_ok=True)

    for model_key, model_label in MODELS.items():
        subset = (
            metrics_df[metrics_df["model_key"] == model_key]
            .set_index("scenario_label")
            .reindex(list(SCENARIOS.values()))
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        bars = ax.bar(
            subset["scenario_label"],
            subset["accuracy"],
            color=[SCENARIO_COLORS[label] for label in subset["scenario_label"]],
            width=0.62,
        )

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_label}: Accuracy Across Four Experimental Settings", fontsize=15, fontweight="bold")
        ax.set_xticks(np.arange(len(subset["scenario_label"])))
        ax.set_xticklabels(subset["scenario_label"], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(within_dir / f"{model_key}_four_setting_comparison.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_setting_bar_chart(metrics_df: pd.DataFrame, scenario_key: str, figure_number: int) -> None:
    subset = (
        metrics_df[metrics_df["scenario_key"] == scenario_key]
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    scenario_label = SCENARIOS[scenario_key]

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    bars = ax.bar(
        subset["model_label"],
        subset["accuracy"],
        color=SCENARIO_COLORS[scenario_label],
        alpha=0.9,
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{scenario_label} Accuracy Ranking Across Models", fontsize=16, fontweight="bold")
    ax.set_xticks(np.arange(len(subset["model_label"])))
    ax.set_xticklabels(subset["model_label"], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    stem = scenario_key.replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"figure_{figure_number}_{stem}_ranking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_confusion_matrix(model_key: str, scenario_key: str) -> tuple[np.ndarray, list[str], list[str]]:
    path = resolve_result_file(model_key, scenario_key, "confusion_matrix.xlsx")
    df = pd.read_excel(path)

    row_labels = [str(value).replace("true_", "").title() for value in df.iloc[:, 0]]
    col_labels = [str(value).replace("pred_", "").title() for value in df.columns[1:]]
    matrix = df.iloc[:, 1:].to_numpy()
    return matrix, row_labels, col_labels


def draw_confusion(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "#0f172a",
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_confusion_figures() -> None:
    matrix, rows, cols = load_confusion_matrix("xgboost", "gaze_high_confidence")
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_confusion(ax, matrix, rows, cols, "XGBoost + Gaze High-Confidence")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_6_xgboost_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    top_models = [
        ("xgboost", "XGBoost"),
        ("extra_trees", "Extra Trees"),
        ("random_forest", "Random Forest"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (model_key, model_label) in zip(axes, top_models):
        matrix, rows, cols = load_confusion_matrix(model_key, "gaze_high_confidence")
        draw_confusion(ax, matrix, rows, cols, model_label)
    fig.suptitle("Top-3 Confusion Matrices Under Gaze High-Confidence", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_7_top3_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_figure() -> None:
    feature_path = resolve_result_file("xgboost", "gaze_high_confidence", "feature_importance.xlsx")
    df = pd.read_excel(feature_path).sort_values("importance", ascending=False).head(20)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    bars = ax.barh(df["feature"], df["importance"], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 XGBoost Feature Importances", fontsize=16, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        ax.text(
            bar.get_width() + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_8_xgboost_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(table_one: pd.DataFrame, table_two: pd.DataFrame) -> None:
    lines = [
        "# Model Comparison Assets",
        "",
        "Generated visual assets for the May 31, 2026 model-comparison report update.",
        "",
        "## Tables",
        "",
        "- `table_1_all_models_all_settings.{csv,xlsx,png}`",
        "- `table_2_gaze_high_confidence_ranking.{csv,xlsx,png}`",
        "",
        "## Figures",
        "",
        "- `figure_1_grouped_accuracy.png`",
        "- `figure_2_baseline_ranking.png`",
        "- `figure_3_high_confidence_ranking.png`",
        "- `figure_4_gaze_baseline_ranking.png`",
        "- `figure_5_gaze_high_confidence_ranking.png`",
        "- `figure_6_xgboost_confusion_matrix.png`",
        "- `figure_7_top3_confusion_matrices.png`",
        "- `figure_8_xgboost_feature_importance.png`",
        "- `within_model_figures/*.png`: one four-setting comparison chart for each algorithm.",
        "",
        "## Quick Summary",
        "",
        f"- Best overall model: {table_two.iloc[0]['Model']} ({table_two.iloc[0]['Accuracy']:.6f} accuracy)",
        f"- Second-best model: {table_two.iloc[1]['Model']} ({table_two.iloc[1]['Accuracy']:.6f} accuracy)",
        f"- Third-best model: {table_two.iloc[2]['Model']} ({table_two.iloc[2]['Accuracy']:.6f} accuracy)",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = load_metrics_frame()
    table_one = build_table_one(metrics_df)
    table_two = build_table_two(metrics_df)

    save_table_files(table_one, "table_1_all_models_all_settings", "Table 1. All Models Across All Experimental Settings")
    save_table_files(table_two, "table_2_gaze_high_confidence_ranking", "Table 2. Gaze High-Confidence Ranking")

    save_grouped_accuracy_chart(metrics_df)
    save_per_model_comparison_charts(metrics_df)
    save_setting_bar_chart(metrics_df, "baseline", 2)
    save_setting_bar_chart(metrics_df, "high_confidence", 3)
    save_setting_bar_chart(metrics_df, "gaze_baseline", 4)
    save_setting_bar_chart(metrics_df, "gaze_high_confidence", 5)
    save_confusion_figures()
    save_feature_importance_figure()
    write_manifest(table_one, table_two)

    print(f"Created assets in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
