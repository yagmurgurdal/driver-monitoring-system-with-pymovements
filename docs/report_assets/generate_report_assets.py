from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "report_assets"

MODEL_DIRS = {
    "Baseline RF": RESULTS_ROOT / "random_forest_baseline",
    "High-Confidence RF": RESULTS_ROOT / "random_forest_high_confidence",
    "Gaze Baseline RF": RESULTS_ROOT / "random_forest_gaze_baseline",
    "Gaze High-Confidence RF": RESULTS_ROOT / "random_forest_gaze_high_confidence",
}


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for label, folder in MODEL_DIRS.items():
        path = folder / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            metrics[label] = json.load(handle)
    return metrics


def save_model_comparison_chart(metrics: dict[str, dict]) -> Path:
    labels = list(metrics.keys())
    accuracy = [metrics[label]["accuracy"] for label in labels]
    macro_f1 = [metrics[label]["macro_f1"] for label in labels]

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar([i - width / 2 for i in x], accuracy, width=width, label="Accuracy", color="#1f77b4")
    bars2 = ax.bar([i + width / 2 for i in x], macro_f1, width=width, label="Macro F1", color="#ff7f0e")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def add_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f1f1f", linewidth=1.3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))


def save_system_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.05, 0.78, 0.18, 0.12, "Raw Driver Videos\n(RGB / IR)", "#dbeafe")
    add_box(ax, 0.29, 0.78, 0.18, 0.12, "Video Standardization\nFFmpeg / Codec Fixes", "#e0f2fe")
    add_box(ax, 0.53, 0.78, 0.18, 0.12, "Frame-Level Features\nFace, EAR, Head Pose", "#dcfce7")
    add_box(ax, 0.77, 0.78, 0.18, 0.12, "Normal-Class Extraction\nRule-Based Window Filtering", "#fef3c7")

    add_box(ax, 0.17, 0.52, 0.22, 0.12, "Window Dataset Creation\nBaseline Behavioral Features", "#ede9fe")
    add_box(ax, 0.43, 0.52, 0.22, 0.12, "Validation Scripts\nWindow / Normal Output Checks", "#fce7f3")
    add_box(ax, 0.69, 0.52, 0.22, 0.12, "Random Forest Training\nBaseline + High Confidence", "#fee2e2")

    add_box(ax, 0.08, 0.24, 0.22, 0.12, "PyMovements Input Extraction\nIris x/y Time Series", "#cffafe")
    add_box(ax, 0.36, 0.24, 0.22, 0.12, "Gaze Window Features\nI-DT / I-VT / Velocity", "#d1fae5")
    add_box(ax, 0.64, 0.24, 0.22, 0.12, "Merged Dataset + Gaze RF\nBest Accuracy / Macro F1", "#fde68a")

    add_box(ax, 0.36, 0.02, 0.28, 0.12, "Real-Time Monitoring\nLive Inference + SQLite + Audio Alert", "#fecaca")

    add_arrow(ax, (0.23, 0.84), (0.29, 0.84))
    add_arrow(ax, (0.47, 0.84), (0.53, 0.84))
    add_arrow(ax, (0.71, 0.84), (0.77, 0.84))

    add_arrow(ax, (0.53, 0.78), (0.28, 0.64))
    add_arrow(ax, (0.86, 0.78), (0.75, 0.64))
    add_arrow(ax, (0.39, 0.58), (0.43, 0.58))
    add_arrow(ax, (0.65, 0.58), (0.69, 0.58))

    add_arrow(ax, (0.53, 0.78), (0.19, 0.36))
    add_arrow(ax, (0.30, 0.30), (0.36, 0.30))
    add_arrow(ax, (0.58, 0.30), (0.64, 0.30))
    add_arrow(ax, (0.50, 0.24), (0.50, 0.14))

    ax.text(0.5, 0.95, "Driver Risk Analysis Pipeline", ha="center", va="center", fontsize=16, fontweight="bold")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "system_pipeline.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def copy_existing_plot(source: Path, target_name: str) -> Path | None:
    if not source.exists():
        return None
    target = OUTPUT_DIR / target_name
    shutil.copy2(source, target)
    return target


def write_summary(metrics: dict[str, dict]) -> Path:
    lines = [
        "# Report Assets",
        "",
        "This folder contains report-ready visual assets generated from the current project outputs.",
        "",
        "## Current Metrics",
        "",
    ]
    for label, payload in metrics.items():
        lines.append(
            f"- {label}: accuracy={payload['accuracy']:.6f}, macro_f1={payload['macro_f1']:.6f}, weighted_f1={payload['weighted_f1']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `model_comparison.png`: grouped bar chart comparing model variants.",
            "- `system_pipeline.png`: high-level processing pipeline diagram.",
            "- `best_model_confusion_matrix.png`: copied from the best current model output.",
            "- `best_model_feature_importance_top25.png`: top 25 feature importance plot for the best current model.",
            "- `best_model_feature_importance_gaze_only.png`: gaze-only importance plot for the best current model.",
        ]
    )
    output_path = OUTPUT_DIR / "README.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    generated = [
        save_model_comparison_chart(metrics),
        save_system_pipeline_diagram(),
        copy_existing_plot(
            RESULTS_ROOT / "random_forest_gaze_high_confidence" / "confusion_matrix.png",
            "best_model_confusion_matrix.png",
        ),
        copy_existing_plot(
            RESULTS_ROOT / "random_forest_gaze_high_confidence" / "feature_importance_top25.png",
            "best_model_feature_importance_top25.png",
        ),
        copy_existing_plot(
            RESULTS_ROOT / "random_forest_gaze_high_confidence" / "feature_importance_gaze_only.png",
            "best_model_feature_importance_gaze_only.png",
        ),
        write_summary(metrics),
    ]

    for path in generated:
        if path is not None:
            print(f"Created: {path}")


if __name__ == "__main__":
    main()
