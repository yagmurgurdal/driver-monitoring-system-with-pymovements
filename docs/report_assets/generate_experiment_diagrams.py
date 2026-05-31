from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "report_assets" / "experiment_diagrams_20260601"

EXPERIMENTS = [
    {
        "key": "baseline",
        "title": "Baseline Setting",
        "architecture_nodes": [
            "DMD Videos",
            "Video Standardization\nand Cleaning",
            "Frame-Level Feature Extraction\n(EAR, Head Pose, Face Validity)",
            "Window Dataset Creation",
            "Baseline Feature Set",
            "Multi-Model Training\nand Evaluation",
        ],
        "flow_nodes": [
            "Input Videos",
            "Standardize Videos",
            "Extract Frame-Level Signals",
            "Apply Data Cleaning\nand Imputation",
            "Compute PERCLOS\nand Window Statistics",
            "Build Baseline Dataset",
            "Train All Models",
            "Evaluate Accuracy,\nF1, Confusion Matrix",
        ],
        "color": "#2563eb",
    },
    {
        "key": "high_confidence",
        "title": "High-Confidence Setting",
        "architecture_nodes": [
            "DMD Videos",
            "Video Standardization\nand Cleaning",
            "Frame-Level Feature Extraction\n(EAR, Head Pose, Face Validity)",
            "Window Dataset Creation",
            "High-Confidence Rule-Based\nWindow Selection",
            "Filtered Baseline Feature Set",
            "Multi-Model Training\nand Evaluation",
        ],
        "flow_nodes": [
            "Input Videos",
            "Standardize Videos",
            "Extract Frame-Level Signals",
            "Apply Data Cleaning\nand Imputation",
            "Compute PERCLOS\nand Window Statistics",
            "Build Baseline Dataset",
            "Filter Windows by\nClass-Specific Rules",
            "Train All Models",
            "Evaluate Accuracy,\nF1, Confusion Matrix",
        ],
        "color": "#0f766e",
    },
    {
        "key": "gaze_baseline",
        "title": "Gaze Baseline Setting",
        "architecture_nodes": [
            "DMD Videos",
            "Video Standardization\nand Cleaning",
            "Frame-Level Feature Extraction\n(EAR, Head Pose, Face Validity)",
            "PyMovements Input Extraction\n(Iris Coordinates)",
            "Gaze Feature Generation",
            "Merge Baseline + Gaze\nWindow Features",
            "Gaze-Enriched Dataset",
            "Multi-Model Training\nand Evaluation",
        ],
        "flow_nodes": [
            "Input Videos",
            "Standardize Videos",
            "Extract Frame-Level Signals",
            "Apply Data Cleaning\nand Imputation",
            "Build Baseline Window Dataset",
            "Extract Iris Coordinates",
            "Generate Gaze Features",
            "Merge Baseline and Gaze Data",
            "Train All Models",
            "Evaluate Accuracy,\nF1, Confusion Matrix",
        ],
        "color": "#d97706",
    },
    {
        "key": "gaze_high_confidence",
        "title": "Gaze High-Confidence Setting",
        "architecture_nodes": [
            "DMD Videos",
            "Video Standardization\nand Cleaning",
            "Frame-Level Feature Extraction\n(EAR, Head Pose, Face Validity)",
            "PyMovements Input Extraction\n(Iris Coordinates)",
            "Gaze Feature Generation",
            "Merge Baseline + Gaze\nWindow Features",
            "High-Confidence Filtering\n+ Gaze Quality Control",
            "Filtered Gaze-Enriched Dataset",
            "Multi-Model Training\nand Evaluation",
        ],
        "flow_nodes": [
            "Input Videos",
            "Standardize Videos",
            "Extract Frame-Level Signals",
            "Apply Data Cleaning\nand Imputation",
            "Build Baseline Window Dataset",
            "Extract Iris Coordinates",
            "Generate Gaze Features",
            "Merge Baseline and Gaze Data",
            "Apply High-Confidence Rules\nand Gaze Usability Check",
            "Train All Models",
            "Evaluate Accuracy,\nF1, Confusion Matrix",
        ],
        "color": "#16a34a",
    },
]


def add_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, wrap=True)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#374151"))


def draw_horizontal_architecture(title: str, nodes: list[str], color: str, output_path: Path) -> None:
    use_two_rows = len(nodes) > 7
    fig_height = 6.6 if use_two_rows else 4.8
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    palette = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7", "#fde68a", "#fce7f3", "#ede9fe", "#fee2e2", "#cffafe"]

    if not use_two_rows:
        n = len(nodes)
        margin_x = 0.04
        gap = 0.02
        width = (1 - 2 * margin_x - gap * (n - 1)) / n
        y = 0.35
        h = 0.28

        centers = []
        for i, node in enumerate(nodes):
            x = margin_x + i * (width + gap)
            add_box(ax, x, y, width, h, node, palette[i % len(palette)])
            centers.append((x + width, y + h / 2))

        for i in range(len(centers) - 1):
            add_arrow(ax, centers[i], (margin_x + (i + 1) * (width + gap), y + h / 2))
    else:
        split_index = (len(nodes) + 1) // 2
        row_nodes = [nodes[:split_index], nodes[split_index:]]
        row_positions = [0.58, 0.18]
        h = 0.2
        row_box_meta: list[tuple[float, float, float, float]] = []

        for row_idx, current_nodes in enumerate(row_nodes):
            n = len(current_nodes)
            margin_x = 0.05
            gap = 0.02
            width = (1 - 2 * margin_x - gap * (n - 1)) / n
            y = row_positions[row_idx]
            for i, node in enumerate(current_nodes):
                x = margin_x + i * (width + gap)
                add_box(ax, x, y, width, h, node, palette[(row_idx * 5 + i) % len(palette)])
                row_box_meta.append((x, y, width, h))
                if i < n - 1:
                    add_arrow(ax, (x + width, y + h / 2), (margin_x + (i + 1) * (width + gap), y + h / 2))

        first_row_last = row_box_meta[split_index - 1]
        second_row_first = row_box_meta[split_index]
        add_arrow(
            ax,
            (first_row_last[0] + first_row_last[2] / 2, first_row_last[1]),
            (second_row_first[0] + second_row_first[2] / 2, second_row_first[1] + second_row_first[3]),
        )

    ax.text(0.5, 0.92, f"{title} Architecture", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_vertical_flowchart(title: str, nodes: list[str], color: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(nodes)
    top = 0.93
    bottom = 0.06
    h = 0.065
    available = top - bottom - n * h
    gap = available / (n - 1) if n > 1 else 0.02
    x = 0.18
    w = 0.64

    palette = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7", "#fde68a", "#fce7f3", "#ede9fe", "#fee2e2", "#cffafe", "#d1fae5", "#fae8ff"]
    ys = []
    current_y = top - h
    for i, node in enumerate(nodes):
        add_box(ax, x, current_y, w, h, node, palette[i % len(palette)])
        ys.append(current_y)
        current_y -= h + gap

    for i in range(len(ys) - 1):
        add_arrow(ax, (0.5, ys[i]), (0.5, ys[i + 1] + h))

    ax.text(0.5, 0.98, f"{title} Flowchart", ha="center", va="top", fontsize=18, fontweight="bold", color=color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        draw_horizontal_architecture(
            experiment["title"],
            experiment["architecture_nodes"],
            experiment["color"],
            OUTPUT_DIR / f"{experiment['key']}_architecture.png",
        )
        draw_vertical_flowchart(
            experiment["title"],
            experiment["flow_nodes"],
            experiment["color"],
            OUTPUT_DIR / f"{experiment['key']}_flowchart.png",
        )

    readme_lines = [
        "# Experiment Diagrams",
        "",
        "Separate architecture and flowchart visuals for each experimental setting.",
        "",
    ]
    for experiment in EXPERIMENTS:
        readme_lines.extend(
            [
                f"## {experiment['title']}",
                "",
                f"- `{experiment['key']}_architecture.png`",
                f"- `{experiment['key']}_flowchart.png`",
                "",
            ]
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Created diagrams in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
