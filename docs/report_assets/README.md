# Report Assets

This folder contains report-ready visual material generated from the current project outputs. It now includes both the original Random Forest-focused assets and the expanded multi-model comparison package used for the thesis/report.

## Asset Groups

### 1. Legacy Random Forest Asset Set

These figures were generated from the original Random Forest experiment track:

- `model_comparison.png`
- `system_pipeline.png`
- `best_model_confusion_matrix.png`
- `best_model_feature_importance_top25.png`
- `best_model_feature_importance_gaze_only.png`

These files are still useful when discussing the four Random Forest variants specifically.

### 2. Expanded Model Comparison Package

The folder `model_comparison_20260531/` contains the current report package for the full algorithm comparison study.

Key outputs:

- `table_1_all_models_all_settings.{csv,xlsx,png}`
- `table_2_gaze_high_confidence_ranking.{csv,xlsx,png}`
- `figure_1_grouped_accuracy.png`
- `figure_2_baseline_ranking.png`
- `figure_3_high_confidence_ranking.png`
- `figure_4_gaze_baseline_ranking.png`
- `figure_5_gaze_high_confidence_ranking.png`
- `figure_6_xgboost_confusion_matrix.png`
- `figure_7_top3_confusion_matrices.png`
- `figure_8_xgboost_feature_importance.png`
- `within_model_figures/*.png`

Current top-ranked result in this package:

- `XGBoost + gaze_high_confidence`
- Accuracy: `0.953010`
- Macro F1: `0.946324`
- Weighted F1: `0.952137`

### 3. Experiment Diagrams

The folder `experiment_diagrams_20260601/` contains separate architecture and flowchart figures for each experimental setting:

- `baseline`
- `high_confidence`
- `gaze_baseline`
- `gaze_high_confidence`

## Suggested Report Placement

- Use the files in `experiment_diagrams_20260601/` in the methodology chapter when explaining each experimental setting.
- Use `model_comparison_20260531/table_1_all_models_all_settings.png` at the beginning of the model-comparison section.
- Use `model_comparison_20260531/within_model_figures/*.png` under the subsection of each algorithm.
- Use `model_comparison_20260531/figure_2_*` to `figure_5_*` when comparing algorithms under the same setting.
- Use `model_comparison_20260531/figure_6_xgboost_confusion_matrix.png` and `figure_7_top3_confusion_matrices.png` in the class-level evaluation subsection.
- Use `model_comparison_20260531/figure_8_xgboost_feature_importance.png` in the interpretability subsection.

## Regeneration

If the experiment outputs change, regenerate these assets with:

```powershell
.venv\Scripts\python.exe docs\report_assets\generate_report_assets.py
.venv\Scripts\python.exe docs\report_assets\generate_model_comparison_assets.py
.venv\Scripts\python.exe docs\report_assets\generate_experiment_diagrams.py
```
