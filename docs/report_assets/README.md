# Report Assets

This folder contains report-ready figures generated from the current project results. The assets here are intended to be copied directly into the thesis/report without having to search through the `results` directories.

## Best Current Model

The best currently validated model is the gaze-supported high-confidence Random Forest model.

- Accuracy: `0.918114`
- Macro F1: `0.927862`
- Weighted F1: `0.915445`
- Result folder: `results/random_forest_gaze_high_confidence`

## Model Summary

The figures in this folder were prepared using the following experiment outputs:

- Baseline RF: accuracy=`0.849193`, macro_f1=`0.732109`, weighted_f1=`0.842881`
- High-Confidence RF: accuracy=`0.904070`, macro_f1=`0.882017`, weighted_f1=`0.897380`
- Gaze Baseline RF: accuracy=`0.884637`, macro_f1=`0.720373`, weighted_f1=`0.875808`
- Gaze High-Confidence RF: accuracy=`0.918114`, macro_f1=`0.927862`, weighted_f1=`0.915445`

## Included Figures

- `model_comparison.png`: comparison chart for the four Random Forest experiment variants.
- `system_pipeline.png`: end-to-end project pipeline diagram from video processing to real-time monitoring.
- `best_model_confusion_matrix.png`: confusion matrix of the best current model.
- `best_model_feature_importance_top25.png`: top 25 most important features of the best current model.
- `best_model_feature_importance_gaze_only.png`: gaze-related feature importance view for the best current model.

## Suggested Report Placement

- Use `system_pipeline.png` in the methodology or system overview section.
- Use `model_comparison.png` in the results section to compare baseline and gaze-supported models.
- Use `best_model_confusion_matrix.png` in the classification results subsection.
- Use the feature importance figures in the discussion or interpretability subsection.

## Regeneration

If the experiment outputs change, regenerate these assets with:

```powershell
.venv\Scripts\python.exe docs\report_assets\generate_report_assets.py
```
