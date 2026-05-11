# Driver Monitoring System with PyMovements

This repository contains an end-to-end driver monitoring workflow built for a graduation project. It combines eye-closure analysis, head-pose estimation, and PyMovements-supported gaze features to classify driver behavior into three states:

- `normal`
- `drowsiness`
- `distraction`

The project goes beyond a single model-training script. It includes dataset preparation, validation utilities, gaze-feature extraction, baseline and gaze-supported Random Forest experiments, result verification, and a real-time monitoring application with database logging and audio alerts.

## Project Goal

The main goal is to build a reproducible pipeline that transforms raw or semi-processed driver videos into structured window-level features for classification. A secondary goal is to move from offline experimentation toward a practical real-time monitoring prototype that can generate live predictions, store session data, and trigger warnings when sustained drowsiness is detected.

## Pipeline Overview

The repository follows a staged workflow:

1. Videos are standardized and processed into frame-level facial, eye, and head-pose measurements.
2. Frame-level signals are converted into window-based behavioral features.
3. A rule-based normal class is derived from safe driving segments.
4. Generated windows are validated against source signals.
5. Random Forest models are trained on baseline features.
6. Iris center coordinates are extracted to build PyMovements-compatible gaze inputs.
7. Gaze features are generated and merged with the baseline window dataset.
8. Gaze-supported Random Forest experiments are trained and evaluated.
9. The trained model is used in a real-time monitoring script with logging and alerts.

## Repository Contents

Key scripts in the project:

- `build_window_dataset.py`: builds window-level datasets for `drowsiness` and `distraction`
- `validate_window_outputs.py`: validates generated windows against source data
- `build_normal_window_dataset.py`: builds the derived `normal` class window dataset
- `validate_normal_window_outputs.py`: validates normal-class windows
- `train_random_forest.py`: trains three-class Random Forest models
- `check_accuracy.py`: independently verifies reported accuracy values
- `run_project_pipeline.py`: orchestrates the main pipeline stages
- `extract_pymovements_inputs.py`: extracts iris-based gaze inputs from videos
- `build_pymovements_window_features.py`: generates gaze-window features from PyMovements input CSV files
- `build_normal_pymovements_window_features.py`: generates aligned gaze-window features for the `normal` class
- `merge_window_with_gaze_features.py`: merges baseline windows with gaze-derived features
- `realtime_driver_monitor.py`: runs real-time monitoring on webcam or video input

## Core Feature Set

The baseline model uses the following window-level features:

- `perclos`
- `perclos_percent`
- `mean_ear`
- `std_ear`
- `min_ear`
- `max_ear`
- `mean_abs_yaw`
- `std_yaw`
- `max_abs_yaw`
- `mean_abs_pitch`
- `std_pitch`
- `max_abs_pitch`
- `mean_abs_roll`
- `std_roll`
- `max_abs_roll`
- `face_detect_ratio`
- `pose_valid_ratio`
- `ear_valid_ratio`

In the gaze-supported setting, these are extended with additional `gaze_*` features generated from PyMovements-compatible iris-coordinate time series.

## Current Experiment Results

Current validated results in the repository:

| Model Variant | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Baseline Random Forest | `0.849193` | `0.732109` | `0.842881` |
| High-Confidence Random Forest | `0.904070` | `0.882017` | `0.897380` |
| Gaze Baseline Random Forest | `0.884637` | `0.720373` | `0.875808` |
| Gaze High-Confidence Random Forest | `0.918114` | `0.927862` | `0.915445` |

The best current result is the gaze-supported high-confidence model.

## Report Assets

Report-ready figures are available in [`docs/report_assets`](docs/report_assets):

- `model_comparison.png`
- `system_pipeline.png`
- `best_model_confusion_matrix.png`
- `best_model_feature_importance_top25.png`
- `best_model_feature_importance_gaze_only.png`

These assets can be regenerated with:

```powershell
.venv\Scripts\python.exe docs\report_assets\generate_report_assets.py
```

## Running the Pipeline

To run the full pipeline:

```powershell
.venv\Scripts\python.exe run_project_pipeline.py --python .venv\Scripts\python.exe
```

To train the baseline and high-confidence Random Forest models:

```powershell
.venv\Scripts\python.exe train_random_forest.py --window-root .\window_dataset --output-dir .\results\random_forest_baseline
.venv\Scripts\python.exe train_random_forest.py --window-root .\window_dataset --output-dir .\results\random_forest_high_confidence --use-high-confidence
```

To train the gaze-supported models:

```powershell
.venv\Scripts\python.exe train_random_forest.py --window-root .\window_dataset_with_gaze --output-dir .\results\random_forest_gaze_baseline --feature-set gaze
.venv\Scripts\python.exe train_random_forest.py --window-root .\window_dataset_with_gaze --output-dir .\results\random_forest_gaze_high_confidence --feature-set gaze --use-high-confidence
```

To independently verify reported accuracy:

```powershell
.venv\Scripts\python.exe check_accuracy.py --results-dir .\results\random_forest_high_confidence
```

Notes:

- `train_random_forest.py` uses the baseline feature set by default.
- Gaze-merged files are expected under `window_dataset_with_gaze` as `*_windows_with_gaze.xlsx`.
- Recent portability fixes allow repo-local dataset roots when hard-coded external paths are unavailable.

## Real-Time Monitoring

First train a baseline model and create `model_bundle.pkl`:

```powershell
.venv\Scripts\python.exe train_random_forest.py --window-root .\window_dataset --output-dir .\results\random_forest_baseline
```

Run the live monitor with a webcam:

```powershell
.venv\Scripts\python.exe realtime_driver_monitor.py --model-bundle .\results\random_forest_baseline\model_bundle.pkl
```

Run it on a video file:

```powershell
.venv\Scripts\python.exe realtime_driver_monitor.py --model-bundle .\results\random_forest_baseline\model_bundle.pkl --video-path C:\path\to\video.mp4
```

The real-time system currently includes:

- rolling window-based live classification
- confidence smoothing and quality gating
- SQLite session logging
- sustained-drowsiness audio alert support

Useful options:

- `--drowsiness-alert-seconds 2.0`
- `--drowsiness-alert-cooldown-sec 4.0`
- `--disable-drowsiness-alert`
- `--disable-db`

The default local database path is `.\results\realtime_monitor.db`.

To inspect stored counts quickly:

```powershell
.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect(r'.\results\realtime_monitor.db'); print(c.execute('select count(*) from frame_measurements').fetchone()); print(c.execute('select count(*) from window_predictions').fetchone())"
```

## Notes

- The baseline and gaze-supported pipelines are both preserved in the repository.
- The real-time path currently focuses on the baseline live feature set; fully online PyMovements-style gaze inference would require a dedicated live gaze-feature layer.
- Several validation and audit scripts exist because reproducibility and dataset consistency were treated as part of the project itself, not as optional cleanup work.

## License

No explicit license file is currently included in the repository. Add one if you plan to distribute or reuse the project publicly.
