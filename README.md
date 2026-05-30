# Driver State Classification Model Comparison with PyMovements

This repository contains an end-to-end driver state classification workflow built for a graduation project. It combines eye-closure analysis, head-pose estimation, and PyMovements-supported gaze features to classify driver behavior into three states:

- `normal`
- `drowsiness`
- `distraction`

The project goes beyond a single model-training script. It includes dataset preparation, validation utilities, gaze-feature extraction, baseline and gaze-supported Random Forest experiments, result verification, and report-ready comparison assets.

## Project Goal

The main goal is to build a reproducible pipeline that transforms raw or semi-processed driver videos into structured window-level features for classification, then compare feature sets and training strategies on the same three-class task.

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
9. Model outputs are summarized through metrics, confusion matrices, feature importance tables, and report visuals.

## Repository Structure

The repository is organized into a small number of code areas:

- `scripts/`: main project code used for dataset building, gaze processing, training, evaluation, and orchestration
- `dataset_preparation/`: earlier preparation and experimentation scripts kept for reference
- `docs/report_assets/`: report figure generation
- `video açma/`: older video access and debugging helpers kept as archival utilities

Within `scripts/`, the folders have clear roles:

- `scripts/dataset`: dataset generation and validation
- `scripts/gaze`: PyMovements input extraction, gaze-window feature generation, merge, and repair utilities
- `scripts/models`: model training and accuracy verification
- `scripts/pipeline`: end-to-end orchestration
- `scripts/utils`: shared helper modules
- `scripts/legacy`: older exploratory scripts kept for reference

## Code Map

This section explains every Python code file currently in the repository.

### `scripts/`

#### Package marker files

- `scripts/__init__.py`: marks `scripts` as a Python package so modules can be run with `python -m`
- `scripts/dataset/__init__.py`: package marker for dataset-related modules
- `scripts/gaze/__init__.py`: package marker for gaze-related modules
- `scripts/models/__init__.py`: package marker for model-related modules
- `scripts/pipeline/__init__.py`: package marker for pipeline modules
- `scripts/utils/__init__.py`: package marker for shared utility modules
- `scripts/legacy/__init__.py`: package marker for legacy modules

#### Dataset generation and validation

- `scripts/dataset/build_window_dataset.py`: builds 3-second window-level Excel datasets for the `drowsiness` and `distraction` classes from frame-level driver feature files
- `scripts/dataset/build_normal_window_dataset.py`: derives the `normal` class windows using rule-based filtering and windowing logic
- `scripts/dataset/validate_window_outputs.py`: checks whether generated non-normal window files are internally consistent with their source frame-level inputs
- `scripts/dataset/validate_normal_window_outputs.py`: validates the derived `normal` class window files
- `scripts/dataset/validate_normal_xlsx.py`: validates `*_normal_frames.xlsx` outputs produced during normal-segment extraction

#### Gaze and PyMovements pipeline

- `scripts/gaze/extract_pymovements_inputs.py`: extracts iris-center time series from videos and writes PyMovements-compatible CSV inputs
- `scripts/gaze/build_pymovements_window_features.py`: converts extracted iris CSV inputs into gaze-window feature Excel files using PyMovements-style statistics
- `scripts/gaze/build_normal_pymovements_inputs.py`: builds PyMovements input CSV files specifically for the derived `normal` class
- `scripts/gaze/build_normal_pymovements_window_features.py`: generates gaze-window features for `normal` samples with window alignment that matches the baseline normal-class pipeline
- `scripts/gaze/merge_window_with_gaze_features.py`: merges baseline window datasets with gaze-derived features to produce `*_windows_with_gaze.xlsx`
- `scripts/gaze/repair_pymovements_inputs.py`: audits PyMovements CSV inputs and repairs only missing, incomplete, or malformed outputs
- `scripts/gaze/repair_pymovements_window_features.py`: audits gaze-window Excel files and repairs only missing or inconsistent outputs

#### Models and evaluation

- `scripts/models/train_random_forest.py`: trains the three-class Random Forest models for baseline and gaze-supported experiments, writes metrics, confusion matrices, feature importance tables, and `model_bundle.pkl`
- `scripts/models/train_rf_baseline.py`: dedicated entry script for the baseline Random Forest experiment
- `scripts/models/train_rf_high_confidence.py`: dedicated entry script for the high-confidence Random Forest experiment
- `scripts/models/train_rf_gaze_baseline.py`: dedicated entry script for the gaze-supported baseline Random Forest experiment
- `scripts/models/train_rf_gaze_high_confidence.py`: dedicated entry script for the gaze-supported high-confidence Random Forest experiment
- `scripts/models/check_accuracy.py`: recalculates accuracy from saved outputs to independently verify reported model performance

#### Orchestration

- `scripts/pipeline/run_project_pipeline.py`: runs the main build, validation, and training steps in sequence and writes a comparison summary workbook

#### Shared utilities

- `scripts/utils/driver_monitoring_features_fixed.py`: shared feature-extraction helpers for face, eye, EAR, and head-pose processing; mainly useful as a reusable low-level feature module

#### Legacy reference code

- `scripts/legacy/normal.py`: older standalone script for extracting normal windows and frames; kept for historical reference after the repository moved to the cleaner `scripts/dataset` workflow

### `dataset_preparation/`

These scripts are mostly exploratory or early-stage preprocessing tools. They are not the main pipeline entry points, but they document intermediate experiments carried out during development.

- `dataset_preparation/convert_to_xlsx.py`: converts prepared data into Excel format and applies workbook-level adjustments
- `dataset_preparation/cropping-resizingmosaics.py`: video preprocessing helper for cropping or resizing mosaic videos
- `dataset_preparation/ear_test1.py`: test script for EAR calculation experiments
- `dataset_preparation/head_pose.py`: test script for head-pose estimation experiments
- `dataset_preparation/imputation.py`: fills or repairs missing values in prepared spreadsheets
- `dataset_preparation/landmark_test.py`: test script for MediaPipe landmark extraction
- `dataset_preparation/pandascsvtest.py`: small pandas-based CSV inspection or testing helper
- `dataset_preparation/perclos.py`: computes or experiments with PERCLOS-related calculations
- `dataset_preparation/perclostest.py`: test script used while validating PERCLOS logic

### `docs/report_assets/`

- `docs/report_assets/generate_report_assets.py`: regenerates the report figures such as model comparison charts, confusion matrices, and feature importance visuals

### `video açma/`

These are older debugging helpers related to opening or screening source videos. They are not part of the main reproducible pipeline.

- `video açma/videoacma.py`: minimal OpenCV video-opening sanity check
- `video açma/videoacma1.py`: tests whether dataset folders and sample MP4 files can be discovered and opened
- `video açma/videoacma2.py`: checks whether a problematic video can be repaired with FFmpeg and then read successfully
- `video açma/videonamechanging.py`: helper for face-detection-based keep/reject filtering of source videos

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
.venv\Scripts\python.exe -m scripts.pipeline.run_project_pipeline --python .venv\Scripts\python.exe
```

To train the baseline and high-confidence Random Forest models:

```powershell
.venv\Scripts\python.exe -m scripts.models.train_rf_baseline
.venv\Scripts\python.exe -m scripts.models.train_rf_high_confidence
```

To train the gaze-supported models:

```powershell
.venv\Scripts\python.exe -m scripts.models.train_rf_gaze_baseline
.venv\Scripts\python.exe -m scripts.models.train_rf_gaze_high_confidence
```

To independently verify reported accuracy:

```powershell
.venv\Scripts\python.exe -m scripts.models.check_accuracy --results-dir .\results\random_forest_high_confidence
```

Notes:

- `scripts.models.train_random_forest` uses the baseline feature set by default.
- Gaze-merged files are expected under `window_dataset_with_gaze` as `*_windows_with_gaze.xlsx`.
- Recent portability fixes allow repo-local dataset roots when hard-coded external paths are unavailable.

## Suggested Usage Order

If you want to understand or rerun the repository in the intended order, the most important code files are:

1. `scripts/dataset/build_window_dataset.py`
2. `scripts/dataset/validate_window_outputs.py`
3. `scripts/dataset/build_normal_window_dataset.py`
4. `scripts/dataset/validate_normal_window_outputs.py`
5. `scripts/gaze/extract_pymovements_inputs.py`
6. `scripts/gaze/build_pymovements_window_features.py`
7. `scripts/gaze/build_normal_pymovements_inputs.py`
8. `scripts/gaze/build_normal_pymovements_window_features.py`
9. `scripts/gaze/merge_window_with_gaze_features.py`
10. `scripts/models/train_random_forest.py`
11. `scripts/models/check_accuracy.py`
12. `scripts/pipeline/run_project_pipeline.py`

If you only want the main experiment path, focus on the `scripts/` directory first. The `dataset_preparation/` and `video açma/` folders are mainly supporting or archival code.

## Notes

- The baseline and gaze-supported pipelines are both preserved in the repository.
- Several validation and audit scripts exist because reproducibility and dataset consistency were treated as part of the project itself, not as optional cleanup work.

## License

No explicit license file is currently included in the repository. Add one if you plan to distribute or reuse the project publicly.
