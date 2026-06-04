# Driver State Classification Model Comparison with PyMovements

This repository contains an end-to-end driver state classification workflow built for a graduation project. It combines eye-closure analysis, head-pose estimation, and PyMovements-supported gaze features to classify driver behavior into three states:

- `normal`
- `drowsiness`
- `distraction`

The project goes beyond a single model-training script. It includes dataset preparation, validation utilities, gaze-feature extraction, four experimental settings, multi-model comparison, result verification, and report-ready tables, charts, and diagrams for the thesis/report.

## Web App Summary

The repository also includes a product-facing Streamlit application for analyzing an uploaded driver video and returning:

- overall class prediction
- confidence and risk score
- window-level prediction timeline
- quality metrics for face, pose, and EAR extraction
- SQLite-backed saved analysis records

The app uses the current best validated model by default:

- model family: `XGBoost`
- setting: `gaze_high_confidence`
- validated accuracy: `0.953010`

To make the app faster than the original research pipeline, the product layer uses a **single MediaPipe pass** for both baseline and gaze-related extraction. The research/training scripts under `scripts/` remain separate and unchanged.

## Project Goal

The main goal is to build a reproducible pipeline that transforms raw or semi-processed driver videos into structured window-level features for classification, then compare feature sets and training strategies on the same three-class task.

## Quick Start

1. Create or activate a virtual environment.
2. Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the app:

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Or use the helper script:

```powershell
.\run_driver_state_app.ps1
```

4. Open [http://localhost:8501](http://localhost:8501) in the browser.

## Pipeline Overview

The repository follows a staged workflow:

1. Videos are standardized and processed into frame-level facial, eye, and head-pose measurements.
2. Frame-level signals are converted into window-based behavioral features.
3. A rule-based normal class is derived from safe driving segments.
4. Generated windows are validated against source signals.
5. Multiple model families are trained under the baseline setting.
6. High-confidence sample selection is applied as a second experimental setting.
7. Iris center coordinates are extracted to build PyMovements-compatible gaze inputs.
8. Gaze features are generated and merged with the baseline window dataset.
9. Gaze baseline and gaze high-confidence experiments are trained and evaluated.
10. Model outputs are summarized through metrics, confusion matrices, feature importance tables, comparison charts, and report visuals.

## Repository Structure

The repository is organized into a small number of code areas:

- `assets/models/`: local model assets such as MediaPipe-compatible `.tflite` files
- `reports/`: generated Excel summaries, validation reports, repair audits, and test report variants
- `docs/forms/`: project application and proposal documents
- `docs/thesis/`: thesis or report draft documents
- `docs/diagrams/`: Draw.io flowcharts and system diagrams
- `scripts/`: main project code used for dataset building, gaze processing, training, evaluation, and orchestration
- `dataset_preparation/`: earlier preparation and experimentation scripts kept for reference
- `docs/report_assets/`: report figure generation, model-comparison charts, and experiment diagrams
- `video açma/`: older video access and debugging helpers kept as archival utilities

Within `scripts/`, the folders have clear roles:

- `scripts/dataset`: dataset generation and validation
- `scripts/gaze`: PyMovements input extraction, gaze-window feature generation, merge, and repair utilities
- `scripts/models`: model training and accuracy verification
- `scripts/pipeline`: end-to-end orchestration
- `scripts/utils`: shared helper modules
- `scripts/legacy`: older exploratory scripts kept for reference

## Organized Artifacts

To keep the repository root cleaner, non-code files are grouped by purpose:

- `docs/forms/`: administrative application forms such as `2209-A_arastirma_onerisi_formu.docx` and `Bitirme_Projesi_Başvuru_Formu.docx`
- `docs/thesis/`: thesis or report draft documents such as `PyMovements-Supported Driver Risk Analysis (11).docx`
- `docs/diagrams/`: editable Draw.io diagrams such as `akış_şeması.drawio` and `sistem_akis_semasi.drawio`
- `assets/models/`: reusable binary model assets such as `blaze_face_short_range.tflite`
- `reports/dataset/`: baseline window dataset exports, summaries, and validation reports
- `reports/gaze/`: PyMovements and merged-gaze summary workbooks
- `reports/gaze/repair/`: missingness and repair audit reports
- `reports/gaze/tests/`: smaller or temporary test-run report files

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

- `scripts/models/random_forest/train_random_forest.py`: shared Random Forest training backend for the four Random Forest scenarios; writes metrics, confusion matrices, feature importance tables, and `model_bundle.pkl`
- `scripts/models/random_forest/train_rf_baseline.py`: entry script for the baseline Random Forest experiment
- `scripts/models/random_forest/train_rf_high_confidence.py`: entry script for the high-confidence Random Forest experiment
- `scripts/models/random_forest/train_rf_gaze_baseline.py`: entry script for the gaze-supported baseline Random Forest experiment
- `scripts/models/random_forest/train_rf_gaze_high_confidence.py`: entry script for the gaze-supported high-confidence Random Forest experiment
- `scripts/models/classical_models/compare_models.py`: compares multiple classical models on the same train/test split
- `scripts/models/classical_models/compare_models_baseline.py`: dedicated entry script for baseline multi-model comparison
- `scripts/models/classical_models/compare_models_high_confidence.py`: dedicated entry script for high-confidence multi-model comparison
- `scripts/models/classical_models/compare_models_gaze_baseline.py`: dedicated entry script for gaze-supported multi-model comparison
- `scripts/models/classical_models/compare_models_gaze_high_confidence.py`: dedicated entry script for gaze-supported high-confidence multi-model comparison
- `scripts/models/classical_models/summarize_model_comparisons.py`: combines all comparison outputs into one master summary workbook
- `scripts/models/classical_models/adaboost`, `decision_tree`, `extra_trees`, `gradient_boosting`, `knn`, `linear_svm`, `logistic_regression`, `rbf_svm`, and `xgboost`: per-model subfolders that keep each non-Random-Forest model definition separate
- `scripts/models/classical_models/scenario_runner.py`: shared scenario execution helper used by the per-model entry scripts
- `scripts/models/run_all_models.ps1`: PowerShell helper that runs every model across every experimental setting sequentially
- `scripts/models/random_forest/check_accuracy.py`: recalculates accuracy from saved outputs to independently verify reported model performance

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

- `docs/report_assets/generate_report_assets.py`: regenerates the original Random Forest-focused report figures
- `docs/report_assets/generate_model_comparison_assets.py`: generates expanded comparison tables and figures for all evaluated models
- `docs/report_assets/generate_experiment_diagrams.py`: generates architecture and flowchart figures for the four experimental settings

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

## Evaluated Model Suite

The repository is not limited to Random Forest. The current comparison framework evaluates ten model families under the same four experimental settings:

- `Random Forest`
- `Extra Trees`
- `Gradient Boosting`
- `Decision Tree`
- `AdaBoost`
- `Logistic Regression`
- `Linear SVM`
- `RBF SVM`
- `K-Nearest Neighbors`
- `XGBoost`

Each model is tested under:

- `baseline`
- `high_confidence`
- `gaze_baseline`
- `gaze_high_confidence`

This means the repository supports both:

- single-model runs, where a specific algorithm such as Random Forest or XGBoost is trained in one scenario
- comparative runs, where all supported algorithms are evaluated under the same scenario and then summarized together

## Current Experiment Results

Current validated results in the repository span all ten evaluated models across the four experimental settings described above.

### Current Best Overall Models

| Rank | Model | Setting | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `XGBoost` | `gaze_high_confidence` | `0.953010` | `0.946324` | `0.952137` |
| 2 | `Extra Trees` | `gaze_high_confidence` | `0.948605` | `0.939588` | `0.946845` |
| 3 | `K-Nearest Neighbors` | `gaze_high_confidence` | `0.932452` | `0.918142` | `0.931709` |
| 4 | `Gradient Boosting` | `gaze_high_confidence` | `0.926579` | `0.921342` | `0.927586` |

### Full Comparison Tables

For the complete model-by-model results, use the generated comparison tables in:

- `docs/report_assets/model_comparison_20260531/table_1_all_models_all_settings.xlsx`
- `docs/report_assets/model_comparison_20260531/table_2_gaze_high_confidence_ranking.xlsx`

These assets summarize all evaluated models rather than highlighting a single algorithm.

## Report Assets

Report-ready figures and tables are available in [`docs/report_assets`](docs/report_assets):

- earlier Random Forest-specific assets in `docs/report_assets/`
- expanded comparison assets in `docs/report_assets/model_comparison_20260531/`
- experiment-setting architecture and flowchart diagrams in `docs/report_assets/experiment_diagrams_20260601/`
- editable XML architecture diagrams for the 16 model variants of the current top 4 algorithms in `docs/diagrams/top4_models_all_variants_20260603/`

The expanded comparison package includes:

- `table_1_all_models_all_settings.png`
- `table_2_gaze_high_confidence_ranking.png`
- `figure_1_grouped_accuracy.png`
- `figure_2_baseline_ranking.png`
- `figure_3_high_confidence_ranking.png`
- `figure_4_gaze_baseline_ranking.png`
- `figure_5_gaze_high_confidence_ranking.png`
- `figure_6_xgboost_confusion_matrix.png`
- `figure_7_top3_confusion_matrices.png`
- `figure_8_xgboost_feature_importance.png`
- `within_model_figures/*.png`

These assets can be regenerated with:

```powershell
.venv\Scripts\python.exe docs\report_assets\generate_report_assets.py
.venv\Scripts\python.exe docs\report_assets\generate_model_comparison_assets.py
.venv\Scripts\python.exe docs\report_assets\generate_experiment_diagrams.py
.venv\Scripts\python.exe docs\report_assets\generate_top4_model_architecture_xml.py
```

## Running The Web App

The Streamlit app lives under `app/`:

- `app/streamlit_app.py`: UI layer
- `app/inference.py`: product-side inference pipeline
- `app/unified_extractor.py`: single-pass MediaPipe extractor
- `app/database.py`: SQLite persistence layer

The app supports two product presets:

- `Hizli Analiz`: uses `xgboost_baseline`, clips the input to the first 30 seconds, and is intended for quick demos
- `Tam Analiz`: uses `xgboost_gaze_high_confidence` on the full video for the strongest prediction quality

Runtime artifacts are written to:

- `app_runtime/uploads/`: uploaded videos
- `app_runtime/analyses/`: exported summaries and window-level prediction files
- `database/driver_state_app.sqlite3`: persisted analysis records

Saved database rows include:

- source video name
- predicted class
- confidence
- risk score
- source label/modality
- quality metrics
- window-level predictions

Notes:

- Uploaded filenames are sanitized for Windows compatibility before saving.
- Runtime outputs and local database files are intentionally ignored by Git.
- Secrets such as `.env`, `.env.*`, and `.env.txt` are ignored by Git.

## Running the Pipeline

To run the full pipeline:

```powershell
.venv\Scripts\python.exe -m scripts.pipeline.run_project_pipeline --python .venv\Scripts\python.exe
```

To train only the Random Forest variants:

```powershell
.venv\Scripts\python.exe -m scripts.models.random_forest.train_rf_baseline
.venv\Scripts\python.exe -m scripts.models.random_forest.train_rf_high_confidence
.venv\Scripts\python.exe -m scripts.models.random_forest.train_rf_gaze_baseline
.venv\Scripts\python.exe -m scripts.models.random_forest.train_rf_gaze_high_confidence
```

To train the other model families individually, use the matching per-model entry points under `scripts.models.classical_models.<model_name>`. For example:

```powershell
.venv\Scripts\python.exe -m scripts.models.classical_models.xgboost.run_baseline
.venv\Scripts\python.exe -m scripts.models.classical_models.xgboost.run_high_confidence
.venv\Scripts\python.exe -m scripts.models.classical_models.knn.run_gaze_baseline
.venv\Scripts\python.exe -m scripts.models.classical_models.adaboost.run_gaze_high_confidence
```

To independently verify a saved Random Forest result:

```powershell
.venv\Scripts\python.exe -m scripts.models.random_forest.check_accuracy --results-dir .\results\random_forest_high_confidence
```

To run scenario-wide comparison scripts that evaluate all supported models under one setting:

```powershell
.venv\Scripts\python.exe -m scripts.models.classical_models.compare_models_baseline
.venv\Scripts\python.exe -m scripts.models.classical_models.compare_models_high_confidence
.venv\Scripts\python.exe -m scripts.models.classical_models.compare_models_gaze_baseline
.venv\Scripts\python.exe -m scripts.models.classical_models.compare_models_gaze_high_confidence
```

The comparison scripts currently evaluate the following model identifiers:

- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `decision_tree`
- `adaboost`
- `logistic_regression`
- `linear_svm`
- `rbf_svm`
- `knn`
- `xgboost`

To run all supported model/scenario combinations sequentially:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\models\run_all_models.ps1
```

Notes:

- `scripts.models.random_forest.train_random_forest` is only one backend in the repository; the broader comparison framework lives under `scripts.models.classical_models`.
- `scripts.models.random_forest.train_random_forest` uses the baseline feature set by default.
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
10. `scripts/models/random_forest/train_random_forest.py`
11. `scripts/models/classical_models/scenario_runner.py`
12. `scripts/models/classical_models/compare_models.py`
13. `scripts/models/run_all_models.ps1`
14. `scripts/models/random_forest/check_accuracy.py`
15. `scripts/pipeline/run_project_pipeline.py`

If you only want the main experiment path, focus on the `scripts/` directory first. The `dataset_preparation/` and `video açma/` folders are mainly supporting or archival code.

## Notes

- The repository now preserves both the original Random Forest workflow and the expanded ten-model comparison framework.
- Several validation and audit scripts exist because reproducibility and dataset consistency were treated as part of the project itself, not as optional cleanup work.

## License

No explicit license file is currently included in the repository. Add one if you plan to distribute or reuse the project publicly.
