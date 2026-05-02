# Driver Monitoring Pipeline

Bu repo, surucu izleme bitirme projesinin pencereleme, dogrulama ve siniflandirma adimlarini icerir.

## Bu bolumde neler var

- `build_window_dataset.py`: `drowsiness` ve `distraction` icin video basina pencere dosyalari uretir.
- `validate_window_outputs.py`: uretilen pencere dosyalarini kaynak veriye karsi dogrular.
- `build_normal_window_dataset.py`: `normal` sinifi icin pencere dosyalari uretir.
- `validate_normal_window_outputs.py`: `normal` pencere dosyalarini dogrular.
- `train_random_forest.py`: 3 sinifli Random Forest modeli egitir.
- `check_accuracy.py`: confusion matrix ve tahmin dosyasindan accuracy hesabini tekrar kontrol eder.
- `run_project_pipeline.py`: tum bu adimlari sirayla calistirir.
- `extract_pymovements_inputs.py`: videolardan PyMovements icin iris `x/y` girdileri uretir.

## Siniflar

- `normal`
- `drowsiness`
- `distraction`

## Model ozellikleri

Modelde kullanilan temel ozellikler:

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

## Calistirma

Tum akisi calistirmak icin:

```powershell
.venv\Scripts\python.exe run_project_pipeline.py --python .venv\Scripts\python.exe
```

Sadece Random Forest egitimi icin:

```powershell
.venv\Scripts\python.exe train_random_forest.py --window-root D:\window --output-dir .\results\random_forest_baseline
.venv\Scripts\python.exe train_random_forest.py --window-root D:\window --output-dir .\results\random_forest_high_confidence --use-high-confidence
```

Accuracy kontrolu icin:

```powershell
.venv\Scripts\python.exe check_accuracy.py --results-dir .\results\random_forest_high_confidence
```
