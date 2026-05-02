import os
import shutil
import csv
import cv2
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- AYARLAR ---
INPUT_DIR = r"D:\DMD Dataset-pymovements\distractionrgb\dmd"
KEEP_DIR = r"D:\DMD Dataset-pymovements\distractionrgb\keep"
REJECT_DIR = r"D:\DMD Dataset-pymovements\distractionrgb\reject"
REPORT_CSV = r"D:\DMD Dataset-pymovements\distractionrgb\filter_report.csv"
MODEL_PATH = "blaze_face_short_range.tflite"

# --- MODEL DOSYASINI KONTROL ET VE İNDİR ---
if not os.path.exists(MODEL_PATH):
    print("Model dosyası bulunamadı, indiriliyor... Lütfen bekleyin.")
    model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    urllib.request.urlretrieve(model_url, MODEL_PATH)
    print("Model başarıyla indirildi.")

os.makedirs(KEEP_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")
MAX_CHECKED = 200      
FRAME_STRIDE = 5       
THRESH_FACE_RATIO = 0.60  

# --- MEDIAPIPE TASKS YAPILANDIRMASI (mp.solutions KULLANMAZ) ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

rows = []
total = keep = reject = 0

for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.lower().endswith(VIDEO_EXTS):
            continue

        total += 1
        path = os.path.join(root, fname)
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            out_name = f"{os.path.basename(root)}__{fname}"
            shutil.copy2(path, os.path.join(REJECT_DIR, out_name))
            rows.append([path, out_name, 0, 0, 0.0, "reject", "cannot_open"])
            reject += 1
            continue

        checked = 0
        face_found = 0
        frame_idx = 0

        while checked < MAX_CHECKED:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if frame_idx % FRAME_STRIDE != 0:
                continue

            checked += 1

            # Kareyi MediaPipe formatına hazırla
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Algılama yap
            detection_result = detector.detect(mp_image)

            if detection_result.detections:
                face_found += 1

        cap.release()

        # Karar Mekanizması
        if checked == 0:
            ratio = 0.0
            decision = "reject"
            reason = "no_frames_read"
        else:
            ratio = face_found / checked
            if ratio >= THRESH_FACE_RATIO:
                decision = "keep"
                reason = "face_ok"
            else:
                decision = "reject"
                reason = "face_low"

        # Klasör yapısını korumak veya çakışmayı önlemek için yeni isim
        out_name = f"{os.path.basename(root)}__{fname}"
        target_dir = KEEP_DIR if decision == "keep" else REJECT_DIR
        shutil.copy2(path, os.path.join(target_dir, out_name))

        if decision == "keep":
            keep += 1
        else:
            reject += 1

        rows.append([path, out_name, checked, face_found, round(ratio, 3), decision, reason])
        print(f"[{decision.upper()}] {out_name} | Oran: {ratio:.2f}")

# Raporu yaz
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source_path", "output_name", "checked_frames", "face_found", "face_ratio", "decision", "reason"])
    w.writerows(rows)

print(f"\nİşlem Tamamlandı. Toplam: {total} | KEEP: {keep} | REJECT: {reject}")