import pandas as pd
import os
import glob

input_folder = r"D:\csvcleaned\distraction\RGB"
output_folder = r"D:\perclos\distraction\RGB"

os.makedirs(output_folder, exist_ok=True)

EAR_THRESHOLD = 0.21
FPS = 30
WINDOW_SEC = 3
WINDOW_SIZE = FPS * WINDOW_SEC

xlsx_files = glob.glob(os.path.join(input_folder, "*.xlsx"))

print(f"Toplam bulunan dosya: {len(xlsx_files)}")

for file_path in xlsx_files:
    try:
        df = pd.read_excel(file_path)

        df["avg_ear"] = pd.to_numeric(df["avg_ear"], errors="coerce")
        df["face_detected"] = pd.to_numeric(df["face_detected"], errors="coerce")
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")

        valid_df = df[(df["face_detected"] == 1) & (df["avg_ear"].notna())].copy()

        if len(valid_df) < WINDOW_SIZE:
            print(f"error: {os.path.basename(file_path)}")
            continue

        valid_df["eye_closed"] = (valid_df["avg_ear"] < EAR_THRESHOLD).astype(int)

        perclos_results = []

        for start in range(0, len(valid_df) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = valid_df.iloc[start:start + WINDOW_SIZE]

            perclos = window["eye_closed"].sum() / len(window)

            perclos_results.append({
                "start_frame": int(window["frame"].iloc[0]),
                "end_frame": int(window["frame"].iloc[-1]),
                "start_time": float(window["time_sec"].iloc[0]),
                "end_time": float(window["time_sec"].iloc[-1]),
                "closed_eye_frames": int(window["eye_closed"].sum()),
                "total_frames": int(len(window)),
                "perclos": round(perclos, 4),
                "perclos_percent": round(perclos * 100, 2)
            })

        perclos_df = pd.DataFrame(perclos_results)

        file_name = os.path.splitext(os.path.basename(file_path))[0] + "_perclos.xlsx"
        output_path = os.path.join(output_folder, file_name)

        perclos_df.to_excel(output_path, index=False)

        print(f"processed: {file_name}")

    except Exception as e:
        print(f"error: {os.path.basename(file_path)} -> {e}")

print("All files processed.")
