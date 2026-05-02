
import pandas as pd

file_path = r"D:\csvcleaned\drowsiness\RGB\drowsinessrgb1.xlsx"

df = pd.read_excel(file_path)

# Sayısal yap
df["avg_ear"] = pd.to_numeric(df["avg_ear"], errors="coerce")
df["face_detected"] = pd.to_numeric(df["face_detected"], errors="coerce")

# Sadece geçerli satırlar
valid_df = df[(df["face_detected"] == 1) & (df["avg_ear"].notna())].copy()

EAR_THRESHOLD = 0.21

# Göz kapalı mı?
valid_df["eye_closed"] = valid_df["avg_ear"] < EAR_THRESHOLD

# Genel PERCLOS
perclos = valid_df["eye_closed"].sum() / len(valid_df)

print("Toplam geçerli kare:", len(valid_df))
print("Kapalı göz kare sayısı:", valid_df["eye_closed"].sum())
print("PERCLOS:", round(perclos, 4))
print("PERCLOS (%):", round(perclos * 100, 2))