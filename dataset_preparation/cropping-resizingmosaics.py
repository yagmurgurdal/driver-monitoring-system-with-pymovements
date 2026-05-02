import cv2
import os

input_video = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\4\mosaic"

output_video = os.path.join(
    os.path.dirname(input_video),
    os.path.splitext(os.path.basename(input_video))[0] + "_panel2_1280x720.mp4"
)

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # güvenlik amaçlı

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("Çözünürlük okunamadı.")
    cap.release()
    exit()

half_w = width // 2
half_h = height // 2

# mp4 çıktı
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (1280, 720))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Panel 2 = sol alt
    panel2 = frame[half_h:height, 0:half_w]

    # 1280x720 resize
    resized = cv2.resize(panel2, (1280, 720), interpolation=cv2.INTER_CUBIC)

    out.write(resized)
    frame_count += 1

cap.release()
out.release()

print("Cropping tamamlandı.")
print("Girdi video :", input_video)
print("Çıktı video :", output_video)
print("FPS         :", fps)
print("Frame sayısı:", frame_count)