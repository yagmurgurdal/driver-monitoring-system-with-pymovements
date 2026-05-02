import cv2
import os
import subprocess
video_path = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\1\s1\RGB\gA_1_s2_2019-03-08T09;21;03+01;00_rgb_mosaic_std_panel2_1280x720.mp4"
ffmpeg_path = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

print("Dosya var mı?:", os.path.exists(video_path))
print("FFmpeg var mı?:", os.path.exists(ffmpeg_path))

cap = cv2.VideoCapture(video_path)
print("OpenCV açtı mı?:", cap.isOpened())

ret, frame = cap.read()
print("İlk frame okunabildi mi?:", ret)
cap.release()

fixed_video = os.path.splitext(video_path)[0] + "_fixed.mp4"

command = [
    ffmpeg_path,
    "-i", video_path,
    "-vcodec", "libx264",
    "-acodec", "aac",
    "-y",
    fixed_video
]

result = subprocess.run(command, capture_output=True, text=True)

print("\n--- FFmpeg return code ---")
print(result.returncode)

print("\n--- FFmpeg stdout ---")
print(result.stdout)

print("\n--- FFmpeg stderr ---")
print(result.stderr)

print("\nOluşan fixed dosya var mı?:", os.path.exists(fixed_video))