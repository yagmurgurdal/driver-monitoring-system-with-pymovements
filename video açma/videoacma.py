import cv2

p = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\1\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face"
cap = cv2.VideoCapture(p)

print("opened:", cap.isOpened())
print("backend:", cap.getBackendName() if cap.isOpened() else None)

ret, frame = cap.read()
print("first frame:", ret, frame.shape if ret else None)

cap.release()
