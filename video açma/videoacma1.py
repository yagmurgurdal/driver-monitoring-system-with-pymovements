import cv2
from pathlib import Path


def test_dataset_access(root_path: str):
    """
    Checks whether videos under the given root path can be accessed by OpenCV.
    Prints basic sanity-check information.
    """
    root = Path(root_path)

    print("Root exists:", root.exists())

    mp4_files = list(root.rglob("*.mp4"))
    print("MP4 count:", len(mp4_files))

    if not mp4_files:
        print("No MP4 files found.")
        return

    first_video = mp4_files[0]
    print("First MP4:", first_video)

    cap = cv2.VideoCapture(str(first_video))

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()
        print("First frame:", ret, frame.shape if ret else None)

    cap.release()


if __name__ == "__main__":
    DATASET_ROOT = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\2"
    test_dataset_access(DATASET_ROOT)
