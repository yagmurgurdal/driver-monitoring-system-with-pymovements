import cv2
import mediapipe as mp
import numpy as np
import csv
import os


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_points):

    if len(eye_points) != 6:
        return None

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return None

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_face_mesh():

    mp_face_mesh = mp.solutions.face_mesh

    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_video(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]

    output_csv = os.path.join(output_folder, f"{base_name}.csv")

    face_mesh = get_face_mesh()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video:", video_name)
    print("FPS:", fps)
    print("Toplam frame:", total_frames)

    with open(output_csv, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "time_sec",
            "left_ear",
            "right_ear",
            "avg_ear",
            "yaw",
            "pitch",
            "roll"
        ])

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            time_sec = frame_count / fps

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(frame_rgb)

            left_ear = None
            right_ear = None
            avg_ear = None
            yaw = None
            pitch = None
            roll = None

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]

                head_pose_ids = [1, 33, 263, 61, 291, 199]

                face_2d = []
                face_3d = []

                for idx in head_pose_ids:

                    lm = face_landmarks.landmark[idx]

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = w

                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ])

                dist_matrix = np.zeros((4, 1))

                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d,
                    face_2d,
                    cam_matrix,
                    dist_matrix
                )

                if success:

                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    pitch = angles[0] * 360
                    yaw = angles[1] * 360
                    roll = angles[2] * 360

                left_eye_ids = [33, 160, 158, 133, 153, 144]
                right_eye_ids = [362, 385, 387, 263, 373, 380]

                left_eye_points = []
                right_eye_points = []

                for idx in left_eye_ids:

                    lm = face_landmarks.landmark[idx]

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    left_eye_points.append((x, y))

                for idx in right_eye_ids:

                    lm = face_landmarks.landmark[idx]

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    right_eye_points.append((x, y))

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)

                if left_ear is not None and right_ear is not None:
                    avg_ear = (left_ear + right_ear) / 2

            writer.writerow([
                frame_count,
                round(time_sec, 3),
                round(left_ear, 4) if left_ear else "",
                round(right_ear, 4) if right_ear else "",
                round(avg_ear, 4) if avg_ear else "",
                round(yaw, 4) if yaw else "",
                round(pitch, 4) if pitch else "",
                round(roll, 4) if roll else ""
            ])

    cap.release()
    face_mesh.close()

    print("CSV oluşturuldu:")
    print(output_csv)


if __name__ == "__main__":

    video_path = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gB\7\s2\RGB\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_mosaic_panel2_1280x720.mp4"

    output_folder = r"D:\DMD Dataset-pymovements\distractionrgb\csv\dmd\gB\7\s2\RGB"

    process_video(video_path, output_folder)