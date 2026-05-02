import cv2
import mediapipe as mp
import numpy as np
import csv
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    """
    6 noktalı EAR hesabı:
    p1, p2, p3, p4, p5, p6
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_path = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\1\s1\RGB\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face_std.mp4"
output_csv = r"D:\DMD Dataset-pymovements\distractionrgb\csv\driver_features_detailed.csv"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("The video could not be displayed.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# CSV
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
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
            for face_landmarks in results.multi_face_landmarks:

                # HEAD POSE LANDMARKS
                head_pose_ids = [1, 33, 263, 61, 291, 199]

                face_2d = []
                face_3d = []

                for idx in head_pose_ids:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

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

                    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll:.2f}", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    nose_lm = face_landmarks.landmark[1]
                    nose_x, nose_y = int(nose_lm.x * w), int(nose_lm.y * h)

                    p2 = (
                        int(nose_x + yaw * 10),
                        int(nose_y - pitch * 10)
                    )
                    cv2.line(frame, (nose_x, nose_y), p2, (255, 0, 0), 3)

                # EAR EYE LANDMARKS
                # Left eye
                left_eye_ids = [33, 160, 158, 133, 153, 144]

                # right eye
                right_eye_ids = [362, 385, 387, 263, 373, 380]

                left_eye_points = []
                right_eye_points = []

                for idx in left_eye_ids:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    left_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                for idx in right_eye_ids:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    right_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0

                cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Avg EAR: {avg_ear:.3f}", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                break

        writer.writerow([
            frame_count,
            round(time_sec, 3),
            round(left_ear, 4) if left_ear is not None else "",
            round(right_ear, 4) if right_ear is not None else "",
            round(avg_ear, 4) if avg_ear is not None else "",
            round(yaw, 4) if yaw is not None else "",
            round(pitch, 4) if pitch is not None else "",
            round(roll, 4) if roll is not None else ""
        ])

        cv2.imshow("DMS Feature Extraction", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
cap.release()
cv2.destroyAllWindows()
print(f"CSV saved: {output_csv}")