import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video yolu
video_path = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\1\s1\RGB\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face_std.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices to be used for head pose
            # 1   -> nose tip
            # 33  -> left eye outer corner
            # 263 -> right eye outer corner
            # 61  -> left mouth corner
            # 291 -> right mouth corner
            # 199 -> chin
            face_2d = []
            face_3d = []

            landmark_ids = [1, 33, 263, 61, 291, 199]

            for idx in landmark_ids:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

            # assume no distortion.
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # solvePnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d,
                face_2d,
                cam_matrix,
                dist_matrix
            )

            if success:
                # Rotation matrix
                rmat, _ = cv2.Rodrigues(rot_vec)

                # Euler angles
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360

                # Print
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # draw a direction line from the tip of the nose.
                nose_lm = face_landmarks.landmark[1]
                nose_x, nose_y = int(nose_lm.x * w), int(nose_lm.y * h)

                p2 = (
                    int(nose_x + yaw * 10),
                    int(nose_y - pitch * 10)
                )

                cv2.line(frame, (nose_x, nose_y), p2, (255, 0, 0), 3)

    cv2.imshow("Head Pose Estimation", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()