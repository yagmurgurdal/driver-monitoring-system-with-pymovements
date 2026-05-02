import cv2
import mediapipe as mp
import os

video_path = r"D:\DMD Dataset-pymovements\distractionrgb\dmd\gA\1\s1\RGB\gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face_std.mp4"

if not os.path.exists(video_path):
    print("Dosya bulunamadı:")
    print(video_path)
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("There may be an issue with the file path.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cv2.namedWindow("Landmark Test", cv2.WINDOW_NORMAL)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished or frame could not be read.")
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            cv2.putText(
                frame,
                "Face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("Landmark Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Exited using ESC.")
            break

cap.release()
cv2.destroyAllWindows()