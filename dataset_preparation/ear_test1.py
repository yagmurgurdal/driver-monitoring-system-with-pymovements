import cv2
import mediapipe as mp
import numpy as np

# EAR hesaplama fonksiyonu
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    if len(eye_points) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h = euclidean_distance(p1, p4)
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

def test_single_video(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video açılamadı -> {video_path}")
        return

    print(f"Test başlatıldı: {video_path}")
    print("Durdurmak için 'ESC' tuşuna basın.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            l_ids = [33, 160, 158, 133, 153, 144]
            r_ids = [362, 385, 387, 263, 373, 380]

            left_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in l_ids]
            right_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in r_ids]

            l_ear = calculate_ear(left_pts)
            r_ear = calculate_ear(right_pts)
            avg_ear = (l_ear + r_ear) / 2.0

            head_ids = [1, 33, 263, 61, 291, 199]
            face_2d = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in head_ids], dtype=np.float64)
            face_3d = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h, face_landmarks.landmark[i].z) for i in head_ids], dtype=np.float64)

            focal_length = w
            cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float64)
            success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1)))

            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                p, y, r = angles[0]*360, angles[1]*360, angles[2]*360

                cv2.putText(frame, f"AVG EAR: {avg_ear:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Pitch: {p:.1f} Yaw: {y:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                for (x, y_coord) in left_pts + right_pts:
                    cv2.circle(frame, (int(x), int(y_coord)), 2, (0, 255, 0), -1)

            print(f"EAR: {avg_ear:.4f} | Pitch: {p:.2f} | Yaw: {y:.2f}", end="\r")

        cv2.imshow("DMS Tekli Video Testi", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nTest sonlandırıldı.")

if __name__ == "__main__":
    video_path = r"C:\videolarim\test_video.mp4" 
    test_single_video(video_path)