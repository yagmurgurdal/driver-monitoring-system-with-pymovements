import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# =========================================================
# CONFIGURATION CONSTANTS
# =========================================================
EAR_SUSPICIOUS_YAW_DEG = 35.0

POSE_VALID_PITCH_RANGE = (-60.0,  60.0)
POSE_VALID_YAW_RANGE   = (-90.0,  90.0)
POSE_VALID_ROLL_RANGE  = (-50.0,  50.0)   # widened from ±45 — mirror-check tilts


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def euclidean_distance_2d(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_pts_2d(face_landmarks, ids, w_img, h_img):
    return [
        (
            face_landmarks.landmark[i].x * w_img,
            face_landmarks.landmark[i].y * h_img,
        )
        for i in ids
    ]


def calculate_ear_robust(eye_points, min_horizontal_px=2.0):
    if len(eye_points) != 6:
        return None, 0
    p1, p2, p3, p4, p5, p6 = eye_points
    v1 = euclidean_distance_2d(p2, p6)
    v2 = euclidean_distance_2d(p3, p5)
    h  = euclidean_distance_2d(p1, p4)
    if h < min_horizontal_px:
        return None, 0
    return float((v1 + v2) / (2.0 * h)), 1


def build_camera_matrix(w_img, h_img):
    # KNOWN APPROXIMATION: focal_length = w_img assumes ~53° horizontal FOV.
    # For IR/embedded cameras, replace with calibrated intrinsics if available.
    # This approximation introduces systematic pose bias at |yaw| > 30°.
    focal_length = w_img
    return np.array(
        [
            [focal_length, 0.0,          w_img / 2.0],
            [0.0,          focal_length, h_img / 2.0],
            [0.0,          0.0,          1.0],
        ],
        dtype=np.float64,
    )


def normalize_angles(pitch, yaw, roll):
    if pitch < -90.0:
        pitch += 180.0
        yaw   = -yaw
        roll  = -roll
    elif pitch > 90.0:
        pitch -= 180.0
        yaw   = -yaw
        roll  = -roll
    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0
    if yaw > 180.0:
        yaw -= 360.0
    elif yaw < -180.0:
        yaw += 360.0
    return pitch, yaw, roll


def is_pose_valid(pitch, yaw, roll):
    p_ok = POSE_VALID_PITCH_RANGE[0] <= pitch <= POSE_VALID_PITCH_RANGE[1]
    y_ok = POSE_VALID_YAW_RANGE[0]   <= yaw   <= POSE_VALID_YAW_RANGE[1]
    r_ok = POSE_VALID_ROLL_RANGE[0]  <= roll  <= POSE_VALID_ROLL_RANGE[1]
    return int(p_ok and y_ok and r_ok)


def get_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def safe_round(val, digits=4):
    if val is None:
        return ""
    return round(float(val), digits)


# =========================================================
# 3D FACE MODEL
# =========================================================
# VALIDATION REQUIRED BEFORE TRUSTING ANY CSV:
#   Turn head clearly RIGHT → yaw must be POSITIVE in the CSV.
#   If yaw is NEGATIVE when turning right, negate all Y values in this model.
FACE_3D_MODEL = np.array(
    [
        [   0.0,    0.0,    0.0],
        [   0.0,  330.0,   65.0],
        [-225.0, -170.0,  135.0],
        [ 225.0, -170.0,  135.0],
        [-150.0,  150.0,  125.0],
        [ 150.0,  150.0,  125.0],
    ],
    dtype=np.float64,
)

HEAD_POSE_IDS = [1, 199, 33, 263, 61, 291]
LEFT_EYE_IDS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]

# KNOWN LIMITATION: zero distortion assumes rectilinear lens.
# IR/embedded cameras typically have k1 in [-0.3, -0.1].
# Pose estimates are systematically biased at |yaw| > 30° without calibration.
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


# =========================================================
# MAIN VIDEO PROCESSING
# =========================================================
def process_video(video_path, output_csv, show_video=False):
    face_mesh = get_face_mesh()
    cap       = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        raw_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_str  = str(raw_total) if raw_total > 0 else "unknown"
        video_name = os.path.basename(video_path)

        print(f"\nProcessing: {video_name}")
        print(f"FPS: {fps:.2f}  |  Total frames: {total_str}")

        n_no_face  = 0
        n_pose_bad = 0
        n_ear_susp = 0

        CSV_COLUMNS = [
            "frame", "time_sec", "face_detected",
            "yaw", "pitch", "roll", "pose_valid",
            "left_ear", "right_ear", "avg_ear", "ear_valid", "ear_suspicious",
        ]

        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)

            cam_matrix = None
            min_h_px   = None
            last_w     = None
            last_h     = None

            frame_count = 0

            while True:
                ret, frame = cap.read()

                # FIX: guard against ret=True with frame=None (corrupted packets
                # in MPEG/AVI containers). Break on clean end-of-stream (not ret),
                # skip on null frame (continue lets the loop try the next packet).
                if not ret:
                    break
                if frame is None:
                    continue

                frame_count += 1
                time_sec = frame_count / fps

                # FIX: explicit channel check — safe regardless of evaluation order.
                # Previously: `if frame.ndim == 2 or frame.shape[2] == 1`
                # which was safe only due to Python's short-circuit `or`.
                n_channels = frame.shape[2] if frame.ndim == 3 else 1
                if n_channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                h_img, w_img = frame.shape[:2]

                # FIX: rebuild cam_matrix if resolution changes mid-video.
                # Previously frozen to first-frame dimensions only.
                if w_img != last_w or h_img != last_h:
                    cam_matrix = build_camera_matrix(w_img, h_img)
                    min_h_px   = max(2.0, w_img * 0.005)
                    last_w, last_h = w_img, h_img

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = face_mesh.process(frame_rgb)

                face_detected  = 0
                pose_valid     = 0
                ear_valid      = 0
                ear_suspicious = 0

                yaw   = None
                pitch = None
                roll  = None

                left_ear  = None
                right_ear = None
                avg_ear   = None

                if results.multi_face_landmarks:
                    face_detected  = 1
                    face_landmarks = results.multi_face_landmarks[0]

                    # --- HEAD POSE ---
                    face_2d = np.array(
                        [
                            [
                                face_landmarks.landmark[idx].x * w_img,
                                face_landmarks.landmark[idx].y * h_img,
                            ]
                            for idx in HEAD_POSE_IDS
                        ],
                        dtype=np.float64,
                    )

                    # FIX: SOLVEPNP_SQPNP is a direct (non-iterative) solver with
                    # no local-minimum problem. SOLVEPNP_ITERATIVE can converge to
                    # wrong solutions at extreme yaw — exactly our distraction poses.
                    # Requires OpenCV >= 4.5. Fall back to SOLVEPNP_EPNP if needed.
                    success, rot_vec, _ = cv2.solvePnP(
                        FACE_3D_MODEL, face_2d, cam_matrix, DIST_COEFFS,
                        flags=cv2.SOLVEPNP_SQPNP,
                    )

                    if success:
                        rmat, _jac = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                        p_raw, y_raw, r_raw = float(angles[0]), float(angles[1]), float(angles[2])
                        pitch, yaw, roll    = normalize_angles(p_raw, y_raw, r_raw)
                        pose_valid          = is_pose_valid(pitch, yaw, roll)

                    # --- EAR ---
                    left_pts  = get_pts_2d(face_landmarks, LEFT_EYE_IDS,  w_img, h_img)
                    right_pts = get_pts_2d(face_landmarks, RIGHT_EYE_IDS, w_img, h_img)

                    left_ear,  l_valid = calculate_ear_robust(left_pts,  min_horizontal_px=min_h_px)
                    right_ear, r_valid = calculate_ear_robust(right_pts, min_horizontal_px=min_h_px)

                    if l_valid and r_valid:
                        avg_ear   = (left_ear + right_ear) / 2.0
                        ear_valid = 1

                        pose_unknown = (yaw is None)
                        yaw_extreme  = (yaw is not None and abs(yaw) > EAR_SUSPICIOUS_YAW_DEG)
                        if pose_unknown or yaw_extreme:
                            ear_suspicious = 1

                    # FIX: counters moved here — outside the show_video block and
                    # before the ESC break, so they are always incremented even
                    # when the user exits with ESC during debugging.
                    # REMOVED: the dead `if not face_detected: n_no_face += 1` that
                    # was previously inside this block — face_detected is always 1
                    # here, so that branch was unreachable. n_no_face is correctly
                    # counted in the else branch below.
                    if not pose_valid:
                        n_pose_bad += 1
                    if ear_suspicious:
                        n_ear_susp += 1

                    # --- VISUAL DEBUG ---
                    if show_video:
                        for (x, y) in left_pts:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                        for (x, y) in right_pts:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                        for idx in HEAD_POSE_IDS:
                            lx = int(face_landmarks.landmark[idx].x * w_img)
                            ly = int(face_landmarks.landmark[idx].y * h_img)
                            cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

                        lines = [
                            (f"Frame: {frame_count}",                            (255, 255, 255)),
                            (f"Face: {face_detected}  PoseOK: {pose_valid}",     (255, 255, 255)),
                            (f"Yaw:   {yaw:.2f}"   if yaw   is not None else "Yaw:   None", (0, 255, 0)),
                            (f"Pitch: {pitch:.2f}" if pitch is not None else "Pitch: None", (0, 255, 0)),
                            (f"Roll:  {roll:.2f}"  if roll  is not None else "Roll:  None", (0, 255, 0)),
                            (f"EAR valid: {ear_valid}  susp: {ear_suspicious}",  (0, 255, 255)),
                            (f"AvgEAR: {avg_ear:.4f}" if avg_ear is not None else "AvgEAR: None", (0, 255, 255)),
                        ]
                        for i, (text, color) in enumerate(lines):
                            cv2.putText(frame, text, (20, 30 + i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                        cv2.imshow("DMS Feature Extraction", frame)

                else:
                    n_no_face += 1

                # FIX: writer.writerow is now BEFORE the ESC check so the current
                # frame is always written before a potential break.
                writer.writerow([
                    frame_count,
                    safe_round(time_sec,  3),
                    face_detected,
                    safe_round(yaw,   4),
                    safe_round(pitch, 4),
                    safe_round(roll,  4),
                    pose_valid,
                    safe_round(left_ear,  4),
                    safe_round(right_ear, 4),
                    safe_round(avg_ear,   4),
                    ear_valid,
                    ear_suspicious,
                ])

                if show_video and (cv2.waitKey(1) & 0xFF == 27):
                    print("[INFO] Stopped by ESC.")
                    break

                # Progress reporting for long videos
                if frame_count % 500 == 0:
                    print(f"  {frame_count}/{total_str} frames...", end="\r")

        pct_no_face  = 100.0 * n_no_face  / max(frame_count, 1)
        pct_pose_bad = 100.0 * n_pose_bad / max(frame_count, 1)
        pct_ear_susp = 100.0 * n_ear_susp / max(frame_count, 1)
        print(f"\n[DONE]    {video_name}")
        print(f"[QUALITY] No face: {pct_no_face:.1f}% | "
              f"Pose invalid: {pct_pose_bad:.1f}% | "
              f"EAR suspicious: {pct_ear_susp:.1f}%")
        print(f"[CSV]     {output_csv}")

    finally:
        cap.release()
        face_mesh.close()
        if show_video:
            cv2.destroyAllWindows()


# =========================================================
# BATCH PROCESSING
# =========================================================
def process_all_videos(input_folder, output_folder, show_video=False):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(video_extensions)
    ]
    if not video_files:
        print("No video files found.")
        return
    print(f"Found {len(video_files)} video(s).\n")
    for video_name in sorted(video_files):
        video_path = os.path.join(input_folder, video_name)
        base_name  = os.path.splitext(video_name)[0]
        output_csv = os.path.join(output_folder, f"{base_name}.csv")
        # Catch per-video failures so a single bad file does not abort the batch
        try:
            process_video(video_path, output_csv, show_video=show_video)
        except Exception as exc:
            print(f"[ERROR] Failed on {video_name}: {exc}")
    print("\nAll videos processed.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    input_folder  = r"D:\dataset\drowsiness\RGB"
    output_folder = r"D:\csv\drowsiness\RGB"
    process_all_videos(
        input_folder=input_folder,
        output_folder=output_folder,
        show_video=False,
    )