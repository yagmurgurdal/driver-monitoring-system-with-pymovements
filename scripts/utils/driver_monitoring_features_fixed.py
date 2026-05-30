import csv
import os

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# GLOBAL PERFORMANCE OPTIONS
# =========================================================
cv2.setUseOptimized(True)


# =========================================================
# CONFIGURATION CONSTANTS
# =========================================================
EAR_SUSPICIOUS_YAW_DEG = 35.0

POSE_VALID_PITCH_RANGE = (-60.0, 60.0)
POSE_VALID_YAW_RANGE = (-90.0, 90.0)
POSE_VALID_ROLL_RANGE = (-50.0, 50.0)

# For pose debugging, full-resolution inference is safer by default.
FAST_MODE = False
FAST_MODE_TARGET_WIDTH = 640

FRAME_STRIDE = 1
CSV_WRITE_BUFFER_SIZE = 256
PROGRESS_PRINT_EVERY = 1000

# Reject visually bad PnP fits before writing angles to CSV.
POSE_MAX_REPROJ_ERROR_PX = 12.0
POSE_FALLBACK_REPROJ_ERROR_PX = 8.0

DEBUG_HEAD_POSE = False

# Set to -1.0 if your input is mirrored and right-turn yaw appears negative.
YAW_SIGN_CORRECTION = 1.0


# =========================================================
# CSV SCHEMA
# =========================================================
CSV_COLUMNS = [
    "frame",
    "time_sec",
    "face_detected",
    "yaw",
    "pitch",
    "roll",
    "pose_valid",
    "left_ear",
    "right_ear",
    "avg_ear",
    "ear_valid",
    "ear_suspicious",
]


# =========================================================
# HEAD POSE LANDMARK / MODEL SETUP
# =========================================================
# Root-cause fix:
# The previous code mixed a generic 3D face model with MediaPipe indices
# that did not match the intended anatomy closely enough. In particular,
# landmark 199 is not a stable chin tip for this 3D model, and landmark 1
# is less reliable as the nose tip than landmark 4 for solvePnP.
#
# To make the correspondence explicit and safe, each 3D model point is bound
# to exactly one MediaPipe index here, in a single ordered definition.
HEAD_POSE_POINTS = [
    ("nose_tip", 4, (0.0, 0.0, 0.0)),
    ("chin", 152, (0.0, 330.0, -65.0)),
    ("left_eye_outer", 33, (-225.0, -170.0, -135.0)),
    ("right_eye_outer", 263, (225.0, -170.0, -135.0)),
    ("left_mouth_corner", 61, (-150.0, 150.0, -125.0)),
    ("right_mouth_corner", 291, (150.0, 150.0, -125.0)),
]

HEAD_POSE_NAMES = tuple(item[0] for item in HEAD_POSE_POINTS)
HEAD_POSE_IDS = tuple(item[1] for item in HEAD_POSE_POINTS)
FACE_3D_MODEL = np.array([item[2] for item in HEAD_POSE_POINTS], dtype=np.float64)

LEFT_EYE_IDS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDS = (362, 385, 387, 263, 373, 380)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

PRIMARY_PNP_FLAG = (
    cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
)
FALLBACK_PNP_FLAG = (
    cv2.SOLVEPNP_EPNP if PRIMARY_PNP_FLAG != cv2.SOLVEPNP_EPNP else None
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def euclidean_distance_2d(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear_robust(eye_points, min_horizontal_px=2.0):
    if eye_points.shape != (6, 2):
        return None, 0

    p1, p2, p3, p4, p5, p6 = eye_points
    v1 = euclidean_distance_2d(p2, p6)
    v2 = euclidean_distance_2d(p3, p5)
    h = euclidean_distance_2d(p1, p4)

    if h < min_horizontal_px:
        return None, 0

    return float((v1 + v2) / (2.0 * h)), 1


def build_camera_matrix(w_img, h_img):
    focal_length = float(w_img)
    return np.array(
        [
            [focal_length, 0.0, w_img / 2.0],
            [0.0, focal_length, h_img / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def wrap_angle_deg(angle_deg):
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def safe_round(val, digits=4):
    if val is None:
        return ""
    return round(float(val), digits)


def flush_rows(writer, rows_buffer):
    if rows_buffer:
        writer.writerows(rows_buffer)
        rows_buffer.clear()


def resize_for_facemesh(frame_bgr, fast_mode):
    if not fast_mode:
        return frame_bgr

    h_img, w_img = frame_bgr.shape[:2]
    if FAST_MODE_TARGET_WIDTH <= 0 or w_img <= FAST_MODE_TARGET_WIDTH:
        return frame_bgr

    scale = FAST_MODE_TARGET_WIDTH / float(w_img)
    resized_h = max(1, int(round(h_img * scale)))
    return cv2.resize(
        frame_bgr,
        (FAST_MODE_TARGET_WIDTH, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )


def fill_points_2d(landmarks, ids, w_img, h_img, out_points):
    for out_idx, lm_idx in enumerate(ids):
        lm = landmarks[lm_idx]
        out_points[out_idx, 0] = lm.x * w_img
        out_points[out_idx, 1] = lm.y * h_img


def compute_reprojection_error(face_2d, rot_vec, trans_vec, cam_matrix):
    projected_2d, _ = cv2.projectPoints(
        FACE_3D_MODEL,
        rot_vec,
        trans_vec,
        cam_matrix,
        DIST_COEFFS,
    )
    projected_2d = projected_2d.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected_2d - face_2d, axis=1)))


def debug_pose_points(frame_idx, face_2d):
    if not DEBUG_HEAD_POSE:
        return

    point_text = ", ".join(
        f"{name}=({pt[0]:.1f},{pt[1]:.1f})"
        for name, pt in zip(HEAD_POSE_NAMES, face_2d)
    )
    print(f"[POSE PTS] frame={frame_idx} {point_text}")


# =========================================================
# HEAD POSE FUNCTIONS
# =========================================================
def rotation_matrix_to_euler_deg(rmat):
    sy = np.sqrt((rmat[0, 0] ** 2) + (rmat[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0

    return np.degrees([pitch, yaw, roll]).astype(np.float64)


def normalize_head_pose_angles(pitch, yaw, roll):
    # Angle conversion itself was not the main root cause, but robust wrapping
    # is still necessary because Euler angles have equivalent 180/-180 forms.
    pitch = wrap_angle_deg(pitch)
    yaw = wrap_angle_deg(yaw) * float(YAW_SIGN_CORRECTION)
    roll = wrap_angle_deg(roll)

    if pitch > 90.0:
        pitch = 180.0 - pitch
        yaw = wrap_angle_deg(yaw + 180.0)
        roll = wrap_angle_deg(roll + 180.0)
    elif pitch < -90.0:
        pitch = -180.0 - pitch
        yaw = wrap_angle_deg(yaw + 180.0)
        roll = wrap_angle_deg(roll + 180.0)

    return (
        wrap_angle_deg(pitch),
        wrap_angle_deg(yaw),
        wrap_angle_deg(roll),
    )


def is_pose_valid(pitch, yaw, roll, reproj_error_px):
    p_ok = POSE_VALID_PITCH_RANGE[0] <= pitch <= POSE_VALID_PITCH_RANGE[1]
    y_ok = POSE_VALID_YAW_RANGE[0] <= yaw <= POSE_VALID_YAW_RANGE[1]
    r_ok = POSE_VALID_ROLL_RANGE[0] <= roll <= POSE_VALID_ROLL_RANGE[1]
    e_ok = reproj_error_px <= POSE_MAX_REPROJ_ERROR_PX
    return int(p_ok and y_ok and r_ok and e_ok)


def debug_pose(frame_idx, raw_angles, normalized_angles, reproj_error_px, method_name):
    if not DEBUG_HEAD_POSE:
        return

    raw_pitch, raw_yaw, raw_roll = raw_angles
    norm_pitch, norm_yaw, norm_roll = normalized_angles
    print(
        f"[POSE DEBUG] frame={frame_idx} method={method_name} "
        f"raw(p,y,r)=({raw_pitch:.2f}, {raw_yaw:.2f}, {raw_roll:.2f}) "
        f"norm(p,y,r)=({norm_pitch:.2f}, {norm_yaw:.2f}, {norm_roll:.2f}) "
        f"reproj_px={reproj_error_px:.3f}"
    )


def solve_pnp_candidate(face_2d, cam_matrix, method, refine):
    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL,
        face_2d,
        cam_matrix,
        DIST_COEFFS,
        flags=method,
    )
    if not success:
        return None

    if refine:
        refine_success, rot_vec_refined, trans_vec_refined = cv2.solvePnP(
            FACE_3D_MODEL,
            face_2d,
            cam_matrix,
            DIST_COEFFS,
            rot_vec,
            trans_vec,
            True,
            cv2.SOLVEPNP_ITERATIVE,
        )
        if refine_success:
            rot_vec, trans_vec = rot_vec_refined, trans_vec_refined

    reproj_error_px = compute_reprojection_error(face_2d, rot_vec, trans_vec, cam_matrix)
    return {
        "rot_vec": rot_vec,
        "trans_vec": trans_vec,
        "reproj_error_px": reproj_error_px,
        "method": method,
    }


def solve_head_pose(face_2d, cam_matrix):
    primary = solve_pnp_candidate(
        face_2d=face_2d,
        cam_matrix=cam_matrix,
        method=PRIMARY_PNP_FLAG,
        refine=True,
    )
    if primary is None:
        if FALLBACK_PNP_FLAG is None:
            return None
        return solve_pnp_candidate(
            face_2d=face_2d,
            cam_matrix=cam_matrix,
            method=FALLBACK_PNP_FLAG,
            refine=True,
        )

    if (
        FALLBACK_PNP_FLAG is not None
        and primary["reproj_error_px"] > POSE_FALLBACK_REPROJ_ERROR_PX
    ):
        fallback = solve_pnp_candidate(
            face_2d=face_2d,
            cam_matrix=cam_matrix,
            method=FALLBACK_PNP_FLAG,
            refine=True,
        )
        if fallback is not None and fallback["reproj_error_px"] < primary["reproj_error_px"]:
            return fallback

    return primary


def extract_pose_angles(pose_solution, frame_count):
    rmat, _ = cv2.Rodrigues(pose_solution["rot_vec"])
    raw_pitch, raw_yaw, raw_roll = rotation_matrix_to_euler_deg(rmat)
    pitch, yaw, roll = normalize_head_pose_angles(raw_pitch, raw_yaw, raw_roll)

    method_name = (
        "SQPNP"
        if pose_solution["method"] == getattr(cv2, "SOLVEPNP_SQPNP", None)
        else "EPNP"
    )
    debug_pose(
        frame_count,
        (raw_pitch, raw_yaw, raw_roll),
        (pitch, yaw, roll),
        pose_solution["reproj_error_px"],
        method_name,
    )
    return pitch, yaw, roll


# =========================================================
# MEDIAPIPE
# =========================================================
def get_face_mesh(fast_mode):
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# =========================================================
# MAIN VIDEO PROCESSING
# =========================================================
def process_video(video_path, output_csv, face_mesh, show_video=False, fast_mode=FAST_MODE):
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_str = str(raw_total) if raw_total > 0 else "unknown"
        video_name = os.path.basename(video_path)

        print(f"\nProcessing: {video_name}")
        print(
            f"FPS: {fps:.2f}  |  Total frames: {total_str}  |  "
            f"Mode: {'FAST' if fast_mode else 'FULL'}"
        )

        n_no_face = 0
        n_pose_bad = 0
        n_ear_susp = 0

        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(CSV_COLUMNS)

            rows_buffer = []
            cam_matrix = None
            min_h_px = None
            last_w = None
            last_h = None

            face_2d = np.empty((len(HEAD_POSE_IDS), 2), dtype=np.float64)
            left_pts = np.empty((len(LEFT_EYE_IDS), 2), dtype=np.float64)
            right_pts = np.empty((len(RIGHT_EYE_IDS), 2), dtype=np.float64)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    continue

                frame_count += 1

                if FRAME_STRIDE > 1 and ((frame_count - 1) % FRAME_STRIDE) != 0:
                    continue

                time_sec = frame_count / fps

                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                h_img, w_img = frame.shape[:2]
                if w_img != last_w or h_img != last_h:
                    cam_matrix = build_camera_matrix(w_img, h_img)
                    min_h_px = max(2.0, w_img * 0.005)
                    last_w, last_h = w_img, h_img

                frame_for_mesh = resize_for_facemesh(frame, fast_mode=fast_mode)
                frame_rgb = cv2.cvtColor(frame_for_mesh, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = face_mesh.process(frame_rgb)

                face_detected = 0
                pose_valid = 0
                ear_valid = 0
                ear_suspicious = 0

                yaw = None
                pitch = None
                roll = None

                left_ear = None
                right_ear = None
                avg_ear = None

                if results.multi_face_landmarks:
                    face_detected = 1
                    landmarks = results.multi_face_landmarks[0].landmark

                    fill_points_2d(landmarks, HEAD_POSE_IDS, w_img, h_img, face_2d)
                    debug_pose_points(frame_count, face_2d)

                    pose_solution = solve_head_pose(face_2d, cam_matrix)

                    if pose_solution is not None:
                        pitch, yaw, roll = extract_pose_angles(pose_solution, frame_count)
                        pose_valid = is_pose_valid(
                            pitch,
                            yaw,
                            roll,
                            pose_solution["reproj_error_px"],
                        )

                        if not pose_valid:
                            pitch = None
                            yaw = None
                            roll = None

                    fill_points_2d(landmarks, LEFT_EYE_IDS, w_img, h_img, left_pts)
                    fill_points_2d(landmarks, RIGHT_EYE_IDS, w_img, h_img, right_pts)

                    left_ear, l_valid = calculate_ear_robust(
                        left_pts,
                        min_horizontal_px=min_h_px,
                    )
                    right_ear, r_valid = calculate_ear_robust(
                        right_pts,
                        min_horizontal_px=min_h_px,
                    )

                    if l_valid and r_valid:
                        avg_ear = (left_ear + right_ear) / 2.0
                        ear_valid = 1

                        pose_unknown = yaw is None
                        yaw_extreme = yaw is not None and abs(yaw) > EAR_SUSPICIOUS_YAW_DEG
                        if pose_unknown or yaw_extreme:
                            ear_suspicious = 1

                    if not pose_valid:
                        n_pose_bad += 1
                    if ear_suspicious:
                        n_ear_susp += 1

                    if show_video:
                        for x, y in left_pts:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                        for x, y in right_pts:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                        for idx in HEAD_POSE_IDS:
                            lm = landmarks[idx]
                            cv2.circle(
                                frame,
                                (int(lm.x * w_img), int(lm.y * h_img)),
                                3,
                                (0, 255, 0),
                                -1,
                            )

                        lines = [
                            (f"Frame: {frame_count}", (255, 255, 255)),
                            (f"Face: {face_detected}  PoseOK: {pose_valid}", (255, 255, 255)),
                            (
                                f"Yaw:   {yaw:.2f}" if yaw is not None else "Yaw:   None",
                                (0, 255, 0),
                            ),
                            (
                                f"Pitch: {pitch:.2f}" if pitch is not None else "Pitch: None",
                                (0, 255, 0),
                            ),
                            (
                                f"Roll:  {roll:.2f}" if roll is not None else "Roll:  None",
                                (0, 255, 0),
                            ),
                            (
                                f"EAR valid: {ear_valid}  susp: {ear_suspicious}",
                                (0, 255, 255),
                            ),
                            (
                                f"AvgEAR: {avg_ear:.4f}" if avg_ear is not None else "AvgEAR: None",
                                (0, 255, 255),
                            ),
                        ]
                        for i, (text, color) in enumerate(lines):
                            cv2.putText(
                                frame,
                                text,
                                (20, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                color,
                                2,
                            )

                        cv2.imshow("DMS Feature Extraction", frame)
                else:
                    n_no_face += 1

                rows_buffer.append(
                    [
                        frame_count,
                        safe_round(time_sec, 3),
                        face_detected,
                        safe_round(yaw, 4),
                        safe_round(pitch, 4),
                        safe_round(roll, 4),
                        pose_valid,
                        safe_round(left_ear, 4),
                        safe_round(right_ear, 4),
                        safe_round(avg_ear, 4),
                        ear_valid,
                        ear_suspicious,
                    ]
                )

                if len(rows_buffer) >= CSV_WRITE_BUFFER_SIZE:
                    flush_rows(writer, rows_buffer)

                if show_video and (cv2.waitKey(1) & 0xFF == 27):
                    print("[INFO] Stopped by ESC.")
                    break

                if frame_count % PROGRESS_PRINT_EVERY == 0:
                    print(f"  {frame_count}/{total_str} frames...", end="\r")

            flush_rows(writer, rows_buffer)

        pct_no_face = 100.0 * n_no_face / max(frame_count, 1)
        pct_pose_bad = 100.0 * n_pose_bad / max(frame_count, 1)
        pct_ear_susp = 100.0 * n_ear_susp / max(frame_count, 1)
        print(f"\n[DONE]    {video_name}")
        print(
            f"[QUALITY] No face: {pct_no_face:.1f}% | "
            f"Pose invalid: {pct_pose_bad:.1f}% | "
            f"EAR suspicious: {pct_ear_susp:.1f}%"
        )
        print(f"[CSV]     {output_csv}")

    finally:
        cap.release()
        if show_video:
            cv2.destroyAllWindows()


# =========================================================
# BATCH PROCESSING
# =========================================================
def process_all_videos(input_folder, output_folder, show_video=False, fast_mode=FAST_MODE):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
    video_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s).\n")

    face_mesh = get_face_mesh(fast_mode=fast_mode)

    try:
        for video_name in sorted(video_files):
            video_path = os.path.join(input_folder, video_name)
            base_name = os.path.splitext(video_name)[0]
            output_csv = os.path.join(output_folder, f"{base_name}.csv")

            try:
                process_video(
                    video_path=video_path,
                    output_csv=output_csv,
                    face_mesh=face_mesh,
                    show_video=show_video,
                    fast_mode=fast_mode,
                )
            except Exception as exc:
                print(f"[ERROR] Failed on {video_name}: {exc}")
    finally:
        face_mesh.close()
        if show_video:
            cv2.destroyAllWindows()

    print("\nAll videos processed.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    input_folder = r"D:\dataset\distraction\RGB"
    output_folder = r"D:\csv1\distraction\RGB"

    process_all_videos(
        input_folder=input_folder,
        output_folder=output_folder,
        show_video=False,
        fast_mode=FAST_MODE,
    )
