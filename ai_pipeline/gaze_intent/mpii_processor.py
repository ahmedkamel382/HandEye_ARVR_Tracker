import os
import cv2
import numpy as np
import pandas as pd
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "MPIIFaceGaze")
OUTPUT_CSV = os.path.join(BASE_DIR, "datasets", "mpii_spatial_features.csv")

mp_model_path = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'assets', 'face_landmarker.task'))

# --- MEDIAPIPE INITIALIZATION ---
base_options = python.BaseOptions(model_asset_path=mp_model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_facial_transformation_matrixes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- LANDMARK INDICES ---
LEFT_IRIS, RIGHT_IRIS = 468, 473
L_CORNERS, R_CORNERS = (33, 133), (362, 263)


def _get_3d_dist(p1, p2, frame_w, frame_h):
    return math.hypot((p1.x - p2.x) * frame_w, (p1.y - p2.y) * frame_h, p1.z - p2.z)


def _get_iris_ratio(landmarks, corners, iris_idx, axis):
    """Measure the iris against the static line joining the eye corners in normalized 2D space."""
    p_corner1, p_corner2 = landmarks[corners[0]], landmarks[corners[1]]
    p_iris = landmarks[iris_idx]

    corner1 = np.array([p_corner1.x, p_corner1.y], dtype=float)
    corner2 = np.array([p_corner2.x, p_corner2.y], dtype=float)
    iris = np.array([p_iris.x, p_iris.y], dtype=float)

    baseline = corner2 - corner1
    baseline_length = float(np.linalg.norm(baseline))
    if baseline_length == 0:
        return 0.5 if axis == "x" else 0.0

    iris_vec = iris - corner1
    scale = float(np.dot(iris_vec, baseline) / (baseline_length ** 2))

    if axis == "x":
        return scale
    if axis == "y":
        proj = scale * baseline
        perp_vec = iris_vec - proj
        norm_dist = float(np.linalg.norm(perp_vec) / baseline_length)
        if perp_vec[1] < 0:
            return -norm_dist
        return norm_dist

    raise ValueError(f"Unsupported iris ratio axis: {axis}")


def _get_head_pose_angles(matrix):
    R = matrix[:3, :3]
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def vector_to_pitch_yaw(gaze_vector):
    """Converts a 3D gaze vector into Pitch and Yaw degrees."""
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)  # Normalize
    x, y, z = gaze_vector

    # Camera coordinate system typically has Z pointing away, Y down, X right
    yaw = math.atan2(-x, -z)
    pitch = math.asin(-y)

    return math.degrees(pitch), math.degrees(yaw)


def process_mpiigaze():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return

    features_list = []
    participants = [f"p{str(i).zfill(2)}" for i in range(15)]  # p00 to p14

    for p in participants:
        p_dir = os.path.join(DATASET_PATH, p)
        txt_path = os.path.join(p_dir, f"{p}.txt")

        if not os.path.exists(txt_path):
            continue

        print(f"Processing {p}...")

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # Process every image in the dataset
        for line in tqdm(lines):
            parts = line.strip().split()
            if len(parts) < 28: continue

            img_rel_path = parts[0]
            img_path = os.path.join(p_dir, img_rel_path)

            if not os.path.exists(img_path): continue

            # Parse Ground Truth 3D Gaze Vector
            fc = np.array([float(parts[21]), float(parts[22]), float(parts[23])])  # Index 21,22,23 are Dim 22-24
            gt = np.array([float(parts[24]), float(parts[25]), float(parts[26])])  # Index 24,25,26 are Dim 25-27
            gaze_vector = gt - fc

            true_pitch, true_yaw = vector_to_pitch_yaw(gaze_vector)

            # Read image and run MediaPipe
            image = cv2.imread(img_path)
            if image is None: continue
            frame_h, frame_w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            results = detector.detect(mp_image)

            if results.face_landmarks and results.facial_transformation_matrixes:
                landmarks = results.face_landmarks[0]
                pose_matrix = results.facial_transformation_matrixes[0]

                l_x = _get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "x")
                r_x = _get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "x")
                student_x = (l_x + r_x) / 2.0

                l_y = _get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "y")
                r_y = _get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "y")
                student_y = (l_y + r_y) / 2.0

                head_pitch, head_yaw, head_roll = _get_head_pose_angles(pose_matrix)

                features_list.append({
                    "student_feature_x": student_x,
                    "student_feature_y": student_y,
                    "head_pitch": head_pitch,
                    "head_yaw": head_yaw,
                    "head_roll": head_roll,
                    "teacher_pitch": true_pitch - head_pitch,
                    "teacher_yaw": true_yaw - head_yaw
                })

    df = pd.DataFrame(features_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully saved {len(df)} screen-gaze samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    process_mpiigaze()