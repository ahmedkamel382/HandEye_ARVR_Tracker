import os
import cv2
import numpy as np
import pandas as pd
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from gaze_intent import extract_gaze_features

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "MPIIFaceGaze")
OUTPUT_CSV_UNMIRRORED = os.path.join(BASE_DIR, "datasets", "mpii_spatial_features_unmirrored.csv")
OUTPUT_CSV_MIRRORED = os.path.join(BASE_DIR, "datasets", "mpii_spatial_features_mirrored.csv")
OUTPUT_CSV_LEGACY = os.path.join(BASE_DIR, "datasets", "mpii_spatial_features.csv")

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


def vector_to_pitch_yaw(gaze_vector):
    """Converts a 3D gaze vector into Pitch and Yaw degrees."""
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)  # Normalize
    x, y, z = gaze_vector

    # Camera coordinate system typically has Z pointing away, Y down, X right
    yaw = math.atan2(-x, -z)
    pitch = math.asin(-y)

    return math.degrees(pitch), math.degrees(yaw)


def _extract_row_from_bgr(bgr_image, true_pitch, true_yaw):
    """Runs MediaPipe FaceLandmarker on a BGR image and returns a feature row dict (or None)."""
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = detector.detect(mp_image)

    if results.face_landmarks and results.facial_transformation_matrixes:
        landmarks = results.face_landmarks[0]
        pose_matrix = results.facial_transformation_matrixes[0]

        student_x, student_y, head_pitch, head_yaw, head_roll = extract_gaze_features(
            landmarks, pose_matrix
        )

        return {
            "student_feature_x": student_x,
            "student_feature_y": student_y,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
            "teacher_pitch": true_pitch - head_pitch,
            "teacher_yaw": true_yaw - head_yaw,
        }
    return None


def process_mpiigaze(
    output_unmirrored_csv=OUTPUT_CSV_UNMIRRORED,
    output_mirrored_csv=OUTPUT_CSV_MIRRORED,
    max_samples=None,
):
    """
    Single-pass dual-view MPIIFaceGaze feature extractor for Hybrid Best-X / Best-Y training:
      1. Unmirrored view (`mpii_spatial_features_unmirrored.csv`) -> algebraically reflected during
         training (`1 - sx`, `-head_yaw`, `-head_roll`), which empirically maximizes horizontal Yaw
         accuracy and live fixation stability (passing all 3 of Engineer 4's X stability limits).
      2. Horizontally mirrored view (`cv2.flip(image, 1)` -> `mpii_spatial_features_mirrored.csv`),
         which aligns MediaPipe's asymmetric 3D face mesh prior with live selfie capture and reduces
         live vertical Pitch jitter (`Y_SD`) by ~39%.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return

    features_unmirrored = []
    features_mirrored = []
    processed_images = 0
    participants = [f"p{str(i).zfill(2)}" for i in range(15)]  # p00 to p14

    for p in participants:
        p_dir = os.path.join(DATASET_PATH, p)
        txt_path = os.path.join(p_dir, f"{p}.txt")

        if not os.path.exists(txt_path):
            continue

        print(f"Processing {p}...")

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # Process every image in the dataset in a single disk read pass
        for line in tqdm(lines):
            parts = line.strip().split()
            if len(parts) < 28:
                continue

            img_rel_path = parts[0]
            img_path = os.path.join(p_dir, img_rel_path)

            if not os.path.exists(img_path):
                continue

            # Parse Ground Truth 3D Gaze Vector
            fc = np.array([float(parts[21]), float(parts[22]), float(parts[23])])  # Index 21,22,23 are Dim 22-24
            gt = np.array([float(parts[24]), float(parts[25]), float(parts[26])])  # Index 24,25,26 are Dim 25-27
            gaze_vector = gt - fc

            true_pitch, true_yaw = vector_to_pitch_yaw(gaze_vector)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # View 1: Unmirrored extraction (Yaw training source)
            row_unmir = _extract_row_from_bgr(image, true_pitch, true_yaw)
            if row_unmir is not None:
                features_unmirrored.append(row_unmir)

            # View 2: Mirrored extraction via cv2.flip(image, 1) (Pitch training source)
            flipped_image = cv2.flip(image, 1)
            row_mir = _extract_row_from_bgr(flipped_image, true_pitch, true_yaw)
            if row_mir is not None:
                features_mirrored.append(row_mir)

            processed_images += 1
            if max_samples is not None and processed_images >= max_samples:
                break

        if max_samples is not None and processed_images >= max_samples:
            break

    df_unmir = pd.DataFrame(features_unmirrored)
    df_mir = pd.DataFrame(features_mirrored)

    if output_unmirrored_csv:
        df_unmir.to_csv(output_unmirrored_csv, index=False)
        print(f"\nSuccessfully saved {len(df_unmir)} unmirrored samples to {output_unmirrored_csv}")

    if output_mirrored_csv:
        df_mir.to_csv(output_mirrored_csv, index=False)
        print(f"Successfully saved {len(df_mir)} mirrored (cv2.flip) samples to {output_mirrored_csv}")
        if output_mirrored_csv == OUTPUT_CSV_MIRRORED and max_samples is None:
            df_mir.to_csv(OUTPUT_CSV_LEGACY, index=False)

    return df_unmir, df_mir


if __name__ == "__main__":
    process_mpiigaze()