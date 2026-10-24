"""
Shared feature extraction for the hand-gesture pipeline.

Used by hand_data_collector.py (logging), model_trainer.py (training) and
hand_tracker.py (inference) so the features are IDENTICAL everywhere.

Why these features (the old 2D pixel-distance approach failed because of it):
  * They are computed from MediaPipe's *world* landmarks (metric 3D, real
    units), not from screen pixels, so hand rotation / tilt relative to the
    camera does not change them the way it changes 2D pixel distances.
  * Joint angles are invariant to rotation, translation and scale.
  * All distances are divided by the 3D palm size (wrist -> middle MCP).

This module has NO mediapipe / cv2 dependency, so it can be unit-tested alone.
"""
import numpy as np

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
MIDDLE_MCP = 9

# (a, b, c) -> angle at joint b between bones b->a and b->c, in degrees.
ANGLE_TRIPLETS = {
    "thumb_cmc": (0, 1, 2), "thumb_mcp": (1, 2, 3), "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6), "index_pip": (5, 6, 7), "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10), "middle_pip": (9, 10, 11), "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14), "ring_pip": (13, 14, 15), "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18), "pinky_pip": (17, 18, 19), "pinky_dip": (18, 19, 20),
}
FINGER_TIPS = {"index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP, "pinky": PINKY_TIP}

FEATURE_NAMES = (
    [f"ang_{n}" for n in ANGLE_TRIPLETS]
    + [f"thumb_to_{f}_tip" for f in FINGER_TIPS]          # pinch ratio = thumb_to_index_tip
    + [f"wrist_to_{f}_tip" for f in ("thumb", *FINGER_TIPS)]
)
N_FEATURES = len(FEATURE_NAMES)  # 24


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def result_to_points(result):
    """MediaPipe HandLandmarkerResult -> (21, 3) world-landmark array, or None."""
    if not result.hand_world_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float64)


def extract_features(points):
    """points: (21, 3) array -> (24,) float array (order = FEATURE_NAMES)."""
    palm = np.linalg.norm(points[MIDDLE_MCP] - points[WRIST])
    if palm < 1e-9:
        palm = 1.0
    angles = [_angle(points[a], points[b], points[c]) for a, b, c in ANGLE_TRIPLETS.values()]
    thumb = points[THUMB_TIP]
    thumb_d = [np.linalg.norm(thumb - points[t]) / palm for t in FINGER_TIPS.values()]
    wrist_d = [np.linalg.norm(points[t] - points[WRIST]) / palm
               for t in (THUMB_TIP, *FINGER_TIPS.values())]
    return np.array(angles + thumb_d + wrist_d, dtype=np.float64)


def load_gesture_model(path):
    """Returns the trained RandomForest, or None if the file doesn't exist."""
    import os
    import joblib
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    if bundle["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Model was trained with different features. Re-run model_trainer.py.")
    model = bundle["model"]
    model.n_jobs = 1  # single-row real-time inference: threads only add overhead
    return model
