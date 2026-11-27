import csv
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, fields
from typing import Tuple, Optional, Any, List

import mediapipe as mp
import numpy as np
import tl2cgen
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore", category=UserWarning)

# --- Landmark indices (MediaPipe Face Landmarker, 478 points) ---
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = 468
RIGHT_IRIS = 473
L_CORNERS = (33, 133)
R_CORNERS = (362, 263)

# --- Canonical eye-aligned MediaPipe blendshape polarity mapping ---
# Empirically verified on MPIIFaceGaze (unflipped and cv::flip mirrored frames):
#   MediaPipe LEFT_EYE (33,160,158,133,153,144) is always image-left (user physical left eye
#   in mirrored webcam frames -> state 1 Left Wink) and corresponds to ARKit *Right blendshapes.
#   MediaPipe RIGHT_EYE (362,385,387,263,373,380) is always image-right (-> state 2 Right Wink)
#   and corresponds to ARKit *Left blendshapes.
EYE_BLENDSHAPE_NAMES = (
    "eyeBlinkRight",      # slot 0: bs_blink_l    (LEFT_EYE)
    "eyeBlinkLeft",       # slot 1: bs_blink_r    (RIGHT_EYE)
    "eyeLookDownRight",   # slot 2: bs_lookdown_l (LEFT_EYE)
    "eyeLookDownLeft",    # slot 3: bs_lookdown_r (RIGHT_EYE)
    "eyeSquintRight",     # slot 4: bs_squint_l   (LEFT_EYE)
    "eyeSquintLeft",      # slot 5: bs_squint_r   (RIGHT_EYE)
    "eyeWideRight",       # slot 6: bs_wide_l     (LEFT_EYE)
    "eyeWideLeft",        # slot 7: bs_wide_r     (RIGHT_EYE)
)

# --- Temporal parameters ---
EAR_WINDOW_SIZE = 10
CALIBRATION_FRAMES = 30

# --- Numerical guards (mirrored in core_engine/include/gaze_math.hpp) ---
EAR_DIST_EPS = 1e-12       # horizontal eye distance below this -> NaN (degenerate eye)
CALIB_EAR_EPS = 1e-6       # legacy EarFeatureState minimum raw EAR guard
IRIS_BASELINE_EPS = 1e-12  # squared eye-corner baseline below this -> neutral iris ratio

# --- HybridBlinkDetector v2 shared spec constants (mirrored in hybrid_blink_spec.json & gaze_math.hpp) ---
HYBRID_CALIB_MIN_EAR_BS = 0.12
HYBRID_CALIB_MIN_EAR_NO_BS = 0.08
HYBRID_CALIB_MAX_ASYM = 0.35
HYBRID_CALIB_MAX_PITCH_DEG = 20.0
HYBRID_CALIB_MAX_BS_BLINK = 0.35
HYBRID_CALIB_EAR_TRIM_LO = 0.20
HYBRID_CALIB_EAR_TRIM_HI = 0.90
HYBRID_CALIB_EAR_QUANTILE = 0.75
HYBRID_CALIB_BS_TRIM_LO = 0.10
HYBRID_CALIB_BS_TRIM_HI = 0.80
HYBRID_CALIB_BS_QUANTILE = 0.25
HYBRID_MAX_BASE_BS = 0.25
HYBRID_DEFAULT_BASE_SY = -0.08

HYBRID_ADAPT_ALPHA = 0.005
HYBRID_ADAPT_MAX_ASYM = 0.25
HYBRID_ADAPT_MAX_PITCH_DEG = 15.0
HYBRID_ADAPT_MAX_RAW_SCORE = 0.22
HYBRID_ADAPT_MAX_BS_BLINK = 0.25
HYBRID_ADAPT_MAX_BS_LOOKDOWN = 0.35

HYBRID_BS_LOOKDOWN_DEADZONE = 0.20
HYBRID_BS_LOOKDOWN_GAIN = 0.30
HYBRID_BS_DENOM_MIN = 0.35
HYBRID_LD_GATE_EAR_LO = 0.34
HYBRID_LD_GATE_EAR_HI = 0.48

HYBRID_EAR_SY_GAIN = 0.90
HYBRID_EAR_SY_MAX_DELTA = 0.20
HYBRID_EAR_PITCH_GAIN = 0.14
HYBRID_EAR_PITCH_MAX_DEG = 25.0
HYBRID_EAR_LOOKDOWN_GAIN = 0.24
HYBRID_EAR_MAX_DROOP_COMP = 0.30
HYBRID_EAR_OPEN_REF = 0.92
HYBRID_EAR_CLOSED_SPAN = 0.55

HYBRID_FUSION_W_BS = 0.55
HYBRID_FUSION_W_EAR = 0.45

HYBRID_EMA_TAU_ATTACK_S = 0.015
HYBRID_EMA_TAU_RELEASE_S = 0.030
HYBRID_EMA_TAU_S = HYBRID_EMA_TAU_RELEASE_S
HYBRID_MAX_GAP_S = 0.250
HYBRID_DEFAULT_DT_S = 1.0 / 30.0

HYBRID_CLOSE_THRESH = 0.48
HYBRID_OPEN_THRESH = 0.30
HYBRID_WINK_ASYM_ENTER = 0.18
HYBRID_WINK_ASYM_EXIT = 0.10
HYBRID_WINK_OTHER_MAX = 0.65
HYBRID_SYM_BAND = 0.14

# --- Pointer span (mirrored in GazeNative::kSpanDeg = 60.0) ---
SPAN_DEG = 60.0

# --- Output sentinel ---
# NaN, not -1.0: -1.0 deg is a physically valid gaze/head angle and cannot signal "invalid".
# Non-finite values are rejected by the downstream DSP stage, so NaN can never seed filter history.
INVALID_ANGLE = float("nan")


@dataclass
class GazeDiagnosticPayload:
    """
    Opt-in per-frame diagnostic telemetry row (--diag-log <csv_path>).
    Unified 32-column schema matching C++ GazeDiagRow / kGazeDiagCsvHeader 1:1.
    Captures timing/validity (frame_id, ts_ms, capture_s, detected, state), the 5-D regressor
    features, bilateral iris ratios, relative/head/absolute angles, normalized X/Y, and the 12 raw
    normalized 2D MediaPipe eye-corner/iris landmark coordinates (33, 133, 468, 362, 263, 473).
    """
    frame_id: int
    ts_ms: int
    capture_s: float
    detected: int
    state: int
    left_iris_x: float
    right_iris_x: float
    sx: float
    left_iris_y: float
    right_iris_y: float
    sy: float
    head_pitch_deg: float
    head_yaw_deg: float
    head_roll_deg: float
    rel_pitch_deg: float
    rel_yaw_deg: float
    abs_pitch_deg: float
    abs_yaw_deg: float
    norm_x: float
    norm_y: float
    # Raw normalized 2D MediaPipe landmark coordinates used to compute left_iris_x/y and right_iris_x/y
    lm33_x: float
    lm33_y: float
    lm133_x: float
    lm133_y: float
    lm468_x: float
    lm468_y: float
    lm362_x: float
    lm362_y: float
    lm263_x: float
    lm263_y: float
    lm473_x: float
    lm473_y: float


_DIAG_FIELD_NAMES = tuple(f.name for f in fields(GazeDiagnosticPayload))


@dataclass
class BlinkDiagnosticPayload:
    """
    Separate opt-in blink telemetry row (--blink-diag-log <csv_path>).
    22-column schema matching C++ BlinkDiagRow / kBlinkDiagCsvHeader 1:1 without touching gaze_diag.csv.
    """
    frame_id: int
    ts_ms: int
    capture_s: float
    detected: int
    state: int
    detector_mode: str
    raw_ear_l: float
    raw_ear_r: float
    base_ear_l: float
    base_ear_r: float
    norm_ear_l: float
    norm_ear_r: float
    bs_blink_l: float
    bs_blink_r: float
    bs_lookdown_l: float
    bs_lookdown_r: float
    bs_squint_l: float
    bs_squint_r: float
    score_l: float
    score_r: float
    sy: float
    head_pitch_deg: float


_BLINK_DIAG_FIELD_NAMES = tuple(f.name for f in fields(BlinkDiagnosticPayload))


def _clamp01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _clamp(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v):
        return lo
    return lo if v < lo else (hi if v > hi else v)


def _trimmed_quantile(values: List[float], low_frac: float, high_frac: float, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    lo = int(math.floor(low_frac * n))
    hi = int(math.floor(high_frac * n))
    if lo < 0:
        lo = 0
    if hi <= lo:
        hi = lo + 1
    if hi > n:
        hi = n
    m = hi - lo
    if m == 1:
        return float(s[lo])
    pos = q * (m - 1)
    idx = int(math.floor(pos))
    frac = pos - idx
    if idx + 1 >= m:
        return float(s[lo + m - 1])
    return float(s[lo + idx] * (1.0 - frac) + s[lo + idx + 1] * frac)


def extract_eye_blendshapes(face_blendshapes) -> Optional[Tuple[float, ...]]:
    """Extract canonical eye-aligned 8-tuple of float32-quantized scores in [0.0, 1.0].

    Order: (bs_blink_l, bs_blink_r, bs_lookdown_l, bs_lookdown_r,
            bs_squint_l, bs_squint_r, bs_wide_l, bs_wide_r).
    Returns None if absent, missing any required channel, non-finite, or out of [0.0, 1.0].
    """
    if face_blendshapes is None:
        return None
    try:
        if isinstance(face_blendshapes, dict):
            scores = face_blendshapes
        else:
            scores = {
                getattr(c, "category_name", ""): getattr(c, "score", float("nan"))
                for c in face_blendshapes
            }
        raw = [float(scores[name]) for name in EYE_BLENDSHAPE_NAMES]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(arr).all():
        return None
    if bool((arr < 0.0).any() or (arr > 1.0).any()):
        return None
    return tuple(float(v) for v in arr.tolist())


def angle_to_norm_coord(angle_deg: float) -> float:
    """Mirrors GazeNative::setPointer: clamp01(0.5 + angle_deg / 60.0), or -1.0 when non-finite."""
    if not math.isfinite(angle_deg):
        return -1.0
    return max(0.0, min(1.0, 0.5 + angle_deg / SPAN_DEG))


def is_valid_output(abs_yaw: float, abs_pitch: float) -> bool:
    """True when process_frame produced a usable absolute gaze (not calibrating, face present)."""
    return math.isfinite(abs_yaw) and math.isfinite(abs_pitch)


def get_3d_dist(p1, p2, frame_w: int, frame_h: int) -> float:
    # Scale normalized coordinates to physical pixels (MediaPipe z scales with width)
    dx = (float(p1.x) - float(p2.x)) * frame_w
    dy = (float(p1.y) - float(p2.y)) * frame_h
    dz = (float(p1.z) - float(p2.z)) * frame_w
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return INVALID_ANGLE
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_ear(face_landmarks, indices, frame_w: int, frame_h: int) -> float:
    """3D pixel-scaled Eye Aspect Ratio.

    P0 safety fix: returns NaN (not 0.0) when frame dimensions are non-positive, any landmark
    is non-finite, or the horizontal eye corner distance |p1 - p4| <= EAR_DIST_EPS so degenerate
    eyes trigger invalid-frame rejection (state = 0) instead of masquerading as Closed (state = 3).
    """
    if frame_w <= 0 or frame_h <= 0:
        return INVALID_ANGLE
    p1, p2, p3, p4, p5, p6 = (face_landmarks[i] for i in indices)
    v_dist_1 = get_3d_dist(p2, p6, frame_w, frame_h)
    v_dist_2 = get_3d_dist(p3, p5, frame_w, frame_h)
    h_dist = get_3d_dist(p1, p4, frame_w, frame_h)
    if not (math.isfinite(v_dist_1) and math.isfinite(v_dist_2) and math.isfinite(h_dist)):
        return INVALID_ANGLE
    if h_dist <= EAR_DIST_EPS:
        return INVALID_ANGLE
    return (v_dist_1 + v_dist_2) / (2.0 * h_dist)


def get_iris_ratio(landmarks, corners: Tuple[int, int], iris_idx: int, axis: str) -> float:
    """Measure the iris against the static line joining the eye corners in normalized 2D space."""
    p_corner1, p_corner2 = landmarks[corners[0]], landmarks[corners[1]]
    p_iris = landmarks[iris_idx]

    c1_x, c1_y = float(p_corner1.x), float(p_corner1.y)
    c2_x, c2_y = float(p_corner2.x), float(p_corner2.y)
    ir_x, ir_y = float(p_iris.x), float(p_iris.y)

    base_x = c2_x - c1_x
    base_y = c2_y - c1_y
    baseline_sq = base_x * base_x + base_y * base_y
    if baseline_sq < IRIS_BASELINE_EPS:
        return 0.5 if axis == "x" else 0.0

    baseline_length = math.sqrt(baseline_sq)
    iris_vx = ir_x - c1_x
    iris_vy = ir_y - c1_y
    scale = (iris_vx * base_x + iris_vy * base_y) / baseline_sq

    if axis == "x":
        return scale
    if axis == "y":
        proj_x = scale * base_x
        proj_y = scale * base_y
        perp_x = iris_vx - proj_x
        perp_y = iris_vy - proj_y
        norm_dist = math.hypot(perp_x, perp_y) / baseline_length
        return -norm_dist if perp_y < 0.0 else norm_dist

    raise ValueError(f"Unsupported iris ratio axis: {axis}")


def get_head_pose_angles(matrix) -> Tuple[float, float, float]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim < 2 or mat.shape[0] < 3 or mat.shape[1] < 3:
        raise ValueError(f"Invalid pose matrix shape: {mat.shape}")
    R = mat[:3, :3]
    if not np.all(np.isfinite(R)):
        raise ValueError("Non-finite rotation matrix entries in get_head_pose_angles")
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(-R[2, 0]), sy)
        roll = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        yaw = math.atan2(float(-R[2, 0]), sy)
        roll = 0.0
    angles = (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))
    if not all(math.isfinite(v) for v in angles):
        raise ValueError("Non-finite Euler angles computed in get_head_pose_angles")
    return angles


def _landmarks_and_pose_finite(landmarks, pose_matrix) -> bool:
    if landmarks is None or len(landmarks) != 478 or pose_matrix is None:
        return False
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            coords = np.asarray([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
            mat = np.asarray(pose_matrix, dtype=np.float32)
        return mat.shape == (4, 4) and np.isfinite(mat).all() and np.isfinite(coords).all()
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def extract_gaze_features(landmarks, pose_matrix) -> Tuple[float, float, float, float, float]:
    """5-D geometric gaze feature vector: (student_x, student_y, head_pitch, head_yaw, head_roll)."""
    l_x = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "x")
    r_x = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "x")
    student_x = (l_x + r_x) / 2.0

    l_y = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "y")
    r_y = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "y")
    student_y = (l_y + r_y) / 2.0

    head_pitch, head_yaw, head_roll = get_head_pose_angles(pose_matrix)
    return student_x, student_y, head_pitch, head_yaw, head_roll


def window_stats(history) -> Tuple[float, float]:
    """Min and population variance over whatever the window currently holds (partial windows included)."""
    n = len(history)
    if n == 0:
        return 1.0, 0.0
    w_min = min(history)
    mean = sum(history) / n
    w_var = sum((x - mean) ** 2 for x in history) / n
    return float(w_min), float(w_var)


def safe_classify_legacy_wink(wink_model, wink_features: Tuple[float, ...]) -> int:
    """P0-guarded wrapper for legacy gesture_model.dll fallback: non-finite inputs/outputs -> 0 Neutral."""
    if wink_model is None or len(wink_features) != 6:
        return 0
    if not all(math.isfinite(float(v)) for v in wink_features):
        return 0
    arr = np.array([wink_features], dtype=np.float32)
    if not np.isfinite(arr).all():
        return 0
    probs = np.asarray(wink_model.predict(tl2cgen.DMatrix(arr)), dtype=np.float64).reshape(-1)
    if probs.size == 0 or not np.isfinite(probs).all():
        return 0
    cls = int(np.argmax(probs))
    return cls if 0 <= cls <= 3 else 0


class EarFeatureState:
    """
    Legacy temporal EAR state (kept for wink_data_collector.py & explicit legacy fallback):
    30-frame calibration, per-eye normalization, and 10-frame rolling window.
    """

    def __init__(self, calibration_frames: int = CALIBRATION_FRAMES, window_size: int = EAR_WINDOW_SIZE):
        self.calibration_frame_count = calibration_frames
        self.window_size = window_size
        self.left_ear_history: deque[float] = deque(maxlen=window_size)
        self.right_ear_history: deque[float] = deque(maxlen=window_size)

        self.is_calibrated = False
        self.current_calibration_frames = 0
        self.left_ear_sum = 0.0
        self.right_ear_sum = 0.0
        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0

    def update(self, raw_left_ear: float, raw_right_ear: float
               ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Feed one frame's raw EARs. Returns None while calibrating (including the frame that completes
        calibration), otherwise the 6-feature wink vector:
        (left_ear, left_min, left_var, right_ear, right_min, right_var).
        """
        if not (math.isfinite(raw_left_ear) and math.isfinite(raw_right_ear)):
            return None
        if not self.is_calibrated:
            if raw_left_ear > CALIB_EAR_EPS and raw_right_ear > CALIB_EAR_EPS:
                self.left_ear_sum += raw_left_ear
                self.right_ear_sum += raw_right_ear
                self.current_calibration_frames += 1

                if self.current_calibration_frames >= self.calibration_frame_count:
                    self.baseline_left_ear = self.left_ear_sum / self.calibration_frame_count
                    self.baseline_right_ear = self.right_ear_sum / self.calibration_frame_count
                    self.is_calibrated = True
            return None

        left_ear = raw_left_ear / self.baseline_left_ear
        right_ear = raw_right_ear / self.baseline_right_ear
        self.left_ear_history.append(left_ear)
        self.right_ear_history.append(right_ear)

        l_min, l_var = window_stats(self.left_ear_history)
        r_min, r_var = window_stats(self.right_ear_history)
        return left_ear, l_min, l_var, right_ear, r_min, r_var

    def reset_tracking_state(self) -> None:
        self.left_ear_history.clear()
        self.right_ear_history.clear()
        if not self.is_calibrated:
            self.current_calibration_frames = 0
            self.left_ear_sum = 0.0
            self.right_ear_sum = 0.0


class HybridBlinkDetector:
    """
    Hybrid Blink v2 state machine (mirrored line-for-line in core_engine/include/gaze_math.hpp):
      - Poison-resistant open-eye baseline calibration (trimmed 75th-percentile EAR + 25th-percentile
        resting blendshape offset + median neutral iris sy) with strict symmetry, pitch & open-eye gates.
      - Slow adaptive open-eye EMA (alpha=0.005) active only during clearly open neutral frames.
      - Per-eye continuous closure score c_L(t), c_R(t) in [0, 1] fusing MediaPipe eyeBlink
        (compensated by eyeLookDown) with pitch/sy-compensated robust 3D EAR.
      - Wall-clock EMA smoothing (tau=40 ms, gap reset >250 ms) independent of FPS and frame skips.
      - Deterministic Schmitt-trigger hysteresis + contralateral asymmetry guard (symmetric blinks
        can NEVER fire Left/Right Wink; firm winks with mild sympathetic squint still register).
    """

    def __init__(self, calibration_frames: int = CALIBRATION_FRAMES, window_size: int = EAR_WINDOW_SIZE):
        self.calibration_frame_count = calibration_frames
        self.window_size = window_size
        # Observational rolling history (kept cleared on tracking loss for backward test compatibility)
        self.left_ear_history: deque[float] = deque(maxlen=window_size)
        self.right_ear_history: deque[float] = deque(maxlen=window_size)

        self.is_calibrated = False
        self.current_calibration_frames = 0
        self._cal_ear_l: List[float] = []
        self._cal_ear_r: List[float] = []
        self._cal_bs_l: List[float] = []
        self._cal_bs_r: List[float] = []
        self._cal_sy: List[float] = []

        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0
        self.baseline_bs_left = 0.0
        self.baseline_bs_right = 0.0
        self.baseline_sy = HYBRID_DEFAULT_BASE_SY

        # Post-calibration temporal filter + Schmitt-trigger state
        self._has_smooth = False
        self._last_ts_ms = -1
        self.score_l = 0.0
        self.score_r = 0.0
        self._eye_closed_l = False
        self._eye_closed_r = False
        self.prev_state = 0

        # Last-frame telemetry snapshot
        self.last_detector_mode = "calibrating"
        self.last_norm_l = INVALID_ANGLE
        self.last_norm_r = INVALID_ANGLE

    def recalibrate(self) -> None:
        self.is_calibrated = False
        self.current_calibration_frames = 0
        self._cal_ear_l.clear()
        self._cal_ear_r.clear()
        self._cal_bs_l.clear()
        self._cal_bs_r.clear()
        self._cal_sy.clear()
        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0
        self.baseline_bs_left = 0.0
        self.baseline_bs_right = 0.0
        self.baseline_sy = HYBRID_DEFAULT_BASE_SY
        self.reset_tracking_state()

    def reset_tracking_state(self) -> None:
        self.left_ear_history.clear()
        self.right_ear_history.clear()
        self._has_smooth = False
        self._last_ts_ms = -1
        self.score_l = 0.0
        self.score_r = 0.0
        self._eye_closed_l = False
        self._eye_closed_r = False
        self.prev_state = 0
        self.last_norm_l = INVALID_ANGLE
        self.last_norm_r = INVALID_ANGLE
        if not self.is_calibrated:
            self.current_calibration_frames = 0
            self._cal_ear_l.clear()
            self._cal_ear_r.clear()
            self._cal_bs_l.clear()
            self._cal_bs_r.clear()
            self._cal_sy.clear()
            self.last_detector_mode = "calibrating"
        else:
            self.last_detector_mode = "invalid"

    def update_calibration(
        self,
        raw_l: float,
        raw_r: float,
        head_pitch_deg: float,
        sy: float,
        blendshapes: Optional[Tuple[float, ...]] = None,
    ) -> bool:
        """Accumulate one frame during startup calibration. Returns True on the frame that completes it."""
        self.last_detector_mode = "calibrating"
        if not (math.isfinite(raw_l) and math.isfinite(raw_r)
                and math.isfinite(head_pitch_deg) and math.isfinite(sy)):
            return False

        has_bs = blendshapes is not None
        min_ear = HYBRID_CALIB_MIN_EAR_BS if has_bs else HYBRID_CALIB_MIN_EAR_NO_BS
        if raw_l <= min_ear or raw_r <= min_ear:
            return False
        max_ear = max(raw_l, raw_r)
        if abs(raw_l - raw_r) / max_ear >= HYBRID_CALIB_MAX_ASYM:
            return False
        if abs(head_pitch_deg) >= HYBRID_CALIB_MAX_PITCH_DEG:
            return False
        if has_bs:
            bs_bl_l, bs_bl_r = blendshapes[0], blendshapes[1]
            if bs_bl_l >= HYBRID_CALIB_MAX_BS_BLINK or bs_bl_r >= HYBRID_CALIB_MAX_BS_BLINK:
                return False

        self._cal_ear_l.append(raw_l)
        self._cal_ear_r.append(raw_r)
        self._cal_sy.append(sy)
        if has_bs:
            self._cal_bs_l.append(blendshapes[0])
            self._cal_bs_r.append(blendshapes[1])
        self.current_calibration_frames += 1

        if self.current_calibration_frames >= self.calibration_frame_count:
            self.baseline_left_ear = _trimmed_quantile(
                self._cal_ear_l, HYBRID_CALIB_EAR_TRIM_LO, HYBRID_CALIB_EAR_TRIM_HI, HYBRID_CALIB_EAR_QUANTILE
            )
            self.baseline_right_ear = _trimmed_quantile(
                self._cal_ear_r, HYBRID_CALIB_EAR_TRIM_LO, HYBRID_CALIB_EAR_TRIM_HI, HYBRID_CALIB_EAR_QUANTILE
            )
            self.baseline_sy = _trimmed_quantile(self._cal_sy, 0.0, 1.0, 0.5)
            if self._cal_bs_l and self._cal_bs_r:
                self.baseline_bs_left = _clamp(
                    _trimmed_quantile(
                        self._cal_bs_l, HYBRID_CALIB_BS_TRIM_LO, HYBRID_CALIB_BS_TRIM_HI, HYBRID_CALIB_BS_QUANTILE
                    ),
                    0.0, HYBRID_MAX_BASE_BS,
                )
                self.baseline_bs_right = _clamp(
                    _trimmed_quantile(
                        self._cal_bs_r, HYBRID_CALIB_BS_TRIM_LO, HYBRID_CALIB_BS_TRIM_HI, HYBRID_CALIB_BS_QUANTILE
                    ),
                    0.0, HYBRID_MAX_BASE_BS,
                )
            else:
                self.baseline_bs_left = 0.0
                self.baseline_bs_right = 0.0
            self.is_calibrated = True
            return True
        return False

    def step(
        self,
        raw_l: float,
        raw_r: float,
        head_pitch_deg: float,
        sy: float,
        ts_ms: int,
        blendshapes: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[int, float, float, float, float, str]:
        """
        Compute one post-calibration blink state step.
        Returns (state, score_l, score_r, norm_l, norm_r, detector_mode).
        """
        norm_l = raw_l / self.baseline_left_ear if self.baseline_left_ear > 0.0 else 0.0
        norm_r = raw_r / self.baseline_right_ear if self.baseline_right_ear > 0.0 else 0.0
        self.last_norm_l = norm_l
        self.last_norm_r = norm_r
        self.left_ear_history.append(norm_l)
        self.right_ear_history.append(norm_r)

        has_bs = blendshapes is not None
        bs_bl_l = blendshapes[0] if has_bs else 0.0
        bs_bl_r = blendshapes[1] if has_bs else 0.0
        bs_ld_l = blendshapes[2] if has_bs else 0.0
        bs_ld_r = blendshapes[3] if has_bs else 0.0

        # 1. Pitch & downward-gaze compensated EAR closure score e_L, e_R in [0, 1]
        # Open-eye gate: when an eye is physically closed (norm_ear + base_droop <= HYBRID_LD_GATE_EAR_LO),
        # MediaPipe eyeLookDown rises from upper-eyelid descent rather than downward gaze and must not suppress winks.
        sy_down = _clamp(sy - self.baseline_sy, 0.0, HYBRID_EAR_SY_MAX_DELTA)
        pitch_down = _clamp(head_pitch_deg, 0.0, HYBRID_EAR_PITCH_MAX_DEG) / HYBRID_EAR_PITCH_MAX_DEG
        base_droop = HYBRID_EAR_SY_GAIN * sy_down + HYBRID_EAR_PITCH_GAIN * pitch_down
        gate_span = HYBRID_LD_GATE_EAR_HI - HYBRID_LD_GATE_EAR_LO
        open_gate_l = _clamp01(((norm_l + base_droop) - HYBRID_LD_GATE_EAR_LO) / gate_span)
        open_gate_r = _clamp01(((norm_r + base_droop) - HYBRID_LD_GATE_EAR_LO) / gate_span)

        ld_excess_l = (max(0.0, bs_ld_l - HYBRID_BS_LOOKDOWN_DEADZONE) * open_gate_l) if has_bs else 0.0
        ld_excess_r = (max(0.0, bs_ld_r - HYBRID_BS_LOOKDOWN_DEADZONE) * open_gate_r) if has_bs else 0.0

        droop_l = _clamp(
            base_droop * open_gate_l + HYBRID_EAR_LOOKDOWN_GAIN * ld_excess_l,
            0.0, HYBRID_EAR_MAX_DROOP_COMP,
        )
        droop_r = _clamp(
            base_droop * open_gate_r + HYBRID_EAR_LOOKDOWN_GAIN * ld_excess_r,
            0.0, HYBRID_EAR_MAX_DROOP_COMP,
        )
        norm_l_comp = norm_l + droop_l
        norm_r_comp = norm_r + droop_r

        ear_score_l = _clamp01((HYBRID_EAR_OPEN_REF - norm_l_comp) / HYBRID_EAR_CLOSED_SPAN)
        ear_score_r = _clamp01((HYBRID_EAR_OPEN_REF - norm_r_comp) / HYBRID_EAR_CLOSED_SPAN)

        # 2. Primary blendshape score (resting-baseline-subtracted + open-gated lookDown) + EAR fusion
        if has_bs:
            denom_l = max(HYBRID_BS_DENOM_MIN, 1.0 - self.baseline_bs_left)
            denom_r = max(HYBRID_BS_DENOM_MIN, 1.0 - self.baseline_bs_right)
            bs_score_l = _clamp01((bs_bl_l - self.baseline_bs_left - HYBRID_BS_LOOKDOWN_GAIN * ld_excess_l) / denom_l)
            bs_score_r = _clamp01((bs_bl_r - self.baseline_bs_right - HYBRID_BS_LOOKDOWN_GAIN * ld_excess_r) / denom_r)
            raw_score_l = HYBRID_FUSION_W_BS * bs_score_l + HYBRID_FUSION_W_EAR * ear_score_l
            raw_score_r = HYBRID_FUSION_W_BS * bs_score_r + HYBRID_FUSION_W_EAR * ear_score_r
            mode = "hybrid_bs_ear"
        else:
            raw_score_l = ear_score_l
            raw_score_r = ear_score_r
            mode = "hybrid_ear_only"

        # 3. Asymmetric attack/release wall-clock EMA smoothing (fast 15 ms onset, crisp 30 ms release)
        if not self._has_smooth:
            self.score_l = raw_score_l
            self.score_r = raw_score_r
            self._has_smooth = True
        else:
            if ts_ms >= 0 and self._last_ts_ms >= 0:
                dt_s = (ts_ms - self._last_ts_ms) / 1000.0
            else:
                dt_s = HYBRID_DEFAULT_DT_S
            if dt_s <= 0.0 or dt_s > HYBRID_MAX_GAP_S:
                self.score_l = raw_score_l
                self.score_r = raw_score_r
                self._eye_closed_l = False
                self._eye_closed_r = False
                self.prev_state = 0
            else:
                tau_l = HYBRID_EMA_TAU_ATTACK_S if raw_score_l > self.score_l else HYBRID_EMA_TAU_RELEASE_S
                tau_r = HYBRID_EMA_TAU_ATTACK_S if raw_score_r > self.score_r else HYBRID_EMA_TAU_RELEASE_S
                alpha_l = 1.0 - math.exp(-dt_s / tau_l)
                alpha_r = 1.0 - math.exp(-dt_s / tau_r)
                self.score_l = self.score_l + alpha_l * (raw_score_l - self.score_l)
                self.score_r = self.score_r + alpha_r * (raw_score_r - self.score_r)
        self._last_ts_ms = ts_ms

        # 4. Deterministic Schmitt-trigger hysteresis + contralateral asymmetry arbitration
        c_l = self.score_l
        c_r = self.score_r
        self._eye_closed_l = (c_l >= HYBRID_OPEN_THRESH) if self._eye_closed_l else (c_l >= HYBRID_CLOSE_THRESH)
        self._eye_closed_r = (c_r >= HYBRID_OPEN_THRESH) if self._eye_closed_r else (c_r >= HYBRID_CLOSE_THRESH)

        diff = c_l - c_r
        abs_diff = abs(diff)
        asym_l = HYBRID_WINK_ASYM_EXIT if self.prev_state == 1 else HYBRID_WINK_ASYM_ENTER
        asym_r = HYBRID_WINK_ASYM_EXIT if self.prev_state == 2 else HYBRID_WINK_ASYM_ENTER

        if self._eye_closed_l and self._eye_closed_r:
            if c_l >= HYBRID_CLOSE_THRESH and c_r < HYBRID_WINK_OTHER_MAX and diff >= asym_l:
                state = 1
            elif c_r >= HYBRID_CLOSE_THRESH and c_l < HYBRID_WINK_OTHER_MAX and (-diff) >= asym_r:
                state = 2
            else:
                state = 3
        elif self.prev_state == 3 and c_l >= HYBRID_OPEN_THRESH and c_r >= HYBRID_OPEN_THRESH and abs_diff < HYBRID_SYM_BAND:
            state = 3
        elif self._eye_closed_l and not self._eye_closed_r:
            if diff >= asym_l and c_r < HYBRID_WINK_OTHER_MAX:
                state = 1
            elif c_l >= HYBRID_CLOSE_THRESH and c_r >= HYBRID_OPEN_THRESH and abs_diff < HYBRID_SYM_BAND:
                state = 3
            else:
                state = 0
        elif self._eye_closed_r and not self._eye_closed_l:
            if (-diff) >= asym_r and c_l < HYBRID_WINK_OTHER_MAX:
                state = 2
            elif c_r >= HYBRID_CLOSE_THRESH and c_l >= HYBRID_OPEN_THRESH and abs_diff < HYBRID_SYM_BAND:
                state = 3
            else:
                state = 0
        else:
            state = 0

        self.prev_state = state
        self.last_detector_mode = mode

        # 5. Slow adaptive open-eye EMA (updates ONLY when clearly open, symmetric, neutral pitch, state == 0)
        min_ear = HYBRID_CALIB_MIN_EAR_BS if has_bs else HYBRID_CALIB_MIN_EAR_NO_BS
        if (
            state == 0
            and raw_l > min_ear
            and raw_r > min_ear
            and (abs(raw_l - raw_r) / max(raw_l, raw_r)) < HYBRID_ADAPT_MAX_ASYM
            and abs(head_pitch_deg) < HYBRID_ADAPT_MAX_PITCH_DEG
            and raw_score_l < HYBRID_ADAPT_MAX_RAW_SCORE
            and raw_score_r < HYBRID_ADAPT_MAX_RAW_SCORE
            and (not has_bs or (
                bs_bl_l < HYBRID_ADAPT_MAX_BS_BLINK
                and bs_bl_r < HYBRID_ADAPT_MAX_BS_BLINK
                and bs_ld_l < HYBRID_ADAPT_MAX_BS_LOOKDOWN
                and bs_ld_r < HYBRID_ADAPT_MAX_BS_LOOKDOWN
            ))
        ):
            a = HYBRID_ADAPT_ALPHA
            self.baseline_left_ear = (1.0 - a) * self.baseline_left_ear + a * raw_l
            self.baseline_right_ear = (1.0 - a) * self.baseline_right_ear + a * raw_r
            if has_bs:
                self.baseline_bs_left = _clamp((1.0 - a) * self.baseline_bs_left + a * bs_bl_l, 0.0, HYBRID_MAX_BASE_BS)
                self.baseline_bs_right = _clamp((1.0 - a) * self.baseline_bs_right + a * bs_bl_r, 0.0, HYBRID_MAX_BASE_BS)

        return state, self.score_l, self.score_r, norm_l, norm_r, mode


class GazeIntent:
    def __init__(
        self,
        diag_log_path: Optional[str] = None,
        blink_diag_log_path: Optional[str] = None,
        blink_mode: str = "hybrid",
    ):
        if blink_mode not in ("hybrid", "legacy"):
            raise ValueError(f"Unsupported blink_mode: {blink_mode!r} (expected 'hybrid' or 'legacy')")
        self.blink_mode = blink_mode
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get(
            'AI_PIPELINE_ROOT',
            os.path.abspath(os.path.join(self.script_dir, '..'))
        )

        # --- 1. LOAD ALL 3 NATIVE AI MODELS ---
        # gesture_model.dll is preserved untouched as an explicit legacy fallback only.
        wink_model_path = os.path.join(self.script_dir, 'models', 'gesture_model.dll')
        pitch_model_path = os.path.join(self.script_dir, 'models', 'gaze_pitch.dll')
        yaw_model_path = os.path.join(self.script_dir, 'models', 'gaze_yaw.dll')

        for path in [wink_model_path, pitch_model_path, yaw_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[CRITICAL ERROR] Production DLL missing at:\n{path}")

        self.wink_model = tl2cgen.Predictor(wink_model_path)
        self.pitch_model = tl2cgen.Predictor(pitch_model_path)
        self.yaw_model = tl2cgen.Predictor(yaw_model_path)

        # --- 2. MEDIAPIPE INITIALIZATION ---
        mp_model_path = os.path.abspath(os.path.join(project_root, '..', 'assets', 'face_landmarker.task'))
        base_options = python.BaseOptions(model_asset_path=mp_model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- 3. TEMPORAL STATE & CALIBRATION ---
        self.hybrid_blink = HybridBlinkDetector()
        self._legacy_ear_state = EarFeatureState()
        # Keep ear_state pointing at the active detector for backward-compatible test introspection
        self.ear_state = self.hybrid_blink if self.blink_mode == "hybrid" else self._legacy_ear_state

        # --- 4. OPT-IN DIAGNOSTIC TELEMETRY LOGGERS (ZERO OVERHEAD WHEN DISABLED) ---
        self._frame_id = 0
        self._diag_file = None
        self._diag_writer = None
        resolved_log = diag_log_path or os.environ.get("GAZE_DIAG_LOG")
        if resolved_log:
            log_abs = os.path.abspath(resolved_log)
            os.makedirs(os.path.dirname(log_abs), exist_ok=True)
            self._diag_file = open(log_abs, mode="w", newline="", encoding="utf-8")
            self._diag_writer = csv.writer(self._diag_file)
            self._diag_writer.writerow(_DIAG_FIELD_NAMES)

        self._blink_diag_file = None
        self._blink_diag_writer = None
        resolved_blink_log = blink_diag_log_path or os.environ.get("BLINK_DIAG_LOG")
        if resolved_blink_log:
            blog_abs = os.path.abspath(resolved_blink_log)
            os.makedirs(os.path.dirname(blog_abs), exist_ok=True)
            self._blink_diag_file = open(blog_abs, mode="w", newline="", encoding="utf-8")
            self._blink_diag_writer = csv.writer(self._blink_diag_file)
            self._blink_diag_writer.writerow(_BLINK_DIAG_FIELD_NAMES)

    @property
    def is_calibrated(self) -> bool:
        return self.ear_state.is_calibrated

    def reset_tracking_state(self) -> None:
        self.hybrid_blink.reset_tracking_state()
        self._legacy_ear_state.reset_tracking_state()

    def _write_diag_row(self, payload: GazeDiagnosticPayload) -> None:
        if self._diag_writer is not None:
            row = []
            for name in _DIAG_FIELD_NAMES:
                val = getattr(payload, name)
                if name == "capture_s":
                    row.append("nan" if math.isnan(val) else f"{val:.9f}")
                else:
                    row.append(val)
            self._diag_writer.writerow(row)

    def _write_blink_diag_row(self, payload: BlinkDiagnosticPayload) -> None:
        if self._blink_diag_writer is not None:
            row = []
            for name in _BLINK_DIAG_FIELD_NAMES:
                val = getattr(payload, name)
                if name == "capture_s":
                    row.append("nan" if math.isnan(val) else f"{val:.9f}")
                else:
                    row.append(val)
            self._blink_diag_writer.writerow(row)

    def shutdown(self):
        if getattr(self, '_diag_file', None) is not None:
            try:
                self._diag_file.flush()
                self._diag_file.close()
            finally:
                self._diag_file = None
                self._diag_writer = None
        if getattr(self, '_blink_diag_file', None) is not None:
            try:
                self._blink_diag_file.flush()
                self._blink_diag_file.close()
            finally:
                self._blink_diag_file = None
                self._blink_diag_writer = None
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        self.shutdown()

    def _emit_invalid_diag(self, frame_id: int, timestamp_ms: int) -> None:
        if self._diag_writer is not None:
            self._write_diag_row(GazeDiagnosticPayload(
                frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                detected=0, state=0,
                left_iris_x=INVALID_ANGLE, right_iris_x=INVALID_ANGLE, sx=INVALID_ANGLE,
                left_iris_y=INVALID_ANGLE, right_iris_y=INVALID_ANGLE, sy=INVALID_ANGLE,
                head_pitch_deg=INVALID_ANGLE, head_yaw_deg=INVALID_ANGLE, head_roll_deg=INVALID_ANGLE,
                rel_pitch_deg=INVALID_ANGLE, rel_yaw_deg=INVALID_ANGLE,
                abs_pitch_deg=INVALID_ANGLE, abs_yaw_deg=INVALID_ANGLE,
                norm_x=-1.0, norm_y=-1.0,
                lm33_x=INVALID_ANGLE, lm33_y=INVALID_ANGLE,
                lm133_x=INVALID_ANGLE, lm133_y=INVALID_ANGLE,
                lm468_x=INVALID_ANGLE, lm468_y=INVALID_ANGLE,
                lm362_x=INVALID_ANGLE, lm362_y=INVALID_ANGLE,
                lm263_x=INVALID_ANGLE, lm263_y=INVALID_ANGLE,
                lm473_x=INVALID_ANGLE, lm473_y=INVALID_ANGLE,
            ))
        if self._blink_diag_writer is not None:
            self._write_blink_diag_row(BlinkDiagnosticPayload(
                frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                detected=0, state=0, detector_mode="invalid",
                raw_ear_l=INVALID_ANGLE, raw_ear_r=INVALID_ANGLE,
                base_ear_l=self.ear_state.baseline_left_ear,
                base_ear_r=self.ear_state.baseline_right_ear,
                norm_ear_l=INVALID_ANGLE, norm_ear_r=INVALID_ANGLE,
                bs_blink_l=INVALID_ANGLE, bs_blink_r=INVALID_ANGLE,
                bs_lookdown_l=INVALID_ANGLE, bs_lookdown_r=INVALID_ANGLE,
                bs_squint_l=INVALID_ANGLE, bs_squint_r=INVALID_ANGLE,
                score_l=INVALID_ANGLE, score_r=INVALID_ANGLE,
                sy=INVALID_ANGLE, head_pitch_deg=INVALID_ANGLE,
            ))

    def process_frame(
        self, rgb_frame_array: np.ndarray, timestamp_ms: int
    ) -> Tuple[float, float, float, float, int, Optional[Any]]:
        """
        Returns (abs_yaw, abs_pitch, head_yaw, head_pitch, state, landmarks).
        When calibrating (state == -1) or when no face / non-finite input is detected
        (state == 0, landmarks is None), all four angles are NaN (INVALID_ANGLE) so invalid
        coordinates never enter downstream DSP filters. Use is_valid_output(abs_yaw, abs_pitch)
        to test validity.
        """
        frame_id = self._frame_id
        self._frame_id += 1
        if rgb_frame_array is None or getattr(rgb_frame_array, "ndim", 0) < 2:
            self.reset_tracking_state()
            self._emit_invalid_diag(frame_id, timestamp_ms)
            return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None
        frame_h, frame_w = int(rgb_frame_array.shape[0]), int(rgb_frame_array.shape[1])
        if frame_w <= 0 or frame_h <= 0:
            self.reset_tracking_state()
            self._emit_invalid_diag(frame_id, timestamp_ms)
            return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_array)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks and results.facial_transformation_matrixes:
            landmarks = results.face_landmarks[0]
            pose_matrix = results.facial_transformation_matrixes[0]
            if not _landmarks_and_pose_finite(landmarks, pose_matrix):
                self.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # Optional eye blendshapes (strictly validated [0.0, 1.0] if provided)
            raw_bs_list = getattr(results, "face_blendshapes", None)
            blendshapes: Optional[Tuple[float, ...]] = None
            if raw_bs_list:
                blendshapes = extract_eye_blendshapes(raw_bs_list[0])
                if blendshapes is None:
                    self.reset_tracking_state()
                    self._emit_invalid_diag(frame_id, timestamp_ms)
                    return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # 1. 3D EAR (degenerate |p1-p4| <= 1e-12 returns NaN -> rejected cleanly)
            raw_left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
            raw_right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
            if not (math.isfinite(raw_left_ear) and math.isfinite(raw_right_ear)):
                self.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # 2. Extract MPIIGaze features + head pose BEFORE calibration/blink evaluation
            l_x = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "x")
            r_x = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "x")
            student_x = (l_x + r_x) / 2.0

            l_y = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "y")
            r_y = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "y")
            student_y = (l_y + r_y) / 2.0

            try:
                head_pitch, head_yaw, head_roll = get_head_pose_angles(pose_matrix)
            except ValueError:
                self.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            if not all(math.isfinite(v) for v in (
                l_x, r_x, student_x, l_y, r_y, student_y, head_pitch, head_yaw, head_roll
            )):
                self.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # 3. Calibration phase (returns state == -1 and NaN sentinel angles, incl. completing frame)
            if not self.is_calibrated:
                if self.blink_mode == "hybrid":
                    self.hybrid_blink.update_calibration(
                        raw_left_ear, raw_right_ear, head_pitch, student_y, blendshapes
                    )
                else:
                    self._legacy_ear_state.update(raw_left_ear, raw_right_ear)

                if self._diag_writer is not None:
                    self._write_diag_row(GazeDiagnosticPayload(
                        frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                        detected=0, state=-1,
                        left_iris_x=l_x, right_iris_x=r_x, sx=student_x,
                        left_iris_y=l_y, right_iris_y=r_y, sy=student_y,
                        head_pitch_deg=head_pitch, head_yaw_deg=head_yaw, head_roll_deg=head_roll,
                        rel_pitch_deg=INVALID_ANGLE, rel_yaw_deg=INVALID_ANGLE,
                        abs_pitch_deg=INVALID_ANGLE, abs_yaw_deg=INVALID_ANGLE,
                        norm_x=-1.0, norm_y=-1.0,
                        lm33_x=float(landmarks[33].x), lm33_y=float(landmarks[33].y),
                        lm133_x=float(landmarks[133].x), lm133_y=float(landmarks[133].y),
                        lm468_x=float(landmarks[468].x), lm468_y=float(landmarks[468].y),
                        lm362_x=float(landmarks[362].x), lm362_y=float(landmarks[362].y),
                        lm263_x=float(landmarks[263].x), lm263_y=float(landmarks[263].y),
                        lm473_x=float(landmarks[473].x), lm473_y=float(landmarks[473].y),
                    ))
                if self._blink_diag_writer is not None:
                    self._write_blink_diag_row(BlinkDiagnosticPayload(
                        frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                        detected=0, state=-1, detector_mode="calibrating",
                        raw_ear_l=raw_left_ear, raw_ear_r=raw_right_ear,
                        base_ear_l=self.ear_state.baseline_left_ear,
                        base_ear_r=self.ear_state.baseline_right_ear,
                        norm_ear_l=INVALID_ANGLE, norm_ear_r=INVALID_ANGLE,
                        bs_blink_l=blendshapes[0] if blendshapes else INVALID_ANGLE,
                        bs_blink_r=blendshapes[1] if blendshapes else INVALID_ANGLE,
                        bs_lookdown_l=blendshapes[2] if blendshapes else INVALID_ANGLE,
                        bs_lookdown_r=blendshapes[3] if blendshapes else INVALID_ANGLE,
                        bs_squint_l=blendshapes[4] if blendshapes else INVALID_ANGLE,
                        bs_squint_r=blendshapes[5] if blendshapes else INVALID_ANGLE,
                        score_l=INVALID_ANGLE, score_r=INVALID_ANGLE,
                        sy=student_y, head_pitch_deg=head_pitch,
                    ))
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, -1, landmarks

            # 4. Predict Screen Gaze (Pitch & Yaw) — validated finite BEFORE mutating blink state
            gaze_features = np.array(
                [[student_x, student_y, head_pitch, head_yaw, head_roll]], dtype=np.float32
            )
            gaze_batch = tl2cgen.DMatrix(gaze_features)

            # Get Relative Eye-in-Socket predictions
            rel_pitch = float(self.pitch_model.predict(gaze_batch).item())
            rel_yaw = float(self.yaw_model.predict(gaze_batch).item())

            # RECOMBINE: Calculate Absolute World Gaze
            abs_pitch = rel_pitch + head_pitch
            abs_yaw = rel_yaw + head_yaw

            if not all(math.isfinite(v) for v in (rel_pitch, rel_yaw, abs_pitch, abs_yaw)):
                self.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # 5. Blink / Wink / Closure Intent State
            if self.blink_mode == "hybrid":
                state, score_l, score_r, norm_l, norm_r, det_mode = self.hybrid_blink.step(
                    raw_left_ear, raw_right_ear, head_pitch, student_y, int(timestamp_ms), blendshapes
                )
            else:
                wink_features = self._legacy_ear_state.update(raw_left_ear, raw_right_ear)
                state = safe_classify_legacy_wink(self.wink_model, wink_features) if wink_features is not None else 0
                norm_l = wink_features[0] if wink_features is not None else INVALID_ANGLE
                norm_r = wink_features[3] if wink_features is not None else INVALID_ANGLE
                score_l = INVALID_ANGLE
                score_r = INVALID_ANGLE
                det_mode = "legacy_rf"

            nx = angle_to_norm_coord(abs_yaw)
            ny = angle_to_norm_coord(abs_pitch)
            det_flag = 1 if (nx >= 0.0 and ny >= 0.0) else 0

            if self._diag_writer is not None:
                self._write_diag_row(GazeDiagnosticPayload(
                    frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                    detected=det_flag, state=state,
                    left_iris_x=l_x, right_iris_x=r_x, sx=student_x,
                    left_iris_y=l_y, right_iris_y=r_y, sy=student_y,
                    head_pitch_deg=head_pitch, head_yaw_deg=head_yaw, head_roll_deg=head_roll,
                    rel_pitch_deg=rel_pitch, rel_yaw_deg=rel_yaw,
                    abs_pitch_deg=abs_pitch, abs_yaw_deg=abs_yaw,
                    norm_x=nx, norm_y=ny,
                    lm33_x=float(landmarks[33].x), lm33_y=float(landmarks[33].y),
                    lm133_x=float(landmarks[133].x), lm133_y=float(landmarks[133].y),
                    lm468_x=float(landmarks[468].x), lm468_y=float(landmarks[468].y),
                    lm362_x=float(landmarks[362].x), lm362_y=float(landmarks[362].y),
                    lm263_x=float(landmarks[263].x), lm263_y=float(landmarks[263].y),
                    lm473_x=float(landmarks[473].x), lm473_y=float(landmarks[473].y),
                ))

            if self._blink_diag_writer is not None:
                self._write_blink_diag_row(BlinkDiagnosticPayload(
                    frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                    detected=det_flag, state=state, detector_mode=det_mode,
                    raw_ear_l=raw_left_ear, raw_ear_r=raw_right_ear,
                    base_ear_l=self.ear_state.baseline_left_ear,
                    base_ear_r=self.ear_state.baseline_right_ear,
                    norm_ear_l=norm_l, norm_ear_r=norm_r,
                    bs_blink_l=blendshapes[0] if blendshapes else INVALID_ANGLE,
                    bs_blink_r=blendshapes[1] if blendshapes else INVALID_ANGLE,
                    bs_lookdown_l=blendshapes[2] if blendshapes else INVALID_ANGLE,
                    bs_lookdown_r=blendshapes[3] if blendshapes else INVALID_ANGLE,
                    bs_squint_l=blendshapes[4] if blendshapes else INVALID_ANGLE,
                    bs_squint_r=blendshapes[5] if blendshapes else INVALID_ANGLE,
                    score_l=score_l, score_r=score_r,
                    sy=student_y, head_pitch_deg=head_pitch,
                ))

            return abs_yaw, abs_pitch, head_yaw, head_pitch, state, landmarks

        # Issue #4 Fix: Face lost -> clear temporal state & return NaN sentinel angles
        self.reset_tracking_state()
        self._emit_invalid_diag(frame_id, timestamp_ms)
        return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None
