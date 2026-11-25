import csv
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, fields
from typing import Tuple, Optional, Any

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

# --- Temporal parameters ---
EAR_WINDOW_SIZE = 10
CALIBRATION_FRAMES = 30

# --- Numerical guards (mirrored in core_engine/include/gaze_math.hpp) ---
EAR_DIST_EPS = 1e-12       # horizontal eye distance below this -> EAR = 0
CALIB_EAR_EPS = 1e-6       # calibration ignores frames whose raw EAR is not above this
IRIS_BASELINE_EPS = 1e-12  # squared eye-corner baseline below this -> neutral iris ratio

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
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_ear(face_landmarks, indices, frame_w: int, frame_h: int) -> float:
    p1, p2, p3, p4, p5, p6 = (face_landmarks[i] for i in indices)
    v_dist_1 = get_3d_dist(p2, p6, frame_w, frame_h)
    v_dist_2 = get_3d_dist(p3, p5, frame_w, frame_h)
    h_dist = get_3d_dist(p1, p4, frame_w, frame_h)
    return (v_dist_1 + v_dist_2) / (2.0 * h_dist) if h_dist > EAR_DIST_EPS else 0.0


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


class EarFeatureState:
    """
    Temporal EAR state shared by the runtime and the data collector: 30-frame dynamic baseline
    calibration, per-eye normalization, and the 10-frame rolling window.
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
        if not self.is_calibrated:
            # Guard against corrupted/zero EAR frames during calibration
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
        """
        Issue #4 Fix: Clears temporal EAR histories on face tracking loss so stale
        pre-loss winks/blinks do not contaminate features upon face reacquisition.
        If initial 30-frame calibration was incomplete when tracking was lost,
        restart the calibration accumulation cleanly.
        """
        self.left_ear_history.clear()
        self.right_ear_history.clear()
        if not self.is_calibrated:
            self.current_calibration_frames = 0
            self.left_ear_sum = 0.0
            self.right_ear_sum = 0.0


class GazeIntent:
    def __init__(self, diag_log_path: Optional[str] = None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get(
            'AI_PIPELINE_ROOT',
            os.path.abspath(os.path.join(self.script_dir, '..'))
        )

        # --- 1. LOAD ALL 3 NATIVE AI MODELS ---
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
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- 3. TEMPORAL STATE & CALIBRATION ---
        self.ear_state = EarFeatureState()

        # --- 4. OPT-IN DIAGNOSTIC TELEMETRY LOGGER (ZERO OVERHEAD WHEN DISABLED) ---
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

    @property
    def is_calibrated(self) -> bool:
        return self.ear_state.is_calibrated

    def reset_tracking_state(self) -> None:
        self.ear_state.reset_tracking_state()

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

    def shutdown(self):
        if getattr(self, '_diag_file', None) is not None:
            try:
                self._diag_file.flush()
                self._diag_file.close()
            finally:
                self._diag_file = None
                self._diag_writer = None
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
            self.ear_state.reset_tracking_state()
            self._emit_invalid_diag(frame_id, timestamp_ms)
            return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None
        frame_h, frame_w = int(rgb_frame_array.shape[0]), int(rgb_frame_array.shape[1])
        if frame_w <= 0 or frame_h <= 0:
            self.ear_state.reset_tracking_state()
            self._emit_invalid_diag(frame_id, timestamp_ms)
            return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_array)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks and results.facial_transformation_matrixes:
            landmarks = results.face_landmarks[0]
            pose_matrix = results.facial_transformation_matrixes[0]
            if not _landmarks_and_pose_finite(landmarks, pose_matrix):
                self.ear_state.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            # 1. Blink Calibration & ML Inference
            raw_left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
            raw_right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
            if not (math.isfinite(raw_left_ear) and math.isfinite(raw_right_ear)):
                self.ear_state.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            wink_features = self.ear_state.update(raw_left_ear, raw_right_ear)
            if wink_features is None:
                if self._diag_writer is not None:
                    l_x = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "x")
                    r_x = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "x")
                    l_y = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "y")
                    r_y = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "y")
                    hp, hy, hr = get_head_pose_angles(pose_matrix)
                    self._write_diag_row(GazeDiagnosticPayload(
                        frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                        detected=0, state=-1,
                        left_iris_x=l_x, right_iris_x=r_x, sx=(l_x + r_x) / 2.0,
                        left_iris_y=l_y, right_iris_y=r_y, sy=(l_y + r_y) / 2.0,
                        head_pitch_deg=hp, head_yaw_deg=hy, head_roll_deg=hr,
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
                # Issue #3 Fix: NaN sentinel angles during calibration
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, -1, landmarks

            # 2. Extract MPIIGaze Features (validate before state inference)
            l_x = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "x")
            r_x = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "x")
            student_x = (l_x + r_x) / 2.0

            l_y = get_iris_ratio(landmarks, L_CORNERS, LEFT_IRIS, "y")
            r_y = get_iris_ratio(landmarks, R_CORNERS, RIGHT_IRIS, "y")
            student_y = (l_y + r_y) / 2.0

            try:
                head_pitch, head_yaw, head_roll = get_head_pose_angles(pose_matrix)
            except ValueError:
                self.ear_state.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            if not all(math.isfinite(v) for v in (
                l_x, r_x, student_x, l_y, r_y, student_y, head_pitch, head_yaw, head_roll
            )):
                self.ear_state.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            wink_batch = tl2cgen.DMatrix(np.array([wink_features], dtype=np.float32))
            state = int(self.wink_model.predict(wink_batch)[0].argmax())

            # 3. Predict Screen Gaze (Pitch & Yaw)
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
                self.ear_state.reset_tracking_state()
                self._emit_invalid_diag(frame_id, timestamp_ms)
                return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None

            if self._diag_writer is not None:
                nx = angle_to_norm_coord(abs_yaw)
                ny = angle_to_norm_coord(abs_pitch)
                self._write_diag_row(GazeDiagnosticPayload(
                    frame_id=frame_id, ts_ms=timestamp_ms, capture_s=timestamp_ms / 1000.0,
                    detected=1 if (nx >= 0.0 and ny >= 0.0) else 0, state=state,
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

            return abs_yaw, abs_pitch, head_yaw, head_pitch, state, landmarks

        # Issue #4 Fix: Face lost -> clear temporal EAR window & return NaN sentinel angles
        self.ear_state.reset_tracking_state()
        self._emit_invalid_diag(frame_id, timestamp_ms)
        return INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, INVALID_ANGLE, 0, None
