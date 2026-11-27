"""Headless unit tests for Engineer 3 (Gaze & Intent pipeline: ai_pipeline/gaze_intent/)."""
from __future__ import annotations

import csv
import math
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure Windows Python's Library\bin exists before importing tl2cgen (non-Conda Windows Python guard).
os.makedirs(os.path.join(sys.prefix, "Library", "bin"), exist_ok=True)
import tl2cgen  # noqa: E402

GAZE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GAZE_DIR.parents[1]
MODELS_DIR = GAZE_DIR / "models"

if str(GAZE_DIR) not in sys.path:
    sys.path.insert(0, str(GAZE_DIR))

import gaze_intent as gi  # noqa: E402
import run_gaze as rg  # noqa: E402
from gaze_intent import (  # noqa: E402
    CALIB_EAR_EPS,
    CALIBRATION_FRAMES,
    EAR_DIST_EPS,
    EAR_WINDOW_SIZE,
    EYE_BLENDSHAPE_NAMES,
    INVALID_ANGLE,
    IRIS_BASELINE_EPS,
    L_CORNERS,
    LEFT_EYE,
    LEFT_IRIS,
    R_CORNERS,
    RIGHT_EYE,
    RIGHT_IRIS,
    SPAN_DEG,
    BlinkDiagnosticPayload,
    EarFeatureState,
    GazeDiagnosticPayload,
    GazeIntent,
    HybridBlinkDetector,
    _BLINK_DIAG_FIELD_NAMES,
    _DIAG_FIELD_NAMES,
    angle_to_norm_coord,
    calculate_ear,
    extract_eye_blendshapes,
    extract_gaze_features,
    get_3d_dist,
    get_head_pose_angles,
    get_iris_ratio,
    is_valid_output,
    safe_classify_legacy_wink,
    window_stats,
)


# ---------------------------------------------------------------------------
# Synthetic Face Landmark & Head-Pose Matrix Helpers (478 points, no webcam)
# ---------------------------------------------------------------------------
N_FACE_LANDMARKS = 478
_HALF_W = 0.035
_APERTURE = 0.012
_LEFT_VERT = (159, 145)
_RIGHT_VERT = (386, 374)


def _place_synthetic_eye(
    pts: np.ndarray,
    eye_ids: tuple[int, int, int, int, int, int],
    vert_ids: tuple[int, int],
    iris_idx: int,
    cx: float,
    cy: float,
    openness: float,
    iris_xy: tuple[float, float],
) -> None:
    """Populate the 6 eye-contour points, vertical lids, and iris center in normalized 2D coordinates.

    Matches the exact landmark semantics of gaze_intent.py:
    - eye_ids = (corner_1, top_1, top_2, corner_2, bottom_2, bottom_1)
    - get_iris_ratio projects the iris onto the corner_1 -> corner_2 baseline:
      iris_xy[0] is the horizontal ratio along the baseline (0.5 = centered),
      iris_xy[1] is the signed perpendicular offset in units of baseline length (0.0 = on baseline).
    """
    ca, ta, tb, cb, bb, ba = eye_ids
    ap = _APERTURE * openness
    pts[ca] = (cx - _HALF_W, cy, 0.0)
    pts[cb] = (cx + _HALF_W, cy, 0.0)
    pts[ta] = (cx - 0.4 * _HALF_W, cy - ap, 0.0)
    pts[ba] = (cx - 0.4 * _HALF_W, cy + ap, 0.0)
    pts[tb] = (cx + 0.4 * _HALF_W, cy - ap, 0.0)
    pts[bb] = (cx + 0.4 * _HALF_W, cy + ap, 0.0)
    pts[vert_ids[0]] = (cx, cy - 1.1 * ap, 0.0)
    pts[vert_ids[1]] = (cx, cy + 1.1 * ap, 0.0)

    baseline_len = 2.0 * _HALF_W
    ix = (cx - _HALF_W) + iris_xy[0] * baseline_len
    iy = cy + iris_xy[1] * baseline_len
    pts[iris_idx] = (ix, iy, 0.0)


def make_synthetic_landmarks(
    k_left: float = 1.0,
    k_right: float = 1.0,
    iris_left: tuple[float, float] = (0.5, 0.0),
    iris_right: tuple[float, float] | None = None,
    center: tuple[float, float] = (0.5, 0.5),
    size: float = 0.25,
    rng: np.random.Generator | None = None,
    noise: float = 0.0,
) -> list[types.SimpleNamespace]:
    """Build 478 synthetic MediaPipe-style face landmark objects with `.x`, `.y`, `.z` attributes."""
    if iris_right is None:
        iris_right = iris_left

    base_rng = np.random.default_rng(1234)
    pts = np.zeros((N_FACE_LANDMARKS, 3), dtype=np.float64)
    pts[:, 0] = center[0] + base_rng.uniform(-1.0, 1.0, N_FACE_LANDMARKS) * size
    pts[:, 1] = center[1] + base_rng.uniform(-1.0, 1.0, N_FACE_LANDMARKS) * size
    pts[:, 2] = base_rng.uniform(-0.02, 0.02, N_FACE_LANDMARKS)

    _place_synthetic_eye(
        pts, LEFT_EYE, _LEFT_VERT, LEFT_IRIS, center[0] - 0.08, center[1] - 0.05, k_left, iris_left
    )
    _place_synthetic_eye(
        pts, RIGHT_EYE, _RIGHT_VERT, RIGHT_IRIS, center[0] + 0.08, center[1] - 0.05, k_right, iris_right
    )

    if rng is not None and noise > 0.0:
        pts[:, :2] += rng.normal(0.0, noise, (N_FACE_LANDMARKS, 2))
        pts[:, 2] += rng.normal(0.0, noise * 0.25, N_FACE_LANDMARKS)

    return [types.SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]


def make_pose_matrix(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Construct a 4x4 homogeneous head-pose matrix from Euler angles (degrees) matching get_head_pose_angles."""
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = (rz @ ry @ rx).astype(np.float32)
    return mat


def make_eye_blendshapes(**overrides: float) -> list[types.SimpleNamespace]:
    """Build MediaPipe-style face_blendshapes category objects with canonical ARKit names."""
    vals = {
        "eyeBlinkRight": 0.05,
        "eyeBlinkLeft": 0.05,
        "eyeLookDownRight": 0.10,
        "eyeLookDownLeft": 0.10,
        "eyeSquintRight": 0.04,
        "eyeSquintLeft": 0.04,
        "eyeWideRight": 0.02,
        "eyeWideLeft": 0.02,
    }
    vals.update(overrides)
    return [types.SimpleNamespace(category_name=k, score=float(v)) for k, v in vals.items()]


class StubFaceDetector:
    """Deterministic in-memory MediaPipe FaceLandmarker stub for headless GazeIntent tests."""

    def __init__(self) -> None:
        self.next_landmarks: list[Any] | None = make_synthetic_landmarks()
        self.next_pose: np.ndarray | None = make_pose_matrix()
        self.next_blendshapes: list[Any] | None = None
        self.received_timestamps: list[int] = []
        self.closed: bool = False

    def detect_for_video(self, mp_image, timestamp_ms: int):
        self.received_timestamps.append(timestamp_ms)
        has_face = self.next_landmarks is not None and len(self.next_landmarks) > 0
        has_pose = self.next_pose is not None
        has_bs = has_face and (self.next_blendshapes is not None)
        return types.SimpleNamespace(
            face_landmarks=[self.next_landmarks] if has_face else [],
            facial_transformation_matrixes=[self.next_pose] if (has_face and has_pose) else [],
            face_blendshapes=[self.next_blendshapes] if has_bs else [],
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def mocked_gaze_tracker(tmp_path):
    """Instantiate GazeIntent with the real Treelite DLLs and a mocked MediaPipe FaceLandmarker."""
    stub_detector = StubFaceDetector()
    with patch("gaze_intent.vision.FaceLandmarker.create_from_options", return_value=stub_detector):
        tracker = GazeIntent()
        try:
            yield tracker, stub_detector
        finally:
            tracker.shutdown()


# ===========================================================================
# Pillar 1: Treelite Model DLL ABI & Sanity Checks
# ===========================================================================
class TestTreeliteModelDlls:
    def test_dll_files_exist_and_non_empty(self):
        for dll_name in ("gesture_model.dll", "gaze_pitch.dll", "gaze_yaw.dll"):
            dll_path = MODELS_DIR / dll_name
            assert dll_path.is_file(), f"Missing deployed model DLL: {dll_path}"
            assert dll_path.stat().st_size > 0, f"Empty model DLL: {dll_path}"

    def test_gesture_model_abi_and_canonical_classes(self):
        predictor = tl2cgen.Predictor(str(MODELS_DIR / "gesture_model.dll"))
        assert predictor.num_feature == 6
        assert list(predictor.num_class) == [4]

        canonical_inputs = {
            0: [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],        # Neutral open eyes
            1: [0.25, 0.25, 0.10, 1.0, 1.0, 0.0],     # Sustained left wink
            2: [1.0, 1.0, 0.0, 0.25, 0.25, 0.10],     # Sustained right wink
            3: [0.25, 0.25, 0.10, 0.25, 0.25, 0.10],  # Sustained both-eyes closed
        }
        for expected_class, feat in canonical_inputs.items():
            dmat = tl2cgen.DMatrix(np.array([feat], dtype=np.float32))
            out = predictor.predict(dmat)
            probs = out[0].reshape(-1)
            assert probs.shape == (4,)
            assert np.all(np.isfinite(probs))
            assert np.all((probs >= 0.0) & (probs <= 1.0))
            assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-5, abs_tol=1e-5)
            assert int(np.argmax(probs)) == expected_class

    def test_gaze_pitch_and_yaw_abi_bounds_and_golden_determinism(self):
        pitch_pred = tl2cgen.Predictor(str(MODELS_DIR / "gaze_pitch.dll"))
        yaw_pred = tl2cgen.Predictor(str(MODELS_DIR / "gaze_yaw.dll"))

        assert pitch_pred.num_feature == 5
        assert yaw_pred.num_feature == 5
        assert list(pitch_pred.num_class) == [1]
        assert list(yaw_pred.num_class) == [1]

        # Golden reference vector: neutral centered iris (sx=0.5, sy=0.0) and zero head pose
        neutral_vec = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        dmat = tl2cgen.DMatrix(neutral_vec)
        pitch_val_1 = float(pitch_pred.predict(dmat).item())
        yaw_val_1 = float(yaw_pred.predict(dmat).item())

        # Bit-exact determinism across repeated predictions
        for _ in range(10):
            assert float(pitch_pred.predict(dmat).item()) == pitch_val_1
            assert float(yaw_pred.predict(dmat).item()) == yaw_val_1

        # Golden regression values for deployed models on first-full-working-demo
        assert math.isclose(pitch_val_1, -17.71785574, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(yaw_val_1, 0.00726980, rel_tol=1e-5, abs_tol=1e-5)

        # Batch of realistic feature vectors stays finite and within plausible angular bounds
        rng = np.random.default_rng(42)
        batch = np.column_stack([
            rng.uniform(0.30, 0.70, 32),
            rng.uniform(-0.15, 0.15, 32),
            rng.uniform(-25.0, 25.0, 32),
            rng.uniform(-35.0, 35.0, 32),
            rng.uniform(-15.0, 15.0, 32),
        ]).astype(np.float32)
        batch_dmat = tl2cgen.DMatrix(batch)
        p_out = pitch_pred.predict(batch_dmat).reshape(-1)
        y_out = yaw_pred.predict(batch_dmat).reshape(-1)

        assert p_out.shape == (32,)
        assert y_out.shape == (32,)
        assert np.all(np.isfinite(p_out))
        assert np.all(np.isfinite(y_out))
        assert np.all(np.abs(p_out) < 90.0)
        assert np.all(np.abs(y_out) < 90.0)


# ===========================================================================
# Pillar 2: Feature Extraction & Geometry (gaze_intent.py & mpii_processor.py)
# ===========================================================================
class TestFeatureExtractionAndGeometry:
    def test_iris_ratio_center_and_horizontal_vertical_shifts(self):
        lm_center = make_synthetic_landmarks(iris_left=(0.5, 0.0), iris_right=(0.5, 0.0))
        assert math.isclose(get_iris_ratio(lm_center, L_CORNERS, LEFT_IRIS, "x"), 0.5, abs_tol=1e-6)
        assert math.isclose(get_iris_ratio(lm_center, R_CORNERS, RIGHT_IRIS, "x"), 0.5, abs_tol=1e-6)
        assert math.isclose(get_iris_ratio(lm_center, L_CORNERS, LEFT_IRIS, "y"), 0.0, abs_tol=1e-6)
        assert math.isclose(get_iris_ratio(lm_center, R_CORNERS, RIGHT_IRIS, "y"), 0.0, abs_tol=1e-6)

        # Horizontal shift: 0.30 vs 0.70 symmetric around 0.5
        lm_left = make_synthetic_landmarks(iris_left=(0.30, 0.0), iris_right=(0.30, 0.0))
        lm_right = make_synthetic_landmarks(iris_left=(0.70, 0.0), iris_right=(0.70, 0.0))
        lx = get_iris_ratio(lm_left, L_CORNERS, LEFT_IRIS, "x")
        rx = get_iris_ratio(lm_right, L_CORNERS, LEFT_IRIS, "x")
        assert math.isclose(lx, 0.30, abs_tol=1e-6)
        assert math.isclose(rx, 0.70, abs_tol=1e-6)
        assert math.isclose((lx + rx) / 2.0, 0.50, abs_tol=1e-6)

        # Vertical shift: negative (above baseline) vs positive (below baseline)
        lm_up = make_synthetic_landmarks(iris_left=(0.5, -0.12), iris_right=(0.5, -0.12))
        lm_down = make_synthetic_landmarks(iris_left=(0.5, 0.12), iris_right=(0.5, 0.12))
        ly_up = get_iris_ratio(lm_up, L_CORNERS, LEFT_IRIS, "y")
        ly_down = get_iris_ratio(lm_down, L_CORNERS, LEFT_IRIS, "y")
        assert math.isclose(ly_up, -0.12, abs_tol=1e-6)
        assert math.isclose(ly_down, 0.12, abs_tol=1e-6)

    def test_iris_ratio_degenerate_baseline_and_invalid_axis(self):
        lm = make_synthetic_landmarks()
        # Collapse eye corners onto the exact same point -> baseline_sq < IRIS_BASELINE_EPS
        lm[L_CORNERS[1]].x = lm[L_CORNERS[0]].x
        lm[L_CORNERS[1]].y = lm[L_CORNERS[0]].y
        assert get_iris_ratio(lm, L_CORNERS, LEFT_IRIS, "x") == 0.5
        assert get_iris_ratio(lm, L_CORNERS, LEFT_IRIS, "y") == 0.0

        with pytest.raises(ValueError, match="Unsupported iris ratio axis"):
            get_iris_ratio(lm, R_CORNERS, RIGHT_IRIS, "z")

    def test_head_pose_angles_identity_known_rotations_and_gimbal_lock(self):
        p0, y0, r0 = get_head_pose_angles(np.eye(4, dtype=np.float32))
        assert math.isclose(p0, 0.0, abs_tol=1e-6)
        assert math.isclose(y0, 0.0, abs_tol=1e-6)
        assert math.isclose(r0, 0.0, abs_tol=1e-6)

        for target_p, target_y, target_r in [
            (12.5, -18.0, 7.25),
            (-15.0, 22.0, -5.0),
            (0.0, 30.0, 0.0),
            (-20.0, 0.0, 10.0),
        ]:
            mat = make_pose_matrix(target_p, target_y, target_r)
            p, y, r = get_head_pose_angles(mat)
            assert math.isclose(p, target_p, abs_tol=1e-4)
            assert math.isclose(y, target_y, abs_tol=1e-4)
            assert math.isclose(r, target_r, abs_tol=1e-4)

        # Gimbal lock branch (yaw = +90 deg -> sy < 1e-6)
        gimbal_mat = make_pose_matrix(10.0, 90.0, 0.0)
        gp, gy, gr = get_head_pose_angles(gimbal_mat)
        assert math.isfinite(gp) and math.isfinite(gy) and math.isfinite(gr)
        assert math.isclose(gy, 90.0, abs_tol=1e-4)
        assert gr == 0.0

    def test_extract_gaze_features_5d_vector(self):
        lm = make_synthetic_landmarks(iris_left=(0.40, -0.05), iris_right=(0.60, 0.05))
        pose = make_pose_matrix(pitch_deg=8.0, yaw_deg=-14.0, roll_deg=3.5)
        sx, sy, hp, hy, hr = extract_gaze_features(lm, pose)
        assert math.isclose(sx, 0.50, abs_tol=1e-6)
        assert math.isclose(sy, 0.00, abs_tol=1e-6)
        assert math.isclose(hp, 8.0, abs_tol=1e-4)
        assert math.isclose(hy, -14.0, abs_tol=1e-4)
        assert math.isclose(hr, 3.5, abs_tol=1e-4)

    def test_calculate_ear_and_window_stats_and_ear_feature_state(self):
        lm_open = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
        lm_closed = make_synthetic_landmarks(k_left=0.2, k_right=0.2)

        ear_open = calculate_ear(lm_open, LEFT_EYE, 640, 480)
        ear_closed = calculate_ear(lm_closed, LEFT_EYE, 640, 480)
        assert ear_open > ear_closed > 0.0

        # Degenerate zero horizontal eye distance -> returns NaN (P0 fix: never mimics closed eye)
        lm_zero = make_synthetic_landmarks()
        lm_zero[LEFT_EYE[3]].x = lm_zero[LEFT_EYE[0]].x
        lm_zero[LEFT_EYE[3]].y = lm_zero[LEFT_EYE[0]].y
        lm_zero[LEFT_EYE[3]].z = lm_zero[LEFT_EYE[0]].z
        assert math.isnan(calculate_ear(lm_zero, LEFT_EYE, 640, 480))

        # window_stats on empty, single-element, and multi-element windows
        assert window_stats([]) == (1.0, 0.0)
        assert window_stats([0.8]) == (0.8, 0.0)
        w_min, w_var = window_stats([1.0, 0.5, 0.5, 1.0])
        assert math.isclose(w_min, 0.5, abs_tol=1e-9)
        assert math.isclose(w_var, 0.0625, abs_tol=1e-9)

        # EarFeatureState 30-frame baseline ignores zero/corrupted EAR frames
        state = EarFeatureState(calibration_frames=5, window_size=4)
        assert state.update(0.0, 0.3) is None
        assert state.current_calibration_frames == 0
        for i in range(5):
            assert state.update(0.30, 0.40) is None
        assert state.is_calibrated
        assert math.isclose(state.baseline_left_ear, 0.30, abs_tol=1e-9)
        assert math.isclose(state.baseline_right_ear, 0.40, abs_tol=1e-9)

        # Post-calibration frame produces normalized 6-D wink vector
        feats = state.update(0.15, 0.40)
        assert feats is not None
        l_ear, l_min, l_var, r_ear, r_min, r_var = feats
        assert math.isclose(l_ear, 0.5, abs_tol=1e-9)
        assert math.isclose(r_ear, 1.0, abs_tol=1e-9)

    def test_mpii_processor_shared_feature_extraction_and_vector_to_pitch_yaw(self):
        stub_detector = StubFaceDetector()
        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=stub_detector):
            import mpii_processor as mp_proc

        # Verify mpii_processor reuses the exact same extract_gaze_features function from gaze_intent
        assert mp_proc.extract_gaze_features is gi.extract_gaze_features

        # Straight-ahead gaze in camera coordinates (0, 0, -1) -> pitch=0, yaw=0
        pitch, yaw = mp_proc.vector_to_pitch_yaw(np.array([0.0, 0.0, -1.0]))
        assert math.isclose(pitch, 0.0, abs_tol=1e-6)
        assert math.isclose(yaw, 0.0, abs_tol=1e-6)

        # Gaze vector tilted up (negative Y in camera space -> positive pitch) and left/right
        pitch_up, yaw_up = mp_proc.vector_to_pitch_yaw(np.array([0.0, -1.0, -1.0]))
        assert math.isclose(pitch_up, 45.0, abs_tol=1e-5)
        assert math.isclose(yaw_up, 0.0, abs_tol=1e-5)


# ===========================================================================
# Pillar 3: Mocked FaceLandmarker — Wink, Blink & Dropout Edge Cases
# ===========================================================================
class TestMockedGazeIntentPipeline:
    def test_calibration_phase_and_incomplete_calibration_dropout_reset(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed 15 open-eye frames (halfway through 30-frame calibration)
        for idx in range(15):
            ay, ap, hy, hp, state, lms = tracker.process_frame(blank, (idx + 1) * 33)
            assert state == -1
            assert not is_valid_output(ay, ap)
            assert math.isnan(ay) and math.isnan(ap) and math.isnan(hy) and math.isnan(hp)
            assert lms is not None
        assert not tracker.is_calibrated
        assert tracker.ear_state.current_calibration_frames == 15

        # Face dropout mid-calibration resets current_calibration_frames back to 0 (Issue #4 fix)
        stub.next_landmarks = None
        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, 16 * 33)
        assert state == 0
        assert lms is None
        assert math.isnan(ay) and math.isnan(ap)
        assert not tracker.is_calibrated
        assert tracker.ear_state.current_calibration_frames == 0

        # Reacquire face: requires a full 30 frames to finish calibration, frame 31 is first active frame
        stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
        for idx in range(CALIBRATION_FRAMES):
            ay, ap, _, _, state, _ = tracker.process_frame(blank, (17 + idx) * 33)
            assert state == -1
            assert not is_valid_output(ay, ap)
        assert tracker.is_calibrated

        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, 60 * 33)
        assert state == 0
        assert is_valid_output(ay, ap)
        assert math.isfinite(hy) and math.isfinite(hp)
        assert lms is not None

    def test_wink_blink_and_double_closure_classification_through_process_frame(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 0

        def step_frames(n_frames: int, k_left: float, k_right: float) -> list[int]:
            nonlocal ts
            stub.next_landmarks = make_synthetic_landmarks(k_left=k_left, k_right=k_right)
            states = []
            for _ in range(n_frames):
                ts += 33
                ay, ap, _, _, s, _ = tracker.process_frame(blank, ts)
                states.append(s)
            return states

        # 1. Complete 30-frame open-eye calibration
        cal_states = step_frames(CALIBRATION_FRAMES, 1.0, 1.0)
        assert all(s == -1 for s in cal_states)

        # 2. Neutral open eyes -> state 0
        neutral_states = step_frames(10, 1.0, 1.0)
        assert all(s == 0 for s in neutral_states)

        # 3. Single-frame natural bilateral blink (1 frame closed, then open):
        # never triggers a false single-eye wink (1 or 2) and immediately recovers to neutral (0)
        blink_frame = step_frames(1, 0.30, 0.30)
        recovery_frames = step_frames(5, 1.0, 1.0)
        assert blink_frame[0] in (0, 3) and blink_frame[0] not in (1, 2)
        assert all(s == 0 for s in recovery_frames)

        # 4. Sustained left wink (left eye closed, right eye open across 10-frame window) -> state 1
        left_wink_states = step_frames(EAR_WINDOW_SIZE, 0.25, 1.0)
        assert left_wink_states[-1] == 1

        # Reset window back to neutral open eyes
        step_frames(EAR_WINDOW_SIZE, 1.0, 1.0)

        # 5. Sustained right wink (left eye open, right eye closed across 10-frame window) -> state 2
        right_wink_states = step_frames(EAR_WINDOW_SIZE, 1.0, 0.25)
        assert right_wink_states[-1] == 2

        # Reset window back to neutral open eyes
        step_frames(EAR_WINDOW_SIZE, 1.0, 1.0)

        # 6. Sustained bilateral eye closure across 10-frame window -> state 3
        both_closed_states = step_frames(EAR_WINDOW_SIZE, 0.25, 0.25)
        assert both_closed_states[-1] == 3

    def test_missing_face_missing_pose_and_post_calibration_dropout_recovery(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 0

        # Calibrate + 5 active left-wink frames
        for _ in range(CALIBRATION_FRAMES):
            ts += 33
            tracker.process_frame(blank, ts)
        stub.next_landmarks = make_synthetic_landmarks(k_left=0.25, k_right=1.0)
        for _ in range(5):
            ts += 33
            tracker.process_frame(blank, ts)
        assert len(tracker.ear_state.left_ear_history) == 5

        # Missing transformation matrix with face present -> treated as invalid/lost face
        stub.next_pose = None
        ts += 33
        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts)
        assert state == 0
        assert lms is None
        assert not is_valid_output(ay, ap)
        # Temporal EAR window is cleared on tracking loss, while calibration baseline is preserved
        assert tracker.is_calibrated
        assert len(tracker.ear_state.left_ear_history) == 0
        assert len(tracker.ear_state.right_ear_history) == 0

        # Restore pose & open eyes -> immediately resumes valid tracking without re-calibrating
        stub.next_pose = make_pose_matrix()
        stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
        ts += 33
        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts)
        assert state == 0
        assert is_valid_output(ay, ap)
        assert lms is not None

    def test_diagnostic_csv_telemetry_logger_32_column_schema(self, tmp_path):
        csv_path = tmp_path / "diag" / "gaze_telemetry.csv"
        stub = StubFaceDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("gaze_intent.vision.FaceLandmarker.create_from_options", return_value=stub):
            tracker = GazeIntent(diag_log_path=str(csv_path))
            # Use 2-frame calibration for fast verification of calibrating, active, and dropout rows
            tracker.hybrid_blink = HybridBlinkDetector(calibration_frames=2, window_size=4)
            tracker.ear_state = tracker.hybrid_blink
            tracker.process_frame(blank, 10)   # frame 0: calibrating (-1)
            tracker.process_frame(blank, 20)   # frame 1: completes calibration (-1)
            tracker.process_frame(blank, 30)   # frame 2: active valid frame (0)
            stub.next_landmarks = None
            tracker.process_frame(blank, 40)   # frame 3: face lost (0, detected=0)
            tracker.shutdown()

        assert csv_path.is_file()
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

        assert len(_DIAG_FIELD_NAMES) == 32
        assert tuple(rows[0]) == _DIAG_FIELD_NAMES
        assert len(rows) == 5  # header + 4 frames

        # Frame 0 (calibrating): detected=0, state=-1, norm_x=-1.0, norm_y=-1.0
        r_calib = dict(zip(_DIAG_FIELD_NAMES, rows[1]))
        assert r_calib["frame_id"] == "0"
        assert r_calib["detected"] == "0"
        assert r_calib["state"] == "-1"
        assert float(r_calib["norm_x"]) == -1.0

        # Frame 2 (active): detected=1, state=0, finite norm_x/norm_y in [0, 1]
        r_active = dict(zip(_DIAG_FIELD_NAMES, rows[3]))
        assert r_active["frame_id"] == "2"
        assert r_active["detected"] == "1"
        assert r_active["state"] == "0"
        assert 0.0 <= float(r_active["norm_x"]) <= 1.0
        assert 0.0 <= float(r_active["norm_y"]) <= 1.0

        # Frame 3 (dropout): detected=0, state=0, norm_x=-1.0
        r_lost = dict(zip(_DIAG_FIELD_NAMES, rows[4]))
        assert r_lost["frame_id"] == "3"
        assert r_lost["detected"] == "0"
        assert r_lost["state"] == "0"
        assert float(r_lost["norm_x"]) == -1.0


# ===========================================================================
# Pillar 4: Coordinate Mapping & Calibration Math (gaze_intent.py & run_gaze.py)
# ===========================================================================
class TestCoordinateMappingAndCalibrationMath:
    def test_angle_to_norm_coord_contract_and_clamping(self):
        assert SPAN_DEG == 60.0
        assert math.isclose(angle_to_norm_coord(0.0), 0.5, abs_tol=1e-9)
        assert math.isclose(angle_to_norm_coord(-15.0), 0.25, abs_tol=1e-9)
        assert math.isclose(angle_to_norm_coord(15.0), 0.75, abs_tol=1e-9)
        assert math.isclose(angle_to_norm_coord(-30.0), 0.0, abs_tol=1e-9)
        assert math.isclose(angle_to_norm_coord(30.0), 1.0, abs_tol=1e-9)

        # Out-of-span angles clamp cleanly to [0.0, 1.0]
        assert angle_to_norm_coord(-75.0) == 0.0
        assert angle_to_norm_coord(75.0) == 1.0

        # Non-finite angles return sentinel -1.0 (matches C++ GazeNative::setPointer)
        assert angle_to_norm_coord(float("nan")) == -1.0
        assert angle_to_norm_coord(float("inf")) == -1.0
        assert angle_to_norm_coord(float("-inf")) == -1.0

    def test_is_valid_output_sentinel_guard(self):
        assert is_valid_output(0.0, 0.0)
        assert is_valid_output(-1.0, -1.0)  # -1.0 deg is a valid physical angle, NOT a sentinel
        assert not is_valid_output(INVALID_ANGLE, 0.0)
        assert not is_valid_output(0.0, INVALID_ANGLE)
        assert not is_valid_output(float("inf"), 0.0)

    def test_run_gaze_poly_features_face_area_and_ridge_calibration_fit(self):
        feat = rg.build_poly_features(12.5, -6.25)
        assert feat.dtype == np.float64
        assert np.allclose(feat, np.array([1.0, 12.5, -6.25], dtype=np.float64))

        assert rg.get_face_area([], 640, 480) == 0.0
        lms = make_synthetic_landmarks(center=(0.5, 0.5), size=0.20)
        area_vga = rg.get_face_area(lms, 640, 480)
        area_hd = rg.get_face_area(lms, 1280, 960)
        assert area_vga > 0.0
        assert math.isclose(area_hd, area_vga * 4.0, rel_tol=1e-5)


# ===========================================================================
# Pillar 5: Timestamp Monotonicity & Sustained-Fixation Stability Regressions
# ===========================================================================
class TestTimestampMonotonicityAndFixationStability:
    def test_strictly_increasing_timestamps_forwarded_to_mediapipe_video_mode(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate 200 rapid back-to-back calls with monotonic millisecond timestamps
        for step in range(1, 201):
            tracker.process_frame(blank, step)

        assert len(stub.received_timestamps) == 200
        diffs = [b - a for a, b in zip(stub.received_timestamps, stub.received_timestamps[1:])]
        assert all(d > 0 for d in diffs), "MediaPipe VIDEO timestamps must be strictly increasing"

    @pytest.mark.parametrize(
        "fixation_name,iris_xy,head_yaw_deg,expected_side",
        [
            ("left_screen_fixation", (0.62, 0.0), -14.0, "left"),
            ("center_screen_fixation", (0.50, 0.0), 0.0, "center"),
            ("right_screen_fixation", (0.38, 0.0), 14.0, "right"),
        ],
    )
    def test_300_frame_sustained_fixation_stability_no_dropouts_or_collapse(
        self, mocked_gaze_tracker, fixation_name, iris_xy, head_yaw_deg, expected_side
    ):
        """Replay a 300-frame (10s @ 30 FPS) steady fixation with realistic landmark micro-jitter.

        Asserts:
        1. Zero false face dropouts or NaN outputs after calibration completes.
        2. Bounded yaw/pitch standard deviation (< 1.5 deg) and peak-to-peak range (< 8.0 deg).
        3. Consistent side-of-screen norm_x classification without collapsing toward center over 10 seconds.
        """
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        rng = np.random.default_rng(2026)
        ts = 0

        # 30-frame calibration with open eyes
        stub.next_pose = make_pose_matrix(0.0, 0.0, 0.0)
        for _ in range(CALIBRATION_FRAMES):
            ts += 33
            stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
            ay, ap, _, _, state, _ = tracker.process_frame(blank, ts)
            assert state == -1

        yaws: list[float] = []
        pitches: list[float] = []
        norm_xs: list[float] = []

        # 300 active fixation frames (10 seconds @ 30 FPS) with realistic micro-jitter
        for _ in range(300):
            ts += 33
            jitter_ix = float(iris_xy[0] + rng.normal(0.0, 0.0025))
            jitter_iy = float(iris_xy[1] + rng.normal(0.0, 0.0025))
            jitter_hy = float(head_yaw_deg + rng.normal(0.0, 0.08))
            jitter_hp = float(rng.normal(0.0, 0.08))

            stub.next_landmarks = make_synthetic_landmarks(
                k_left=1.0, k_right=1.0, iris_left=(jitter_ix, jitter_iy), iris_right=(jitter_ix, jitter_iy)
            )
            stub.next_pose = make_pose_matrix(pitch_deg=jitter_hp, yaw_deg=jitter_hy, roll_deg=0.0)

            ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts)
            assert is_valid_output(ay, ap), f"Unexpected NaN dropout during {fixation_name}"
            assert state == 0, f"Unexpected wink/blink trigger during steady open-eye {fixation_name}"
            assert lms is not None

            yaws.append(ay)
            pitches.append(ap)
            norm_xs.append(angle_to_norm_coord(ay))

        yaw_arr = np.array(yaws, dtype=np.float64)
        pitch_arr = np.array(pitches, dtype=np.float64)
        norm_x_arr = np.array(norm_xs, dtype=np.float64)

        # Stability bounds across 300 frames
        assert float(np.std(yaw_arr)) < 1.5
        assert float(np.std(pitch_arr)) < 1.5
        assert float(np.ptp(yaw_arr)) < 8.0

        # No drift/collapse between first 50 frames and last 50 frames
        first_mean_nx = float(np.mean(norm_x_arr[:50]))
        last_mean_nx = float(np.mean(norm_x_arr[-50:]))
        assert abs(last_mean_nx - first_mean_nx) < 0.03

        if expected_side == "left":
            assert np.all(norm_x_arr < 0.40)
        elif expected_side == "right":
            assert np.all(norm_x_arr > 0.60)
        else:
            assert np.all((norm_x_arr > 0.40) & (norm_x_arr < 0.60))


# ===========================================================================
# Pillar 6: Real-Asset Offline Smoke & Negative Asset Regressions (#9 - P2)
# ===========================================================================
class TestRealAssetSmokeAndNegativeRegressions:
    def test_real_face_landmarker_and_dlls_blank_frame_smoke(self):
        """Instantiate real GazeIntent (no MediaPipe mock) with shipped assets and process blank RGB frames."""
        task_path = REPO_ROOT / "assets" / "face_landmarker.task"
        assert task_path.is_file(), f"Missing deployed MediaPipe task asset: {task_path}"
        for dll_name in ("gesture_model.dll", "gaze_pitch.dll", "gaze_yaw.dll"):
            assert (MODELS_DIR / dll_name).is_file(), f"Missing deployed DLL: {dll_name}"

        tracker = GazeIntent()
        try:
            assert tracker.detector is not None
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            for ts_ms in (33, 66):
                ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts_ms)
                assert math.isnan(ay) and math.isnan(ap)
                assert math.isnan(hy) and math.isnan(hp)
                assert state == 0
                assert lms is None
                assert not is_valid_output(ay, ap)
            assert not tracker.is_calibrated
        finally:
            tracker.shutdown()
        assert tracker.detector is None

    def test_missing_face_landmarker_task_raises_explicit_error(self, tmp_path, monkeypatch):
        """Missing assets/face_landmarker.task must raise an explicit error rather than succeeding silently."""
        fake_ai_root = tmp_path / "ai_pipeline"
        fake_ai_root.mkdir()
        (tmp_path / "assets").mkdir()
        monkeypatch.setenv("AI_PIPELINE_ROOT", str(fake_ai_root))

        with pytest.raises((FileNotFoundError, RuntimeError, ValueError, OSError)):
            GazeIntent()

    def test_corrupt_face_landmarker_task_raises_explicit_error(self, tmp_path, monkeypatch):
        """Corrupt/invalid assets/face_landmarker.task must raise an explicit error."""
        fake_ai_root = tmp_path / "ai_pipeline"
        fake_ai_root.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "face_landmarker.task").write_bytes(b"corrupt-mediapipe-task-archive")
        monkeypatch.setenv("AI_PIPELINE_ROOT", str(fake_ai_root))

        with pytest.raises((RuntimeError, ValueError, OSError)):
            GazeIntent()

    def test_missing_treelite_dll_raises_explicit_error(self, tmp_path):
        """Missing production Treelite DLL under gaze_intent/models must raise FileNotFoundError."""
        import importlib.util

        isolated_gaze_dir = tmp_path / "ai_pipeline" / "gaze_intent"
        isolated_gaze_dir.mkdir(parents=True)
        isolated_py = isolated_gaze_dir / "gaze_intent.py"
        isolated_py.write_text(
            (GAZE_DIR / "gaze_intent.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        spec = importlib.util.spec_from_file_location("isolated_gaze_intent_missing_dll", str(isolated_py))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with pytest.raises(FileNotFoundError, match="Production DLL missing"):
            mod.GazeIntent()


# ===========================================================================
# Pillar 7: Hybrid Blink v2 Safety, Polarity Lock, Hysteresis & Telemetry
# ===========================================================================
class TestHybridBlinkV2AndSafety:
    def test_hybrid_blink_spec_matches_module_constants(self):
        import json

        spec_path = GAZE_DIR / "hybrid_blink_spec.json"
        assert spec_path.is_file()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        slots = tuple(s["mediapipe_name"] for s in spec["bridge_packing"]["eye_aligned_blendshape_slots"])
        assert slots == EYE_BLENDSHAPE_NAMES
        assert spec["calibration"]["calibration_frames"] == CALIBRATION_FRAMES
        assert math.isclose(spec["closure_scoring"]["fusion_weight_bs"], gi.HYBRID_FUSION_W_BS, abs_tol=1e-12)
        assert math.isclose(spec["closure_scoring"]["fusion_weight_ear"], gi.HYBRID_FUSION_W_EAR, abs_tol=1e-12)
        assert math.isclose(spec["closure_scoring"]["lookdown_gate_ear_lo"], gi.HYBRID_LD_GATE_EAR_LO, abs_tol=1e-12)
        assert math.isclose(spec["closure_scoring"]["lookdown_gate_ear_hi"], gi.HYBRID_LD_GATE_EAR_HI, abs_tol=1e-12)
        assert math.isclose(spec["hysteresis"]["close_thresh"], gi.HYBRID_CLOSE_THRESH, abs_tol=1e-12)
        assert math.isclose(spec["hysteresis"]["open_thresh"], gi.HYBRID_OPEN_THRESH, abs_tol=1e-12)
        assert math.isclose(spec["temporal_smoothing"]["ema_tau_attack_seconds"], gi.HYBRID_EMA_TAU_ATTACK_S, abs_tol=1e-12)
        assert math.isclose(spec["temporal_smoothing"]["ema_tau_release_seconds"], gi.HYBRID_EMA_TAU_RELEASE_S, abs_tol=1e-12)
        assert math.isclose(spec["bridge_packing"]["blendshape_range_eps"], gi.BLENDSHAPE_RANGE_EPS, abs_tol=1e-12)
        assert spec["bridge_packing"]["sentinel_slot"] == 8
        assert math.isclose(spec["bridge_packing"]["sentinel_value"], 1.0, abs_tol=1e-12)
        assert "missing_blendshapes" in spec["bridge_packing"]["policies"]
        assert "invalid_blendshapes" in spec["bridge_packing"]["policies"]
        assert "dsp_recorder_landmark_count" in spec["bridge_packing"]["policies"]

    def test_degenerate_eye_and_legacy_nan_guard_never_emit_closure(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 0
        for _ in range(CALIBRATION_FRAMES):
            ts += 33
            tracker.process_frame(blank, ts)
        assert tracker.is_calibrated

        # Degenerate collapsed horizontal eye corners (|p1 - p4| == 0) -> rejected cleanly as state 0 (never 3!)
        lm_deg = make_synthetic_landmarks()
        lm_deg[LEFT_EYE[3]].x = lm_deg[LEFT_EYE[0]].x
        lm_deg[LEFT_EYE[3]].y = lm_deg[LEFT_EYE[0]].y
        lm_deg[LEFT_EYE[3]].z = lm_deg[LEFT_EYE[0]].z
        stub.next_landmarks = lm_deg
        ts += 33
        ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
        assert state == 0 and lms is None and not is_valid_output(ay, ap)

        # Legacy classifier guard: non-finite features OR non-finite model output -> 0 Neutral (never 3 Closure)
        assert safe_classify_legacy_wink(tracker.wink_model, (float("nan"), 1.0, 0.0, 1.0, 1.0, 0.0)) == 0
        assert safe_classify_legacy_wink(tracker.wink_model, (1.0, float("inf"), 0.0, 1.0, 1.0, 0.0)) == 0

        class _NanForest:
            def predict(self, _):
                return np.array([[0.1, float("nan"), 0.2, 0.3]], dtype=np.float32)

        assert safe_classify_legacy_wink(_NanForest(), (1.0, 1.0, 0.0, 1.0, 1.0, 0.0)) == 0

    def test_poisoned_startup_calibration_recovers_true_baseline(self):
        det = HybridBlinkDetector()
        # Interleave 15 blink-poisoned frames (low EAR / high blendshape) + 30 open frames
        for i in range(45):
            if i % 3 == 0:
                # Blink frame during startup: rejected by min_ear/bs/asym gate
                det.update_calibration(0.08, 0.08, 0.0, 0.0, (0.85, 0.85, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0))
            else:
                det.update_calibration(0.30, 0.32, 0.0, 0.0, (0.06, 0.07, 0.1, 0.1, 0.04, 0.04, 0.02, 0.02))
        assert det.is_calibrated
        assert math.isclose(det.baseline_left_ear, 0.30, abs_tol=1e-6)
        assert math.isclose(det.baseline_right_ear, 0.32, abs_tol=1e-6)
        assert math.isclose(det.baseline_bs_left, 0.06, abs_tol=1e-6)
        assert math.isclose(det.baseline_bs_right, 0.07, abs_tol=1e-6)

        # Mildly drooping frames inside the calibration window are still trimmed by 75th-percentile EAR
        det2 = HybridBlinkDetector()
        for i in range(CALIBRATION_FRAMES):
            ear_val = 0.20 if i < 10 else 0.32
            bs_val = 0.22 if i < 10 else 0.05
            det2.update_calibration(ear_val, ear_val, 0.0, 0.0, (bs_val, bs_val, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0))
        assert det2.is_calibrated
        assert math.isclose(det2.baseline_left_ear, 0.32, abs_tol=1e-6)
        assert math.isclose(det2.baseline_bs_left, 0.05, abs_tol=1e-6)

    def test_mirrored_blendshape_polarity_lock_left_vs_right_click(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 0

        # Calibrate with neutral open eyes + resting blendshapes
        stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
        stub.next_blendshapes = make_eye_blendshapes(eyeBlinkRight=0.05, eyeBlinkLeft=0.05)
        for _ in range(CALIBRATION_FRAMES):
            ts += 33
            tracker.process_frame(blank, ts)
        assert tracker.is_calibrated

        # Empirically locked polarity:
        # Image-left LEFT_EYE (user physical left eye in mirrored webcam -> state=1 Left Click) = eyeBlinkRight
        stub.next_landmarks = make_synthetic_landmarks(k_left=0.25, k_right=1.0)
        stub.next_blendshapes = make_eye_blendshapes(eyeBlinkRight=0.85, eyeBlinkLeft=0.05)
        for _ in range(6):
            ts += 33
            _, _, _, _, st_l, _ = tracker.process_frame(blank, ts)
        assert st_l == 1, "eyeBlinkRight + LEFT_EYE closure MUST emit state=1 (Left Wink / Left Click)"

        # Return to neutral
        stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=1.0)
        stub.next_blendshapes = make_eye_blendshapes(eyeBlinkRight=0.05, eyeBlinkLeft=0.05)
        for _ in range(6):
            ts += 33
            tracker.process_frame(blank, ts)

        # Image-right RIGHT_EYE (user physical right eye in mirrored webcam -> state=2 Right Click) = eyeBlinkLeft
        stub.next_landmarks = make_synthetic_landmarks(k_left=1.0, k_right=0.25)
        stub.next_blendshapes = make_eye_blendshapes(eyeBlinkRight=0.05, eyeBlinkLeft=0.85)
        for _ in range(6):
            ts += 33
            _, _, _, _, st_r, _ = tracker.process_frame(blank, ts)
        assert st_r == 2, "eyeBlinkLeft + RIGHT_EYE closure MUST emit state=2 (Right Wink / Right Click)"

    def test_symmetric_blink_sympathetic_squint_downward_gaze_hysteresis_and_fps(self):
        def make_calibrated() -> HybridBlinkDetector:
            d = HybridBlinkDetector()
            for _ in range(CALIBRATION_FRAMES):
                d.update_calibration(0.30, 0.30, 0.0, 0.0, (0.05, 0.05, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02))
            return d

        # 1. Symmetric blink NEVER emits 1 or 2 across entire closure sweep
        d = make_calibrated()
        ts = 0
        for lvl in (0.10, 0.30, 0.55, 0.75, 0.90, 0.90, 0.65, 0.40, 0.10):
            ts += 33
            ear = 0.30 * (1.0 - 0.75 * lvl)
            st, *_ = d.step(ear, ear, 0.0, 0.0, ts, (lvl, lvl, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0))
            assert st in (0, 3), f"Symmetric blink emitted single-eye wink state={st}"

        # 2. Sympathetic-squint firm left wink (c_L high, c_R moderate ~0.48) -> state 1
        d = make_calibrated()
        ts = 0
        for _ in range(8):
            ts += 33
            st, cl, cr, *_ = d.step(0.08, 0.18, 0.0, 0.0, ts, (0.88, 0.50, 0.10, 0.10, 0.30, 0.20, 0.0, 0.0))
        assert st == 1 and cl > 0.75 and 0.35 < cr < 0.60

        # 3. Downward gaze (eyeLookDown=0.65, sy=+0.06, head_pitch=+10 deg, natural lid droop EAR 0.21) -> stays 0
        d = make_calibrated()
        ts = 0
        for _ in range(15):
            ts += 33
            st, cl, cr, *_ = d.step(0.21, 0.21, 10.0, 0.06, ts, (0.24, 0.24, 0.65, 0.65, 0.05, 0.05, 0.0, 0.0))
            assert st == 0, f"Downward gaze caused false trigger state={st} (cl={cl:.3f}, cr={cr:.3f})"

        # 4. Schmitt-trigger hysteresis: +-0.02 jitter around 0.52 causes ZERO flips after settling
        d = make_calibrated()
        ts = 0
        # Enter state 1 cleanly first
        for _ in range(8):
            ts += 33
            st, *_ = d.step(0.09, 0.30, 0.0, 0.0, ts, (0.82, 0.05, 0.1, 0.1, 0.1, 0.04, 0.0, 0.0))
        assert st == 1
        # Oscillate right around 0.52 (between open_thresh=0.32 and close_thresh=0.52) -> stays locked in state 1!
        states = []
        for k in range(30):
            ts += 33
            jitter = 0.02 if (k % 2 == 0) else -0.02
            bs_l = 0.54 + jitter
            ear_l = 0.165 - jitter * 0.05
            st, *_ = d.step(ear_l, 0.30, 0.0, 0.0, ts, (bs_l, 0.05, 0.1, 0.1, 0.05, 0.04, 0.0, 0.0))
            states.append(st)
        assert all(s == 1 for s in states), f"Hysteresis chattered on +-0.02 jitter: {states}"

        # 5. Wall-clock EMA invariance across 15 / 30 / 60 FPS (same physical 120ms pulse reaches same state)
        final_scores = []
        for dt_ms in (16, 33, 66):
            d_fps = make_calibrated()
            ts = 0
            # Start from open at t=0
            d_fps.step(0.30, 0.30, 0.0, 0.0, ts, (0.05, 0.05, 0.1, 0.1, 0.04, 0.04, 0.02, 0.02))
            while ts < 132:
                ts += dt_ms
                st, cl, cr, *_ = d_fps.step(0.08, 0.30, 0.0, 0.0, ts, (0.88, 0.05, 0.1, 0.1, 0.1, 0.04, 0.0, 0.0))
            assert st == 1
            final_scores.append(cl)
        assert max(final_scores) - min(final_scores) < 0.05

    def test_opt_in_22_col_blink_diag_does_not_alter_32_col_gaze_diag(self, tmp_path):
        gaze_csv = tmp_path / "gaze_32.csv"
        blink_csv = tmp_path / "blink_22.csv"
        stub = StubFaceDetector()
        stub.next_blendshapes = make_eye_blendshapes()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("gaze_intent.vision.FaceLandmarker.create_from_options", return_value=stub):
            tracker = GazeIntent(diag_log_path=str(gaze_csv), blink_diag_log_path=str(blink_csv))
            tracker.hybrid_blink = HybridBlinkDetector(calibration_frames=2, window_size=4)
            tracker.ear_state = tracker.hybrid_blink
            tracker.process_frame(blank, 10)
            tracker.process_frame(blank, 20)
            tracker.process_frame(blank, 30)
            stub.next_landmarks = None
            tracker.process_frame(blank, 40)
            tracker.shutdown()

        with open(gaze_csv, "r", encoding="utf-8", newline="") as fg:
            g_rows = list(csv.reader(fg))
        with open(blink_csv, "r", encoding="utf-8", newline="") as fb:
            b_rows = list(csv.reader(fb))

        assert len(_DIAG_FIELD_NAMES) == 32 and len(g_rows[0]) == 32
        assert len(_BLINK_DIAG_FIELD_NAMES) == 22 and len(b_rows[0]) == 22
        assert len(g_rows) == 5 and len(b_rows) == 5
        active_b = dict(zip(_BLINK_DIAG_FIELD_NAMES, b_rows[3]))
        assert active_b["detector_mode"] == "hybrid_bs_ear"
        assert active_b["detected"] == "1"
        assert active_b["state"] == "0"

    def test_open_eye_lookdown_gate_and_asymmetric_fast_attack_ema(self):
        d = HybridBlinkDetector()
        for _ in range(CALIBRATION_FRAMES):
            d.update_calibration(0.30, 0.30, -4.0, -0.08, (0.11, 0.18, 0.32, 0.34, 0.32, 0.26, 0.0, 0.0))
        assert d.is_calibrated

        # 1. Start from neutral open-eye frame
        ts = 1000
        st0, cl0, cr0, *_ = d.step(0.30, 0.30, -4.0, -0.08, ts, (0.11, 0.18, 0.32, 0.34, 0.32, 0.26, 0.0, 0.0))
        assert st0 == 0 and cl0 < 0.05 and cr0 < 0.05

        # 2. Real quick left wink with elevated bs_lookdown=0.55 (eyelid-descent artifact):
        #    MUST enter state=1 within <=2 frames (<=66 ms) and NEVER be suppressed by bs_lookdown.
        entered_frame = None
        for f_idx in range(1, 5):
            ts += 33
            st, cl, cr, *_ = d.step(0.09, 0.23, -4.0, -0.03, ts, (0.50, 0.34, 0.55, 0.55, 0.68, 0.34, 0.0, 0.0))
            if st == 1 and entered_frame is None:
                entered_frame = f_idx
        assert entered_frame is not None and entered_frame <= 2, f"Slow/missed wink onset: frame={entered_frame}"

        # 3. Return to open-eye downward gaze (norm_ear=0.70, bs_lookdown=0.65, sy=+0.02, pitch=+10 deg):
        #    open_gate == 1.0 -> full lookDown compensation keeps state == 0.
        for _ in range(10):
            ts += 33
            st, cl, cr, *_ = d.step(0.21, 0.21, 10.0, 0.02, ts, (0.24, 0.24, 0.65, 0.65, 0.10, 0.10, 0.0, 0.0))
        assert st == 0 and cl < gi.HYBRID_CLOSE_THRESH and cr < gi.HYBRID_CLOSE_THRESH

    def test_blendshape_float_tolerance_and_clamping(self):
        # Minor float32/TFLite rounding overshoot within [-1e-4, 1+1e-4] accepted & clamped to [0.0, 1.0]
        bs = make_eye_blendshapes(eyeBlinkRight=-1e-6, eyeBlinkLeft=1.0000001)
        out = gi.extract_eye_blendshapes(bs)
        assert out is not None and len(out) == 8
        assert out[0] == 0.0 and out[1] == 1.0

        # Out-of-tolerance values (> 1e-4 outside [0, 1]) and non-finite still rejected
        for bad_val in (-0.01, -2e-4, 1.0002, 1.01, float("nan"), float("inf"), float("-inf")):
            bad_bs = make_eye_blendshapes(eyeBlinkRight=bad_val)
            assert gi.extract_eye_blendshapes(bad_bs) is None

    def test_capture_s_drives_ema_over_artificial_ts_ms(self, tmp_path):
        d = HybridBlinkDetector()
        open_bs = (0.05, 0.05, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02)
        left_bs = (0.85, 0.05, 0.10, 0.10, 0.25, 0.04, 0.0, 0.02)
        for _ in range(CALIBRATION_FRAMES):
            d.update_calibration(0.30, 0.30, 0.0, 0.0, open_bs)

        # Seed smooth state at capture_s=1.000, ts_ms=1000
        st0, cl0, *_ = d.step(0.30, 0.30, 0.0, 0.0, ts_ms=1000, blendshapes=open_bs, capture_s=1.000)
        assert st0 == 0 and cl0 < 1e-6

        # Artificial ts_ms only advances +1ms (1001), while real capture_s advances +0.080s (1.080):
        # EMA MUST follow capture_s (alpha = 1 - exp(-0.080/0.015) = 0.9952 -> state 1, cl > 0.80).
        st1, cl1, *_ = d.step(0.08, 0.30, 0.0, 0.0, ts_ms=1001, blendshapes=left_bs, capture_s=1.080)
        assert st1 == 1 and cl1 > 0.80

        # Gap > 0.250s on capture_s (+0.300s) resets EMA immediately even when ts_ms only advances +1ms
        st2, cl2, *_ = d.step(0.30, 0.30, 0.0, 0.0, ts_ms=1002, blendshapes=open_bs, capture_s=1.380)
        assert st2 == 0 and cl2 < 0.01

        # GazeIntent.process_frame forwards capture_s into both EMA and 22-col blink diag CSV
        blink_csv = tmp_path / "blink_cap_s.csv"
        stub = StubFaceDetector()
        stub.next_blendshapes = make_eye_blendshapes(eyeBlinkRight=0.05, eyeBlinkLeft=0.05)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("gaze_intent.vision.FaceLandmarker.create_from_options", return_value=stub):
            tracker = GazeIntent(blink_diag_log_path=str(blink_csv))
            tracker.hybrid_blink = HybridBlinkDetector(calibration_frames=2, window_size=4)
            tracker.ear_state = tracker.hybrid_blink
            tracker.process_frame(blank, 1000, capture_s=1.000)
            tracker.process_frame(blank, 1001, capture_s=1.033)
            tracker.process_frame(blank, 1002, capture_s=1.250)
            tracker.shutdown()
        with open(blink_csv, "r", encoding="utf-8", newline="") as fb:
            rows = list(csv.reader(fb))
        last = dict(zip(_BLINK_DIAG_FIELD_NAMES, rows[-1]))
        assert math.isclose(float(last["capture_s"]), 1.250, abs_tol=1e-6)
        assert last["ts_ms"] == "1002"


# ===========================================================================
# Pillar 8: Blink v2 Cued Data Collector Schema, State Machine & Audit Guards
# ===========================================================================
class TestWinkDataCollectorV2:
    def test_cue_event_state_machine_transitions_and_auto_return(self):
        import wink_data_collector as wdc

        sm = wdc.CueEventStateMachine()
        assert sm.current(0.0) == (0, "neutral")

        # Trigger left wink at t=10.0s: prepare [0, 0.25) -> active [0.25, 0.90) -> release [0.90, 1.15) -> neutral
        sm.trigger("left_wink", 10.0)
        assert sm.current(10.00) == (-1, "prepare")
        assert sm.current(10.24) == (-1, "prepare")
        assert sm.current(10.25) == (1, "active_left_wink")
        assert sm.current(10.89) == (1, "active_left_wink")
        assert sm.current(10.90) == (-1, "release")
        assert sm.current(11.14) == (-1, "release")
        assert sm.current(11.15) == (0, "neutral")
        assert sm.current(12.00) == (0, "neutral")

        # Downward gaze negative cue emits cue_label=0 during active_downward_gaze
        sm.trigger("downward_gaze", 20.0)
        assert sm.current(20.50) == (0, "active_downward_gaze")
        sm.reset_neutral()
        assert sm.current(20.60) == (0, "neutral")

        with pytest.raises(ValueError, match="Unknown cue_name"):
            sm.trigger("invalid_cue", 21.0)

    def test_collector_row_validation_rejects_nan_oob_and_invalid_labels(self):
        import wink_data_collector as wdc

        valid_kwargs = dict(
            subject_id="subj_test",
            session_id="20261127_150000",
            frame_id=42,
            ts_ms=1400,
            capture_s=1.4,
            detected=1,
            cue_label=1,
            cue_phase="active_left_wink",
            raw_ear_l=0.10,
            raw_ear_r=0.30,
            base_ear_l=0.30,
            base_ear_r=0.30,
            norm_ear_l=0.333333,
            norm_ear_r=1.0,
            blendshapes=(0.75, 0.08, 0.20, 0.15, 0.30, 0.10, 0.0, 0.02),
            sx=0.50,
            sy=-0.04,
            head_pitch_deg=-2.5,
            head_yaw_deg=1.0,
            head_roll_deg=0.5,
            hybrid_state=1,
        )
        row = wdc.build_collector_row(**valid_kwargs)
        assert len(row) == len(wdc.BLINK_COLLECTOR_HEADER) == 29
        assert row[0] == wdc.BLINK_COLLECTOR_SCHEMA_VERSION

        # Non-finite EAR rejected
        bad_nan = dict(valid_kwargs, raw_ear_l=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            wdc.build_collector_row(**bad_nan)

        # Out-of-bounds blendshape rejected
        bad_bs = dict(valid_kwargs, blendshapes=(1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="Blendshape"):
            wdc.build_collector_row(**bad_bs)

        # Invalid cue_label rejected
        bad_lbl = dict(valid_kwargs, cue_label=99)
        with pytest.raises(ValueError, match="cue_label"):
            wdc.build_collector_row(**bad_lbl)

    def test_collector_csv_writer_enforces_schema_and_refuses_legacy_drift(self, tmp_path):
        import wink_data_collector as wdc

        # 1. Fresh file gets exact 29-column v2 header and allows valid append
        out_csv = tmp_path / "blink_v2_session_test.csv"
        row = wdc.build_collector_row(
            subject_id="s1",
            session_id="sess1",
            frame_id=1,
            ts_ms=33,
            capture_s=0.033,
            detected=1,
            cue_label=0,
            cue_phase="neutral",
            raw_ear_l=0.30,
            raw_ear_r=0.30,
            base_ear_l=0.30,
            base_ear_r=0.30,
            norm_ear_l=1.0,
            norm_ear_r=1.0,
            blendshapes=(0.05, 0.05, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02),
            sx=0.50,
            sy=0.0,
            head_pitch_deg=0.0,
            head_yaw_deg=0.0,
            head_roll_deg=0.0,
            hybrid_state=0,
        )
        with wdc.BlinkCollectorCsvWriter(str(out_csv)) as w:
            w.write_row(row)
        # Re-open same valid v2 file succeeds
        with wdc.BlinkCollectorCsvWriter(str(out_csv)) as w:
            w.write_row(row)

        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3
        assert tuple(rows[0]) == wdc.BLINK_COLLECTOR_HEADER
        assert len(rows[1]) == 29 and len(rows[2]) == 29

        # 2. Legacy 8-column or 9-column file MUST be refused (prevents silent schema drift!)
        legacy_csv = tmp_path / "legacy_blink_dataset.csv"
        legacy_csv.write_text("Timestamp,Left_EAR,Left_Min_10f,Left_Var_10f,Right_EAR,Right_Min_10f,Right_Var_10f,Label\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Schema mismatch"):
            with wdc.BlinkCollectorCsvWriter(str(legacy_csv)):
                pass

        # 3. Default session output path is non-destructive and never targets legacy blink_dataset.csv
        default_path = wdc.default_session_output_path(str(tmp_path), "20261127_151500")
        assert default_path.endswith("blink_v2_session_20261127_151500.csv")
        assert "blink_dataset.csv" not in default_path

