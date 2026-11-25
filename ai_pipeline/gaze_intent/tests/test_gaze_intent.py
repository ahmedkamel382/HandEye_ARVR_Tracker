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
    INVALID_ANGLE,
    IRIS_BASELINE_EPS,
    L_CORNERS,
    LEFT_EYE,
    LEFT_IRIS,
    R_CORNERS,
    RIGHT_EYE,
    RIGHT_IRIS,
    SPAN_DEG,
    EarFeatureState,
    GazeDiagnosticPayload,
    GazeIntent,
    _DIAG_FIELD_NAMES,
    angle_to_norm_coord,
    calculate_ear,
    extract_gaze_features,
    get_3d_dist,
    get_head_pose_angles,
    get_iris_ratio,
    is_valid_output,
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


class StubFaceDetector:
    """Deterministic in-memory MediaPipe FaceLandmarker stub for headless GazeIntent tests."""

    def __init__(self) -> None:
        self.next_landmarks: list[Any] | None = make_synthetic_landmarks()
        self.next_pose: np.ndarray | None = make_pose_matrix()
        self.received_timestamps: list[int] = []
        self.closed: bool = False

    def detect_for_video(self, mp_image, timestamp_ms: int):
        self.received_timestamps.append(timestamp_ms)
        has_face = self.next_landmarks is not None and len(self.next_landmarks) > 0
        has_pose = self.next_pose is not None
        return types.SimpleNamespace(
            face_landmarks=[self.next_landmarks] if has_face else [],
            facial_transformation_matrixes=[self.next_pose] if (has_face and has_pose) else [],
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

        # Degenerate zero horizontal eye distance -> returns 0.0 without ZeroDivisionError
        lm_zero = make_synthetic_landmarks()
        lm_zero[LEFT_EYE[3]].x = lm_zero[LEFT_EYE[0]].x
        lm_zero[LEFT_EYE[3]].y = lm_zero[LEFT_EYE[0]].y
        lm_zero[LEFT_EYE[3]].z = lm_zero[LEFT_EYE[0]].z
        assert calculate_ear(lm_zero, LEFT_EYE, 640, 480) == 0.0

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
        import importlib.util

        spec = importlib.util.spec_from_file_location("mpii_processor_lazy_test", GAZE_DIR / "mpii_processor.py")
        mp_proc = importlib.util.module_from_spec(spec)
        blocked_optional = {
            "pandas": None,
            "tqdm": None,
            "cv2": None,
            "mediapipe": None,
            "mediapipe.tasks": None,
            "mediapipe.tasks.python": None,
            "mediapipe.tasks.python.vision": None,
        }
        with patch.dict(sys.modules, blocked_optional):
            spec.loader.exec_module(mp_proc)
            # Pure math helper works with all optional heavy dependencies absent
            pitch, yaw = mp_proc.vector_to_pitch_yaw(np.array([0.0, 0.0, -1.0]))
            assert math.isclose(pitch, 0.0, abs_tol=1e-6)
            assert math.isclose(yaw, 0.0, abs_tol=1e-6)

            pitch_up, yaw_up = mp_proc.vector_to_pitch_yaw(np.array([0.0, -1.0, -1.0]))
            assert math.isclose(pitch_up, 45.0, abs_tol=1e-5)
            assert math.isclose(yaw_up, 0.0, abs_tol=1e-5)

        # Verify mpii_processor reuses the exact same extract_gaze_features function without creating FaceLandmarker
        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options") as mock_create:
            assert mp_proc.extract_gaze_features is gi.extract_gaze_features
            mock_create.assert_not_called()

    def test_mpii_processor_lazy_detector_and_exception_safe_close(self, tmp_path, monkeypatch):
        import importlib.util
        import types

        spec = importlib.util.spec_from_file_location("mpii_processor_lifecycle_test", GAZE_DIR / "mpii_processor.py")
        mp_proc = importlib.util.module_from_spec(spec)
        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options") as mock_create:
            spec.loader.exec_module(mp_proc)
            mock_create.assert_not_called()

        # 1. Missing dataset returns early without constructing FaceLandmarker
        monkeypatch.setattr(mp_proc, "DATASET_PATH", str(tmp_path / "missing_dataset"))
        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options") as mock_create:
            mp_proc.process_mpiigaze()
            mock_create.assert_not_called()

        # Prepare a minimal 1-image synthetic MPIIFaceGaze directory
        ds_dir = tmp_path / "MPIIFaceGaze"
        p00_dir = ds_dir / "p00"
        p00_dir.mkdir(parents=True)
        img_path = p00_dir / "sample.jpg"
        img_path.write_bytes(b"fake-image")
        fields = ["sample.jpg"] + ["0.0"] * 20 + ["0.0", "0.0", "0.0", "0.0", "0.0", "-1.0", "extra"]
        (p00_dir / "p00.txt").write_text(" ".join(fields) + "\n", encoding="utf-8")
        out_csv = tmp_path / "out.csv"

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.COLOR_BGR2RGB = 4
        fake_cv2.imread = lambda _p: np.zeros((16, 16, 3), dtype=np.uint8)
        fake_cv2.cvtColor = lambda img, _code: img

        saved_rows = {}
        class _FakeDF:
            def __init__(self, rows):
                self._rows = list(rows)
            def __len__(self):
                return len(self._rows)
            def to_csv(self, path, index=False):
                saved_rows["path"] = path
                saved_rows["rows"] = self._rows
        fake_pandas = types.ModuleType("pandas")
        fake_pandas.DataFrame = _FakeDF
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda it: it

        monkeypatch.setattr(mp_proc, "DATASET_PATH", str(ds_dir))
        monkeypatch.setattr(mp_proc, "OUTPUT_CSV", str(out_csv))

        # 2. Normal completion constructs FaceLandmarker lazily, extracts row, and closes detector
        class _TrackingDetector:
            def __init__(self, should_raise=False):
                self.closed = 0
                self.detected = 0
                self.should_raise = should_raise
            def detect(self, _mp_image):
                self.detected += 1
                if self.should_raise:
                    raise RuntimeError("simulated detector failure")
                return types.SimpleNamespace(
                    face_landmarks=[make_synthetic_landmarks()],
                    facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
                )
            def close(self):
                self.closed += 1

        det_ok = _TrackingDetector(should_raise=False)
        with patch.dict(sys.modules, {"cv2": fake_cv2, "pandas": fake_pandas, "tqdm": fake_tqdm}), \
             patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=det_ok) as mock_create:
            mp_proc.process_mpiigaze()
            assert mock_create.call_count == 1
            assert det_ok.detected == 1
            assert det_ok.closed == 1
            assert len(saved_rows["rows"]) == 1

        # 3. Exception during detect still closes detector cleanly (no resource leak)
        det_err = _TrackingDetector(should_raise=True)
        with patch.dict(sys.modules, {"cv2": fake_cv2, "pandas": fake_pandas, "tqdm": fake_tqdm}), \
             patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=det_err):
            with pytest.raises(RuntimeError, match="simulated detector failure"):
                mp_proc.process_mpiigaze()
            assert det_err.closed == 1


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
            tracker.ear_state = EarFeatureState(calibration_frames=2, window_size=4)
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


    def test_nonfinite_landmarks_pose_and_prediction_rejected_and_recovers(self, mocked_gaze_tracker):
        tracker, stub = mocked_gaze_tracker
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 0

        # Non-finite pose matrix directly raises ValueError in get_head_pose_angles
        for bad in (float("nan"), float("inf"), float("-inf")):
            bad_pose = make_pose_matrix()
            bad_pose[1, 2] = bad
            with pytest.raises(ValueError):
                get_head_pose_angles(bad_pose)
        with pytest.raises(ValueError):
            get_head_pose_angles(np.eye(2, dtype=np.float32))

        # 1. Mid-calibration NaN resets incomplete calibration back to 0
        for _ in range(10):
            ts += 33
            tracker.process_frame(blank, ts)
        assert tracker.ear_state.current_calibration_frames == 10

        bad_lm = make_synthetic_landmarks()
        bad_lm[468].x = float("nan")
        stub.next_landmarks = bad_lm
        ts += 33
        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts)
        assert state == 0 and lms is None and not is_valid_output(ay, ap)
        assert math.isnan(hy) and math.isnan(hp)
        assert tracker.ear_state.current_calibration_frames == 0

        # Fewer than 478 landmarks -> rejected cleanly without IndexError
        stub.next_landmarks = make_synthetic_landmarks()[:477]
        ts += 33
        ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
        assert state == 0 and lms is None and not is_valid_output(ay, ap)

        # Complete full calibration after reset
        stub.next_landmarks = make_synthetic_landmarks()
        stub.next_pose = make_pose_matrix()
        for _ in range(CALIBRATION_FRAMES):
            ts += 33
            tracker.process_frame(blank, ts)
        assert tracker.is_calibrated

        # 2. Post-calibration valid -> invalid (NaN/+-Inf in landmark, pose, or predictor) -> valid recovery
        ts += 33
        ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
        assert state == 0 and lms is not None and is_valid_output(ay, ap)

        for bad in (float("nan"), float("inf"), float("-inf")):
            lm_corrupt = make_synthetic_landmarks()
            lm_corrupt[33].y = bad
            stub.next_landmarks = lm_corrupt
            stub.next_pose = make_pose_matrix()
            ts += 33
            ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
            assert state == 0 and lms is None and not is_valid_output(ay, ap)

            stub.next_landmarks = make_synthetic_landmarks()
            pose_corrupt = make_pose_matrix()
            pose_corrupt[0, 0] = bad
            stub.next_pose = pose_corrupt
            ts += 33
            ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
            assert state == 0 and lms is None and not is_valid_output(ay, ap)

        # Model predictor outputting NaN/+Inf is also rejected cleanly
        class _BadModel:
            def predict(self, _):
                return np.array([[float("nan")]], dtype=np.float32)

        orig_yaw_model = tracker.yaw_model
        tracker.yaw_model = _BadModel()
        stub.next_landmarks = make_synthetic_landmarks()
        stub.next_pose = make_pose_matrix()
        ts += 33
        ay, ap, _, _, state, lms = tracker.process_frame(blank, ts)
        assert state == 0 and lms is None and not is_valid_output(ay, ap)
        tracker.yaw_model = orig_yaw_model

        # Valid frame immediately recovers with calibration preserved
        ts += 33
        ay, ap, hy, hp, state, lms = tracker.process_frame(blank, ts)
        assert tracker.is_calibrated
        assert state == 0 and lms is not None and is_valid_output(ay, ap)
        assert math.isfinite(hy) and math.isfinite(hp)


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

