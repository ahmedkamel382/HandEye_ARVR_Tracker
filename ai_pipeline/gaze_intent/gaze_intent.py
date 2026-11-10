import math
import os
import warnings
from collections import deque
from typing import Tuple, Optional, Any

import cv2
import mediapipe as mp
import numpy as np
import tl2cgen
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore", category=UserWarning)


def get_head_pose_angles(matrix) -> Tuple[float, float, float]:
    R = matrix[:3, :3]
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
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class GazeIntent:
    def __init__(self):
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

        # --- 3. LANDMARK INDICES ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473
        self.L_CORNERS = (33, 133)
        self.R_CORNERS = (362, 263)

        # --- 4. TEMPORAL STATE & CALIBRATION ---
        self.window_size = 10
        self.left_ear_history: deque[float] = deque(maxlen=self.window_size)
        self.right_ear_history: deque[float] = deque(maxlen=self.window_size)

        self.is_calibrated = False
        self.calibration_frame_count = 30
        self.current_calibration_frames = 0
        self.left_ear_sum = 0.0
        self.right_ear_sum = 0.0
        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0

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

    def shutdown(self):
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        self.shutdown()

    @staticmethod
    def _get_3d_dist(p1, p2, frame_w: int, frame_h: int) -> float:
        # Scale normalized coordinates to physical pixels (MediaPipe z scales with width)
        dx = (float(p1.x) - float(p2.x)) * frame_w
        dy = (float(p1.y) - float(p2.y)) * frame_h
        dz = (float(p1.z) - float(p2.z)) * frame_w
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @classmethod
    def _calculate_ear(cls, face_landmarks, indices: list, frame_w: int, frame_h: int) -> float:
        p1, p2, p3, p4, p5, p6 = (face_landmarks[i] for i in indices)
        v_dist_1 = cls._get_3d_dist(p2, p6, frame_w, frame_h)
        v_dist_2 = cls._get_3d_dist(p3, p5, frame_w, frame_h)
        h_dist = cls._get_3d_dist(p1, p4, frame_w, frame_h)
        return (v_dist_1 + v_dist_2) / (2.0 * h_dist) if h_dist > 1e-12 else 0.0

    @classmethod
    def _get_iris_ratio(cls, landmarks, corners: Tuple[int, int], iris_idx: int, axis: str) -> float:
        """Measure the iris against the static line joining the eye corners in normalized 2D space."""
        p_corner1, p_corner2 = landmarks[corners[0]], landmarks[corners[1]]
        p_iris = landmarks[iris_idx]

        c1_x, c1_y = float(p_corner1.x), float(p_corner1.y)
        c2_x, c2_y = float(p_corner2.x), float(p_corner2.y)
        ir_x, ir_y = float(p_iris.x), float(p_iris.y)

        base_x = c2_x - c1_x
        base_y = c2_y - c1_y
        baseline_sq = base_x * base_x + base_y * base_y
        if baseline_sq < 1e-12:
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

    @staticmethod
    def _get_head_pose_angles(transformation_matrix) -> Tuple[float, float, float]:
        return get_head_pose_angles(transformation_matrix)

    @staticmethod
    def _window_stats(history: deque[float]) -> Tuple[float, float]:
        """Pure-Python min and population variance to avoid NumPy generic type warnings."""
        n = len(history)
        if n == 0:
            return 1.0, 0.0
        w_min = min(history)
        mean = sum(history) / n
        w_var = sum((x - mean) ** 2 for x in history) / n
        return float(w_min), float(w_var)

    def process_frame(
        self, rgb_frame_array: np.ndarray, timestamp_ms: int
    ) -> Tuple[float, float, float, float, int, Optional[Any]]:
        """
        Returns (raw_abs_yaw, raw_abs_pitch, head_yaw, head_pitch, state, landmarks)
        When calibrating (state == -1) or when no face is detected (landmarks is None),
        returns (-1.0, -1.0, 0.0, 0.0, state, landmarks) so invalid coordinates never
        enter downstream DSP filters.
        """
        frame_h, frame_w, _ = rgb_frame_array.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_array)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks and results.facial_transformation_matrixes:
            landmarks = results.face_landmarks[0]
            pose_matrix = results.facial_transformation_matrixes[0]

            # 1. Blink Calibration & ML Inference
            raw_left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, frame_w, frame_h)
            raw_right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, frame_w, frame_h)

            if not self.is_calibrated:
                # Guard against corrupted/zero EAR frames during calibration
                if raw_left_ear > 1e-6 and raw_right_ear > 1e-6:
                    self.left_ear_sum += raw_left_ear
                    self.right_ear_sum += raw_right_ear
                    self.current_calibration_frames += 1

                    if self.current_calibration_frames >= self.calibration_frame_count:
                        self.baseline_left_ear = self.left_ear_sum / self.calibration_frame_count
                        self.baseline_right_ear = self.right_ear_sum / self.calibration_frame_count
                        self.is_calibrated = True

                # Issue #3 Fix: Return -1.0 sentinel coordinates during calibration
                return -1.0, -1.0, 0.0, 0.0, -1, landmarks

            left_ear = raw_left_ear / self.baseline_left_ear
            right_ear = raw_right_ear / self.baseline_right_ear
            self.left_ear_history.append(left_ear)
            self.right_ear_history.append(right_ear)

            l_min, l_var = self._window_stats(self.left_ear_history)
            r_min, r_var = self._window_stats(self.right_ear_history)

            wink_batch = tl2cgen.DMatrix(
                np.array([[left_ear, l_min, l_var, right_ear, r_min, r_var]], dtype=np.float32)
            )
            state = int(self.wink_model.predict(wink_batch)[0].argmax())

            # 2. Extract MPIIGaze Features
            l_x = self._get_iris_ratio(landmarks, self.L_CORNERS, self.LEFT_IRIS, "x")
            r_x = self._get_iris_ratio(landmarks, self.R_CORNERS, self.RIGHT_IRIS, "x")
            student_x = (l_x + r_x) / 2.0

            l_y = self._get_iris_ratio(landmarks, self.L_CORNERS, self.LEFT_IRIS, "y")
            r_y = self._get_iris_ratio(landmarks, self.R_CORNERS, self.RIGHT_IRIS, "y")
            student_y = (l_y + r_y) / 2.0

            head_pitch, head_yaw, head_roll = get_head_pose_angles(pose_matrix)

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

            return abs_yaw, abs_pitch, head_yaw, head_pitch, state, landmarks

        # Issue #4 Fix: Face lost -> clear temporal EAR window & return invalid (-1.0, -1.0)
        self.reset_tracking_state()
        return -1.0, -1.0, 0.0, 0.0, 0, None