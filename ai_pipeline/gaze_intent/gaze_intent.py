import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import joblib
from collections import deque
import warnings
import math
from typing import Tuple

# Suppress scikit-learn version warnings during real-time execution
warnings.filterwarnings("ignore", category=UserWarning)


class GazeIntent:
    def __init__(self):
        # --- 1. PATH RESOLUTION ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.environ.get('AI_PIPELINE_ROOT', os.path.abspath(os.path.join(self.script_dir, '..')))
        model_path = os.path.join(self.script_dir, 'gesture_model.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Production Model missing at:\n{model_path}")

        # --- 2. LOAD AI MODEL ---
        self.model = joblib.load(model_path)

        # --- 3. MediaPipe INITIALIZATION ---
        mp_model_path = os.path.abspath(os.path.join(project_root, '..', 'assets', 'face_landmarker.task'))
        base_options = python.BaseOptions(model_asset_path=mp_model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- 4. LANDMARK INDICES ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473

        self.LEFT_CORNERS = (33, 133)
        self.RIGHT_CORNERS = (362, 263)

        self.LEFT_VERTICAL_BOUNDS = (159, 145)
        self.RIGHT_VERTICAL_BOUNDS = (386, 374)

        # --- 5. TEMPORAL STATE & CALIBRATION ---
        self.window_size = 10
        self.left_ear_history = deque(maxlen=self.window_size)
        self.right_ear_history = deque(maxlen=self.window_size)

        self.frame_timestamp_ms = 0

        # Dynamic Baseline Normalization Variables
        self.is_calibrated = False
        self.calibration_frame_count = 30
        self.current_calibration_frames = 0
        self.left_ear_sum = 0.0
        self.right_ear_sum = 0.0
        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0

    def shutdown(self):
        """Safely release MediaPipe resources to prevent memory leaks in production."""
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        self.shutdown()

    @staticmethod
    def _get_3d_dist(p1, p2, frame_w: int, frame_h: int) -> float:
        """Centralized helper to calculate true 3D Euclidean distance between two landmarks."""
        return math.hypot((p1.x - p2.x) * frame_w,
                          (p1.y - p2.y) * frame_h,
                          p1.z - p2.z)

    @classmethod
    def _calculate_ear(cls, face_landmarks, frame_w: int, frame_h: int, indices: list) -> float:
        """Calculates Eye Aspect Ratio (EAR) using the DRY 3D distance helper."""
        p1, p2, p3, p4, p5, p6 = (face_landmarks[i] for i in indices)

        v_dist_1 = cls._get_3d_dist(p2, p6, frame_w, frame_h)
        v_dist_2 = cls._get_3d_dist(p3, p5, frame_w, frame_h)
        h_dist = cls._get_3d_dist(p1, p4, frame_w, frame_h)

        return (v_dist_1 + v_dist_2) / (2.0 * h_dist) if h_dist > 0 else 0.0

    @staticmethod
    def _calculate_bounding_box_area(face_landmarks, frame_w: int, frame_h: int) -> float:
        """Calculates face bounding box area for normalization."""
        x_coords = [lm.x * frame_w for lm in face_landmarks]
        y_coords = [lm.y * frame_h for lm in face_landmarks]
        return float((max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)))

    def _get_rolling_features(self, history: deque, current_ear: float) -> Tuple[float, float]:
        """Calculates rolling min and variance safely."""
        if len(history) == self.window_size:
            ear_array = np.array(history)
            return float(np.min(ear_array)), float(np.var(ear_array))
        return float(current_ear), 0.0

    @classmethod
    def _get_eye_gaze_ratio(cls, landmarks, frame_w: int, frame_h: int, bounds: Tuple[int, int],
                            iris_idx: int) -> float:
        """Generic method to calculate gaze ratio across a specified eye axis."""
        p_bound1 = landmarks[bounds[0]]
        p_bound2 = landmarks[bounds[1]]
        p_iris = landmarks[iris_idx]

        axis_length = cls._get_3d_dist(p_bound1, p_bound2, frame_w, frame_h)
        if axis_length == 0:
            return 0.5

        iris_dist = cls._get_3d_dist(p_iris, p_bound2, frame_w, frame_h)
        return float(iris_dist / axis_length)

    def _calculate_gaze(self, landmarks, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """Calculates normalized X and Y cursor coordinates based on iris position."""
        l_ratio_x = self._get_eye_gaze_ratio(landmarks, frame_w, frame_h, self.LEFT_CORNERS, self.LEFT_IRIS)
        r_ratio_x = self._get_eye_gaze_ratio(landmarks, frame_w, frame_h, self.RIGHT_CORNERS, self.RIGHT_IRIS)
        gaze_x = (l_ratio_x + r_ratio_x) / 2.0

        l_ratio_y = self._get_eye_gaze_ratio(landmarks, frame_w, frame_h, self.LEFT_VERTICAL_BOUNDS, self.LEFT_IRIS)
        r_ratio_y = self._get_eye_gaze_ratio(landmarks, frame_w, frame_h, self.RIGHT_VERTICAL_BOUNDS, self.RIGHT_IRIS)
        gaze_y = (l_ratio_y + r_ratio_y) / 2.0

        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    @staticmethod
    def _normalize_ear(raw_ear: float, baseline: float) -> float:
        """Normalizes the current EAR against the user's calibrated resting baseline."""
        return raw_ear / baseline if baseline > 0 else 0.0

    def process_frame(self, rgb_frame_array: np.ndarray, timestamp_ms: int) -> Tuple[float, float, int]:
        """
        THE C++ BRIDGE METHOD.
        Returns a Tuple: (Cursor X, Cursor Y, Eye State)
        State -1 indicates the system is currently calibrating.
        """
        gaze_x, gaze_y = -1.0, -1.0
        state = 0  # 0 = Neutral

        frame_h, frame_w, _ = rgb_frame_array.shape
        self.frame_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_array)
        results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # 1. Extract raw EAR
            raw_left_ear = self._calculate_ear(landmarks, frame_w, frame_h, self.LEFT_EYE)
            raw_right_ear = self._calculate_ear(landmarks, frame_w, frame_h, self.RIGHT_EYE)

            # 2. Gaze Math (Runs independently of calibration)
            gaze_x, gaze_y = self._calculate_gaze(landmarks, frame_w, frame_h)

            # 3. Handle Dynamic Calibration
            if not self.is_calibrated:
                self.left_ear_sum += raw_left_ear
                self.right_ear_sum += raw_right_ear
                self.current_calibration_frames += 1

                if self.current_calibration_frames >= self.calibration_frame_count:
                    self.baseline_left_ear = self.left_ear_sum / self.calibration_frame_count
                    self.baseline_right_ear = self.right_ear_sum / self.calibration_frame_count
                    self.is_calibrated = True

                # Return -1 to tell C++ to hold the cursor and show a "Calibrating" UI
                return float(gaze_x), float(gaze_y), -1

            # 4. Normalize the EAR based on the user's baseline
            left_ear = self._normalize_ear(raw_left_ear, self.baseline_left_ear)
            right_ear = self._normalize_ear(raw_right_ear, self.baseline_right_ear)

            # 5. Proceed with AI Inference using Normalized data
            self.left_ear_history.append(left_ear)
            self.right_ear_history.append(right_ear)

            l_min, l_var = self._get_rolling_features(self.left_ear_history, left_ear)
            r_min, r_var = self._get_rolling_features(self.right_ear_history, right_ear)
            bounding_box_area = self._calculate_bounding_box_area(landmarks, frame_w, frame_h)

            features = np.array([[left_ear, l_min, l_var, right_ear, r_min, r_var, bounding_box_area]])
            state = int(self.model.predict(features)[0])

        return float(gaze_x), float(gaze_y), int(state)