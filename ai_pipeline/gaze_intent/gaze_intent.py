import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import joblib
from collections import deque
import warnings
from typing import Tuple

# Suppress scikit-learn version warnings during real-time execution
warnings.filterwarnings("ignore", category=UserWarning)


class GazeIntent:
    def __init__(self):
        # --- 1. PATH RESOLUTION ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Locate the trained Random Forest model
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
            running_mode=vision.RunningMode.IMAGE,
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

        # Eye corners for horizontal gaze calculation (Outer, Inner)
        self.LEFT_CORNERS = (33, 133)
        self.RIGHT_CORNERS = (362, 263)

        # Eyelid centers for vertical gaze calculation (Top, Bottom)
        self.LEFT_VERTICAL_BOUNDS = (159, 145)
        self.RIGHT_VERTICAL_BOUNDS = (386, 374)

        # --- 5. TEMPORAL STATE ---
        self.window_size = 15
        self.left_ear_history = deque(maxlen=self.window_size)
        self.right_ear_history = deque(maxlen=self.window_size)

    def close(self):
        """Safely release MediaPipe resources to prevent memory leaks in production."""
        if hasattr(self, 'detector'):
            self.detector.close()

    def __del__(self):
        self.close()

    @staticmethod
    def _calculate_ear(face_landmarks, frame_w: int, frame_h: int, indices: list) -> float:
        """Calculates Eye Aspect Ratio (EAR)."""
        coords = [np.array([face_landmarks[idx].x * frame_w, face_landmarks[idx].y * frame_h]) for idx in indices]
        p1, p2, p3, p4, p5, p6 = coords
        v_dist_1 = float(np.linalg.norm(p2 - p6))
        v_dist_2 = float(np.linalg.norm(p3 - p5))
        h_dist = float(np.linalg.norm(p1 - p4))
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

    @staticmethod
    def _get_horizontal_eye_gaze_ratio(landmarks, frame_w: int, corners: Tuple[int, int], iris_idx: int) -> float:
        """Calculates the horizontal gaze ratio for a single eye."""
        p_outer = landmarks[corners[0]]
        p_inner = landmarks[corners[1]]
        p_iris = landmarks[iris_idx]

        outer_x = p_outer.x * frame_w
        inner_x = p_inner.x * frame_w
        iris_x = p_iris.x * frame_w

        min_x = min(outer_x, inner_x)
        max_x = max(outer_x, inner_x)
        eye_width = max_x - min_x

        if eye_width == 0:
            return 0.5

        return float((iris_x - min_x) / eye_width)

    @staticmethod
    def _get_vertical_eye_gaze_ratio(landmarks, frame_h: int, bounds: Tuple[int, int], iris_idx: int) -> float:
        """Calculates the vertical gaze ratio for a single eye relative to the eyelids."""
        p_top = landmarks[bounds[0]]
        p_bottom = landmarks[bounds[1]]
        p_iris = landmarks[iris_idx]

        top_y = p_top.y * frame_h
        bottom_y = p_bottom.y * frame_h
        iris_y = p_iris.y * frame_h

        min_y = min(top_y, bottom_y)
        max_y = max(top_y, bottom_y)
        eye_height = max_y - min_y

        if eye_height == 0:
            return 0.5

        return float((iris_y - min_y) / eye_height)

    def _calculate_gaze(self, landmarks, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """
        Calculates normalized X and Y cursor coordinates based on iris position relative to eye boundaries.
        Returns values between 0.0 and 1.0.
        """
        # Horizontal ratio (X)
        l_ratio_x = self._get_horizontal_eye_gaze_ratio(landmarks, frame_w, self.LEFT_CORNERS, self.LEFT_IRIS)
        r_ratio_x = self._get_horizontal_eye_gaze_ratio(landmarks, frame_w, self.RIGHT_CORNERS, self.RIGHT_IRIS)
        gaze_x = (l_ratio_x + r_ratio_x) / 2.0

        # Vertical ratio (Y)
        l_ratio_y = self._get_vertical_eye_gaze_ratio(landmarks, frame_h, self.LEFT_VERTICAL_BOUNDS, self.LEFT_IRIS)
        r_ratio_y = self._get_vertical_eye_gaze_ratio(landmarks, frame_h, self.RIGHT_VERTICAL_BOUNDS, self.RIGHT_IRIS)
        gaze_y = (l_ratio_y + r_ratio_y) / 2.0

        # Clamp between 0.0 and 1.0 for absolute bounds safety
        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    # noinspection DuplicatedCode
    def process_frame(self, frame_array: np.ndarray) -> Tuple[float, float, int]:
        """
        THE C++ BRIDGE METHOD.
        Takes a raw BGR numpy array from OpenCV.
        Returns a Tuple: (Cursor X, Cursor Y, Eye State)
        """
        gaze_x, gaze_y = -1.0, -1.0
        state = 0  # 0 = Neutral

        frame = cv2.flip(frame_array, 1)
        frame_h, frame_w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # Extract EAR for both eyes
            left_ear = self._calculate_ear(landmarks, frame_w, frame_h, self.LEFT_EYE)
            right_ear = self._calculate_ear(landmarks, frame_w, frame_h, self.RIGHT_EYE)

            self.left_ear_history.append(left_ear)
            self.right_ear_history.append(right_ear)

            # Calculate Rolling Features
            l_min, l_var = self._get_rolling_features(self.left_ear_history, left_ear)
            r_min, r_var = self._get_rolling_features(self.right_ear_history, right_ear)
            bounding_box_area = self._calculate_bounding_box_area(landmarks, frame_w, frame_h)

            # AI Inference
            features = np.array([[left_ear, l_min, l_var, right_ear, r_min, r_var, bounding_box_area]])
            state = int(self.model.predict(features)[0])

            # Gaze Math
            gaze_x, gaze_y = self._calculate_gaze(landmarks, frame_w, frame_h)

        return float(gaze_x), float(gaze_y), int(state)