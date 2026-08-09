import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from typing import Tuple


class HandTracker:
    """
    THE C++ BRIDGE CLASS.

    Standalone Python hand-tracking module. Zero OS-level logic -- its only
    job is to turn a raw webcam frame into a standardized (Cursor X, Cursor Y,
    Gesture State) payload for the Pybind11 core engine to consume.

    Unlike GazeIntent (which needs a trained Random Forest because blink
    nuance depends on each user's facial anatomy), hand poses are mechanically
    rigid -- Pinch and Fist are reliably detected with pure vector geometry
    on the 21 MediaPipe hand landmarks. No training pipeline, no .pkl model,
    virtually zero extra CPU cost next to the eye-tracking model running in
    parallel on the same machine.
    """

    def __init__(self):
        # --- 1. PATH RESOLUTION ---
        # Mirrors GazeIntent's layout exactly:
        # script_dir      -> ai_pipeline/hands/
        # project_root    -> ai_pipeline/            (one level up)
        # assets folder   -> <repo_root>/assets/     (one more level up)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get('AI_PIPELINE_ROOT', os.path.abspath(os.path.join(self.script_dir, '..')))

        model_path = os.path.abspath(os.path.join(project_root, '..', 'assets', 'hand_landmarker.task'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Hand model missing at:\n{model_path}")

        # --- 2. MediaPipe INITIALIZATION ---
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # --- 3. LANDMARK INDICES ---
        # Standard MediaPipe 21-point hand topology.
        self.WRIST = 0

        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

        # (tip, mcp) pairs for the four fingers used in Fist detection.
        # Thumb is intentionally excluded -- its curl geometry is different
        # from the other four fingers and would need its own rule.
        self.CURL_FINGER_PAIRS = [
            (self.INDEX_TIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_MCP),
        ]

        # --- 4. TUNING THRESHOLDS ---
        # Pinch distance is normalized against palm size (wrist -> middle MCP)
        # so the threshold stays valid whether the hand is close to or far
        # from the camera.
        # NOTE: raised from 0.35 -> 0.5 after live testing with
        # hand_data_collector.py showed real pinches landing around 0.46,
        # above the old threshold (which caused them to be misread as Fist,
        # since the other 3 fingers naturally curl a bit during a pinch too).
        self.PINCH_THRESHOLD_RATIO = 0.5

        # A finger counts as "curled" (closed) when its tip sits closer to
        # the wrist than its own MCP knuckle does. Fist = at least 3 of the
        # 4 tracked fingers curled.
        self.FIST_MIN_CURLED_FINGERS = 3

    @staticmethod
    def _landmark_xy(landmarks, idx: int, frame_w: int, frame_h: int) -> np.ndarray:
        """Converts a normalized MediaPipe landmark to pixel coordinates."""
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    def _get_palm_size(self, landmarks, frame_w: int, frame_h: int) -> float:
        """Distance from wrist to middle-finger MCP, used as a scale reference."""
        wrist = self._landmark_xy(landmarks, self.WRIST, frame_w, frame_h)
        middle_mcp = self._landmark_xy(landmarks, self.MIDDLE_MCP, frame_w, frame_h)
        return float(np.linalg.norm(wrist - middle_mcp))

    def _is_pinching(self, landmarks, frame_w: int, frame_h: int, palm_size: float) -> bool:
        """
        Detects a Pinch by measuring the Euclidean distance between the
        index fingertip (Landmark 8) and thumb tip (Landmark 4), normalized
        by palm size so the threshold works at any distance from the camera.
        """
        if palm_size == 0:
            return False

        thumb_tip = self._landmark_xy(landmarks, self.THUMB_TIP, frame_w, frame_h)
        index_tip = self._landmark_xy(landmarks, self.INDEX_TIP, frame_w, frame_h)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

        return (pinch_dist / palm_size) < self.PINCH_THRESHOLD_RATIO

    def _is_fist(self, landmarks, frame_w: int, frame_h: int) -> bool:
        """
        Detects a closed Fist by checking, for each of the 4 main fingers,
        whether its tip sits closer to the wrist than its own MCP knuckle
        does (i.e. the finger is curled inward rather than extended outward).
        """
        wrist = self._landmark_xy(landmarks, self.WRIST, frame_w, frame_h)
        curled_count = 0

        for tip_idx, mcp_idx in self.CURL_FINGER_PAIRS:
            tip = self._landmark_xy(landmarks, tip_idx, frame_w, frame_h)
            mcp = self._landmark_xy(landmarks, mcp_idx, frame_w, frame_h)

            tip_to_wrist = np.linalg.norm(tip - wrist)
            mcp_to_wrist = np.linalg.norm(mcp - wrist)

            if tip_to_wrist < mcp_to_wrist:
                curled_count += 1

        return curled_count >= self.FIST_MIN_CURLED_FINGERS

    # noinspection DuplicatedCode
    def process_frame(self, frame_array: np.ndarray) -> Tuple[float, float, int]:
        """
        THE C++ BRIDGE METHOD.
        Takes a raw BGR numpy array from OpenCV.
        Returns a Tuple: (Cursor X, Cursor Y, Gesture State)

        Gesture State values:
            0 = Neutral / Hover
            1 = Pinch   (e.g. Left Click / Drag)
            2 = Fist    (e.g. Pause / Alternate action)
        """
        cursor_x, cursor_y = -1.0, -1.0
        state = 0  # 0 = Neutral

        frame = cv2.flip(frame_array, 1)
        frame_h, frame_w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]

            # Cursor position: normalized index fingertip (Landmark 8),
            # already in [0.0, 1.0] range directly from MediaPipe -- no
            # frame_w/frame_h division needed, same pattern as GazeIntent's
            # iris .y usage.
            cursor_x = max(0.0, min(1.0, landmarks[self.INDEX_TIP].x))
            cursor_y = max(0.0, min(1.0, landmarks[self.INDEX_TIP].y))

            # Gesture classification via pure geometric heuristics.
            palm_size = self._get_palm_size(landmarks, frame_w, frame_h)

            if self._is_pinching(landmarks, frame_w, frame_h, palm_size):
                state = 1
            elif self._is_fist(landmarks, frame_w, frame_h):
                state = 2
            else:
                state = 0

        return float(cursor_x), float(cursor_y), int(state)