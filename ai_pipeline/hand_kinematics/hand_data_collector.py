import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import time
import os
from collections import deque


class HandDataCollector:
    """
    Standalone visual debug + CSV logging tool for the hand-tracking pipeline.

    Mirrors data_collector.py's structure exactly, but for hand geometry
    instead of eye geometry. Lets you SEE the Pinch/Fist heuristics from
    hand_tracker.py working live on your own webcam, and (optionally) log
    labeled frames to CSV in case a future Random Forest (Path 2) is ever
    needed for more complex gestures.
    """

    def __init__(self, output_file=None):
        # --- 1. DIRECTORY SETUP ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        if output_file is None:
            self.output_file = os.path.join(self.script_dir, 'datasets', 'hand_dataset.csv')
        else:
            self.output_file = output_file

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # --- 2. MODEL INITIALIZATION ---
        # Same two-levels-up resolution as hand_tracker.py:
        # ai_pipeline/hand_kinematics/ -> ai_pipeline/ -> <repo_root>/assets/
        model_path = os.path.abspath(os.path.join(self.script_dir, '..', '..', 'assets', 'hand_landmarker.task'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at:\n{model_path}")

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

        # --- 3. LANDMARK MAPPING (same indices as hand_tracker.py) ---
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

        self.CURL_FINGER_PAIRS = [
            (self.INDEX_TIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_MCP),
        ]

        # Full 21-point connection topology, just for the visual skeleton.
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        # --- 4. TUNING THRESHOLDS (same as hand_tracker.py) ---
        # Raised from 0.35 -> 0.5 after live testing showed real pinches
        # landing around 0.46, which the old threshold misclassified as Fist.
        self.PINCH_THRESHOLD_RATIO = 0.5
        self.FIST_MIN_CURLED_FINGERS = 3

        # --- 5. STATE & TEMPORAL WINDOW ---
        self.current_label = 0

        # 15 frames (~0.5s at 30fps) sliding window, same pattern as
        # data_collector.py's EAR history -- kept here for feature parity
        # in case a future Random Forest ever needs rolling min/var.
        self.window_size = 15
        self.pinch_ratio_history = deque(maxlen=self.window_size)

    @staticmethod
    def _landmark_xy(landmarks, idx, frame_w, frame_h):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    def _get_palm_size(self, landmarks, frame_w, frame_h):
        wrist = self._landmark_xy(landmarks, self.WRIST, frame_w, frame_h)
        middle_mcp = self._landmark_xy(landmarks, self.MIDDLE_MCP, frame_w, frame_h)
        return float(np.linalg.norm(wrist - middle_mcp))

    def _calculate_pinch_ratio(self, landmarks, frame_w, frame_h, palm_size):
        """Normalized thumb-to-index distance (lower = closer to pinching)."""
        if palm_size == 0:
            return 1.0
        thumb_tip = self._landmark_xy(landmarks, self.THUMB_TIP, frame_w, frame_h)
        index_tip = self._landmark_xy(landmarks, self.INDEX_TIP, frame_w, frame_h)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        return pinch_dist / palm_size

    def _count_curled_fingers(self, landmarks, frame_w, frame_h):
        wrist = self._landmark_xy(landmarks, self.WRIST, frame_w, frame_h)
        curled_count = 0
        for tip_idx, mcp_idx in self.CURL_FINGER_PAIRS:
            tip = self._landmark_xy(landmarks, tip_idx, frame_w, frame_h)
            mcp = self._landmark_xy(landmarks, mcp_idx, frame_w, frame_h)
            if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist):
                curled_count += 1
        return curled_count

    def _get_rolling_features(self, history, current_value):
        """Helper method to calculate rolling min and variance to prevent duplicated code."""
        if len(history) == self.window_size:
            arr = np.array(history)
            return float(np.min(arr)), float(np.var(arr))
        return float(current_value), 0.0

    def run(self):
        file_exists = os.path.isfile(self.output_file)

        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    'Timestamp',
                    'Pinch_Ratio', 'Pinch_Min_15f', 'Pinch_Var_15f',
                    'Curled_Finger_Count',
                    'Cursor_X', 'Cursor_Y',
                    'Label'
                ])

            cap = cv2.VideoCapture(0)

            print("--- HAND DATA COLLECTOR STARTED ---")
            print("Press 'SPACE' for NEUTRAL (Hover/Default)")
            print("Press '1' for PINCH (Left Click)")
            print("Press '2' for FIST (Alternate Action)")
            print("Press 'Q' to save and exit.")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.detector.detect(mp_image)

                pinch_ratio = 1.0
                pinch_min = pinch_var = 0.0
                curled_count = 0
                cursor_x = cursor_y = -1.0
                detected_state = 0

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]

                    # 1. Geometry
                    palm_size = self._get_palm_size(landmarks, frame_w, frame_h)
                    pinch_ratio = self._calculate_pinch_ratio(landmarks, frame_w, frame_h, palm_size)
                    curled_count = self._count_curled_fingers(landmarks, frame_w, frame_h)

                    # 2. Update rolling history
                    self.pinch_ratio_history.append(pinch_ratio)
                    pinch_min, pinch_var = self._get_rolling_features(self.pinch_ratio_history, pinch_ratio)

                    # 3. Cursor position (normalized index fingertip)
                    cursor_x = max(0.0, min(1.0, landmarks[self.INDEX_TIP].x))
                    cursor_y = max(0.0, min(1.0, landmarks[self.INDEX_TIP].y))

                    # 4. Live detection preview (what hand_tracker.py would output)
                    if pinch_ratio < self.PINCH_THRESHOLD_RATIO:
                        detected_state = 1
                    elif curled_count >= self.FIST_MIN_CURLED_FINGERS:
                        detected_state = 2
                    else:
                        detected_state = 0

                    # 5. Save to CSV
                    writer.writerow([
                        time.time(),
                        pinch_ratio, pinch_min, pinch_var,
                        curled_count,
                        cursor_x, cursor_y,
                        self.current_label
                    ])

                    # 6. Draw Debug Skeleton
                    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
                    for start_idx, end_idx in self.HAND_CONNECTIONS:
                        cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
                    for (x, y) in points:
                        cv2.circle(frame, (x, y), 4, (0, 128, 255), -1)

                    # Highlight thumb tip + index tip in red (the Pinch pair)
                    cv2.circle(frame, points[self.THUMB_TIP], 6, (0, 0, 255), -1)
                    cv2.circle(frame, points[self.INDEX_TIP], 6, (0, 0, 255), -1)

                # --- UI OVERLAY ---
                label_text = {
                    0: "LABEL: NEUTRAL",
                    1: "LABEL: PINCH",
                    2: "LABEL: FIST",
                }
                detected_text = {
                    0: "DETECTED: Neutral",
                    1: "DETECTED: Pinch",
                    2: "DETECTED: Fist",
                }
                colors = {
                    0: (0, 255, 0),    # Green
                    1: (0, 255, 255),  # Yellow
                    2: (0, 0, 255),    # Red
                }

                cv2.putText(frame, label_text[self.current_label], (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colors[self.current_label], 2)
                cv2.putText(frame, detected_text[detected_state], (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[detected_state], 2)

                cv2.putText(frame, f"Pinch Ratio: {pinch_ratio:.3f} (thresh {self.PINCH_THRESHOLD_RATIO})",
                            (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Curled Fingers: {curled_count}/4",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Cursor: ({cursor_x:.2f}, {cursor_y:.2f})",
                            (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow('Hand Data Collector (Press Q to Quit)', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q')]:
                    break
                elif key == ord(' '):
                    self.current_label = 0
                elif key == ord('1'):
                    self.current_label = 1
                elif key == ord('2'):
                    self.current_label = 2

        cap.release()
        cv2.destroyAllWindows()
        print(f"Data successfully saved to:\n{self.output_file}")


if __name__ == "__main__":
    collector = HandDataCollector()
    collector.run()