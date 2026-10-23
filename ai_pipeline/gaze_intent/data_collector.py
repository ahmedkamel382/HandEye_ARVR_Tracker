import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import time
import os
import math
from collections import deque


class DataCollector:
    def __init__(self, output_file=None):
        # --- 1. DIRECTORY SETUP ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        if output_file is None:
            self.output_file = os.path.join(self.script_dir, 'datasets', 'blink_dataset.csv')
        else:
            self.output_file = output_file

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # --- 2. MODEL INITIALIZATION ---
        model_path = os.path.abspath(os.path.join(self.script_dir, '..', '..', 'assets', 'face_landmarker.task'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at:\n{model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- 3. LANDMARK MAPPING ---
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        self.LEFT_IRIS_INDEX = 468
        self.RIGHT_IRIS_INDEX = 473

        # --- 4. STATE & TEMPORAL WINDOW ---
        self.current_label = 0

        self.window_size = 10
        self.left_ear_history = deque(maxlen=self.window_size)
        self.right_ear_history = deque(maxlen=self.window_size)

        # --- 5. CALIBRATION VARIABLES ---
        self.is_calibrated = False
        self.calibration_frame_count = 30
        self.current_calibration_frames = 0
        self.left_ear_sum = 0.0
        self.right_ear_sum = 0.0
        self.baseline_left_ear = 1.0
        self.baseline_right_ear = 1.0

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

    def _get_rolling_features(self, history: deque, current_ear: float) -> tuple[float, float]:
        """Calculates rolling min and variance safely."""
        if len(history) == self.window_size:
            ear_array = np.array(history)
            return float(np.min(ear_array)), float(np.var(ear_array))
        return float(current_ear), 0.0

    def run(self):
        file_exists = os.path.isfile(self.output_file)

        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    'Timestamp',
                    'Left_EAR', 'Left_Min_10f', 'Left_Var_10f',
                    'Right_EAR', 'Right_Min_10f', 'Right_Var_10f',
                    'BoundingBox_Area', 'Label'
                ])

            cap = cv2.VideoCapture(0)

            print("--- DATA COLLECTOR STARTED ---")
            print("Look neutrally at the camera for calibration...")

            frame_timestamp_ms = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_timestamp_ms += int(1000 / 30)

                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                results = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

                if results.face_landmarks:
                    first_face_landmarks = results.face_landmarks[0]

                    raw_left_ear = self._calculate_ear(first_face_landmarks, frame_w, frame_h, self.LEFT_EYE_INDICES)
                    raw_right_ear = self._calculate_ear(first_face_landmarks, frame_w, frame_h, self.RIGHT_EYE_INDICES)

                    if not self.is_calibrated:
                        self.left_ear_sum += raw_left_ear
                        self.right_ear_sum += raw_right_ear
                        self.current_calibration_frames += 1

                        cv2.putText(frame,
                                    f"CALIBRATING: {self.current_calibration_frames}/{self.calibration_frame_count}",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                        if self.current_calibration_frames >= self.calibration_frame_count:
                            self.baseline_left_ear = self.left_ear_sum / self.calibration_frame_count
                            self.baseline_right_ear = self.right_ear_sum / self.calibration_frame_count
                            self.is_calibrated = True

                            print("\nCalibration Complete!")
                            print("Press 'SPACE' for NEUTRAL (Hover/Default)")
                            print("Press '1' for LEFT WINK (Left Click)")
                            print("Press '2' for RIGHT WINK (Right Click)")
                            print("Press '3' for SUSTAINED CLOSURE (Pause/Sleep)")
                            print("Press 'Q' to save and exit.")
                    else:
                        left_ear = raw_left_ear / self.baseline_left_ear if self.baseline_left_ear > 0 else 0.0
                        right_ear = raw_right_ear / self.baseline_right_ear if self.baseline_right_ear > 0 else 0.0

                        self.left_ear_history.append(left_ear)
                        self.right_ear_history.append(right_ear)

                        l_min_ear, l_var_ear = self._get_rolling_features(self.left_ear_history, left_ear)
                        r_min_ear, r_var_ear = self._get_rolling_features(self.right_ear_history, right_ear)

                        bounding_box_area = self._calculate_bounding_box_area(first_face_landmarks, frame_w, frame_h)

                        writer.writerow([
                            time.time(),
                            left_ear, l_min_ear, l_var_ear,
                            right_ear, r_min_ear, r_var_ear,
                            bounding_box_area, self.current_label
                        ])

                        label_text = {
                            0: "STATE: NEUTRAL (HOVER)",
                            1: "STATE: LEFT WINK (LEFT CLICK)",
                            2: "STATE: RIGHT WINK (RIGHT CLICK)",
                            3: "STATE: SUSTAINED CLOSURE (PAUSE)"
                        }
                        colors = {
                            0: (0, 255, 0),
                            1: (0, 255, 255),
                            2: (255, 255, 0),
                            3: (0, 0, 255)
                        }

                        cv2.putText(frame, label_text[self.current_label], (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[self.current_label], 2)

                        cv2.putText(frame, f"L EAR: {left_ear:.3f} | R EAR: {right_ear:.3f}", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"L Min: {l_min_ear:.3f} | R Min: {r_min_ear:.3f}", (30, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        cv2.putText(frame, f"L Var: {l_var_ear:.5f} | R Var: {r_var_ear:.5f}", (30, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # Draw Debug Elements
                    for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 2, (0, 255, 0), -1)

                    for idx in [self.LEFT_IRIS_INDEX, self.RIGHT_IRIS_INDEX]:
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 3, (0, 0, 255), -1)

                cv2.imshow('Data Collector (Press Q to Quit)', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q')]:
                    break
                elif key == ord(' '):
                    self.current_label = 0
                elif key == ord('1'):
                    self.current_label = 1
                elif key == ord('2'):
                    self.current_label = 2
                elif key == ord('3'):
                    self.current_label = 3

        cap.release()
        cv2.destroyAllWindows()
        print(f"Data successfully saved to:\n{self.output_file}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()