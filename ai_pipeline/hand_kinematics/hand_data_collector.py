import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import time
import os

from hand_features import FEATURE_NAMES, extract_features, result_to_points, load_gesture_model


class HandDataCollector:
    """
    Collects LABELED training data for the gesture Random Forest.

    Controls:
        SPACE -> start recording as NEUTRAL (label 0)
        1     -> start recording as PINCH   (label 1)
        2     -> start recording as FIST    (label 2)
        X     -> PAUSE (nothing is saved)   <- starts paused on purpose
        Q     -> save & quit

    Tips for data that actually generalizes (this is what fixes the old bug):
      * For EACH gesture, slowly rotate/tilt the hand toward and away from the
        camera, move it closer/farther, use both slight angles of the wrist.
      * Include "in-between" poses: half-open hand, loose fist, wide pinch,
        relaxed hand with fingers slightly bent -> label them NEUTRAL.
      * Aim for ~1500+ frames per class, over 2+ separate runs (each run is a
        'session'; the trainer uses sessions for an honest accuracy estimate).
    """

    PAUSED = -1

    def __init__(self, output_file=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_file = output_file or os.path.join(self.script_dir, 'datasets', 'hand_dataset_v2.csv')
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        self.assets_dir = os.path.abspath(os.path.join(self.script_dir, '..', '..', 'assets'))
        model_path = os.path.join(self.assets_dir, 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at:\n{model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            # VIDEO mode, matching hand_tracker.py: lets MediaPipe track the
            # hand forward between frames instead of re-detecting the palm
            # from scratch every frame, which is more robust on hard poses
            # (tight Fist near the face) -- see hand_tracker.py's docstring.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            # Matches hand_tracker.py: lowered so hard poses (tight Fist near
            # the face) are still detected instead of dropping frames.
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # VIDEO mode requires a strictly increasing timestamp (ms) per call.
        self._start_time = time.time()
        self._last_timestamp_ms = -1

        # Optional live preview of the trained model (if one already exists).
        self.rf = load_gesture_model(os.path.join(self.assets_dir, 'hand_gesture_rf.pkl'))

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ]
        self.current_label = self.PAUSED
        self.counts = {0: 0, 1: 0, 2: 0}
        self.session_id = int(time.time())

    def run(self):
        file_exists = os.path.isfile(self.output_file)
        names = {self.PAUSED: "PAUSED (not saving)", 0: "NEUTRAL", 1: "PINCH", 2: "FIST"}
        colors = {self.PAUSED: (180, 180, 180), 0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Session', *FEATURE_NAMES, 'Label'])

            cap = cv2.VideoCapture(0)
            print("--- HAND DATA COLLECTOR STARTED (paused) ---")
            print("SPACE=Neutral  1=Pinch  2=Fist  X=Pause  Q=Save & quit")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int((time.time() - self._start_time) * 1000)
                if timestamp_ms <= self._last_timestamp_ms:
                    timestamp_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = timestamp_ms

                result = self.detector.detect_for_video(mp_image, timestamp_ms)
                points = result_to_points(result)

                pred_text = ""
                if points is not None:
                    feats = extract_features(points)

                    if self.current_label != self.PAUSED:
                        writer.writerow([time.time(), self.session_id, *feats, self.current_label])
                        self.counts[self.current_label] += 1

                    if self.rf is not None:
                        pred_text = f"MODEL: {names[int(self.rf.predict(feats.reshape(1, -1))[0])]}"

                    lms = result.hand_landmarks[0]
                    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in lms]
                    for a, b in self.HAND_CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
                    for p in pts:
                        cv2.circle(frame, p, 4, (0, 128, 255), -1)
                    cv2.circle(frame, pts[4], 6, (0, 0, 255), -1)
                    cv2.circle(frame, pts[8], 6, (0, 0, 255), -1)
                else:
                    cv2.putText(frame, "No hand detected", (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, f"LABEL: {names[self.current_label]}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colors[self.current_label], 2)
                cv2.putText(frame, f"Saved  N:{self.counts[0]}  P:{self.counts[1]}  F:{self.counts[2]}",
                            (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if pred_text:
                    cv2.putText(frame, pred_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow('Hand Data Collector (Q to quit)', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                elif key == ord(' '):
                    self.current_label = 0
                elif key == ord('1'):
                    self.current_label = 1
                elif key == ord('2'):
                    self.current_label = 2
                elif key in (ord('x'), ord('X')):
                    self.current_label = self.PAUSED

        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved {self.counts} to:\n{self.output_file}")


if __name__ == "__main__":
    HandDataCollector().run()
