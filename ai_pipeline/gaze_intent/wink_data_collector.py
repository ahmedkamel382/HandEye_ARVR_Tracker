import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time
import os

# Training/serving parity: EAR maths, calibration, rolling window and face-loss reset come from the
# runtime module itself, so recorded features are computed exactly as GazeIntent.process_frame computes them.
from gaze_intent import (
    EarFeatureState, calculate_ear, LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS,
)


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
            min_tracking_confidence=0.5,
            # Same options as the runtime, so "face present" means the same thing in both
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- 3. STATE & TEMPORAL WINDOW (shared with the runtime) ---
        self.current_label = 0
        self.ear_state = EarFeatureState()

    def run(self):
        file_exists = os.path.isfile(self.output_file)

        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    'Timestamp',
                    'Left_EAR', 'Left_Min_10f', 'Left_Var_10f',
                    'Right_EAR', 'Right_Min_10f', 'Right_Var_10f',
                    'Label'
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

                if results.face_landmarks and results.facial_transformation_matrixes:
                    first_face_landmarks = results.face_landmarks[0]

                    raw_left_ear = calculate_ear(first_face_landmarks, LEFT_EYE, frame_w, frame_h)
                    raw_right_ear = calculate_ear(first_face_landmarks, RIGHT_EYE, frame_w, frame_h)

                    was_calibrated = self.ear_state.is_calibrated
                    features = self.ear_state.update(raw_left_ear, raw_right_ear)

                    if features is None:
                        cv2.putText(frame,
                                    f"CALIBRATING: {self.ear_state.current_calibration_frames}"
                                    f"/{self.ear_state.calibration_frame_count}",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                        if not was_calibrated and self.ear_state.is_calibrated:
                            print("\nCalibration Complete!")
                            print("Press 'SPACE' for NEUTRAL (Hover/Default)")
                            print("Press '1' for LEFT WINK (Left Click)")
                            print("Press '2' for RIGHT WINK (Right Click)")
                            print("Press '3' for SUSTAINED CLOSURE (Pause/Sleep)")
                            print("Press 'Q' to save and exit.")
                    else:
                        left_ear, l_min_ear, l_var_ear, right_ear, r_min_ear, r_var_ear = features

                        writer.writerow([
                            time.time(),
                            left_ear, l_min_ear, l_var_ear,
                            right_ear, r_min_ear, r_var_ear,
                            self.current_label
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
                    for idx in LEFT_EYE + RIGHT_EYE:
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 2, (0, 255, 0), -1)

                    for idx in (LEFT_IRIS, RIGHT_IRIS):
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 3, (0, 0, 255), -1)
                else:
                    # Face lost: identical to the runtime (clear EAR windows, restart incomplete calibration)
                    self.ear_state.reset_tracking_state()
                    cv2.putText(frame, "NO FACE - window reset", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
        self.detector.close()
        print(f"Data successfully saved to:\n{self.output_file}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
