import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import time
import os


class DataCollector:
    def __init__(self, output_file=None):
        # 1. Dynamically locate the directory of this script (ai_pipeline folder)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Set the target for the CSV file into a new 'datasets' folder
        if output_file is None:
            self.output_file = os.path.join(self.script_dir, 'datasets', 'blink_dataset.csv')
        else:
            self.output_file = output_file

        # Automatically create the 'datasets' folder if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # 3. Path to the model in the assets folder (up two levels from ai_pipeline)
        # Stepping out of gaze_intent, and then out of ai_pipeline
        model_path = os.path.abspath(os.path.join(self.script_dir, '..', '..', 'assets', 'face_landmarker.task'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at:\n{model_path}\n"
                f"Please ensure 'face_landmarker.task' is inside the 'assets' folder at the root of the project."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Map both eyes based on the 478-point mesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # --- Map the exact center of the Irises ---
        self.LEFT_IRIS_INDEX = 468
        self.RIGHT_IRIS_INDEX = 473

        self.current_label = 0

    @staticmethod
    def _calculate_ear(face_landmarks, frame_w, frame_h, indices):
        """Calculates EAR for a specific eye given its landmark indices."""
        coords = []
        for idx in indices:
            lm = face_landmarks[idx]
            coords.append(np.array([lm.x * frame_w, lm.y * frame_h]))

        p1, p2, p3, p4, p5, p6 = coords

        v_dist_1 = np.linalg.norm(p2 - p6)
        v_dist_2 = np.linalg.norm(p3 - p5)
        h_dist = np.linalg.norm(p1 - p4)

        return float((v_dist_1 + v_dist_2) / (2.0 * h_dist))

    @staticmethod
    def _calculate_bbox_area(face_landmarks, frame_w, frame_h):
        x_coords = [lm.x * frame_w for lm in face_landmarks]
        y_coords = [lm.y * frame_h for lm in face_landmarks]

        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return float(width * height)

    def run(self):
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Avg_EAR', 'BBox_Area', 'Label'])

            cap = cv2.VideoCapture(0)

            print("--- DATA COLLECTOR STARTED ---")
            print("Press '0' to label data as OPEN (Default)")
            print("Press '1' to label data as INTENTIONAL CLICK")
            print("Press '2' to label data as SUSTAINED CLOSURE")
            print("Press 'Q' to save and exit.")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # --- MIRROR THE FRAME ---
                frame = cv2.flip(frame, 1)

                frame_h, frame_w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.detector.detect(mp_image)

                avg_ear = 0.0
                bbox_area = 0.0

                if results.face_landmarks:
                    first_face_landmarks = results.face_landmarks[0]

                    # 1. Math Extractions for BOTH eyes
                    left_ear = self._calculate_ear(first_face_landmarks, frame_w, frame_h, self.LEFT_EYE_INDICES)
                    right_ear = self._calculate_ear(first_face_landmarks, frame_w, frame_h, self.RIGHT_EYE_INDICES)

                    # Average the two for a much more stable signal
                    avg_ear = (left_ear + right_ear) / 2.0

                    bbox_area = self._calculate_bbox_area(first_face_landmarks, frame_w, frame_h)

                    # 2. Write to CSV (Iris coordinates are purposefully excluded here)
                    writer.writerow([time.time(), avg_ear, bbox_area, self.current_label])

                    # 3. Visual Debugging (Draw BOTH eyelids in Green)
                    for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 2, (0, 255, 0), -1)

                    # 4. Visual Debugging (Draw BOTH irises in Red to verify gaze tracking works)
                    for idx in [self.LEFT_IRIS_INDEX, self.RIGHT_IRIS_INDEX]:
                        lm = first_face_landmarks[idx]
                        pos = (int(lm.x * frame_w), int(lm.y * frame_h))
                        cv2.circle(frame, pos, 3, (0, 0, 255), -1)

                # Overlay current state on the video feed
                label_text = {0: "STATE: OPEN", 1: "STATE: CLICKING", 2: "STATE: SUSTAINED CLOSURE"}
                colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}

                cv2.putText(frame, label_text[self.current_label], (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colors[self.current_label], 2)
                cv2.putText(frame, f"Avg EAR: {avg_ear:.3f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Data Collector (Press Q to Quit)', frame)

                # Keyboard listener for labeling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('0'):
                    self.current_label = 0
                elif key == ord('1'):
                    self.current_label = 1
                elif key == ord('2'):
                    self.current_label = 2

        cap.release()
        cv2.destroyAllWindows()
        print(f"Data successfully saved to:\n{self.output_file}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()