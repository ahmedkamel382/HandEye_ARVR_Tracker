import cv2
import time
from gaze_intent import GazeIntent


def main():
    tracker = GazeIntent()
    cap = cv2.VideoCapture(0)

    print("Starting test... Press 'q' to exit.")

    # 1. Initialize a timestamp tracker before the loop
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Calculate the current timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 3. Pass the timestamp into process_frame
        gaze_x, gaze_y, state = tracker.process_frame(rgb_frame, timestamp_ms)

        # UI Overlay for Debugging
        cv2.putText(frame, f"X: {gaze_x:.2f} Y: {gaze_y:.2f} STATE: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gaze Intent Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.shutdown()  # Ensure clean teardown


if __name__ == "__main__":
    main()