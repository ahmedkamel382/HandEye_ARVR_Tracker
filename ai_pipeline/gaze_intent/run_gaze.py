import cv2
from gaze_intent import GazeIntent


def main():
    # Initialize your pipeline
    try:
        tracker = GazeIntent()
    except FileNotFoundError as e:
        print(e)
        return

    # Open the default webcam
    cap = cv2.VideoCapture(0)
    print("Starting test... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Pass the raw OpenCV frame to your processor
        gaze_x, gaze_y, state = tracker.process_frame(frame)

        # Overlay the output on the video feed
        text = f"X: {gaze_x:.2f} | Y: {gaze_y:.2f} | State: {state}"

        # Mirror the frame for intuitive viewing
        display_frame = cv2.flip(frame, 1)
        cv2.putText(display_frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gaze Intent - Live Test", display_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()