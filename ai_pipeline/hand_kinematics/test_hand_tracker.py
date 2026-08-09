"""
Simulates exactly what the C++ core engine will do on every frame:
open the webcam, grab a frame, and call HandTracker.process_frame(frame)
to get back the standardized (Cursor X, Cursor Y, Gesture State) payload.

No CSV, no labeling controls -- this is a bare Interface test. The only
UI is a text overlay showing exactly what process_frame() returned, same
style as hand_data_collector.py, so you can watch the raw bridge output
live instead of reading the terminal.

Run from ai_pipeline/hand_kinematics/:
    python test_hand_tracker.py
"""

import cv2
from hand_tracker import HandTracker

STATE_NAMES = {
    0: "Neutral",
    1: "Pinch",
    2: "Fist",
}

STATE_COLORS = {
    0: (0, 255, 0),    # Green
    1: (0, 255, 255),  # Yellow
    2: (0, 0, 255),    # Red
}


def main():
    print("Loading HandTracker...")
    tracker = HandTracker()
    print("Model loaded. Starting webcam. Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # --- THE EXACT CALL C++ WILL MAKE ---
        cursor_x, cursor_y, state = tracker.process_frame(frame)

        state_name = STATE_NAMES.get(state, f"Unknown({state})")
        color = STATE_COLORS.get(state, (255, 255, 255))

        # Also print to terminal, useful for copy-pasting values into a log.
        print(f"X: {cursor_x:6.3f}  |  Y: {cursor_y:6.3f}  |  State: {state} ({state_name})")

        # --- ON-SCREEN OVERLAY (mirrors hand_data_collector.py's style) ---
        # Note: the frame here is raw from OpenCV (not mirrored), because
        # process_frame() does its own internal cv2.flip(). We mirror the
        # display copy separately just for a natural on-screen preview.
        display_frame = cv2.flip(frame, 1)

        cv2.putText(display_frame, f"process_frame() OUTPUT", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"STATE: {state} ({state_name})", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_frame, f"Cursor X: {cursor_x:.3f}", (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Cursor Y: {cursor_y:.3f}", (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("test_hand_tracker (press q to quit)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nStopped.")


if __name__ == "__main__":
    main()