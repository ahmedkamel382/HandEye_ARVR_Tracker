import cv2
import time
import numpy as np
from sklearn.linear_model import Ridge
from gaze_intent import GazeIntent

STATE_LABELS = {
    -1: ("INITIALIZING...", (0, 165, 255)),
    0: ("NEUTRAL", (0, 255, 0)),
    1: ("LEFT WINK", (255, 191, 0)),
    2: ("RIGHT WINK", (255, 0, 255)),
    3: ("BLINK (DOUBLE)", (0, 0, 255))
}


def draw_hud_text(frame, text, pos, font_scale=0.8, color=(0, 255, 0), thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def build_poly_features(yaw, pitch):
    return np.array([1.0, yaw, pitch], dtype=np.float64)


def get_face_area(landmarks, frame_w, frame_h):
    if not landmarks: return 0.0
    x_coords = [lm.x * frame_w for lm in landmarks]
    y_coords = [lm.y * frame_h for lm in landmarks]
    return float((max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)))


def main():
    tracker = GazeIntent()
    cap = cv2.VideoCapture(0)

    print("Starting 9-Point Calibration... Follow the dots.")
    start_time = time.time()

    # 9-point calibration setup
    calib_active = True
    calib_step = 0
    calib_duration = 3.0  # seconds per dot
    step_timer = time.time()

    # Data collection containers
    current_dot_samples = []
    collected_features = []  # Poly feature vectors
    target_screen_pts = []  # (x, y) ground-truth screen pixels

    # Regression models
    reg_x = None
    reg_y = None
    baseline_areas = []
    baseline_face_area = 1.0

    calib_grid = None  # Will compute once frame dimensions are known
    smooth_cx, smooth_cy, smooth_size, face_bbox = None, None, None, None
    cursor_smooth_x, cursor_smooth_y = None, None
    last_cursor_x, last_cursor_y = None, None
    CROP_SIZE = 500

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        raw_h, raw_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 1. RUN TRACKER ON RAW FRAME (Perfects 3D Head Pose)
        rel_yaw, rel_pitch, head_yaw, head_pitch, state, landmarks = tracker.process_frame(rgb_frame, timestamp_ms)
        state_text, state_color = STATE_LABELS.get(state, (f"STATE: {state}", (0, 255, 0)))

        # 2. CALCULATE UI CROP
        if landmarks:
            raw_x_coords = [lm.x * raw_w for lm in landmarks]
            raw_y_coords = [lm.y * raw_h for lm in landmarks]

            target_cx = (min(raw_x_coords) + max(raw_x_coords)) / 2.0
            target_cy = (min(raw_y_coords) + max(raw_y_coords)) / 2.0
            target_size = max(max(raw_x_coords) - min(raw_x_coords), max(raw_y_coords) - min(raw_y_coords)) * 1.5

            smooth_factor = 0.15
            if smooth_cx is None:
                smooth_cx, smooth_cy, smooth_size = target_cx, target_cy, target_size
            else:
                smooth_cx = smooth_cx * (1 - smooth_factor) + target_cx * smooth_factor
                smooth_cy = smooth_cy * (1 - smooth_factor) + target_cy * smooth_factor
                smooth_size = smooth_size * (1 - smooth_factor) + target_size * smooth_factor

            crop_w = int(smooth_size)
            crop_x1 = int(smooth_cx - (crop_w / 2.0))
            crop_y1 = int(smooth_cy - (crop_w / 2.0))
        else:
            smooth_cx, smooth_cy, smooth_size = None, None, None
            cursor_smooth_x, cursor_smooth_y = None, None
            last_cursor_x, last_cursor_y = None, None
            crop_x1, crop_y1, crop_w = 0, 0, raw_w

        # 3. EXTRACT AND PAD SQUARE CROP
        safe_x1, safe_y1 = max(0, crop_x1), max(0, crop_y1)
        safe_x2, safe_y2 = min(raw_w, crop_x1 + crop_w), min(raw_h, crop_y1 + crop_w)
        crop_bgr = frame[safe_y1:safe_y2, safe_x1:safe_x2]

        pad_top = max(0, -crop_y1)
        pad_bot = max(0, (crop_y1 + crop_w) - raw_h)
        pad_left = max(0, -crop_x1)
        pad_right = max(0, (crop_x1 + crop_w) - raw_w)

        if pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0:
            crop_bgr = cv2.copyMakeBorder(crop_bgr, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))

        cropped_frame = cv2.resize(crop_bgr, (CROP_SIZE, CROP_SIZE)) if crop_bgr.size > 0 else cv2.resize(frame,
                                                                                                          (CROP_SIZE,
                                                                                                           CROP_SIZE))

        def map_to_crop(rx, ry):
            return int(((rx - crop_x1) / crop_w) * CROP_SIZE), int(((ry - crop_y1) / crop_w) * CROP_SIZE)

        if calib_grid is None:
            margin_x, margin_y = int(CROP_SIZE * 0.10), int(CROP_SIZE * 0.15)
            mid_x, mid_y = CROP_SIZE // 2, CROP_SIZE // 2
            right_x, bot_y = CROP_SIZE - margin_x, CROP_SIZE - margin_y
            calib_grid = [(margin_x, margin_y), (mid_x, margin_y), (right_x, margin_y), (margin_x, mid_y),
                          (mid_x, mid_y), (right_x, mid_y), (margin_x, bot_y), (mid_x, bot_y), (right_x, bot_y)]

        if landmarks:
            for idx in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                cx, cy = map_to_crop(landmarks[idx].x * raw_w, landmarks[idx].y * raw_h)
                cv2.circle(cropped_frame, (cx, cy), 2, (0, 255, 0), -1)
            for idx in [468, 473]:
                cx, cy = map_to_crop(landmarks[idx].x * raw_w, landmarks[idx].y * raw_h)
                cv2.circle(cropped_frame, (cx, cy), 3, (0, 0, 255), -1)

        overlay = cropped_frame.copy()
        cv2.rectangle(overlay, (0, 0), (CROP_SIZE, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, cropped_frame, 0.4, 0, cropped_frame)

        if state == -1:
            draw_hud_text(cropped_frame, "STATUS: INITIALIZING EYE SENSORS...", (20, 50), 0.85, (0, 165, 255), 2)
            step_timer = time.time()
        elif calib_active:
            elapsed = time.time() - step_timer
            remain = max(0.0, calib_duration - elapsed)
            tx, ty = calib_grid[calib_step]
            pulse = int(5 * np.sin(time.time() * 8))
            cv2.circle(cropped_frame, (tx, ty), 22 + pulse, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(cropped_frame, (tx, ty), 12, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(cropped_frame, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
            draw_hud_text(cropped_frame, f"CALIBRATION: DOT {calib_step + 1}/9  ({remain:.1f}s)", (20, 50), 0.85,
                          (0, 255, 255), 2)

            if elapsed > 1.2:
                current_dot_samples.append((rel_yaw, rel_pitch))
                baseline_areas.append(get_face_area(landmarks, raw_w, raw_h))  # Calculate multiplier on RAW frame

            if elapsed >= calib_duration:
                med_yaw = float(np.median([s[0] for s in current_dot_samples])) if current_dot_samples else rel_yaw
                med_pitch = float(np.median([s[1] for s in current_dot_samples])) if current_dot_samples else rel_pitch
                collected_features.append(build_poly_features(med_yaw, med_pitch))
                target_screen_pts.append([tx, ty])
                current_dot_samples = []
                calib_step += 1
                step_timer = time.time()

                if calib_step >= len(calib_grid):
                    baseline_face_area = float(np.median(baseline_areas))
                    A, targets = np.array(collected_features, dtype=np.float64), np.array(target_screen_pts,
                                                                                          dtype=np.float64)
                    reg_x, reg_y = Ridge(alpha=0.5).fit(A, targets[:, 0]), Ridge(alpha=0.5).fit(A, targets[:, 1])
                    calib_active = False
        else:
            feat = build_poly_features(rel_yaw, rel_pitch).reshape(1, -1)
            raw_cx, raw_cy = float(reg_x.predict(feat)[0]), float(reg_y.predict(feat)[0])

            current_area = get_face_area(landmarks, raw_w, raw_h)
            raw_multiplier = float(np.sqrt(baseline_face_area / current_area)) if current_area > 0 else 1.0
            multiplier = min(raw_multiplier, 2.0)  # Cap to prevent noise explosion

            mid_x, mid_y = CROP_SIZE // 2, CROP_SIZE // 2
            scaled_cx = mid_x + ((raw_cx - mid_x) * multiplier)
            scaled_cy = mid_y + ((raw_cy - mid_y) * multiplier)

            if cursor_smooth_x is None:
                cursor_smooth_x, cursor_smooth_y = scaled_cx, scaled_cy
            else:
                c_alpha = 0.20  # Screen smoothing factor
                cursor_smooth_x = (c_alpha * scaled_cx) + ((1 - c_alpha) * cursor_smooth_x)
                cursor_smooth_y = (c_alpha * scaled_cy) + ((1 - c_alpha) * cursor_smooth_y)

            cursor_x = int(np.clip(cursor_smooth_x, 15, CROP_SIZE - 15))
            cursor_y = int(np.clip(cursor_smooth_y, 15, CROP_SIZE - 15))

            # Deadzone: Ignore micro-movements smaller than 8 pixels
            if last_cursor_x is None:
                last_cursor_x, last_cursor_y = cursor_x, cursor_y
            else:
                # Calculate pixel distance between new frame and old frame
                dist = np.hypot(cursor_x - last_cursor_x, cursor_y - last_cursor_y)
                if dist < 8.0:
                    cursor_x, cursor_y = last_cursor_x, last_cursor_y
                else:
                    last_cursor_x, last_cursor_y = cursor_x, cursor_y

            cv2.circle(cropped_frame, (cursor_x, cursor_y), 15, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(cropped_frame, (cursor_x, cursor_y), 18, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(cropped_frame, (cursor_x, cursor_y), 2, (0, 0, 0), -1, cv2.LINE_AA)

            draw_hud_text(cropped_frame,
                          f"EYE: (Y:{rel_yaw:.1f} P:{rel_pitch:.1f}) | HEAD: (Y:{head_yaw:.1f} P:{head_pitch:.1f})",
                          (20, 35), 0.65, (255, 255, 255), 2)
            draw_hud_text(cropped_frame, f"STATE: {state_text}", (20, 70), 0.85, state_color, 2)

        cv2.imshow('Gaze Intent Test', cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    tracker.shutdown()


if __name__ == "__main__":
    main()
