"""In-window index-finger pointer demo. Opens the camera only when run directly.

Run from ai_pipeline/hand_kinematics:
    python demo_index_pointer.py --camera 0 --margin 0.1
"""

import argparse
from collections import deque
from dataclasses import dataclass
import statistics
import time

import cv2

from index_tracker import IndexFingerTracker, map_pointer


@dataclass(frozen=True)
class CoordinateStats:
    x_sd: float
    x_range: float
    y_sd: float
    y_range: float


@dataclass(frozen=True)
class JitterSnapshot:
    count: int
    raw: CoordinateStats
    final: CoordinateStats


def _coordinate_stats(samples, x_index, y_index):
    xs = [sample[x_index] for sample in samples]
    ys = [sample[y_index] for sample in samples]
    if not xs:
        return CoordinateStats(0.0, 0.0, 0.0, 0.0)
    return CoordinateStats(
        float(statistics.pstdev(xs)),
        float(max(xs) - min(xs)),
        float(statistics.pstdev(ys)),
        float(max(ys) - min(ys)),
    )


class RollingJitter:
    def __init__(self, window):
        window = int(window)
        if window < 2:
            raise ValueError("diagnostic window must be at least 2 samples")
        self.window = window
        self._samples = deque(maxlen=window)

    def add(self, raw_x, raw_y, final_x, final_y):
        self._samples.append((
            float(raw_x),
            float(raw_y),
            float(final_x),
            float(final_y),
        ))

    def clear(self):
        self._samples.clear()

    def __len__(self):
        return len(self._samples)

    def snapshot(self):
        samples = list(self._samples)
        return JitterSnapshot(
            count=len(samples),
            raw=_coordinate_stats(samples, 0, 1),
            final=_coordinate_stats(samples, 2, 3),
        )


def update_jitter(jitter, result, tracking_region):
    if not result.valid:
        jitter.clear()
        return None
    mapped = map_pointer(result.camera_x, result.camera_y, tracking_region)
    jitter.add(result.camera_x, result.camera_y, result.x, result.y)
    return mapped


def _format_axis(x_sd, x_range, y_sd, y_range):
    return (f"X sd={x_sd:.4f} rng={x_range:.4f}  "
            f"Y sd={y_sd:.4f} rng={y_range:.4f}")


def format_jitter(snapshot, window):
    if snapshot.count == 0:
        return [
            f"JITTER n=0/{window}",
            "waiting for valid frames",
        ]
    return [
        f"JITTER n={snapshot.count}/{window}",
        f"RAW L8   {_format_axis(snapshot.raw.x_sd, snapshot.raw.x_range, snapshot.raw.y_sd, snapshot.raw.y_range)}",
        f"POINTER  {_format_axis(snapshot.final.x_sd, snapshot.final.x_range, snapshot.final.y_sd, snapshot.final.y_range)}",
        "valid-only; not pure model jitter",
    ]


def draw_text_panel(display, lines, top=10, right=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    line_height = 18
    padding = 8
    text_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    )
    panel_width = text_width + 2 * padding
    panel_height = len(lines) * line_height + 2 * padding
    x = max(5, display.shape[1] - panel_width - right)
    y = top
    cv2.rectangle(display, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.rectangle(display, (x, y), (x + panel_width, y + panel_height), (80, 80, 80), 1)
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (x + padding, y + padding + (index + 1) * line_height - 5),
            font,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Index-finger landmark-8 pointer demo")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="normalized margin removed from each side of the tracking region")
    parser.add_argument("--smoothing", type=float, default=0.35,
                        help="EMA weight for new samples; 1.0 disables smoothing")
    parser.add_argument("--diagnostics", action="store_true",
                        help="show raw, mapped, final coordinates and rolling jitter")
    parser.add_argument("--diag-window", type=int, default=120,
                        help="number of valid samples in the rolling diagnostic window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.margin < 0.5:
        raise SystemExit("--margin must satisfy 0 <= margin < 0.5")
    if not 0.0 < args.smoothing <= 1.0:
        raise SystemExit("--smoothing must satisfy 0 < smoothing <= 1")
    if args.diagnostics and args.diag_window < 2:
        raise SystemExit("--diag-window must be at least 2")

    region = (args.margin, args.margin, 1.0 - args.margin, 1.0 - args.margin)
    tracker = IndexFingerTracker(tracking_region=region, smoothing_alpha=args.smoothing)
    jitter = RollingJitter(args.diag_window) if args.diagnostics else None
    jitter_snapshot = jitter.snapshot() if jitter is not None else None
    diag_refresh_count = 0
    cap = None
    previous_time = time.perf_counter()

    try:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a camera frame.")
                break

            result = tracker.track(frame)
            display = cv2.flip(frame, 1)
            height, width = display.shape[:2]

            mapped = None
            if args.diagnostics:
                mapped = update_jitter(jitter, result, tracker.tracking_region)
                if result.valid:
                    diag_refresh_count += 1
                    if jitter_snapshot.count == 0 or diag_refresh_count >= 10:
                        jitter_snapshot = jitter.snapshot()
                        diag_refresh_count = 0
                else:
                    jitter_snapshot = jitter.snapshot()
                    diag_refresh_count = 0

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now

            left = int(region[0] * width)
            top = int(region[1] * height)
            right = int(region[2] * width)
            bottom = int(region[3] * height)
            cv2.rectangle(display, (left, top), (right, bottom), (255, 0, 255), 1)

            if result.valid:
                landmark_x = min(width - 1, max(0, int(result.camera_x * width)))
                landmark_y = min(height - 1, max(0, int(result.camera_y * height)))
                pointer_x = min(width - 1, max(0, int(result.x * width)))
                pointer_y = min(height - 1, max(0, int(result.y * height)))

                cv2.circle(display, (landmark_x, landmark_y), 6, (0, 255, 255), -1)
                landmark_label = "RAW L8 (mirror)" if args.diagnostics else "LANDMARK 8"
                landmark_label_limit = width - 100 if args.diagnostics else width - 80
                cv2.putText(display, landmark_label,
                            (min(landmark_x + 8, landmark_label_limit), max(15, landmark_y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                if args.diagnostics:
                    mapped_x = min(width - 1, max(0, int(mapped[0] * width)))
                    mapped_y = min(height - 1, max(0, int(mapped[1] * height)))
                    cv2.circle(display, (mapped_x, mapped_y), 4, (255, 255, 0), -1)
                    cv2.putText(display, "MAPPED PRE-EMA",
                                (min(mapped_x + 6, width - 120), min(height - 5, mapped_y + 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

                cv2.line(display, (landmark_x, landmark_y), (pointer_x, pointer_y),
                         (180, 180, 180), 1, cv2.LINE_AA)
                cv2.circle(display, (pointer_x, pointer_y), 12, (255, 0, 255), 2)
                cv2.line(display, (pointer_x - 16, pointer_y), (pointer_x + 16, pointer_y),
                         (255, 0, 255), 1)
                cv2.line(display, (pointer_x, pointer_y - 16), (pointer_x, pointer_y + 16),
                         (255, 0, 255), 1)
                if args.diagnostics:
                    cv2.putText(display, "PTR (map+EMA)",
                                (min(pointer_x + 16, width - 110), min(height - 5, pointer_y + 26)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

                cv2.putText(display,
                            f"TRACKED  pointer=({result.x:.3f}, {result.y:.3f})",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(display, "TRACKING LOST", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(display, "No fresh landmark-8 position", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

            if args.diagnostics:
                if result.valid:
                    live_lines = [
                        f"RAW L8 (mirror) ({result.camera_x:.3f}, {result.camera_y:.3f})",
                        f"MAPPED PRE-EMA  ({mapped[0]:.3f}, {mapped[1]:.3f})",
                        f"POINTER ({result.x:.3f}, {result.y:.3f})",
                    ]
                else:
                    live_lines = ["RAW L8 (mirror)  --", "MAPPED PRE-EMA  --", "POINTER --"]
                live_lines.append(
                    f"region=({region[0]:.2f}, {region[1]:.2f}, {region[2]:.2f}, {region[3]:.2f}) "
                    f"alpha={args.smoothing:.2f}"
                )
                draw_text_panel(display, live_lines + format_jitter(jitter_snapshot, args.diag_window))

            cv2.putText(display, f"FPS {fps:.1f}   q / Esc to quit", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Index Finger Pointer", display)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if args.diagnostics:
            print("\n".join(format_jitter(jitter.snapshot(), args.diag_window)))
        tracker.close()


if __name__ == "__main__":
    main()
