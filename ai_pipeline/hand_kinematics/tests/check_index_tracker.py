"""Focused index-tip tracker checks. No camera or model file required.

Run from ai_pipeline/hand_kinematics:
    python tests/check_index_tracker.py
"""

import math
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
from mediapipe.tasks.python import vision

import index_tracker as it
from demo_index_pointer import RollingJitter, update_jitter


ok = True


def check(name, condition, extra=""):
    global ok
    ok &= bool(condition)
    print(("PASS " if condition else "FAIL ") + name + (f"  [{extra}]" if extra else ""))


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Res:
    def __init__(self, hand_landmarks=()):
        self.hand_landmarks = list(hand_landmarks)
        self.hand_world_landmarks = []


class FakeDetector:
    def __init__(self, script):
        self.script = list(script)
        self.index = 0
        self.timestamps = []
        self.seen = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        self.seen.append(image.numpy_view().copy())
        result = self.script[self.index] if self.index < len(self.script) else Res()
        self.index += 1
        return result

    def close(self):
        self.closed = True


def hand_at(x, y):
    hand = [LM(0.5, 0.5) for _ in range(21)]
    hand[8] = LM(x, y)
    return Res([hand])


def make(script, **kwargs):
    fake = FakeDetector(script)
    kwargs.setdefault("model_path", "dummy.task")
    with mock.patch("index_tracker.os.path.exists", return_value=True), \
         mock.patch.object(vision.HandLandmarker, "create_from_options", return_value=fake):
        tracker = it.IndexFingerTracker(**kwargs)
    return tracker, fake


FRAME = np.full((480, 640, 3), 128, np.uint8)


tracker, _ = make([hand_at(0.25, 0.75)], smoothing_alpha=1.0)
result = tracker.track(FRAME)
check("valid landmark 8 returns a result", result.valid and result.status == it.STATUS_TRACKED)
check("valid pointer coordinates", np.allclose((result.x, result.y), (0.25, 0.75)))
check("valid camera coordinates", np.allclose((result.camera_x, result.camera_y), (0.25, 0.75)))
check("result reports landmark index 8", result.landmark_index == 8)
check("tracking_valid and status are exposed", tracker.tracking_valid and tracker.get_status()[0] == "TRACKED")


hand = [LM(0.5, 0.5) for _ in range(21)]
hand[8] = LM(0.2, 0.3)
hand[12] = LM(0.9, 0.1)
tracker, _ = make([Res([hand])], smoothing_alpha=1.0)
result = tracker.track(FRAME)
check("uses landmark 8, not landmark 12", np.allclose((result.camera_x, result.camera_y), (0.2, 0.3)))


corners = [
    ((0.0, 0.0), (0.0, 0.0)),
    ((0.1, 0.1), (0.0, 0.0)),
    ((0.9, 0.9), (1.0, 1.0)),
    ((1.0, 1.0), (1.0, 1.0)),
    ((-0.5, 1.5), (0.0, 1.0)),
]
script = [hand_at(x, y) for (x, y), _ in corners]
tracker, _ = make(script, tracking_region=(0.1, 0.1, 0.9, 0.9), smoothing_alpha=1.0)
for (x, y), expected in corners:
    result = tracker.track(FRAME)
    check(f"corner/clamp ({x}, {y})", np.allclose((result.x, result.y), expected, atol=1e-9),
          f"got {(result.x, result.y)}")


check("region maps to full pointer range", np.allclose(it.map_pointer(0.1, 0.1, (0.1, 0.1, 0.9, 0.9)), (0.0, 0.0)))
check("region map preserves axis direction", np.allclose(it.map_pointer(0.2, 0.2, (0.1, 0.1, 0.9, 0.9)), (0.125, 0.125)))
check("pointer mapping clamps low", np.allclose(it.map_pointer(-1.0, -1.0), (0.0, 0.0)))
check("pointer mapping clamps high", np.allclose(it.map_pointer(2.0, 2.0), (1.0, 1.0)))


margin = 0.05
region = (margin, margin, 1.0 - margin, 1.0 - margin)
check("margin 0.05 keeps center mapped to center",
      np.allclose(it.map_pointer(0.5, 0.5, region), (0.5, 0.5)))
check("margin 0.05 lower region boundary maps to zero",
      np.allclose(it.map_pointer(margin, margin, region), (0.0, 0.0)))
check("margin 0.05 upper region boundary maps to one",
      np.allclose(it.map_pointer(1.0 - margin, 1.0 - margin, region), (1.0, 1.0)))
check("margin 0.05 interior mapping",
      np.allclose(it.map_pointer(0.2, 0.8, region), (1.0 / 6.0, 5.0 / 6.0)))
check("margin 0.05 maximum marker offset is the margin",
      np.isclose(abs((1.0 - margin) - it.map_pointer(1.0 - margin, 1.0 - margin, region)[0]), margin))
check("camera coordinates are clamped", np.allclose(
    (lambda r: (r.camera_x, r.camera_y))(make([hand_at(-0.2, 1.2)], smoothing_alpha=1.0)[0].track(FRAME)),
    (0.0, 1.0),
))

frame = np.zeros((480, 640, 3), np.uint8)
frame[:, :320] = (0, 0, 255)
frame[:, 320:] = (255, 0, 0)
tracker, fake = make([hand_at(0.5, 0.5)], smoothing_alpha=1.0)
tracker.track(frame)
expected_mirrored = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
check("detector receives a horizontally mirrored frame", np.array_equal(fake.seen[0], expected_mirrored))


tracker, _ = make([hand_at(0.2, 0.2), hand_at(0.8, 0.8)], smoothing_alpha=0.5)
first = tracker.track(FRAME)
second = tracker.track(FRAME)
check("first smoothed sample equals raw", np.allclose((first.x, first.y), (0.2, 0.2)))
check("EMA uses configured alpha", np.allclose((second.x, second.y), (0.5, 0.5)), f"got {(second.x, second.y)}")


tracker, _ = make([hand_at(0.2, 0.2), Res(), hand_at(0.8, 0.8)], smoothing_alpha=0.5)
tracker.track(FRAME)
lost = tracker.track(FRAME)
reacquired = tracker.track(FRAME)
check("missing hand invalidates immediately", not lost.valid and lost.x is None and lost.status == "LOST")
check("smoothing history resets after loss", np.allclose((reacquired.x, reacquired.y), (0.8, 0.8)))


tracker, _ = make([Res()], smoothing_alpha=1.0)
missing = tracker.track(FRAME)
legacy_missing = tracker.process_frame(FRAME)
check("missing result has no stale coordinates", missing.x is None and missing.camera_x is None)
check("legacy no-hand fallback is (-1,-1,0)", legacy_missing == (-1.0, -1.0, 0))
check("global tracking validity is false after loss", not tracker.tracking_valid)


malformed_results = [
    None,
    Res([]),
    Res([[]]),
    Res([[LM(0.5, 0.5)] * 8]),
    Res([[LM(0.5, 0.5)] * 8 + [LM(math.nan, 0.5)] + [LM(0.5, 0.5)] * 12]),
]
for index, malformed in enumerate(malformed_results):
    tracker, _ = make([malformed], smoothing_alpha=1.0)
    try:
        result = tracker.track(FRAME)
        check(f"malformed result {index} is handled as loss", not result.valid and result.x is None)
    except Exception as exc:
        check(f"malformed result {index} is handled as loss", False, repr(exc))


tracker, _ = make([hand_at(0.3, 0.4)], smoothing_alpha=1.0)
output = tracker.process_frame(FRAME)
check("process_frame returns (float, float, int)", isinstance(output[0], float) and isinstance(output[1], float) and isinstance(output[2], int))
check("process_frame preserves pointer and no-action state", np.allclose(output[:2], (0.3, 0.4)) and output[2] == 0)
check("process_frame exposes validity separately", tracker.tracking_valid)


tracker, fake = make([hand_at(0.5, 0.5)] * 200, smoothing_alpha=1.0)
for _ in range(200):
    tracker.track(FRAME)
check("VIDEO timestamps strictly increase", all(a < b for a, b in zip(fake.timestamps, fake.timestamps[1:])))


tracker, fake = make([hand_at(0.5, 0.5)], smoothing_alpha=1.0)
tracker.close()
tracker.close()
check("close releases the detector and is idempotent", fake.closed)


try:
    it.map_pointer(0.5, 0.5, (0.9, 0.1, 0.2, 0.9))
    check("invalid tracking region is rejected", False)
except ValueError:
    check("invalid tracking region is rejected", True)

try:
    make([], smoothing_alpha=0.0)
    check("invalid smoothing alpha is rejected", False)
except ValueError:
    check("invalid smoothing alpha is rejected", True)


jitter = RollingJitter(3)
for sample in (
    (0.2, 0.4, 0.3, 0.5),
    (0.4, 0.4, 0.5, 0.5),
    (0.9, 0.4, 0.7, 0.5),
):
    jitter.add(*sample)
snapshot = jitter.snapshot()
check("jitter reports the valid sample count", snapshot.count == 3 and len(jitter) == 3)
check("jitter reports raw X population SD and range",
      np.isclose(snapshot.raw.x_sd, 0.294392029) and np.isclose(snapshot.raw.x_range, 0.7))
check("jitter reports raw Y population SD and range",
      np.isclose(snapshot.raw.y_sd, 0.0) and np.isclose(snapshot.raw.y_range, 0.0))
check("jitter reports final X population SD and range",
      np.isclose(snapshot.final.x_sd, 0.163299316) and np.isclose(snapshot.final.x_range, 0.4))
check("jitter reports final Y population SD and range",
      np.isclose(snapshot.final.y_sd, 0.0) and np.isclose(snapshot.final.y_range, 0.0))
check("jitter keeps raw and final coordinate sets separate",
      not np.isclose(snapshot.raw.x_range, snapshot.final.x_range))


jitter = RollingJitter(2)
jitter.add(0.2, 0.4, 0.3, 0.5)
jitter.add(0.4, 0.4, 0.5, 0.5)
jitter.add(0.9, 0.4, 0.7, 0.5)
snapshot = jitter.snapshot()
check("jitter window evicts oldest samples", snapshot.count == 2)
check("jitter stats use only the current window",
      np.isclose(snapshot.raw.x_sd, 0.25) and np.isclose(snapshot.raw.x_range, 0.5)
      and np.isclose(snapshot.final.x_sd, 0.1) and np.isclose(snapshot.final.x_range, 0.2))


jitter = RollingJitter(5)
valid_result = it.FingerTrackingResult(
    x=0.4,
    y=0.6,
    camera_x=0.5,
    camera_y=0.5,
    valid=True,
    status=it.STATUS_TRACKED,
)
mapped = update_jitter(jitter, valid_result, (0.0, 0.0, 1.0, 1.0))
check("valid diagnostics add one sample and return the mapped point",
      mapped == (0.5, 0.5) and len(jitter) == 1)
lost_result = it.FingerTrackingResult(
    x=None,
    y=None,
    camera_x=None,
    camera_y=None,
    valid=False,
    status=it.STATUS_LOST,
)
mapped_lost = update_jitter(jitter, lost_result, (0.0, 0.0, 1.0, 1.0))
check("lost diagnostics return no mapped point and clear the window",
      mapped_lost is None and len(jitter) == 0)
snapshot = jitter.snapshot()
check("cleared jitter snapshot has zero count and zero statistics",
      snapshot.count == 0 and snapshot.raw.x_sd == 0.0 and snapshot.final.y_range == 0.0)


try:
    RollingJitter(1)
    check("diagnostic window below two samples is rejected", False)
except ValueError:
    check("diagnostic window below two samples is rejected", True)


print("\nALL PASS" if ok else "\nSOME FAILED")
if __name__ == "__main__":
    sys.exit(0 if ok else 1)
