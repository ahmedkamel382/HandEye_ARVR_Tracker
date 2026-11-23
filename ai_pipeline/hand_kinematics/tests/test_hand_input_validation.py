"""Regression tests for HandEye Tester hand-input validation (DSP message item 6).

Run with:
    .venv\\Scripts\\python.exe -m pytest ai_pipeline/hand_kinematics/tests/test_hand_input_validation.py

No camera, no model file and no MediaPipe detector are used: the detector result is
stubbed. The contract under test is that malformed or non-finite landmarks can never
become a fresh cursor coordinate, matching the native Core Engine safeguard.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import hand_tracker as ht
from hand_tracker import HandTracker, STATUS_DETECTED, STATUS_LOST
from hand_features import FEATURE_NAMES  # noqa: F401 - ensures module import order matches production


class P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Result:
    def __init__(self, image, world):
        self.hand_landmarks = [image] if image else []
        self.hand_world_landmarks = [world] if world else []


def hand(n, value=0.25):
    return [P(0.1 + 0.01 * i, value, 0.0) for i in range(n)]


def invalid_image(at, value):
    landmarks = hand(21)
    landmarks[at] = P(value, 0.3, 0.0)
    return landmarks


def invalid_world(at, value):
    landmarks = hand(21)
    landmarks[at] = P(value, 0.3, 0.0)
    return landmarks


# ---------------------------------------------------------------- _parse contract

def test_valid_21_21_is_accepted():
    points, lm_xy = HandTracker._parse(Result(hand(21), hand(21)))
    assert points.shape == (21, 3)
    assert lm_xy.shape == (21, 2)
    assert np.isfinite(points).all() and np.isfinite(lm_xy).all()


@pytest.mark.parametrize("n_img,n_wld", [(20, 22), (22, 20), (21, 20), (20, 21),
                                          (21, 0), (0, 21), (0, 0)])
def test_malformed_counts_are_rejected(n_img, n_wld):
    assert HandTracker._parse(Result(hand(n_img), hand(n_wld))) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_image_coordinates_are_rejected(value):
    assert HandTracker._parse(Result(invalid_image(8, value), hand(21))) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_world_coordinates_are_rejected(value):
    assert HandTracker._parse(Result(hand(21), invalid_world(8, value))) is None


def test_invalid_to_valid_recovery():
    assert HandTracker._parse(Result(hand(20), hand(22))) is None
    assert HandTracker._parse(Result(invalid_image(8, float("inf")), hand(21))) is None
    points, lm_xy = HandTracker._parse(Result(hand(21), hand(21)))
    assert points.shape == (21, 3) and lm_xy.shape == (21, 2)


# ---------------------------------------------------------------- process_frame contract

class FakeDetector:
    def __init__(self, result):
        self.result = result

    def detect_for_video(self, image, ts):
        return self.result

    def detect(self, image):
        return self.result

    def close(self):
        pass


def make_tracker(result):
    """Build a HandTracker without loading MediaPipe or the gesture model."""
    t = object.__new__(HandTracker)
    t.enhancer = None
    t.detector = FakeDetector(result)
    t.roi_detector = None
    t._start_time = 0.0
    t._last_timestamp_ms = -1
    t.enable_roi_recovery = False
    t.rf = None
    t.predict_cursor = False
    t._votes = __import__("collections").deque(maxlen=t.SMOOTHING_FRAMES)
    t.last_confidence = 0.0
    t.status = STATUS_LOST
    t.tracking_valid = False
    t.stats = ht.TrackerStats()
    t._missed_frames = 0
    t._last_output = (-1.0, -1.0, 0)
    t._last_bbox = None
    t._last_cursor = None
    t._vel = np.zeros(2)
    return t


@pytest.mark.parametrize("result", [
    Result(hand(20), hand(22)),
    Result(invalid_image(8, float("inf")), hand(21)),
    Result(invalid_image(8, float("nan")), hand(21)),
    Result(hand(21), invalid_world(8, float("inf"))),
])
def test_invalid_payload_never_becomes_fresh_coordinate(result):
    t = make_tracker(result)
    x, y, state = t.process_frame(np.zeros((8, 8, 3), np.uint8))
    assert t.status == STATUS_LOST
    assert t.tracking_valid is False
    assert (x, y) == (-1.0, -1.0)


def test_valid_payload_still_tracks_after_invalid_frame():
    bad = make_tracker(Result(hand(20), hand(22)))
    assert bad.process_frame(np.zeros((8, 8, 3), np.uint8))[:2] == (-1.0, -1.0)

    good = make_tracker(Result(hand(21), hand(21)))
    x, y, state = good.process_frame(np.zeros((8, 8, 3), np.uint8))
    assert good.status == STATUS_DETECTED
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
