"""Tester vs native hand-validation equivalence (DSP message item 6) — NO camera.

Proves the HandEye Tester path (HandSource -> HandTracker.process_frame) maps the
same malformed/non-finite payloads to LOST as the native Core Engine, and that a
valid payload still produces TRACKED. The MediaPiPe detector is stubbed; no model,
camera, mouse or Windows GUI is used.

Run from repo root:
    .venv\\Scripts\\python.exe -m pytest core_engine/os_integration/tester/tests/test_hand_validation_equivalence.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

TESTER = Path(__file__).resolve().parents[1]
AI = Path(__file__).resolve().parents[4] / "ai_pipeline"
sys.path.insert(0, str(TESTER))
sys.path.insert(0, str(AI / "hand_kinematics"))

import hand_tracker as ht                      # noqa: E402
from hand_tracker import HandTracker           # noqa: E402
from tester_engine import HandSource, TesterConfig  # noqa: E402
import osi_capi                                # noqa: E402


class P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Result:
    def __init__(self, img, wld):
        self.hand_landmarks = [img] if img else []
        self.hand_world_landmarks = [wld] if wld else []


def hand(n):
    return [P(0.1 + 0.01 * i, 0.3, 0.0) for i in range(n)]


def bad_img(value):
    h = hand(21); h[8] = P(value, 0.3, 0.0); return h


class FakeDetector:
    def __init__(self, result):
        self.result = result
    def detect_for_video(self, image, ts):
        return self.result
    def detect(self, image):
        return self.result
    def close(self):
        pass


def make_source(result):
    """Real HandTracker.process_frame with a stubbed detector, driven by HandSource."""
    src = object.__new__(HandSource)
    src.kind = "hand"
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
    t.status = ht.STATUS_LOST
    t.tracking_valid = False
    t.stats = ht.TrackerStats()
    t._missed_frames = 0
    t._last_output = (-1.0, -1.0, 0)
    t._last_bbox = None
    t._last_cursor = None
    t._vel = np.zeros(2)
    src.tracker = t
    return src


FRAME = np.zeros((8, 8, 3), np.uint8)


@pytest.mark.parametrize("result", [
    Result(hand(20), hand(22)),
    Result(hand(22), hand(20)),
    Result(bad_img(float("nan")), hand(21)),
    Result(bad_img(float("inf")), hand(21)),
    Result(bad_img(float("-inf")), hand(21)),
])
def test_tester_rejects_invalid_payload_as_lost(result):
    src = make_source(result)
    obs = src.process(FRAME)
    assert obs.tracking == osi_capi.LOST
    assert obs.point is None
    assert obs.status == "LOST"


def test_tester_publishes_valid_payload_as_tracked():
    src = make_source(Result(hand(21), hand(21)))
    obs = src.process(FRAME)
    assert obs.tracking == osi_capi.TRACKED
    assert obs.point is not None
    assert 0.0 <= obs.point[0] <= 1.0 and 0.0 <= obs.point[1] <= 1.0


def test_tester_invalid_to_valid_recovery():
    bad = make_source(Result(hand(20), hand(22)))
    assert bad.process(FRAME).tracking == osi_capi.LOST
    good = make_source(Result(hand(21), hand(21)))
    assert good.process(FRAME).tracking == osi_capi.TRACKED
