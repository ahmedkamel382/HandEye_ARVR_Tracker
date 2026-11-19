"""Run with:  python ai_pipeline/bridge/tests/check_hand_landmark_diag.py

Needs mediapipe and numpy. No camera and no model file are used: the detector is
stubbed, so the branch logic can be checked without loading hand_landmarker.task.
"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import landmark_extractors as le

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))


class P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class R:
    def __init__(self, img=None, world=None):
        self.hand_landmarks = self._wrap(img)
        self.hand_world_landmarks = self._wrap(world)

    @staticmethod
    def _wrap(points):
        if not points:
            return []
        return [points] if isinstance(points[0], P) else points


def hand(n, offset=0.0):
    return [P(0.2 + 0.02 * i + offset, 0.3 + 0.01 * i) for i in range(n)]


def describe(img=None, world=None, seq=1, ts=1000, mode="VIDEO"):
    return le._describe_hand(R(img, world), seq, ts, mode)


# ---------------------------------------------------------------- packing paths
packed = le._pack_hand(R(hand(21), hand(21)))
check("full result: 42 rows", packed.shape == (42, 3), str(packed.shape))
check("full result: image rows first", np.allclose(packed[0], [0.2, 0.3, 0.0]))
check("full result: world rows second", np.allclose(packed[21], [0.2, 0.3, 0.0]))
check("both empty: (0, 3)", le._pack_hand(R()).shape == (0, 3))
check("image only: (0, 3)", le._pack_hand(R(hand(21), [])).shape == (0, 3))
check("world only: (0, 3)", le._pack_hand(R([], hand(21))).shape == (0, 3))
malformed = le._pack_hand(R(hand(20), hand(21)))
check("malformed result is returned unchanged (41 rows)", malformed.shape == (41, 3), str(malformed.shape))

# ---------------------------------------------------------------- cause labels
d = describe()
check("label: both empty", d["reason"] == "BOTH_EMPTY" and d["n"] == 0, d["reason"])
d = describe(hand(21), [])
check("label: image only", d["reason"] == "IMAGE_ONLY" and d["img_n"] == 21 and d["n"] == 0, d["reason"])
d = describe([], hand(21))
check("label: world only", d["reason"] == "WORLD_ONLY" and d["world_n"] == 21 and d["n"] == 0, d["reason"])
d = describe(hand(21), hand(21))
check("label: normal", d["reason"] == "OK" and d["n"] == 42 and d["img_n"] == 21 and d["world_n"] == 21, d["reason"])
d = describe(hand(20), hand(21))
check("label: malformed", d["reason"] == "MALFORMED" and d["n"] == 41, d["reason"])

# ---------------------------------------------------------------- bounding box
img = [P(0.1, 0.2), P(0.5, 0.6), P(0.3, 0.4)] + [P(0.3, 0.4) for _ in range(18)]
d = describe(img, hand(21))
check("bbox min/max", d["min_x"] == 0.1 and d["max_x"] == 0.5 and d["min_y"] == 0.2 and d["max_y"] == 0.6, str(d))
check("edge margins", abs(d["edge_l"] - 0.1) < 1e-9 and abs(d["edge_r"] - 0.5) < 1e-9
      and abs(d["edge_t"] - 0.2) < 1e-9 and abs(d["edge_b"] - 0.4) < 1e-9, str(d))

# ---------------------------------------------------------------- enable switch
saved = os.environ.pop("HAND_LANDMARK_DIAG", None)
check("diagnostic is off by default", le._diag_enabled() is False)
os.environ["HAND_LANDMARK_DIAG"] = "true"
check("diagnostic enables on true", le._diag_enabled() is True)
os.environ["HAND_LANDMARK_DIAG"] = "0"
check("diagnostic stays off on 0", le._diag_enabled() is False)
if saved is None:
    os.environ.pop("HAND_LANDMARK_DIAG", None)
else:
    os.environ["HAND_LANDMARK_DIAG"] = saved

# ---------------------------------------------------------------- CSV writer
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "diag.csv")
    writer = le._HandDiagWriter(path)
    writer.write(describe(hand(21), [], seq=7, ts=350, mode="VIDEO"))
    writer.close()
    writer.close()                                        # idempotent
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("csv: one row", len(rows) == 1)
    check("csv: complete header", set(rows[0].keys()) == set(le._DIAG_FIELDS))
    check("csv: reason recorded", rows[0]["reason"] == "IMAGE_ONLY")
    check("csv: counts recorded", rows[0]["img_n"] == "21" and rows[0]["world_n"] == "0" and rows[0]["n"] == "0")
    check("csv: seq and timestamp recorded", rows[0]["seq"] == "7" and rows[0]["ts_ms"] == "350")

# ---------------------------------------------------------------- failure safety
class Boom:
    def write(self, info):
        raise RuntimeError("boom")


check("logging failure cannot break packing", le._pack_hand(R(hand(21), hand(21)), Boom()).shape == (42, 3))

# ---------------------------------------------------------------- process_frame
class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.closed = False
        self.calls = []

    def detect(self, image):
        self.calls.append(("IMAGE", image))
        return self.result

    def detect_for_video(self, image, ts):
        self.calls.append(("VIDEO", ts))
        return self.result

    def close(self):
        self.closed = True


def make_extractor(result, video=False):
    ex = object.__new__(le.HandLandmarkExtractor)          # no MediaPipe model load
    ex._video = video
    ex._frame = 0
    ex._diag = None
    ex.detector = FakeDetector(result)
    return ex


frame = np.zeros((16, 16, 3), np.uint8)
ex = make_extractor(R(hand(21), hand(21)))
out = ex.process_frame(frame, 1000)
check("process_frame: 42 rows", out.shape == (42, 3), str(out.shape))
check("process_frame: frame counter", ex._frame == 1)
check("process_frame: IMAGE dispatch", ex.detector.calls[0][0] == "IMAGE")

ex = make_extractor(R(hand(21), hand(21)), video=True)
ex.process_frame(frame, 1050)
check("process_frame: VIDEO dispatch", ex.detector.calls[0][0] == "VIDEO" and ex.detector.calls[0][1] == 1050)

ex = make_extractor(R(hand(21), []))
out = ex.process_frame(frame, 1100)
check("process_frame: image-only still returns (0, 3)", out.shape == (0, 3))
check("process_frame: frame counter advances on empty", ex._frame == 1)

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "run.csv")
    ex = make_extractor(R(hand(21), []))
    ex._diag = le._HandDiagWriter(path)
    det = ex.detector
    ex.process_frame(frame, 1150)
    ex.process_frame(frame, 1200)
    ex.shutdown()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("process_frame: diagnostic writes one row per frame", len(rows) == 2)
    check("process_frame: diagnostic rows classified", [r["reason"] for r in rows] == ["IMAGE_ONLY", "IMAGE_ONLY"])
    check("process_frame: detector closed by shutdown", det.closed)
    check("process_frame: diagnostic closed by shutdown", ex._diag is None)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
