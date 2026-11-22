"""Fixture matrix for HandLandmarkExtractor packing (NEW A / NEW B). Run: python -m pytest ai_pipeline/tests -q
MediaPipe is stubbed: only the validation/packing path is exercised."""
import os, sys, types
import numpy as np
sys.modules.setdefault("mediapipe", types.ModuleType("mediapipe"))
for m in ("mediapipe.tasks", "mediapipe.tasks.python", "mediapipe.tasks.python.vision"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["mediapipe.tasks"].python = sys.modules["mediapipe.tasks.python"]
sys.modules["mediapipe.tasks.python"].vision = sys.modules["mediapipe.tasks.python.vision"]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bridge"))
import landmark_extractors as le

class LM:
    def __init__(s, x, y, z): s.x, s.y, s.z = x, y, z
def lms(n, v=0.5): return [LM(v, v, 0.0) for _ in range(n)]
def bad(n, field, val):
    l = lms(n); setattr(l[8], field, val); return l

def run(img, wld):
    packed, d = le.pack_hand(img, wld)
    print(f"n_image={d['n_image']} n_world={d['n_world']} packed_n={d['packed_n']} valid={d['valid']}")
    return packed, d

def test_valid_21_21():
    p, d = run(lms(21), lms(21)); assert p.shape == (42, 3) and d["valid"] and d["packed_n"] == 42

def test_malformed_counts_rejected():
    for a, b in [(20, 22), (22, 20), (21, 0), (0, 21), (0, 0)]:
        p, d = run(lms(a), lms(b)); assert p.shape == (0, 3) and not d["valid"] and d["packed_n"] == 0
    assert run(lms(21), None)[1]["has_world"] is False and run(None, lms(21))[1]["has_image"] is False

def test_nonfinite_rejected():
    for v in (float("nan"), float("inf"), float("-inf")):
        for f in "xyz":
            p, d = run(bad(21, f, v), lms(21)); assert p.shape == (0, 3) and d["reason"] == "nonfinite"
            p, d = run(lms(21), bad(21, f, v)); assert p.shape == (0, 3)

def test_exception_path_propagates():   # C++ PyLandmarker counts exceptions and treats the frame as "nothing detected"
    class Boom: pass
    try: le.pack_hand([Boom()] * 21, lms(21)); assert False
    except AttributeError: pass


class R:   # fake MediaPipe result
    def __init__(s, i, w): s.hand_landmarks = [i] if i is not None else []; s.hand_world_landmarks = [w] if w is not None else []

def _fixtures():
    nan = float("nan")
    return [(lms(21), lms(21)), (lms(20), lms(22)), (lms(21), None), (None, lms(21)), (None, None),
            (bad(21, "x", nan), lms(21)), (lms(21), bad(21, "z", float("inf")))]

def test_pack_hand_result_and_diag_reason():
    assert le._pack_hand(R(lms(21), lms(21))).shape == (42, 3)
    for i, w in _fixtures()[1:]:
        assert le._pack_hand(R(i, w)).shape == (0, 3)
    assert le._describe_hand(R(lms(20), lms(22)), 0, 0, "VIDEO")["n"] == 0
    assert le._describe_hand(R(bad(21, "x", float("inf")), lms(21)), 0, 0, "VIDEO")["reason"] == "NONFINITE"

def test_diag_on_off_equivalence():   # HAND_LANDMARK_DIAG writer must not change the returned array
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "d.csv")
    w = le._HandDiagWriter(path)
    for i, wl in _fixtures():
        a = le._pack_hand(R(i, wl)); b = le._pack_hand(R(i, wl), w, 1, 1, "VIDEO")
        assert a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    w.close(); assert os.path.getsize(path) > 0
