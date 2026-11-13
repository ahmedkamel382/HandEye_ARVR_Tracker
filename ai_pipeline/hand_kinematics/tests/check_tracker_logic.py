"""Run with:  python tests/check_<name>.py   (needs: pip install mediapipe opencv-python numpy). No camera or model file required."""
import os
import os
"""
Exercises the REAL HandTracker.__init__ / process_frame with fake MediaPipe landmarkers
(the .task model can't be downloaded in this sandbox). This validates the control logic
(grace period, ROI recovery + coordinate mapping, click safety, stats, contract) -
NOT the real-world detection quality, which needs a live camera.
"""
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from unittest import mock
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import hand_tracker as ht

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))

class LM:
    def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z

class Res:
    def __init__(self, hand_landmarks=(), world=()):
        self.hand_landmarks, self.hand_world_landmarks = list(hand_landmarks), list(world)

_rng = np.random.default_rng(1)
_WORLD = [LM(*p) for p in (_rng.normal(0, 0.04, (21, 3)))]

def hand(tip_x, tip_y, spread=0.05):
    """Result with a hand whose index tip (landmark 8) is exactly (tip_x, tip_y); bbox spans +-spread."""
    offs = np.linspace(-spread, spread, 21)
    lms = [LM(tip_x + offs[i], tip_y + offs[(i * 5) % 21]) for i in range(21)]
    lms[8] = LM(tip_x, tip_y)
    return Res([lms], [_WORLD])

EMPTY = Res()

class FakeVideo:
    def __init__(self, script): self.script, self.i, self.ts, self.seen = list(script), 0, [], []
    def detect_for_video(self, img, ts):
        self.ts.append(ts); self.seen.append(img.numpy_view().copy())
        r = self.script[self.i] if self.i < len(self.script) else EMPTY
        self.i += 1
        return r

class FakeImage:
    def __init__(self, script): self.script, self.i, self.shapes, self.contig = list(script), 0, [], []
    def detect(self, img):
        v = img.numpy_view(); self.shapes.append(v.shape); self.contig.append(v.flags['C_CONTIGUOUS'])
        r = self.script[self.i] if self.i < len(self.script) else EMPTY
        self.i += 1
        return r

def make(primary, roi=(), state=0, **kw):
    fv, fi = FakeVideo(primary), FakeImage(roi)
    kw.setdefault('enable_lowlight', True)
    with mock.patch('hand_tracker.os.path.exists', return_value=True), \
         mock.patch('hand_tracker.load_gesture_model', return_value=None), \
         mock.patch.object(vision.HandLandmarker, 'create_from_options', side_effect=[fv, fi] if kw.get('enable_roi_recovery', True) else [fv]):
        t = ht.HandTracker(**kw)
    t._classify = lambda feats: (state, 0.9)       # control the classifier output
    return t, fv, fi

FRAME = np.full((480, 640, 3), 128, np.uint8)       # bright enough: enhancer stays off

def run(t, n): return [t.process_frame(FRAME) for _ in range(n)]

# ---- startup state: no prior observation is LOST, never HELD ---------------------------
t, fv, fi = make([EMPTY], state=0, enable_roi_recovery=False)
startup = t.process_frame(FRAME)
check("startup no hand: status LOST + invalid coords",
      startup == (-1.0, -1.0, 0) and t.status == ht.STATUS_LOST
      and not t.tracking_valid and t.get_status() == (ht.STATUS_LOST, False))
check("startup no hand: missed_frames == 1", t.missed_frames == 1)

# ---- contract & plain detection --------------------------------------------------------
t, fv, fi = make([hand(0.4, 0.5)] * 3, state=2, enable_roi_recovery=False)
outs = run(t, 3)
check("detected: returns (float,float,int)", all(isinstance(o[0], float) and isinstance(o[1], float) and isinstance(o[2], int) for o in outs))
check("detected: cursor = index tip, state voted", np.allclose(outs[-1][:2], (0.4, 0.5)) and outs[-1][2] == 2)
check("detected: status DETECTED + tracking_valid", t.status == ht.STATUS_DETECTED and t.tracking_valid)
check("detected: missed_frames == 0", t.missed_frames == 0)
check("get_status remains (status, tracking_valid)", t.get_status() == (t.status, t.tracking_valid))
try:
    t.missed_frames = 99
    missed_frames_readonly = False
except AttributeError:
    missed_frames_readonly = True
check("missed_frames is read-only", missed_frames_readonly)
check("ROI detector not created when disabled", t.roi_detector is None)

# ---- grace period: hold, then lost -----------------------------------------------------
t, fv, fi = make([hand(0.4, 0.5)] * 3 + [EMPTY] * 8 + [hand(0.6, 0.6)], state=2, enable_roi_recovery=False)
o = run(t, 3); last_good = o[-1]
held = []
for i in range(1, 7):
    held.append(t.process_frame(FRAME))
    check(f"miss {i}: HELD, retained output, missed_frames {i}",
          t.status == ht.STATUS_HELD and not t.tracking_valid and t.missed_frames == i)
check("short dropout (<=6): output held EXACTLY", all(h == last_good for h in held))
l1 = t.process_frame(FRAME)
check("7th missed frame: LOST -> (-1,-1,0), missed_frames 7",
      l1 == (-1.0, -1.0, 0) and t.status == ht.STATUS_LOST
      and not t.tracking_valid and t.missed_frames == 7)
l2 = t.process_frame(FRAME)
check("8th missed frame stays LOST, missed_frames 8",
      l2 == (-1.0, -1.0, 0) and t.status == ht.STATUS_LOST and t.missed_frames == 8)
check("lost: vote buffer + bbox cleared", len(t._votes) == 0 and t._last_bbox is None)
back = t.process_frame(FRAME)
check("re-acquire after LOST restores observed behavior",
      np.allclose(back[:2], (0.6, 0.6)) and t.status == ht.STATUS_DETECTED
      and t.tracking_valid and t.missed_frames == 0)

# ---- ROI recovery: mapping, crop geometry, status --------------------------------------
# last bbox: tip (0.5,0.5), spread .05 -> bbox 0.1 x 0.1 -> 64x48 px -> side=max(64,48)*2=128 px
roi_hand = hand(0.25, 0.75, spread=0.05)            # index tip at crop-normalized (0.25, 0.75)
t, fv, fi = make([hand(0.5, 0.5)] + [EMPTY], roi=[roi_hand], state=2)
t.process_frame(FRAME)
check('fixture bbox is exactly 0.45..0.55 in x and y', np.allclose(t._last_bbox, (0.45, 0.45, 0.55, 0.55)), str(np.round(t._last_bbox, 4)))
out = t.process_frame(FRAME)
exp = ((320 - 64 + 0.25 * 128) / 640, (240 - 64 + 0.75 * 128) / 480)
check("ROI used after primary miss", fi.i == 1 and t.status == ht.STATUS_RECOVERED and t.tracking_valid)
check("ROI crop is 128x128, C-contiguous", fi.shapes == [(128, 128, 3)] and fi.contig == [True], str(fi.shapes))
check("ROI landmarks mapped back to full-frame coords", np.allclose(out[:2], exp, atol=1e-6), f"got {out[:2]} expected {exp}")
check("ROI recovery counted as primary miss in stats", t.stats.primary_misses == 1 and t.stats.counts[ht.STATUS_RECOVERED] == 1)

# crop near the right edge is shifted inward, not shrunk
t, fv, fi = make([hand(0.97, 0.5, spread=0.02)] + [EMPTY], roi=[EMPTY], state=0)
t.process_frame(FRAME); t.process_frame(FRAME)
check("edge: crop keeps its size (shifted inward)", fi.shapes and fi.shapes[0][0] >= ht.HandTracker.ROI_MIN_PX and fi.shapes[0][1] >= ht.HandTracker.ROI_MIN_PX, str(fi.shapes))
check("ROI miss falls through to HELD", t.status == ht.STATUS_HELD)

# no ROI attempt once LOST (full-frame search only)
t, fv, fi = make([hand(0.5, 0.5)] + [EMPTY] * 10, roi=[], state=0)
run(t, 11)
calls_during_grace = fi.i
check("ROI attempted until LOST is declared, never afterwards (7 of 10 missed frames)", calls_during_grace == ht.HandTracker.GRACE_FRAMES + 1, f"{calls_during_grace} ROI calls")

# ---- click safety ---------------------------------------------------------------------
t, fv, fi = make([hand(0.5, 0.5)] * 4 + [EMPTY] * 5, state=0, enable_roi_recovery=False)
run(t, 4); t._classify = lambda f: (1, 0.9)
states_held = [o[2] for o in run(t, 5)]
check("held frames can never start a Pinch", all(s == 0 for s in states_held))
t, fv, fi = make([hand(0.5, 0.5)] * 5, state=0, enable_roi_recovery=False)
run(t, 4); t._classify = lambda f: (1, 0.9)
single = t.process_frame(FRAME)
check("single noisy Pinch frame loses the 5-frame majority vote", single[2] == 0)

# ---- cursor prediction ----------------------------------------------------------------
xs = [0.40, 0.41, 0.42, 0.43, 0.44]
t, fv, fi = make([hand(x, 0.5) for x in xs] + [EMPTY] * 3, state=2, enable_roi_recovery=False, predict_cursor=True)
run(t, 5)
v = 0.0
for i in range(1, 5): v = 0.5 * v + 0.5 * 0.01
exp_x, got = 0.44, []
for k in (1, 2, 3):
    exp_x += v * 0.6 ** k
    got.append(t.process_frame(FRAME)[0])
check("predict_cursor: damped velocity extrapolation", np.allclose(got, [0.44 + v * 0.6, 0.44 + v * 0.6 + v * 0.36, exp_x]), f"{np.round(got, 5)}")
check("predict_cursor: state untouched while predicting", t._last_output[2] == 2)
t, *_ = make([hand(x, 0.5) for x in xs] + [EMPTY], state=2, enable_roi_recovery=False, predict_cursor=False)
run(t, 5); a = t._last_output; b = t.process_frame(FRAME)
check("predict_cursor=False (default): exact hold", a == b)

# ---- incomplete result (landmarks but no world landmarks) is a miss, never an exception ----
bad = Res([hand(0.5, 0.5).hand_landmarks[0]], [])
t, *_ = make([bad], state=0, enable_roi_recovery=False)
try:
    out = t.process_frame(FRAME); check("missing world landmarks handled as a miss", t.status in (ht.STATUS_HELD, ht.STATUS_LOST))
except Exception as e:
    check("missing world landmarks handled as a miss", False, repr(e))

# ---- timestamps ------------------------------------------------------------------------
t, fv, fi = make([hand(0.5, 0.5)] * 200, state=0, enable_roi_recovery=False)
run(t, 200)
check("VIDEO timestamps strictly increasing over 200 rapid calls", all(a < b for a, b in zip(fv.ts, fv.ts[1:])))

# ---- stats -----------------------------------------------------------------------------
script = [hand(0.5, 0.5)] * 10 + [EMPTY] * 3 + [hand(0.5, 0.5)] * 5 + [EMPTY] * 8 + [hand(0.5, 0.5)] * 2
t, *_ = make(script, state=0, enable_roi_recovery=False)
run(t, len(script)); s = t.stats.summary()
check("stats: frames/detected/held/lost", (s['frames'], s['detected_pct']) == (28, round(100 * 17 / 28, 2)) and t.stats.counts[ht.STATUS_HELD] == 9 and t.stats.counts[ht.STATUS_LOST] == 2, str({k: s[k] for k in ('frames','detected_pct','held_pct','lost_pct')}))
check("stats: 2 dropout events, longest 8, mean 5.5", (s['dropout_events'], s['longest_dropout_frames'], s['mean_dropout_frames']) == (2, 8, 5.5))
check("stats: effective dropout % = (held+lost)/frames", s['effective_dropout_pct'] == round(100 * 11 / 28, 2))
check("stats: latency fields present", s['latency_p50_ms'] >= 0 and s['latency_p95_ms'] >= s['latency_p50_ms'])
t.reset_stats(); check("reset_stats", t.stats.frames == 0)

# ---- low-light integration -------------------------------------------------------------
DARK = np.full((480, 640, 3), 25, np.uint8)
t, fv, fi = make([hand(0.5, 0.5)] * 30, state=0, enable_roi_recovery=False, enable_lowlight=True)
for _ in range(30): t.process_frame(DARK)
check("low-light ON: detector receives a brightened frame", fv.seen[-1].mean() > 60, f"input 25 -> {fv.seen[-1].mean():.0f}")
check("low-light ON: stats count enhanced frames", t.stats.summary()['enhanced_pct'] > 50)
t, fv, fi = make([hand(0.5, 0.5)] * 5, state=0, enable_roi_recovery=False, enable_lowlight=False)
for _ in range(5): t.process_frame(DARK)
check("low-light OFF: frame reaches detector unmodified", abs(fv.seen[-1].mean() - 25) < 1e-6 and t.enhancer is None)
t, fv, fi = make([hand(0.5, 0.5)] * 5, state=0, enable_roi_recovery=False, enable_lowlight=True)
for _ in range(5): t.process_frame(FRAME)
check("normal light: frame reaches detector unmodified (baseline unchanged)", abs(fv.seen[-1].mean() - 128) < 1e-6)

print("\nALL PASS" if ok else "\nSOME FAILED")
if __name__ == "__main__":
    sys.exit(0 if ok else 1)
