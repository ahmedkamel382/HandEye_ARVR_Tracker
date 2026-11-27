#!/usr/bin/env python
"""Parity test: the C++ ports versus the team's REAL Python files, frame by frame.

  gaze: ai_pipeline/gaze_intent/gaze_intent.py   (GazeIntent.process_frame + the real wink / pitch / yaw .dll models,
        random head-pose matrices; compares final_yaw, final_pitch and the state)
  hand: ai_pipeline/hand_kinematics/hand_tracker.py (HandTracker.process_frame: world-landmark features,
        the real hand_gesture_rf.pkl in assets/, 5-frame vote, 0.55 gate, 6-frame dropout grace)

MediaPipe is replaced by a stub that simply hands the Python code the same synthetic landmarks the C++
side receives, so no camera, no .task files and no MediaPipe install are needed. Everything else in those
files (EAR, calibration, features, head pose, the Treelite models, scikit-learn, thresholds) is the real code.
Needs Windows (the gaze models are .dll files) and the tl2cgen package.

  python run_parity.py --exe core_parity_check --ai-root <repo>/ai_pipeline [--frames 3000] [--seed 1]

If this fails after someone edits hand_tracker.py / gaze_intent.py (new threshold, new feature...),
the C++ in core_engine/include/{hand_math,gaze_math}.hpp must be updated to match.
"""
import argparse
import collections
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fake_ai", "bridge"))
import synthetic_landmarks as syn  # noqa: E402

W, H = 640, 480
W2, H2 = 1280, 720          # a second resolution: the pixel-area feature and the pixel distances depend on it


# ---------------------------------------------------------------- MediaPipe stub -------------------------
class _Detector:
    def __init__(self):
        self.next_face = []          # list of landmark objects, or []
        self.next_hand = []          # image landmarks of the hand
        self.next_world = []         # world landmarks of the hand
        self.next_pose = np.eye(4, dtype=np.float32)   # facial transformation matrix of the face
        self.next_blendshapes = None  # optional list of blendshape category objects, or None

    def detect_for_video(self, image, ts):
        has_bs = bool(self.next_face) and (self.next_blendshapes is not None)
        return types.SimpleNamespace(
            face_landmarks=[self.next_face] if self.next_face else [],
            facial_transformation_matrixes=[self.next_pose] if self.next_face else [],
            face_blendshapes=[self.next_blendshapes] if has_bs else [],
            hand_landmarks=[self.next_hand] if self.next_hand else [],
            hand_world_landmarks=[self.next_world] if self.next_world else [])

    def detect(self, image):
        return types.SimpleNamespace(
            hand_landmarks=[self.next_hand] if self.next_hand else [],
            hand_world_landmarks=[self.next_world] if self.next_world else [])

    def close(self):
        pass


def install_mediapipe_stub():
    det = _Detector()
    mp = types.ModuleType("mediapipe")
    mp.Image = lambda image_format=None, data=None: types.SimpleNamespace(data=data)
    mp.ImageFormat = types.SimpleNamespace(SRGB=1)
    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")
    python.BaseOptions = lambda **kw: types.SimpleNamespace(**kw)
    python.vision = vision
    tasks.python = python
    mp.tasks = tasks
    vision.RunningMode = types.SimpleNamespace(IMAGE=0, VIDEO=1)
    vision.FaceLandmarkerOptions = vision.HandLandmarkerOptions = lambda **kw: types.SimpleNamespace(**kw)
    vision.FaceLandmarker = types.SimpleNamespace(create_from_options=lambda o: det)
    vision.HandLandmarker = types.SimpleNamespace(create_from_options=lambda o: det)
    for name, mod in (("mediapipe", mp), ("mediapipe.tasks", tasks),
                      ("mediapipe.tasks.python", python), ("mediapipe.tasks.python.vision", vision)):
        sys.modules[name] = mod
    return det


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_objs(arr):
    return [types.SimpleNamespace(x=float(a), y=float(b), z=float(c)) for a, b, c in arr.tolist()]


def fmt(arr):
    return " ".join("%.9g" % v for v in arr.reshape(-1).tolist())


# Canonical eye-aligned blendshape order matching landmark_extractors.EYE_BLENDSHAPE_NAMES
_BS_NAMES = (
    "eyeBlinkRight",      # slot 0: bs_blink_l (image-left LEFT_EYE)
    "eyeBlinkLeft",       # slot 1: bs_blink_r (image-right RIGHT_EYE)
    "eyeLookDownRight",   # slot 2: bs_look_down_l
    "eyeLookDownLeft",    # slot 3: bs_look_down_r
    "eyeSquintRight",     # slot 4: bs_squint_l
    "eyeSquintLeft",      # slot 5: bs_squint_r
    "eyeWideRight",       # slot 6: bs_wide_l
    "eyeWideLeft",        # slot 7: bs_wide_r
)


def make_bs_objs(vals):
    return [types.SimpleNamespace(category_name=name, score=float(v)) for name, v in zip(_BS_NAMES, vals)]


def pack_face_with_optional_bs(lm, pose, bs_vals):
    base = syn.pack_face(lm, pose)
    if bs_vals is None:
        return base
    extra = np.zeros((3, 3), dtype=np.float32)
    extra[0, 0], extra[0, 1], extra[0, 2] = bs_vals[0], bs_vals[1], bs_vals[2]
    extra[1, 0], extra[1, 1], extra[1, 2] = bs_vals[3], bs_vals[4], bs_vals[5]
    extra[2, 0], extra[2, 1], extra[2, 2] = bs_vals[6], bs_vals[7], 0.0
    return np.vstack((base, extra)).astype(np.float32)


# ---------------------------------------------------------------- scenario generators --------------------
def gen_face_sequences(rng, frames):
    """Yield sequences of (lm_or_None, bs_vals_or_None, dt_ms) covering both n=484 (EAR-only)
    and n=487 (blendshape + EAR) hybrid blink regimes, including poisoned startup, winks,
    sympathetic squint, symmetric blinks, downward gaze, threshold jitter, degenerate eye corners,
    dropouts, and irregular 15/30/60 FPS timestamps."""
    seqs = []
    per_seq = max(220, frames // 4)
    for seq_idx in range(4):
        use_bs = (seq_idx % 2 == 1)
        seq, kl, kr, iris = [], 1.0, 1.0, np.array([0.5, 0.5])
        bs = [0.06, 0.06, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02]
        left = 0
        for i in range(per_seq):
            # Variable FPS: 15 FPS (66ms), 30 FPS (33ms), 60 FPS (16ms)
            dt_ms = (16, 33, 66)[(seq_idx + i // 40) % 3]
            if i < 40 and seq_idx == 1 and (i % 5 == 0):
                # Blink-poisoned startup frame
                lm_f = syn.make_face(0.25, 0.25, tuple(iris), rng=rng, noise=0.0005)
                bs_f = [0.85, 0.85, 0.10, 0.10, 0.10, 0.10, 0.0, 0.0]
                seq.append((lm_f, bs_f, dt_ms))
                continue
            if left == 0:
                left = int(rng.integers(12, 45))
                if i > 45:
                    mode = int(rng.integers(0, 9))
                    if mode == 0:      # open neutral
                        kl, kr = 1.0, 1.0
                        bs = [0.05, 0.05, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02]
                    elif mode == 1:    # firm left wink
                        kl, kr = 0.25, 1.0
                        bs = [0.85, 0.08, 0.10, 0.10, 0.25, 0.05, 0.01, 0.02]
                    elif mode == 2:    # firm right wink
                        kl, kr = 1.0, 0.25
                        bs = [0.08, 0.85, 0.10, 0.10, 0.05, 0.25, 0.02, 0.01]
                    elif mode == 3:    # symmetric blink / closure
                        kl, kr = 0.22, 0.22
                        bs = [0.88, 0.88, 0.15, 0.15, 0.10, 0.10, 0.0, 0.0]
                    elif mode == 4:    # sympathetic-squint left wink
                        kl, kr = 0.25, 0.65
                        bs = [0.88, 0.50, 0.10, 0.10, 0.30, 0.20, 0.0, 0.0]
                    elif mode == 5:    # downward gaze with heavy lid droop
                        kl, kr = 0.62, 0.62
                        bs = [0.22, 0.22, 0.68, 0.68, 0.05, 0.05, 0.0, 0.0]
                    elif mode == 6:    # near-threshold +-0.02 jitter around 0.52
                        kl, kr = 0.50, 1.0
                        bs = [0.54, 0.06, 0.10, 0.10, 0.05, 0.05, 0.0, 0.0]
                    elif mode == 7:    # random openness
                        kl = float(rng.uniform(0.15, 1.15))
                        kr = float(rng.uniform(0.15, 1.15))
                        bs = [
                            float(np.clip(1.0 - kl, 0.0, 1.0)),
                            float(np.clip(1.0 - kr, 0.0, 1.0)),
                            float(rng.uniform(0.0, 0.7)),
                            float(rng.uniform(0.0, 0.7)),
                            0.05, 0.05, 0.02, 0.02,
                        ]
                    else:              # occasional degenerate / closed right eye
                        kl, kr = 1.0, (0.0 if rng.random() < 0.2 else 1.0)
                        bs = [0.05, 0.05, 0.10, 0.10, 0.04, 0.04, 0.02, 0.02]
            left -= 1
            iris = np.clip(iris + rng.normal(0, 0.04, 2), 0.0, 1.0)
            if i > 45 and rng.random() < 0.03:
                seq.append((None, None, dt_ms))
                continue
            lm_f = syn.make_face(kl, kr, tuple(iris), center=(0.5 + rng.normal(0, .02), 0.5 + rng.normal(0, .02)),
                                 size=float(rng.uniform(0.18, 0.3)), rng=rng, noise=0.0008)
            # Inject occasional degenerate collapsed eye corner (h_dist == 0) post-calibration
            if i > 60 and rng.random() < 0.01:
                lm_f[133] = lm_f[33]
            bs_out = None
            if use_bs:
                jitter = rng.normal(0.0, 0.015, 2)
                bs_out = list(bs)
                bs_out[0] = float(np.clip(bs_out[0] + jitter[0], 0.0, 1.0))
                bs_out[1] = float(np.clip(bs_out[1] + jitter[1], 0.0, 1.0))
            seq.append((lm_f, bs_out, dt_ms))
        seqs.append(seq)
    return seqs


def random_pose(rng):
    """4x4 float32 head-pose matrix (what MediaPipe returns): a random rotation, now and then gimbal-locked
    (yaw = +-90 deg) to exercise the singular branch of the Euler-angle code."""
    pitch, yaw, roll = np.radians(rng.uniform([-35.0, -45.0, -20.0], [35.0, 45.0, 20.0]))
    if rng.random() < 0.02:
        yaw = np.radians(90.0 if rng.random() < 0.5 else -90.0)
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = rng.uniform(-5.0, 5.0, 3)
    return m


def gen_hand_frames(rng, n):
    """Frames are None (no hand) or a (42,3) packed array: image landmarks + world landmarks."""
    frames = []
    miss_run = 0
    while len(frames) < n:
        r = rng.random()
        if miss_run > 0:                                                         # a run of detection misses
            frames.append(None); miss_run -= 1
        elif r < 0.03:
            miss_run = int(rng.integers(1, 12))                                  # exercises the 6-frame grace period
        elif r < 0.04:
            img = np.full((syn.N_HAND, 3), 0.5, dtype=np.float32)               # degenerate: palm size 0
            frames.append(syn.pack_hand(img, np.zeros((syn.N_HAND, 3), dtype=np.float32)))
        else:
            tip = (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.6)))
            curled = tuple(bool(c) for c in (rng.random(4) < 0.5))
            scale = float(rng.uniform(0.6, 1.6))
            thumb = None
            if rng.random() < 0.6:                                              # straddle the pinch threshold
                thumb = (tip[0] + rng.normal(0, 0.05), tip[1] + rng.normal(0, 0.05))
            img = syn.make_hand(tip, curled, thumb, pinch=bool(rng.random() < 0.3), scale=scale, rng=rng, noise=0.002)
            frames.append(syn.pack_hand(img, syn.make_hand_world(img, rng, noise=0.003)))
            # hold the same pose for a few frames so the 5-frame vote sees runs, like a real hand does
            for _ in range(int(rng.integers(0, 6))):
                img2 = img + rng.normal(0.0, 0.0005, img.shape).astype(np.float32)
                frames.append(syn.pack_hand(img2, syn.make_hand_world(img2, rng, noise=0.003)))
    return frames[:n]


# ---------------------------------------------------------------- main ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--ai-root", required=True, help="the repo's ai_pipeline folder")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    gaze_py = os.path.join(a.ai_root, "gaze_intent", "gaze_intent.py")
    hand_py = os.path.join(a.ai_root, "hand_kinematics", "hand_tracker.py")
    hand_pkl = os.path.abspath(os.path.join(a.ai_root, "..", "assets", "hand_gesture_rf.pkl"))
    models_dir = os.path.join(a.ai_root, "gaze_intent", "models")
    for p in (gaze_py, hand_py, os.path.join(os.path.dirname(hand_py), "hand_features.py"), hand_pkl,
              os.path.join(models_dir, "gesture_model.dll"), os.path.join(models_dir, "gaze_pitch.dll"),
              os.path.join(models_dir, "gaze_yaw.dll")):
        if not os.path.exists(p):
            print(f"PARITY: missing {p}", file=sys.stderr)
            return 2

    tmp = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(tmp.name, "ai_pipeline"))
    os.makedirs(os.path.join(tmp.name, "assets"))
    for f in ("hand_landmarker.task", "face_landmarker.task"):
        open(os.path.join(tmp.name, "assets", f), "wb").close()           # only has to exist
    shutil.copy(hand_pkl, os.path.join(tmp.name, "assets", "hand_gesture_rf.pkl"))   # the REAL hand model
    os.environ["AI_PIPELINE_ROOT"] = os.path.join(tmp.name, "ai_pipeline")
    detector = install_mediapipe_stub()
    sys.path.insert(0, os.path.dirname(hand_py))                              # hand_tracker.py does "from hand_features import ..."
    gaze_mod = load_module(gaze_py, "real_gaze_intent")
    hand_mod = load_module(hand_py, "real_hand_tracker")

    rng = np.random.default_rng(a.seed)
    lines, expected, kinds = [], [], []

    # ---- gaze: 4 sequences x 2 resolutions (both n=484 EAR-only and n=487 blendshapes+EAR) ----
    for (w, h) in ((W, H), (W2, H2)):
        for seq in gen_face_sequences(rng, a.frames // 2):
            gi = gaze_mod.GazeIntent()
            lines.append("S")
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            ts_ms = 0
            for lm, bs_vals, dt_ms in seq:
                ts_ms += int(dt_ms)
                pose = random_pose(rng)
                bs_f32 = [float(np.float32(v)) for v in bs_vals] if bs_vals is not None else None
                detector.next_face = as_objs(lm) if lm is not None else []
                detector.next_pose = pose
                detector.next_blendshapes = make_bs_objs(bs_f32) if (lm is not None and bs_f32 is not None) else None
                final_yaw, final_pitch, head_yaw, head_pitch, state, landmarks = gi.process_frame(blank, ts_ms)
                expected.append((final_yaw, final_pitch, state)); kinds.append("gaze")
                packed = pack_face_with_optional_bs(lm, pose, bs_f32) if lm is not None else None
                n = 0 if packed is None else len(packed)
                lines.append(f"FT {w} {h} {ts_ms} {n} " + (fmt(packed) if packed is not None else ""))
            gi.shutdown()
            detector.next_blendshapes = None

    # ---- hand ----
    # HandNative (the C++ side under test) does not run Python ROI recovery,
    # so the Python reference must use the same full-frame-only configuration.
    ht = hand_mod.HandTracker(enable_roi_recovery=False)
    for (w, h) in ((W, H), (W2, H2)):
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for lm in gen_hand_frames(rng, a.frames // 2):
            detector.next_hand = as_objs(lm[:syn.N_HAND]) if lm is not None else []
            detector.next_world = as_objs(lm[syn.N_HAND:]) if lm is not None else []
            x, y, s = ht.process_frame(blank)
            expected.append((x, y, s)); kinds.append("hand")
            n = 0 if lm is None else len(lm)
            lines.append(f"H {w} {h} {n} " + (fmt(lm) if lm is not None else ""))

    rec = os.path.join(tmp.name, "recording.txt")
    with open(rec, "w") as f:
        f.write("\n".join(lines) + "\n")
    proc = subprocess.run([a.exe, rec, models_dir], capture_output=True, text=True)
    if proc.returncode != 0:
        print("PARITY: core_parity_check failed:\n" + proc.stderr, file=sys.stderr)
        return 2
    got = [ln.split() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(got) != len(expected):
        print(f"PARITY: expected {len(expected)} result lines, got {len(got)}", file=sys.stderr)
        return 2

    bad = collections.Counter()
    shown = 0
    states = {"gaze": collections.Counter(), "hand": collections.Counter()}
    for i, (kind, (ex, ey, es), g) in enumerate(zip(kinds, expected, got)):
        gx, gy, gs = float(g[0]), float(g[1]), int(g[2])
        states[kind][es] += 1
        # NaN != NaN under IEEE-754: the gaze angles are NaN while calibrating / without a face (INVALID_ANGLE),
        # so two NaNs count as a match; a NaN against a number is still a mismatch (abs(nan) < eps is False).
        def same(a, b):
            return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 2e-6
        ok = (gs == es) and same(gx, ex) and same(gy, ey)
        if not ok:
            bad[kind] += 1
            if shown < 8:
                shown += 1
                print(f"  MISMATCH {kind} #{i}: python=({ex:.6f},{ey:.6f},{es})  c++=({gx:.6f},{gy:.6f},{gs})")

    for kind in ("gaze", "hand"):
        total = sum(states[kind].values())
        print(f"{kind}: {total} frames, states {dict(sorted(states[kind].items()))}, mismatches {bad[kind]}")
    fail = sum(bad.values()) > 0
    # Guard against a vacuous test: the hand must not be stuck in one state.
    if len(states["hand"]) < 2:
        print("PARITY: warning - the real hand model gave a single state on the synthetic hands "
              "(parity of the classes is covered by --selftest-hand-forest)", file=sys.stderr)
    if len(states["gaze"]) < 2:
        print("PARITY: warning - the wink model gave a single gaze state on the synthetic data", file=sys.stderr)
    print("PARITY", "FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())