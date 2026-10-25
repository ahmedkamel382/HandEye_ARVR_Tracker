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

    def detect_for_video(self, image, ts):
        return types.SimpleNamespace(
            face_landmarks=[self.next_face] if self.next_face else [],
            facial_transformation_matrixes=[self.next_pose] if self.next_face else [],
            hand_landmarks=[self.next_hand] if self.next_hand else [],
            hand_world_landmarks=[self.next_world] if self.next_world else [])

    def detect(self, image):
        return types.SimpleNamespace(hand_landmarks=[self.next_hand] if self.next_hand else [])

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


# ---------------------------------------------------------------- scenario generators --------------------
def gen_face_sequences(rng, frames):
    """Yield sequences (list of landmark arrays, 'None' = no face). Eyes open at first (calibration), then
    random wink / blink / closure segments with random openness, moving iris, noise, dropouts."""
    seqs = []
    per_seq = max(200, frames // 4)
    for _ in range(4):
        seq, kl, kr, iris = [], 1.0, 1.0, np.array([0.5, 0.5])
        left = 0
        for i in range(per_seq):
            if left == 0:
                left = int(rng.integers(15, 60))
                if i > 45:
                    mode = rng.integers(0, 6)
                    kl = [1.0, 0.25, 1.0, 0.25, float(rng.uniform(0.1, 1.2)), 1.0][mode]
                    kr = [1.0, 1.0, 0.25, 0.25, float(rng.uniform(0.1, 1.2)), 0.0 if rng.random() < .1 else 1.0][mode]
            left -= 1
            iris = np.clip(iris + rng.normal(0, 0.04, 2), 0.0, 1.0)
            if i > 45 and rng.random() < 0.03:
                seq.append(None)
                continue
            seq.append(syn.make_face(kl, kr, tuple(iris), center=(0.5 + rng.normal(0, .02), 0.5 + rng.normal(0, .02)),
                                     size=float(rng.uniform(0.18, 0.3)), rng=rng, noise=0.0008))
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

    # ---- gaze: 4 sequences x 2 resolutions ----
    for (w, h) in ((W, H), (W2, H2)):
        for seq in gen_face_sequences(rng, a.frames // 2):
            gi = gaze_mod.GazeIntent()
            lines.append("S")
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            for ts, lm in enumerate(seq):
                pose = random_pose(rng)
                detector.next_face = as_objs(lm) if lm is not None else []
                detector.next_pose = pose
                final_yaw, final_pitch, head_yaw, head_pitch, state, landmarks = gi.process_frame(blank, ts * 16 + 1)
                expected.append((final_yaw, final_pitch, state)); kinds.append("gaze")
                packed = syn.pack_face(lm, pose) if lm is not None else None      # what the engine receives
                n = 0 if packed is None else len(packed)
                lines.append(f"F {w} {h} {n} " + (fmt(packed) if packed is not None else ""))
            gi.shutdown()

    # ---- hand ----
    ht = hand_mod.HandTracker()
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
        ok = (gs == es) and abs(gx - ex) < 2e-6 and abs(gy - ey) < 2e-6
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