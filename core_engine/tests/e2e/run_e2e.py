#!/usr/bin/env python
"""End-to-end test of the whole engine, no camera and no MediaPipe.

Draws a 5.5 s, 60 FPS synthetic video whose pixels carry a known scene (see
tests/fake_ai/bridge/landmark_extractors.py), runs the real core_engine on it in dry-run mode
(--os log, the mouse is never touched) and checks what comes out of every stage:

  capture -> mirror + RGB (colour block, disc position) -> Python extractors -> native hand/gaze math
  -> forest -> 1-Euro filter -> OS integration (Eng 5, dry run: nothing reaches the real mouse)
  + frame accounting (no frame unaccounted for), no Python errors, no leaked frame slots.

  python run_e2e.py --exe core_engine --ai-root <repo>/core_engine/tests/fake_ai
Needs: python with opencv (cv2) and numpy.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

W, H, FPS = 640, 480, 60
# (name, frames, face, hand, pinch)
PHASES = [("face only", 60, True, False, False),
          ("hand", 90, True, True, False),
          ("hand + pinch", 90, True, True, True),
          ("hand released", 60, True, True, False),
          ("nothing", 30, False, False, False)]
DISC_X, DISC_Y = 100, 240          # in the SOURCE video: the engine mirrors, so the hand lands at x = 1 - 100/640 = 0.84


def draw(face, hand, pinch):
    f = np.zeros((H, W, 3), dtype=np.uint8)
    def block(row, color):
        f[row * 40:row * 40 + 40, W - 40:W] = color         # source right edge == left edge after mirroring
    white = (255, 255, 255)
    if face: block(0, white)
    if pinch: block(3, white)
    block(4, (0, 0, 255))                                    # pure red in BGR: must arrive as R high / B low
    if hand: cv2.circle(f, (DISC_X, DISC_Y), 25, white, -1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--ai-root", required=True)
    a = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    video = os.path.join(tmp.name, "scene.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not vw.isOpened():
        print("E2E: cannot create the test video (MJPG writer missing in this OpenCV build)", file=sys.stderr)
        return 2
    total = 0
    for _, n, face, hand, pinch in PHASES:
        frame = draw(face, hand, pinch)
        for _ in range(n):
            vw.write(frame); total += 1
    vw.release()

    # stand-in hand model (trained on synthetic hands): the real one is checked by the parity test
    sys.path.insert(0, os.path.join(a.ai_root, "bridge"))
    import hand_features_ref
    hand_pkl = os.path.join(tmp.name, "standin_hand_rf.pkl")
    hand_features_ref.train_standin_model(hand_pkl)

    cmd = [a.exe, "--video", video, "--ai-root", a.ai_root, "--hand-model", hand_pkl,
           "--start-active", "--os", "log"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = p.stdout
    print(out)
    if p.stderr.strip():
        print("--- stderr ---\n" + p.stderr)

    checks = []
    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))

    check("engine exit code 0", p.returncode == 0, f"exit {p.returncode}")
    m = re.search(r"captured (\d+) \| processed (\d+) \| evicted (\d+) \| skipped (\d+) \| stale (\d+) \| python errors (\d+)", out)
    if not m:
        check("summary line present", False)
        return 1
    cap, proc, evi, skp, stale, pyerr = map(int, m.groups())
    check("every frame accounted for", cap == proc + evi + skp + stale, f"{cap} = {proc}+{evi}+{skp}+{stale}")
    check("all video frames captured", cap == total - 1, f"{cap} of {total - 1} (first frame is the startup probe)")
    check("no Python errors (RGB order, contiguity, timestamps)", pyerr == 0, f"{pyerr}")
    check("no leaked frame slots", "(no leaks)" in out)
    m = re.search(r"os stage: (\d+) frames submitted", out)
    check("frames reached the OS stage", bool(m) and int(m.group(1)) > 50, m.group(1) if m else "no os line")
    check("hand is mirrored (x ~ 0.84, y ~ 0.50)", bool(re.search(r"hand\(0\.8[2-6],0\.(4[89]|5[0-2]) s=", out)))
    check("pinch state reported by the native hand math", bool(re.search(r"hand\([\d.]+,[\d.]+ s=1\)", out)))
    # ---- --check (start-up self test): must pass on a good setup and fail with a hint on bad ones ----
    base = [a.exe, "--check", "--ai-root", a.ai_root, "--hand-model", hand_pkl]
    c = subprocess.run(base + ["--video", video], capture_output=True, text=True, timeout=120)
    check("--check passes on a good setup (exit 0)", c.returncode == 0 and "ALL CHECKS PASSED" in c.stdout,
          f"exit {c.returncode}")
    c = subprocess.run(base + ["--video", os.path.join(tmp.name, "missing.avi")], capture_output=True, text=True, timeout=120)
    check("--check fails (exit 1) when the video source is missing",
          c.returncode == 1 and "[FAIL] video source" in c.stdout, f"exit {c.returncode}")
    c = subprocess.run([a.exe, "--check", "--ai-root", os.path.join(tmp.name, "nope"), "--video", video],
                       capture_output=True, text=True, timeout=120)
    check("--check fails (exit 1) with a hint when --ai-root is wrong",
          c.returncode == 1 and "--ai-root" in c.stdout, f"exit {c.returncode}")
    c = subprocess.run([a.exe, "--no-such-option"], capture_output=True, text=True, timeout=60)
    check("unknown option -> exit 1", c.returncode == 1, f"exit {c.returncode}")
    c = subprocess.run([a.exe, "--video", os.path.join(tmp.name, "missing.avi"), "--ai-root", a.ai_root],
                       capture_output=True, text=True, timeout=120)
    check("unreadable video -> exit 1, engine does not hang", c.returncode == 1, f"exit {c.returncode}")
    print("\nE2E", "PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
