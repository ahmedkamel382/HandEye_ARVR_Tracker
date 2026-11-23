# -*- coding: utf-8 -*-
"""UNSCORED live Tester-path hand capture (item 6) — LOG ONLY, no mouse.

Drives the real HandEye Tester HandSource -> HandTracker on camera 0 and writes a
CSV of frame/time/status/tracking/x/y/state. No OS injection, no GUI, no scored
trial. Use together with the native --dsp-record-dir trace for comparison.

Usage:
  python capture_tester_hand_live.py --out <dir>/tester.csv --seconds 8
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
TESTER = REPO / "core_engine" / "os_integration" / "tester"
sys.path.insert(0, str(TESTER))
os.environ.setdefault("AI_PIPELINE_ROOT", str(REPO / "ai_pipeline"))

from tester_engine import HandSource, TesterConfig  # noqa: E402
import osi_capi  # noqa: E402

TRACK_NAME = {osi_capi.TRACKED: "TRACKED", osi_capi.HELD: "HELD", osi_capi.LOST: "LOST"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise RuntimeError("refusing to overwrite %s" % out)

    cfg = TesterConfig(lowlight=False, roi_recovery=False, real_mouse=False)
    src = HandSource("hand", cfg)
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise RuntimeError("camera %d could not be opened" % a.camera)

    t0 = time.monotonic()
    n = 0
    rows = []
    try:
        while time.monotonic() - t0 < a.seconds:
            ok, bgr = cap.read()
            if not ok:
                break
            obs = src.process(bgr)
            n += 1
            x, y = obs.point if obs.point else (-1.0, -1.0)
            rows.append((n, round(time.monotonic() - t0, 6), obs.status,
                         TRACK_NAME.get(obs.tracking, str(obs.tracking)),
                         round(x, 6), round(y, 6), obs.state, round(obs.ms, 3)))
    finally:
        cap.release()
        src.close()

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "status", "tracking", "x", "y", "state", "tester_ms"])
        w.writerows(rows)
    states = {}
    for r in rows:
        states[r[3]] = states.get(r[3], 0) + 1
    print("UNSCORED tester capture: %d frames, %s, %s" % (len(rows), states, out))
    print("NO mouse movement was performed (real_mouse=False).")


if __name__ == "__main__":
    main()
