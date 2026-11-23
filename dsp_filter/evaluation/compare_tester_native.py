# -*- coding: utf-8 -*-
"""Compare UNSCORED live Tester-path vs native Core hand observations (item 6).

Reads the Tester CSV from capture_tester_hand_live.py and the native E4DSP trace
from core_engine.exe --dsp-record-dir, both taken with a static hand hold. Reports
frame counts, TRACKED/HELD/LOST distributions and mean x/y for fresh observations.
No camera, no scoring, no threshold change.

Usage:
    python compare_tester_native.py --tester tester.csv --native native_run/trace.txt
"""
import argparse
import csv
import re
import statistics
import sys

F = re.compile(r"^E4REC F\s+(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")


def read_tester(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"t": float(r["t_s"]), "state": r["tracking"],
                         "x": float(r["x"]), "y": float(r["y"]), "status": r["status"]})
    return rows


def read_native(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = F.match(line.strip())
            if not m:
                continue
            d = dict(KV.findall(m.group(1)))
            trk = int(d.get("trk", -1))
            rows.append({"t": float(d.get("input_t", "0")),
                         "state": {0: "TRACKED", 1: "HELD", 2: "LOST"}.get(trk, str(trk)),
                         "x": float(d.get("ix", "nan")), "y": float(d.get("iy", "nan")),
                         "ok": int(d.get("ok", 0)), "n": int(d.get("n", 0))})
    return rows


def summary(name, rows, fresh_fn):
    fresh = [r for r in rows if fresh_fn(r)]
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    xs = [r["x"] for r in fresh if r["x"] >= 0]
    ys = [r["y"] for r in fresh if r["y"] >= 0]
    print("%s: frames=%d states=%s fresh=%d" % (name, len(rows), counts, len(fresh)))
    if xs and ys:
        print("  mean x=%.4f sd=%.4f range=%.4f | mean y=%.4f sd=%.4f range=%.4f" %
              (statistics.mean(xs), statistics.pstdev(xs), max(xs) - min(xs),
               statistics.mean(ys), statistics.pstdev(ys), max(ys) - min(ys)))
    else:
        print("  no fresh coordinates")
    return counts, (statistics.mean(xs) if xs else None), (statistics.mean(ys) if ys else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tester", required=True)
    ap.add_argument("--native", required=True)
    a = ap.parse_args()
    t = read_tester(a.tester)
    n = read_native(a.native)
    tc, tx, ty = summary("TESTER (HandSource->HandTracker)", t,
                         lambda r: r["state"] == "TRACKED")
    nc, nx, ny = summary("NATIVE (core_engine E4DSP)", n,
                         lambda r: r["state"] == "TRACKED" and r["ok"] == 1 and r["n"] == 42)
    print("\nComparison:")
    t_tracked = tc.get("TRACKED", 0)
    n_tracked = nc.get("TRACKED", 0)
    print("  TRACKED frames: tester=%d native=%d (different durations; compare ratios and means)" %
          (t_tracked, n_tracked))
    print("  tester TRACKED ratio=%.3f native TRACKED ratio=%.3f" %
          (t_tracked / max(len(t), 1), n_tracked / max(len(n), 1)))
    if tx is not None and nx is not None:
        print("  |mean x difference|=%.4f  |mean y difference|=%.4f" % (abs(tx - nx), abs(ty - ny)))
    print("\nVerdict: %s" % ("CONSISTENT" if t_tracked and n_tracked else "INCONCLUSIVE"))
    print("UNSCORED comparison only. Static hold; no mouse movement; no scored trial.")


if __name__ == "__main__":
    main()
