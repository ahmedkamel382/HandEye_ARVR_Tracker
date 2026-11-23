# -*- coding: utf-8 -*-
"""Evaluate an UNSCORED native FAST readiness capture against the original gates.

This tool does not score a FAST trial. It consumes the native E4DSP numerical trace
produced by `core_engine.exe --dsp-record-dir` and reports whether the tracking input
met the original H1 readiness gates. It preserves every threshold; none is relaxed.

Usage:
    python check_unscored_fast_readiness.py --trace TRACE_DIR/trace.txt \
        --cues CUES_JSON [--mode fast|slow] [--out report.json]
    python check_unscored_fast_readiness.py --self-test
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

F = re.compile(r"^E4REC F\s+(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")

# Original readiness gates from the DSP Engineer's 23 Nov 2026 handoff (H1).
GATES = {
    "fast_ms_min": 150.0, "fast_ms_max": 800.0,
    "amplitude_min": 0.15, "y_drift_max": 0.05,
    "fresh_gap_max_ms": 100.0, "crossing_samples_min": 3,
    "fixation_samples_min": 25, "fixation_sd_max": 0.010,
    "fixation_range_max": 0.030, "fixation_slope_max": 0.005,
    "dwell_min_ms": 500.0,
}
DURATION = {"fast": 0.5, "slow": 3.0}


def parse_trace(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = F.match(line.strip())
            if not m:
                continue
            d = dict(KV.findall(m.group(1)))
            try:
                r = {
                    "frame": int(d.get("frame", -1)),
                    "t": float(d.get("input_t", d.get("t", d.get("cap", "0")))),
                    "n": int(d.get("n", 0)),
                    "ok": int(d.get("ok", 0)),
                    "x": float(d.get("ix", d.get("x", "nan"))),
                    "y": float(d.get("iy", d.get("y", "nan"))),
                    "trk": int(d.get("trk", -1)),
                }
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def fresh(rows):
    return [r for r in rows
            if r["ok"] == 1 and r["n"] == 42 and r["trk"] == 0
            and math.isfinite(r["x"]) and math.isfinite(r["y"])]


def window(rows, t0, t1):
    return [r for r in rows if t0 <= r["t"] <= t1]


def slope_per_second(points):
    if len(points) < 2:
        return 0.0
    t0 = points[0]["t"]
    xs = [p["t"] - t0 for p in points]
    ys = [p["x"] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((v - mx) ** 2 for v in xs)
    return 0.0 if den == 0 else sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den


def fixation(points):
    if len(points) < GATES["fixation_samples_min"]:
        return {"qualified": False, "reason": "fewer than %d eligible samples" % GATES["fixation_samples_min"],
                "samples": len(points)}
    xs = [p["x"] for p in points]
    sd = statistics.pstdev(xs)
    rng = max(xs) - min(xs)
    sl = abs(slope_per_second(points))
    dwell_ms = (points[-1]["t"] - points[0]["t"]) * 1000.0
    checks = {
        "samples": len(points) >= GATES["fixation_samples_min"],
        "sd": sd <= GATES["fixation_sd_max"],
        "range": rng <= GATES["fixation_range_max"],
        "slope": sl <= GATES["fixation_slope_max"],
        "dwell": dwell_ms >= GATES["dwell_min_ms"],
    }
    return {"qualified": all(checks.values()), "checks": checks,
            "samples": len(points), "sd": sd, "range": rng,
            "slope": sl, "dwell_ms": dwell_ms}


def evaluate_leg(rows, move_t, duration, expect_increase, label):
    stop_t = move_t + duration
    move_win = window(rows, move_t - 0.5, stop_t + 0.5)
    pts = fresh(move_win)
    # The cue schedule holds each mark for several seconds. A fixation gate needs
    # >=25 eligible samples, which at 12.4 FPS requires ~2 s; use a 2.9 s window.
    baseline = fresh(window(rows, move_t - 3.0, move_t - 0.1))
    endpoint = fresh(window(rows, stop_t + 0.1, stop_t + 3.0))
    out = {"leg": label, "move_t": move_t, "eligible_in_move_window": len(pts)}

    if len(pts) < GATES["crossing_samples_min"]:
        out.update(verdict="INCONCLUSIVE", reason="too few fresh samples in movement window")
        return out

    xs = [p["x"] for p in pts]
    amplitude = max(xs) - min(xs)
    base_x = statistics.mean([p["x"] for p in baseline]) if baseline else xs[0]
    end_x = statistics.mean([p["x"] for p in endpoint]) if endpoint else xs[-1]
    direction = 1 if expect_increase else -1
    signed = direction * (end_x - base_x)
    out.update(amplitude=amplitude, baseline_x=base_x, endpoint_x=end_x, signed_travel=signed)

    if signed <= 0:
        out.update(verdict="FAIL", reason="movement direction is opposite to protocol")
        return out

    lo = base_x + direction * GATES["amplitude_min"] * 0.1 * (amplitude / GATES["amplitude_min"])
    hi = base_x + direction * amplitude * 0.9
    cross = [p for p in pts if min(lo, hi) <= p["x"] <= max(lo, hi)]
    if len(cross) < GATES["crossing_samples_min"]:
        out.update(verdict="INCONCLUSIVE", reason="too few samples spanning 10-90 percent")
        return out
    t10, t90 = min(p["t"] for p in cross), max(p["t"] for p in cross)
    duration_ms = (t90 - t10) * 1000.0

    times = sorted(p["t"] for p in pts)
    max_gap_ms = max(((b - a) * 1000.0 for a, b in zip(times, times[1:])), default=0.0)
    base_y = statistics.mean([p["y"] for p in baseline]) if baseline else pts[0]["y"]
    end_y = statistics.mean([p["y"] for p in endpoint]) if endpoint else pts[-1]["y"]
    y_drift = abs(end_y - base_y)

    f_base = fixation(baseline)
    f_end = fixation(endpoint)
    checks = {
        "duration": GATES["fast_ms_min"] <= duration_ms <= GATES["fast_ms_max"],
        "amplitude": amplitude >= GATES["amplitude_min"],
        "y_drift": y_drift <= GATES["y_drift_max"],
        "fresh_gap": max_gap_ms <= GATES["fresh_gap_max_ms"],
        "crossing_samples": len(cross) >= GATES["crossing_samples_min"],
        "baseline_fixation": f_base["qualified"],
        "endpoint_fixation": f_end["qualified"],
    }
    out.update(duration_ms=duration_ms, max_gap_ms=max_gap_ms, y_drift=y_drift,
               crossing_samples=len(cross), baseline_fixation=f_base,
               endpoint_fixation=f_end, checks=checks)
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return out


def detect_moves(rows, duration, min_travel=0.10):
    """Locate the two FAST legs from the raw trace itself (direction-agnostic).

    Finds the two strongest monotonic x transitions whose 10-90% crossing takes
    roughly the configured movement duration. Returns (t1, t2, increasing_first).
    """
    pts = fresh(rows)
    best = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dt = pts[j]["t"] - pts[i]["t"]
            if 0.10 <= dt <= duration + 0.5:
                travel = pts[j]["x"] - pts[i]["x"]
                if abs(travel) >= min_travel:
                    best.append((abs(travel), pts[i]["t"], pts[j]["t"], travel > 0))
    # Keep the two non-overlapping strongest transitions.
    best.sort(reverse=True)
    chosen = []
    for item in best:
        if all(abs(item[1] - c[1]) > duration + 1.0 for c in chosen):
            chosen.append(item)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("could not locate two FAST legs in trace")
    chosen.sort(key=lambda c: c[1])
    return chosen[0][1], chosen[1][1], chosen[0][3]


def move_times_from_cues(cues_path, ready_trace_t=None):
    """Return trace-time for the two MOVE cues.

    Cue `at_seconds` are relative to the cue clock start (engine readiness), NOT to
    the E4DSP trace clock. Without `ready_trace_t` (the trace time when the cue clock
    started) the returned values are cue-relative and must not be used to index the
    trace. Callers must either supply the anchor or use --auto-moves.
    """
    with open(cues_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cues = data.get("cues", data) if isinstance(data, dict) else data
    times = []
    for c in cues:
        msg = c.get("message", "") if isinstance(c, dict) else ""
        if msg.startswith("MOVE!"):
            times.append(float(c["at_seconds"]))
    if len(times) != 2:
        raise ValueError("expected exactly two MOVE cues, found %d" % len(times))
    if ready_trace_t is None:
        raise ValueError(
            "cue at_seconds are cue-clock relative; supply --ready-trace-t (trace time "
            "when the cue clock started) or use --auto-moves instead")
    return ready_trace_t + times[0], ready_trace_t + times[1]


def evaluate(trace, move1, move2, mode, out_increasing=True):
    rows = parse_trace(trace)
    duration = DURATION[mode]
    legs = [
        evaluate_leg(rows, move1, duration, out_increasing, "OUTWARD_LEG"),
        evaluate_leg(rows, move2, duration, not out_increasing, "RETURN_LEG"),
    ]
    verdict = "PASS" if all(l["verdict"] == "PASS" for l in legs) else (
        "INCONCLUSIVE" if all(l["verdict"] != "FAIL" for l in legs) else "FAIL")
    return {"scored": False, "trace": os.path.abspath(trace), "mode": mode,
            "gates": GATES, "legs": legs, "verdict": verdict,
            "note": "UNSCORED readiness result only; not a scored FAST trial outcome."}


def self_test():
    import tempfile
    # Cue anchor: at_seconds must be offset by the ready trace time; unanchored use is rejected.
    with tempfile.TemporaryDirectory() as td:
        cues = os.path.join(td, "cues.json")
        with open(cues, "w", encoding="utf-8") as fh:
            json.dump({"cues": [{"at_seconds": 0, "message": "Hold left."},
                                {"at_seconds": 11.0, "message": "MOVE! Go right."},
                                {"at_seconds": 25.5, "message": "MOVE! Go left."}]}, fh)
        assert move_times_from_cues(cues, 4.5) == (15.5, 30.0)
        try:
            move_times_from_cues(cues)
        except ValueError:
            pass
        else:
            raise AssertionError("unanchored cue times must be rejected")
    print("SELF-TEST cue anchor: OK")
    # Clean synthetic capture: 30 Hz, baseline 1 s, fast outward/return, endpoints held.
    rows, t = [], 0.0
    def add(x, y, trk=0, ok=1, n=42):
        nonlocal t
        rows.append("E4REC F frame=%d input_t=%.6f n=%d ok=%d ix=%.6f iy=%.6f trk=%d" %
                    (len(rows), t, n, ok, x, y, trk))
        t += 1.0 / 30.0
    for _ in range(90): add(0.20, 0.50)   # 3.0 s baseline hold
    for i in range(15): add(0.20 + 0.50 * i / 14.0, 0.50)
    for _ in range(120): add(0.70, 0.50)  # 4.0 s hold at far mark
    for i in range(15): add(0.70 - 0.50 * i / 14.0, 0.50)
    for _ in range(120): add(0.20, 0.50)  # 4.0 s return hold
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "trace.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        rep = evaluate(p, 3.0, 7.5, "fast")
        assert rep["verdict"] == "PASS", json.dumps(rep, indent=2)[:1500]
        print("SELF-TEST clean capture: PASS (expected PASS)")
        # Inject a >100 ms fresh-data gap (4 frames at 30 Hz) in the outward leg.
        gapped = [r for r in rows]
        idx = next(i for i, r in enumerate(gapped) if "frame=95 " in r)
        for j in range(idx, idx + 4):
            gapped[j] = gapped[j].replace("ok=1", "ok=0")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(gapped) + "\n")
        rep2 = evaluate(p, 3.0, 7.5, "fast")
        assert rep2["verdict"] != "PASS", json.dumps(rep2, indent=2)[:1500]
        print("SELF-TEST gapped capture:", rep2["verdict"], "(expected non-PASS)")
    print("SELF-TEST OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace")
    ap.add_argument("--cues")
    ap.add_argument("--ready-trace-t", type=float,
                    help="trace time when the cue clock started (anchors --cues at_seconds)")
    ap.add_argument("--mode", choices=sorted(DURATION), default="fast")
    ap.add_argument("--move1-t", type=float)
    ap.add_argument("--move2-t", type=float)
    ap.add_argument("--auto-moves", action="store_true",
                    help="detect the two FAST legs from the trace (direction-agnostic)")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if not a.trace:
        ap.error("--trace is required unless --self-test is used")
    if a.auto_moves:
        m1, m2, inc = detect_moves(parse_trace(a.trace), DURATION[a.mode])
        print("Auto-detected moves: t=%.3f, t=%.3f, outward x increasing=%s" % (m1, m2, inc))
    elif a.cues:
        m1, m2 = move_times_from_cues(a.cues, a.ready_trace_t)
        inc = True
    elif a.move1_t is not None and a.move2_t is not None:
        m1, m2 = a.move1_t, a.move2_t
        inc = True
    else:
        ap.error("provide --auto-moves, --cues, or both --move1-t and --move2-t")
    rep = evaluate(a.trace, m1, m2, a.mode, inc)
    text = json.dumps(rep, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
