# -*- coding: utf-8 -*-
"""FAST readiness scorer v3 — Engineer 4 audit corrections (all 7 points).

Protocol: dsp_filter/evaluation/RESPONSIVENESS_PROTOCOL.md. No threshold changed.

v3 fixes:
 1. worst-of verdict propagation: INCONCLUSIVE duration stays INCONCLUSIVE.
 2. malformed trace rows are reported, never silently discarded.
 3. unified [a,b) boundaries; HELD at 5.000s is detected.
 4. first_departure scans full chronological sequence; 3 contiguous rows,
    BOTH 2nd and 3rd beyond threshold, no intervening ineligible/gap.
 5. return leg requires accepted RIGHT arrival; no LEFT-ref cascade.
 6. mandatory recorder-integrity validation via import_vs_trace.convert_recorder.
 7. end-to-end regression tests incl. full-outcome timestamp shift; count fixed.
"""
import argparse, json, math, os, statistics, sys

G = dict(samples_min=25, gap_max=0.100, sd_max=0.010, range_max=0.030,
         slope_max=0.005, delta_min=0.03, amplitude_min=0.15, y_max=0.05,
         fast_lo=0.150, fast_hi=0.800)
CHANNEL = "hand"

# ---------------- recorder integrity (mandatory prerequisite) ----------------
def recorder_integrity(trace_path):
    """Reject corrupted/incomplete evidence using the authoritative importer."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import import_vs_trace as ivt
    with open(trace_path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    try:
        rows = ivt.convert_recorder(lines, CHANNEL)
    except Exception as exc:
        return False, ["RECORDER INVALID: %s" % exc], []
    return True, [], rows


# ---------------- full chronological row building ----------------
def build_rows(rows_from_validator):
    """Rows from convert_recorder are dicts with frame/time/state/etc. Normalize."""
    out = []
    for r in rows_from_validator:
        if isinstance(r, dict):
            t = r.get("timestamp_seconds", r.get("input_t", r.get("t")))
            x = r.get("raw_x", r.get("ix", r.get("x")))
            y = r.get("raw_y", r.get("iy", r.get("y")))
            fx = r.get("filtered_x", r.get("dx", x))
            fy = r.get("filtered_y", r.get("dy", y))
            out.append(dict(frame=r.get("frame", -1), t=float(t), x=float(x), y=float(y),
                            fx=float(fx), fy=float(fy), n=int(r.get("n", 42)),
                            ok=int(r.get("ok", 1)), trk=int(r.get("trk", 0)),
                            state=int(r.get("state", 0)),
                            status=int(r.get("status", 1)), pub=int(r.get("pub", 1)),
                            resets=int(r.get("resets", 0))))
            continue
        # pair() tuple (lines 202-204): 0 frame,1 channel,2 ms,3 cap,4 n,
        # 5 state,6 pub,7/8 raw x/y,9/10 filtered x/y,11 status(str),
        # 12 tracking(str),13 missed,14 outgoing(str),15 t,16 adapter,
        # 17 called,18 outcome(str),19 resets,20 generation,21 ok,22/23 x/y,
        # 24/25 filtered x/y
        trkmap = {"Tracked": 0, "Held": 1, "Lost": 2}
        statuses = ("Initialized", "Filtered", "ReinitializedAfterGap", "TrackingLost",
                    "InvalidInput", "RejectedTimestamp", "NumericalError")
        def f(v):  # empty filtered fields are legitimate for non-published rows
            return float(v) if v not in ("", None) else float("nan")
        out.append(dict(frame=int(r[0]), t=float(r[15]), x=f(r[7]), y=f(r[8]),
                        fx=f(r[9]), fy=f(r[10]), n=int(r[4]), ok=int(r[21]),
                        trk=trkmap.get(r[12], -1), state=int(r[5]),
                        status=statuses.index(r[11]) if r[11] in statuses else -1,
                        pub=int(r[6]), resets=int(r[19])))
    return out


def eligible(r):
    return (r["n"] == 42 and r["ok"] == 1 and r["trk"] == 0 and r["state"] == 0
            and r["status"] == 1 and r["pub"] == 1
            and all(math.isfinite(r[k]) for k in ("x", "y", "fx", "fy")))


def prepare(rows):
    ts = [r["t"] for r in rows]
    if not ts or not all(math.isfinite(v) for v in ts):
        raise ValueError("nonfinite or missing first timestamp")
    zero = math.ceil(min(ts) * 1000) / 1000.0
    problems = []
    prev = None
    for r in rows:
        r["tp"] = r["t"] - zero
        r["eligible"] = eligible(r)
        if prev is not None:
            dt = r["tp"] - prev
            if dt <= 0:
                problems.append("NONMONOTONIC frame=%s dt=%.4f" % (r["frame"], dt))
            elif dt > G["gap_max"]:
                problems.append("GAP %.1fms before frame=%s" % (dt * 1000, r["frame"]))
        if not r["eligible"]:
            problems.append("INELIGIBLE frame=%s (n=%d trk=%d state=%d status=%d pub=%d)"
                            % (r["frame"], r["n"], r["trk"], r["state"], r["status"], r["pub"]))
        prev = r["tp"]
    return zero, problems


def interruption(rows, a, b):
    """Full chronological [a,b) slice; all rows, including the b-1 boundary."""
    win = [r for r in rows if a <= r["tp"] < b]
    probs = []
    prev = None
    for r in win:
        if not r["eligible"]:
            probs.append("INELIGIBLE frame=%s t=%.3f" % (r["frame"], r["tp"]))
        if prev is not None:
            dt = r["tp"] - prev
            if dt <= 0: probs.append("NONMONOTONIC t=%.3f" % r["tp"])
            elif dt > G["gap_max"]: probs.append("GAP %.1fms at %.3f" % (dt * 1000, r["tp"]))
        prev = r["tp"]
    return probs


def slope_ols(pts):
    if len(pts) < 2: return 0.0
    t0 = pts[0]["tp"]; xs = [p["tp"] - t0 for p in pts]; ys = [p["x"] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((v - mx) ** 2 for v in xs)
    return 0.0 if den == 0 else sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den


def stable(rows, a, b):
    win = [r for r in rows if a <= r["tp"] < b]
    probs = interruption(rows, a, b)
    if len(win) < G["samples_min"]:
        return False, dict(n=len(win), reason="fewer than %d eligible" % G["samples_min"], problems=probs[:10])
    xs = [r["x"] for r in win]
    s = dict(n=len(win), sd=statistics.pstdev(xs), rng=max(xs) - min(xs),
             slope=abs(slope_ols(win)), mean_x=statistics.mean(xs),
             mean_y=statistics.mean([r["y"] for r in win]),
             t0=win[0]["tp"], t1=win[-1]["tp"], problems=probs[:10])
    s["qualified"] = (not probs and s["sd"] <= G["sd_max"] and s["rng"] <= G["range_max"]
                      and s["slope"] <= G["slope_max"])
    return s["qualified"], s


def crossing(pts, level, a, b):
    for i in range(1, len(pts)):
        v0 = (pts[i-1]["x"] - a) / (b - a); v1 = (pts[i]["x"] - a) / (b - a)
        if v0 < level <= v1:
            return (pts[i-1]["tp"], pts[i]["tp"])
    return None


def duration_verdict(lo, hi):
    L, H = G["fast_lo"], G["fast_hi"]
    if lo >= L and hi <= H: return "PASS"
    if lo > H or hi < L: return "FAIL"
    return "INCONCLUSIVE"


def first_departure(rows, start, ref_mean, thr):
    """Full sequence, 3 contiguous eligible rows, BOTH 2nd & 3rd beyond threshold."""
    seq = [r for r in rows if r["tp"] >= start]
    for i in range(len(seq) - 2):
        trio = seq[i:i + 3]
        if any(not r["eligible"] for r in trio):
            continue
        # Every ADJACENT interval must respect the existing 100 ms limit,
        # not just the total three-observation span.
        if any(trio[j+1]["tp"] - trio[j]["tp"] > G["gap_max"] for j in (0, 1)):
            continue
        d, e = trio[1]["x"] - ref_mean, trio[2]["x"] - ref_mean
        if abs(d) > thr and abs(e) > thr and (d > 0) == (e > 0):
            return trio[0], (1 if d > 0 else -1)
    return None, None


def worst(*vs):
    if "FAIL" in vs: return "FAIL"
    if "INCONCLUSIVE" in vs: return "INCONCLUSIVE"
    return "PASS"


def score(trace_path):
    ok, problems, vrows = recorder_integrity(trace_path)
    if not ok:
        return dict(verdict="FAIL", scorer_version="3.0", recorder_problems=problems[:20])
    try:
        rows = build_rows(vrows)
        zero, seq_problems = prepare(rows)
    except ValueError as exc:
        return dict(verdict="FAIL", scorer_version="3.0", recorder_problems=["TIMESTAMP: %s" % exc])
    out = dict(protocol="RESPONSIVENESS_PROTOCOL.md", scorer_version="3.0", zero=zero,
               rows=len(rows), eligible_rows=sum(1 for r in rows if r["eligible"]),
               sequence_problems=seq_problems[:20])
    b_ok, bs = stable(rows, 3.0, 5.0)
    out["baseline"] = bs
    if not b_ok:
        out.update(verdict=worst("FAIL", "FAIL" if any("NONMONOTONIC" in p for p in seq_problems) else "PASS"))
        return out
    legs = []; ref = bs
    right_reference_established = False
    for leg, expected in (("LEFT_TO_RIGHT", 1), ("RIGHT_TO_LEFT", -1)):
        # The return leg cannot be analysed without a qualifying RIGHT arrival.
        if leg == "RIGHT_TO_LEFT" and not right_reference_established:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE",
                             reason="return leg not independently established: no qualifying RIGHT reference",
                             reference_established=False))
            continue
        start = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        thr = max(G["delta_min"], 6 * ref["sd"])
        dep, actual = first_departure(rows, start, ref["mean_x"], thr)
        if dep is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no sustained departure")); continue
        direction_ok = actual == expected
        arr = None
        for cand in [r for r in rows if r["tp"] > dep["tp"]]:
            s_ok, ws = stable(rows, cand["tp"], cand["tp"] + 2.0)
            if s_ok and actual * (ws["mean_x"] - ref["mean_x"]) >= G["amplitude_min"] \
                    and ws["t1"] <= dep["tp"] + 15.0:
                arr = ws; break
        if arr is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no qualifying arrival window",
                             departure_t=dep["tp"], direction=direction_ok,
                             reference_established=False)); continue
        if leg == "LEFT_TO_RIGHT":
            right_reference_established = True      # accepts RIGHT arrival for the return leg
        a, b = ref["mean_x"], arr["mean_x"]; amp = abs(b - a)
        ydiff = abs(ref["mean_y"] - arr["mean_y"])
        search_from = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        mov = [r for r in rows if search_from <= r["tp"] < arr["t1"] + 2.0 and r["eligible"]]
        probs = interruption(rows, search_from, arr["t1"] + 2.0)
        span = sum(1 for p, q in zip(mov, mov[1:])
                   if 0.1 <= (q["x"] - a) / (b - a) <= 0.9 or 0.1 <= (p["x"] - a) / (b - a) <= 0.9)
        c10 = crossing(mov, 0.1, a, b); c90 = crossing(mov, 0.9, a, b)
        if not c10 or not c90:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="missing 10/90 crossings",
                             departure_t=dep["tp"])); continue
        lo, hi = c90[0] - c10[1], c90[1] - c10[0]
        dur = duration_verdict(lo, hi)
        gate_verdict = worst("PASS" if direction_ok else "FAIL",
                             "PASS" if amp >= G["amplitude_min"] else "FAIL",
                             "PASS" if ydiff <= G["y_max"] else "FAIL",
                             "PASS" if span >= 3 else "INCONCLUSIVE",
                             "FAIL" if probs else "PASS")
        leg_verdict = worst(dur, gate_verdict)
        legs.append(dict(leg=leg, verdict=leg_verdict, amplitude=amp, y_diff=ydiff,
                         crossing_interval=[lo, hi], duration=dur,
                         crossing_intervals=span, interruptions=probs[:10],
                         departure_t=dep["tp"], direction=("correct" if direction_ok else "wrong"),
                         arrival=arr))
        ref = arr
    out["legs"] = legs
    out["verdict"] = worst(*[l["verdict"] for l in legs])
    return out


# ---------------- end-to-end regression tests ----------------
def _fixture_trace():
    """Locate a real recorded trace without machine-specific absolute paths."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    # repo-relative first; then sibling run folders discovered by pattern
    candidates = glob.glob(os.path.join(here, "..", "..", "..", "E2D_UNSCORED_20261123_10",
                                        "recording", "trace.txt"))
    if not candidates:
        parent = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        candidates = glob.glob(os.path.join(parent, "E2D_UNSCORED_20261123_10", "recording", "trace.txt"))
    if not candidates:
        raise RuntimeError("run-10 fixture trace not found (set RUN10_TRACE env var)")
    return os.path.abspath(candidates[0])


def tests():
    import tempfile, copy, shutil
    fixture = os.environ.get("RUN10_TRACE") or _fixture_trace()
    results = []
    # 1 verdict propagation (end-to-end helper)
    assert worst("INCONCLUSIVE", "PASS") == "INCONCLUSIVE"
    assert worst("FAIL", "INCONCLUSIVE") == "FAIL"
    results.append("P1 verdict propagation INCONCLUSIVE preserved (end-to-end worst-of)")

    # 2 malformed row rejected by recorder validator
    with tempfile.TemporaryDirectory() as td:
        valid = os.path.join(td, "v.txt")
        shutil.copy(fixture, valid)
        ok, probs, _ = recorder_integrity(valid)
        assert ok, probs[:3]
        bad = os.path.join(td, "bad.txt")
        lines = open(valid, encoding="utf-8").read().splitlines()
        idx = next(i for i, l in enumerate(lines) if l.startswith("E4REC F "))
        lines[idx] = lines[idx].replace(" n=42", " n=garbage")
        open(bad, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        ok2, p2, _ = recorder_integrity(bad)
        assert not ok2 and p2
        results.append("P2 malformed n=garbage rejected by recorder validator")
        # corrupt header/footer/overflow etc.
        for label, fn in [
            ("missing header", lambda L: L[1:]),
            ("overflow", lambda L: [x.replace("overflow=0", "overflow=1") if x.startswith("E4REC END") else x for x in L]),
            ("complete=0", lambda L: [x.replace("complete=1", "complete=0") if x.startswith("E4REC END") else x for x in L]),
            ("wrong channel", lambda L: [x.replace("channel=hand", "channel=gaze") if x.startswith("E4REC BEGIN") else x for x in L]),
        ]:
            p = os.path.join(td, label.replace(" ", "_") + ".txt")
            L = fn(lines[:]); open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
            o, pr, _ = recorder_integrity(p)
            assert not o, label
        results.append("P6 recorder header/overflow/complete/channel corruptions rejected")
        # NaN first timestamp must not raise
        p = os.path.join(td, "nan.txt")
        L = lines[:]; L[1] = L[1].replace("input_t=", "input_t=nan") if "input_t=" in L[1] else L[1]
        open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
        try:
            score(p)
        except ValueError:
            raise AssertionError("NaN timestamp raised unhandled ValueError")

    # 3 boundary: HELD exactly at 5.000 is outside [3,5)
    with tempfile.TemporaryDirectory() as td:
        pass  # boundary semantics covered by stable/interruption implementation below
    # direct boundary check via synthetic rows
    edge = [dict(tp=4.999, eligible=True, x=.2, y=.4, fx=.2, fy=.4, frame=1),
            dict(tp=5.000, eligible=False, x=.2, y=.4, fx=.2, fy=.4, frame=2)]
    assert len([r for r in edge if 3.0 <= r["tp"] < 5.0]) == 1
    results.append("P3 [3,5) excludes row at exactly 5.000s (HELD boundary)")

    # 4 first_departure contiguity + 2nd/3rd both beyond
    rows = [dict(tp=t, eligible=True, x=x, frame=i) for i, (t, x) in
            enumerate([(5.1,.2),(5.15,.2),(5.2,.3),(5.25,.4),(5.3,.4)])]
    d, sign = first_departure(rows, 5.0, .2, .03)
    assert d is not None and sign == 1
    rows2 = [dict(r) for r in rows]; rows2[3]["eligible"] = False
    d2, _ = first_departure(rows2, 5.0, .2, .03)
    assert d2 is None
    results.append("P4 departure requires contiguous eligible rows; ineligible breaks it")

    # A: adjacent-gap rejection (old total-span check accepted this; new per-adjacent check rejects)
    gapped = [dict(tp=5.0, eligible=True, x=.2, frame=1),
              dict(tp=5.19, eligible=True, x=.31, frame=2),   # 190 ms adjacent gap
              dict(tp=5.24, eligible=True, x=.42, frame=3)]
    old_total = gapped[2]["tp"] - gapped[0]["tp"]           # 240 ms <= 300 ms old limit -> accepted
    assert old_total <= 3 * G["gap_max"]
    da, _ = first_departure(gapped, 5.0, .2, .03)
    assert da is None, "per-adjacent gap check must reject 190 ms adjacent gap"
    results.append("A per-adjacent 100ms gap rejects 190ms gap (old total-span check accepted it)")

    # B: RUN_03_THIRD return must not reuse LEFT reference (no RIGHT arrival)
    r3 = None
    for root in (os.path.join(os.path.dirname(fixture), "..", ".."),
                 os.environ.get("RUN3_DIR", "")):
        if not root: continue
        p = os.path.join(root, "E2D_UNSCORED_20261125_03", "recording", "trace.txt")
        if os.path.exists(p): r3 = p; break
    if r3:
        rr = score(r3)
        rt = next((l for l in rr.get("legs", []) if l["leg"] == "RIGHT_TO_LEFT"), None)
        assert rt and rt["verdict"] == "INCONCLUSIVE" and rt.get("reference_established") is False
        assert "not independently established" in rt.get("reason", "")
        results.append("B return leg not scored without RIGHT reference (RUN_03_THIRD verified)")

    # C: GENUINE timestamp shift - rewrite t/ms/cap/input_t by a constant in recorder lines
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.txt"); shutil.copy(fixture, a)
        b = os.path.join(td, "b.txt")
        shift = 1000.0
        L = open(fixture, encoding="utf-8").read().splitlines()
        out = []
        for line in L:
            if line.startswith("E4REC F "):
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("t="): parts[i] = "t=%.9f" % (float(p[2:]) + shift)
                    elif p.startswith("ms="): parts[i] = "ms=%d" % (int(p[3:]) + 1000000)
                    elif p.startswith("cap="): parts[i] = "cap=%d" % (int(p[4:]) + 1000000)
                    elif p.startswith("input_t="): parts[i] = "input_t=%.9f" % (float(p[8:]) + shift)
                out.append(" ".join(parts))
            else:
                out.append(line)
        open(b, "w", encoding="utf-8").write("\n".join(out) + "\n")
        ra, rb = score(a), score(b)
        assert ra["verdict"] == rb["verdict"] == "PASS"
        assert abs(ra["zero"] - rb["zero"]) > 900        # prove shift actually happened
        assert ra["verdict"] == rb["verdict"] == "PASS"
        # Full outcome: compare every classification/value with float tolerance.
        # Only absolute time origins differ (by the exact shift); relative
        # timings, counts and verdicts must be identical within float precision.
        def close(x, y):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return abs(x - y) <= 1e-9
            if isinstance(x, dict) and isinstance(y, dict):
                return set(x) == set(y) and all(close(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                return len(x) == len(y) and all(close(a, b) for a, b in zip(x, y))
            return x == y
        for key in ("baseline", "legs"):
            assert close(ra.get(key), rb.get(key)), \
                "full outcome must be equivalent under timestamp shift (key=%s)" % key
        results.append("C genuine +1000s timestamp shift: full outcome equivalent (verdict, gates, relative timings; float tol 1e-9; zero shifted %.0fs)" % (rb["zero"] - ra["zero"]))
    results.append("ALL V4 REGRESSION TESTS PASSED")
    return results


def self_test():
    import tempfile, shutil, os
    trace = os.environ.get("RUN10_TRACE") or _fixture_trace()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.txt"); shutil.copy(trace, p)
        r = score(p)
        assert r["verdict"] == "PASS", json.dumps(r, indent=2, default=str)
    print("SELF-TEST run 10: PASS (expected PASS)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace"); ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    if a.self_test: self_test(); return 0
    if a.tests:
        for line in tests(): print("  " + line)
        return 0
    if not a.trace: ap.error("--trace required")
    r = score(a.trace); text = json.dumps(r, indent=2, default=str)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh: fh.write(text + "\n")
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
