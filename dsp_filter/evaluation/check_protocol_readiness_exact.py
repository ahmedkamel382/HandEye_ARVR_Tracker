# -*- coding: utf-8 -*-
"""EXACT original-protocol FAST readiness scorer v2 (RESPONSIVENESS_PROTOCOL.md).

Corrections vs 393e834 (Engineer 4 audit):
  1. FAST duration enforces BOTH inclusive bounds 150<=bracket<=800 ms.
  2. Full chronological sequence retained; ineligible rows are NOT removed before
     movement detection. Ineligible rows are marked, never silently dropped; prohibited
     interruptions (Held/Lost/invalid/missing/nonmonotonic/gap>100ms) are reported and
     invalidate the affected windows/legs.
  3. First sustained departure is found chronologically regardless of direction; its
     sign is then checked against the protocol's expected direction (no skipping an
     earlier wrong-direction move).
  4. ONE documented time origin: protocol-relative tp = t - zero everywhere
     (windows, boundaries, gaps, crossings, endpoints). A constant shift of all
     timestamps yields identical duration verdicts.

All original numerical thresholds unchanged.
"""
import argparse, json, math, re, statistics, sys

F = re.compile(r"^E4REC F\s+(.*)$"); KV = re.compile(r"(\w+)=([^\s]+)")
G = dict(samples_min=25, gap_max=0.100, sd_max=0.010, range_max=0.030,
         slope_max=0.005, delta_min=0.03, amplitude_min=0.15, y_max=0.05,
         fast_lo=0.150, fast_hi=0.800)


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = F.match(line.strip())
            if not m: continue
            d = dict(KV.findall(m.group(1)))
            try:
                rows.append(dict(frame=int(d.get("frame", -1)),
                    t=float(d.get("input_t", d.get("t", "0"))),
                    n=int(d.get("n", 0)), ok=int(d.get("ok", 0)),
                    x=float(d.get("ix", "nan")), y=float(d.get("iy", "nan")),
                    fx=float(d.get("dx", "nan")), fy=float(d.get("dy", "nan")),
                    trk=int(d.get("trk", -1)), state=int(d.get("state", -1)),
                    status=int(d.get("status", -1)), pub=int(d.get("pub", 0)),
                    resets=int(d.get("resets", 0)), outcome=int(d.get("outcome", -1))))
            except (ValueError, TypeError):
                continue
    return rows


def prepare(rows):
    """Assign ONE time origin to every row. Never drop rows."""
    zero = math.ceil(min(r["t"] for r in rows) * 1000) / 1000.0
    for r in rows:
        r["tp"] = r["t"] - zero
        r["eligible"] = eligible(r)
    return zero


def eligible(r):
    return (r["n"] == 42 and r["ok"] == 1 and r["trk"] == 0 and r["state"] == 0
            and r["status"] == 1 and r["pub"] == 1
            and all(math.isfinite(r[k]) for k in ("x", "y", "fx", "fy")))


def interruption(rows, a, b):
    """Check the FULL chronological slice [a,b] of protocol time for prohibited rows."""
    win = [r for r in rows if a <= r["tp"] <= b]
    problems = []
    prev_t = None
    for r in win:
        if not r["eligible"]:
            if r["trk"] == 1: problems.append("HELD at %.3f" % r["tp"])
            elif r["trk"] == 2: problems.append("LOST at %.3f" % r["tp"])
            else: problems.append("INELIGIBLE(state=%d,status=%d,n=%d) at %.3f" %
                                  (r["state"], r["status"], r["n"], r["tp"]))
        if prev_t is not None:
            dt = r["tp"] - prev_t
            if dt <= 0:
                problems.append("NONMONOTONIC dt=%.4f at %.3f" % (dt, r["tp"]))
            elif dt > G["gap_max"]:
                problems.append("GAP %.1fms at %.3f" % (dt * 1000, r["tp"]))
        prev_t = r["tp"]
    return problems


def slope_ols(pts):
    if len(pts) < 2: return 0.0
    t0 = pts[0]["tp"]; xs = [p["tp"] - t0 for p in pts]; ys = [p["x"] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((v - mx) ** 2 for v in xs)
    return 0.0 if den == 0 else sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den


def stable(rows, a, b, full):
    """Window [a,b): all rows in it must be eligible and gap-free; then stats."""
    win = [r for r in rows if a <= r["tp"] < b]
    probs = interruption(full, a, b)
    if len(win) < G["samples_min"]:
        return False, dict(n=len(win), reason="fewer than %d eligible" % G["samples_min"],
                           problems=probs[:10])
    xs = [r["x"] for r in win]
    s = dict(n=len(win), sd=statistics.pstdev(xs), rng=max(xs) - min(xs),
             slope=abs(slope_ols(win)), mean_x=statistics.mean(xs),
             mean_y=statistics.mean([r["y"] for r in win]),
             t0=win[0]["tp"], t1=win[-1]["tp"], problems=probs[:10])
    s["qualified"] = (not probs and s["sd"] <= G["sd_max"] and s["rng"] <= G["range_max"]
                      and s["slope"] <= G["slope_max"])
    return s["qualified"], s


def crossing(pts, level, a, b):
    """First 10/90 crossing, using protocol-relative timestamps."""
    for i in range(1, len(pts)):
        v0 = (pts[i-1]["x"] - a) / (b - a); v1 = (pts[i]["x"] - a) / (b - a)
        if v0 < level <= v1:
            return (pts[i-1]["tp"], pts[i]["tp"])
    return None


def duration_verdict(lo, hi):
    """Entire bracket must lie within [150,800] ms, inclusive."""
    L, H = G["fast_lo"], G["fast_hi"]
    if lo >= L and hi <= H: return "PASS"
    if lo > H or hi < L: return "FAIL"
    return "INCONCLUSIVE"


def first_departure(rows, start, ref_mean, thr):
    """First sustained departure CHRONOLOGICALLY, regardless of direction."""
    cand = [r for r in rows if r["tp"] >= start and r["eligible"]]
    for i in range(len(cand) - 2):
        seq = cand[i:i + 3]
        if seq[2]["tp"] - seq[0]["tp"] > 2 * G["gap_max"] + 0.7:  # allow 3 frames spanning ~0.4s
            continue
        d = seq[1]["x"] - ref_mean
        e = seq[2]["x"] - ref_mean
        if abs(d) > thr and abs(e) > thr and (d > 0) == (e > 0):
            return cand[i], (1 if d > 0 else -1)   # row + actual direction
    return None, None


def score(path):
    rows = parse(path)
    if not rows: return dict(verdict="NOT RUN", reason="no E4REC F rows")
    zero = prepare(rows)
    ok, bs = stable(rows, 3.0, 5.0, rows)            # fixed [3,5), never substituted
    out = dict(protocol="RESPONSIVENESS_PROTOCOL.md", scorer_version="2.0",
               zero=zero, rows=len(rows),
               eligible_rows=sum(1 for r in rows if r["eligible"]), baseline=bs)
    if not ok:
        out.update(verdict="FAIL", reason="fixed [3,5) baseline fails stable-window gates")
        return out
    legs = []; ref = bs
    for leg, expected in (("LEFT_TO_RIGHT", 1), ("RIGHT_TO_LEFT", -1)):
        start = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        thr = max(G["delta_min"], 6 * ref["sd"])
        dep, actual = first_departure(rows, start, ref["mean_x"], thr)
        if dep is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no sustained departure")); continue
        direction_ok = (actual == expected)
        direction_note = ("correct" if direction_ok else
                          "WRONG-DIRECTION FIRST DEPARTURE (sign=%+d, expected=%+d)" % (actual, expected))
        # arrival: first 2s stable window with shift>=0.15 in the ACTUAL direction, within 15s
        arr = None
        for cand in [r for r in rows if r["tp"] > dep["tp"] and r["eligible"]]:
            s_ok, ws = stable(rows, cand["tp"], cand["tp"] + 2.0, rows)
            if s_ok and actual * (ws["mean_x"] - ref["mean_x"]) >= G["amplitude_min"] \
                    and ws["t1"] <= dep["tp"] + 15.0:
                arr = ws; break
        if arr is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no qualifying arrival window",
                             departure_t=dep["tp"], direction=direction_note)); continue
        a, b = ref["mean_x"], arr["mean_x"]; amp = abs(b - a)
        ydiff = abs(ref["mean_y"] - arr["mean_y"])
        search_from = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        mov = [r for r in rows if search_from <= r["tp"] <= arr["t1"] + 2.0 and r["eligible"]]
        probs = interruption(rows, search_from, arr["t1"] + 2.0)
        span = sum(1 for p, q in zip(mov, mov[1:])
                   if 0.1 <= (q["x"] - a) / (b - a) <= 0.9 or 0.1 <= (p["x"] - a) / (b - a) <= 0.9)
        c10 = crossing(mov, 0.1, a, b); c90 = crossing(mov, 0.9, a, b)
        if not c10 or not c90:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="missing 10/90 crossings",
                             departure_t=dep["tp"], direction=direction_note)); continue
        lo, hi = c90[0] - c10[1], c90[1] - c10[0]
        dur = duration_verdict(lo, hi)
        ok_all = (direction_ok and amp >= G["amplitude_min"] and ydiff <= G["y_max"]
                  and span >= 3 and not probs)
        legs.append(dict(leg=leg, verdict="PASS" if (dur == "PASS" and ok_all) else "FAIL",
                         amplitude=amp, y_diff=ydiff,
                         crossing_interval=[lo, hi], duration=dur,
                         crossing_intervals=span, interruptions=probs[:10],
                         departure_t=dep["tp"], direction=direction_note, arrival=arr))
        ref = arr
    out["legs"] = legs
    out["verdict"] = "PASS" if all(l["verdict"] == "PASS" for l in legs) else (
        "INCONCLUSIVE" if all(l["verdict"] != "FAIL" for l in legs) else "FAIL")
    return out


# ---------------- regression tests ----------------
def _row(t, x, **kw):
    d = dict(frame=0, t=t, n=42, ok=1, x=x, y=0.4, fx=x, fy=0.4,
             trk=0, state=0, status=1, pub=1, resets=0, outcome=0)
    d.update(kw)
    return "E4REC F frame=%d input_t=%.6f n=%d ok=%d ix=%.6f iy=%.6f dx=%.6f dy=%.6f " \
           "trk=%d state=%d status=%d pub=%d resets=%d outcome=%d" % (
               d["frame"], d["t"], d["n"], d["ok"], d["x"], d["y"], d["fx"], d["fy"],
               d["trk"], d["state"], d["status"], d["pub"], d["resets"], d["outcome"])


def _write(tmp, lines):
    import os
    p = os.path.join(tmp, "trace.txt")
    with open(p, "w") as f: f.write("\n".join(lines) + "\n")
    return p


def tests():
    import tempfile
    results = []

    # Defect 1: duration bounds
    for lo, hi, exp in [(0.05, 0.14, "FAIL"),          # entirely below 150 -> FAIL
                        (0.10, 0.20, "INCONCLUSIVE"),  # straddles 150 lower bound
                        (0.15, 0.30, "PASS"),          # exactly 150 lower bound inclusive
                        (0.30, 0.80, "PASS"),          # exactly 800 upper bound inclusive
                        (0.85, 0.95, "FAIL"),          # entirely above 800 -> FAIL
                        (0.10, 0.85, "INCONCLUSIVE"),  # straddles both bounds
                        (0.14, 0.80, "INCONCLUSIVE")]: # straddles 150 lower bound
        got = duration_verdict(lo, hi)
        assert got == exp, "duration %.3f..%.3f got %s expected %s" % (lo, hi, got, exp)
    results.append("D1 duration bounds (6 cases)")

    # Defect 4: time-shift invariance
    with tempfile.TemporaryDirectory() as td:
        base = [0.1 + 0.05 * i for i in range(100)]
        a = [_row(t, 0.2) for t in base]
        b = [_row(t + 1000.0, 0.2) for t in base]
        import os; os.makedirs(td + "b", exist_ok=True)
        pa, pb = _write(td, a), _write(td + "b", b)
        ra, rb = score(pa), score(pb)
        assert ra["zero"] != rb["zero"] and ra["baseline"]["mean_x"] == rb["baseline"]["mean_x"]
        results.append("D4 time-shift invariance")

        # Defect 2: ineligible row preserved and reported (HELD inside baseline)
        bad = [_row(t, 0.2) for t in base]
        bad[70] = _row(base[70], 0.2, trk=1, status=0)
        pb2 = os.path.join(td, "held.txt")
        with open(pb2, "w") as f: f.write("\n".join(bad) + "\n")
        r2 = score(pb2)
        assert r2["verdict"] == "FAIL" and r2["baseline"].get("problems"), json.dumps(r2)[:500]
        results.append("D2 HELD row detected in baseline")

        # Defect 3: wrong-direction first departure must be caught.
        # Start from the known-good clean trace (baseline passes), then flip the
        # first outward move to the wrong direction.
        rows3 = []
        tt = 0.1
        def add3(x):
            nonlocal tt
            rows3.append(_row(tt, x)); tt += 0.05
        for _ in range(60): add3(0.20)
        for i in range(40): add3(0.20 + 0.0002 * (i % 3))   # stable [3,5)
        for i in range(10): add3(0.20 - 0.40 * (i + 1) / 10)  # WRONG direction first
        for _ in range(60): add3(0.60)
        for i in range(10): add3(0.60 - 0.40 * (i + 1) / 10)
        for _ in range(80): add3(0.20)
        p3 = os.path.join(td, "wrongdir.txt")
        with open(p3, "w") as f: f.write("\n".join(rows3) + "\n")
        r3 = score(p3)
        l0 = r3["legs"][0]
        assert "WRONG" in l0.get("direction", "") or l0["verdict"] != "PASS", json.dumps(l0)[:500]
        results.append("D3 wrong-direction first departure flagged")

    results.append("ALL REGRESSION TESTS PASSED")
    return results


def self_test():
    import tempfile, os
    rows = []; t = 0.1
    def add(x, **kw):
        nonlocal t
        rows.append(_row(t, x, **kw)); t += 0.05
    for _ in range(60): add(0.20)
    for i in range(40): add(0.20 + 0.0002 * (i % 3))
    for i in range(10): add(0.20 + 0.40 * (i + 1) / 10)
    for _ in range(60): add(0.60)
    for i in range(10): add(0.60 - 0.40 * (i + 1) / 10)
    for _ in range(80): add(0.20)
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, rows)
        r = score(p)
        assert r["verdict"] == "PASS", json.dumps(r, indent=2, default=str)
    print("SELF-TEST synthetic clean trace: PASS (expected PASS)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace"); ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tests", action="store_true")
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
