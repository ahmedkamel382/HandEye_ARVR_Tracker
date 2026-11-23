# -*- coding: utf-8 -*-
"""EXACT original-protocol FAST readiness scorer (RESPONSIVENESS_PROTOCOL.md).

Use this instead of the earlier check_unscored_fast_readiness.py. It implements:
- fixed protocol zero = ceil(first stored timestamp * 1000)/1000
- eligible rows: n=42, ok=1, trk=Tracked, state=Neutral(0), status=Filtered(1), published, finite raw/filtered X/Y
- initial LEFT reference = FIXED [3,5) protocol seconds (never substituted)
- stable window: >=25 eligible, no gap>100ms, SD<=0.01, range<=0.03, |OLS slope|<=0.005/s
- first departure: 3 consecutive eligible rows beyond max(0.03, 6*ref_SD)
- first arrival: 2s stable window with mean shift >=0.15 within 15s
- amplitude >=0.15, endpoint Y diff <=0.05, >=3 intervals spanning raw 10-90
- raw 10-90 DURATION INTERVAL: [raw90_low-raw10_high, raw90_high-raw10_low]
- fast PASS requires interval upper bound <=0.8s; straddle=INCONCLUSIVE; lower>limit=FAIL
No threshold was relaxed.

Usage: python check_protocol_readiness_exact.py --trace TRACE [--out report.json]
       python check_protocol_readiness_exact.py --self-test
"""
import argparse, json, math, re, statistics, sys

F = re.compile(r"^E4REC F\s+(.*)$"); KV = re.compile(r"(\w+)=([^\s]+)")
G = dict(samples_min=25, gap_max=0.100, sd_max=0.010, range_max=0.030,
         slope_max=0.005, delta_min=0.03, amplitude_min=0.15, y_max=0.05,
         fast_hi=0.80)


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = F.match(line.strip())
            if not m: continue
            d = dict(KV.findall(m.group(1)))
            try:
                rows.append(dict(frame=int(d.get("frame", -1)), t=float(d.get("input_t", d.get("t", "0"))),
                    n=int(d.get("n", 0)), ok=int(d.get("ok", 0)), x=float(d.get("ix", "nan")),
                    y=float(d.get("iy", "nan")), fx=float(d.get("dx", "nan")), fy=float(d.get("dy", "nan")),
                    trk=int(d.get("trk", -1)), state=int(d.get("state", -1)),
                    status=int(d.get("status", -1)), pub=int(d.get("pub", 0)),
                    resets=int(d.get("resets", 0)), outcome=int(d.get("outcome", -1))))
            except (ValueError, TypeError): continue
    return rows


def eligible(r):
    return (r["n"] == 42 and r["ok"] == 1 and r["trk"] == 0 and r["state"] == 0
            and r["status"] == 1 and r["pub"] == 1
            and all(math.isfinite(r[k]) for k in ("x", "y", "fx", "fy")))


def slope_ols(pts):
    if len(pts) < 2: return 0.0
    t0 = pts[0]["t"]; xs = [p["t"] - t0 for p in pts]; ys = [p["x"] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((v - mx) ** 2 for v in xs)
    return 0.0 if den == 0 else sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den


def stable(pts):
    if len(pts) < G["samples_min"]:
        return False, dict(n=len(pts), reason="fewer than %d eligible" % G["samples_min"])
    xs = [p["x"] for p in pts]
    s = dict(n=len(pts), sd=statistics.pstdev(xs), rng=max(xs) - min(xs),
             slope=abs(slope_ols(pts)), mean_x=statistics.mean(xs),
             mean_y=statistics.mean([p["y"] for p in pts]),
             t0=pts[0]["t"], t1=pts[-1]["t"])
    gaps_ok = all((b["t"] - a["t"]) <= G["gap_max"] for a, b in zip(pts, pts[1:]))
    s["gap_ok"] = gaps_ok
    s["qualified"] = (gaps_ok and s["sd"] <= G["sd_max"] and s["rng"] <= G["range_max"]
                      and s["slope"] <= G["slope_max"])
    return s["qualified"], s


def crossing(pts, level, a, b):
    for i in range(1, len(pts)):
        v0 = (pts[i-1]["x"] - a) / (b - a); v1 = (pts[i]["x"] - a) / (b - a)
        if v0 < level <= v1:
            return (pts[i-1]["t"], pts[i]["t"])
    return None


def interval_verdict(lo, hi, limit):
    if hi <= limit: return "PASS"
    if lo > limit: return "FAIL"
    return "INCONCLUSIVE"


def score(path):
    rows = parse(path)
    if not rows: return dict(verdict="NOT RUN", reason="no E4REC F rows")
    zero = math.ceil(min(r["t"] for r in rows) * 1000) / 1000.0
    for r in rows: r["tp"] = r["t"] - zero
    E = [r for r in rows if eligible(r)]
    base = [r for r in E if 3.0 <= r["tp"] < 5.0]      # fixed baseline, never substituted
    ok, bs = stable(base)
    out = dict(protocol="RESPONSIVENESS_PROTOCOL.md", zero=zero,
               rows=len(rows), eligible=len(E), baseline=bs)
    if not ok:
        out.update(verdict="FAIL", reason="fixed [3,5) baseline fails stable-window gates")
        return out
    legs = []; ref = bs
    for leg, exp in (("LEFT_TO_RIGHT", 1), ("RIGHT_TO_LEFT", -1)):
        thr = max(G["delta_min"], 6 * ref["sd"])
        start = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        after = [r for r in E if r["tp"] >= start]
        dep = next((after[i] for i in range(len(after) - 2)
                    if all(exp * (after[j]["x"] - ref["mean_x"]) > thr for j in (i, i+1, i+2))), None)
        if dep is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no first departure")); continue
        arr = None
        for cand in [r for r in E if r["tp"] > dep["tp"]]:
            w = [r for r in E if cand["tp"] <= r["tp"] < cand["tp"] + 2.0]
            okw, ws = stable(w)
            if okw and exp * (ws["mean_x"] - ref["mean_x"]) >= G["amplitude_min"] \
                    and ws["t1"] <= dep["tp"] + 15.0:
                arr = ws; break
        if arr is None:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="no qualifying arrival window")); continue
        a, b = ref["mean_x"], arr["mean_x"]; amp = abs(b - a)
        ydiff = abs(ref["mean_y"] - arr["mean_y"])
        # Crossings are searched in the candidate search from the end of the current
        # reference window (not from the departure row), so the 10% crossing before
        # departure is included.
        search_from = 5.0 if leg == "LEFT_TO_RIGHT" else ref["t1"]
        mov = [r for r in E if search_from <= r["tp"] <= arr["t1"] + 2.0]
        gaps_ok = all((q["t"] - p["t"]) <= G["gap_max"] for p, q in zip(mov, mov[1:]))
        span = sum(1 for p, q in zip(mov, mov[1:])
                   if 0.1 <= (q["x"] - a) / (b - a) <= 0.9 or 0.1 <= (p["x"] - a) / (b - a) <= 0.9)
        c10 = crossing(mov, 0.1, a, b); c90 = crossing(mov, 0.9, a, b)
        if not c10 or not c90:
            legs.append(dict(leg=leg, verdict="INCONCLUSIVE", reason="missing 10/90 crossings")); continue
        lo, hi = c90[0] - c10[1], c90[1] - c10[0]
        dur = interval_verdict(lo, hi, G["fast_hi"])
        ok_all = (amp >= G["amplitude_min"] and ydiff <= G["y_max"] and span >= 3 and gaps_ok)
        legs.append(dict(leg=leg, verdict="PASS" if (dur == "PASS" and ok_all) else "FAIL",
                         amplitude=amp, y_diff=ydiff, crossing_interval=[lo, hi], duration=dur,
                         crossing_intervals=span, gaps_ok=gaps_ok,
                         departure_t=dep["tp"], arrival=arr))
        ref = arr
    out["legs"] = legs
    out["verdict"] = "PASS" if all(l["verdict"] == "PASS" for l in legs) else (
        "INCONCLUSIVE" if all(l["verdict"] != "FAIL" for l in legs) else "FAIL")
    return out


def self_test():
    rows = []; t = 0.1
    def add(x):
        nonlocal t
        rows.append("E4REC F frame=%d input_t=%.6f n=42 ok=1 ix=%.6f iy=%.6f dx=%.6f dy=%.6f "
                    "trk=0 state=0 status=1 pub=1 resets=0 outcome=0" % (len(rows), t, x, 0.4, x, 0.4))
        t += 0.05
    for _ in range(60): add(0.20)                       # pre-baseline filler
    for i in range(40): add(0.20 + 0.0002 * (i % 3))    # baseline [3,5) stays ~0.20 and stable
    for i in range(10): add(0.20 + 0.40 * (i+1) / 10)
    for _ in range(60): add(0.60)
    for i in range(10): add(0.60 - 0.40 * (i+1) / 10)
    for _ in range(80): add(0.20)
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "synthetic.txt")
        with open(p, "w") as fh: fh.write("\n".join(rows) + "\n")
        r = score(p)
        assert r["verdict"] == "PASS", json.dumps(r, indent=2, default=str)
    print("SELF-TEST synthetic clean trace: PASS (expected PASS)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace"); ap.add_argument("--out"); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    if not a.trace: ap.error("--trace required")
    r = score(a.trace); text = json.dumps(r, indent=2, default=str)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh: fh.write(text + "\n")
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
