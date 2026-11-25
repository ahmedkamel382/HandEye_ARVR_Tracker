# Exact-protocol re-score of unscored run `E2D_UNSCORED_20261123_06`

**Date:** 23 Nov 2026
**Tool:** `dsp_filter/evaluation/check_protocol_readiness_exact.py` (new; implements `RESPONSIVENESS_PROTOCOL.md` exactly)
**Result: FAIL — confirms Codex's finding.**

## What changed versus the earlier calculator

| Aspect | Earlier tool (wrong) | Exact tool (correct) |
|---|---|---|
| Time origin | raw trace time | `ceil(first stored timestamp × 1000)/1000` = **2.168 s** for run 06 |
| LEFT baseline | window relative to detected move | **fixed `[3,5)` protocol seconds** (trace 5.168–7.168) |
| Eligible row | n=42, ok=1, trk=0, finite x/y | + Neutral state 0, `Filtered` status 1, published point, finite raw **and** filtered X/Y |
| 10–90 duration | point estimate from sample min/max | **interval** `[raw90_low − raw10_high, raw90_high − raw10_low]`; PASS needs upper bound ≤0.8 s |
| Verdict | point thresholds | PASS/FAIL/INCONCLUSIVE conservative interval rules |

## Run 06 exact result

- Rows 687, eligible 686, protocol zero 2.168 s.
- Fixed `[3,5)` baseline: **25 eligible rows**, SD 0.004483 (≤0.010), range 0.013085 (≤0.030), gaps OK, but **OLS slope 0.0063329/s > 0.005/s** → baseline NOT stable.
- Verdict: **FAIL** — "fixed [3,5) baseline fails stable-window gates".

This reproduces Codex's reported `0.006333/s` exactly. My earlier claim that run 06 passed all gates was based on the incorrect calculator; it is withdrawn.

## Self-test

`python dsp_filter/evaluation/check_protocol_readiness_exact.py --self-test` → **SELF-TEST synthetic clean trace: PASS (expected PASS)**. The synthetic trace has a stable `[3,5)` baseline, one outward and one return leg with 10–90 intervals [0.35,0.45] and [0.30,0.40], both within 0.15–0.80 s.

## Consequence

A genuinely qualifying unscored run must be captured with:
- the operator holding **absolutely still at LEFT during protocol seconds 3–5** (the fixed baseline window, not merely "at LEFT before MOVE"),
- one outward and one return leg within 0.15–0.80 s,
- no gaps >100 ms, ≥3 intervals spanning 10–90, endpoint Y difference ≤0.05,
- and scored with this exact tool.

No threshold was relaxed. Trial 5 remains blocked on H2 executable approval and is not affected by this re-score.
