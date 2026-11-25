# Scorer defect resolution — Engineer 4 audit response

**Date:** 25 Nov 2026. **Branch:** `engineer2/dsp-report-and-preview-overlays`.
**File:** `dsp_filter/evaluation/check_protocol_readiness_exact.py` (now v2.0).

## Four defects corrected (actual code changes)

| # | Defect | Correction |
|---|---|---|
| 1 | No minimum FAST duration (only upper bound checked) | `duration_verdict(lo,hi)` now enforces the **entire bracket within [150,800] ms inclusive**: PASS iff `lo>=0.150 and hi<=0.800`; FAIL if `lo>0.800` or `hi<0.150`; else INCONCLUSIVE. |
| 2 | Ineligible rows removed before movement detection | `prepare()` **never drops rows**; each row is marked `eligible`. New `interruption(rows,a,b)` inspects the FULL chronological slice for Held/Lost/invalid/nonmonotonic/gap>100ms and reports them; affected windows/legs fail. |
| 3 | Searched only expected direction | `first_departure()` finds the **first sustained departure chronologically regardless of sign**, then checks its direction vs expected; wrong-direction first departure is recorded and fails the leg. No later move is chosen to force a PASS. |
| 4 | Mixed timestamp origins | One documented origin: `tp = t - zero` used for baselines, boundaries, gaps, crossings, endpoints. Verified by a shifted-timestamp test: identical verdicts after constant offset. |

## Regression tests (`--tests`)

```
D1 duration bounds (6 cases)      — <150 FAIL, =150 PASS, =800 PASS, >800 FAIL, both straddles INCONCLUSIVE
D4 time-shift invariance           — +1000s shift gives identical baseline/verdict
D2 HELD row detected in baseline   — HELD inside [3,5) now FAILS the baseline
D3 wrong-direction first departure — flagged, leg cannot PASS
ALL REGRESSION TESTS PASSED
```
`--self-test` (synthetic protocol-clean trace) → PASS.

## Historical H1 verdicts reconciled

| Run | v2 verdict | Baseline slope | Notes |
|---|---|---|---|
| 06 | FAIL | 0.00633286 | matches Codex's 0.006333 |
| 07 | FAIL | 0.00249591 | baseline OK; legs too slow (preserved) |
| 08 | FAIL | 0.00676197 | unstable baseline |
| 09 | FAIL | 0.01954266 | unstable baseline |
| 10 | **PASS** | 0.00122968 | preserved legitimate PASS |

Run 10 v2 numbers (exact match to Engineer 4's reproduction):
- outward: amplitude 0.232273, Y diff 0.037391, 10–90 bracket **[0.4950986, 0.6561693]**, 8 crossings, direction correct;
- return: amplitude 0.213059, Y diff 0.001546, 10–90 bracket **[0.6405350, 0.7999113]**, 10 crossings, direction correct.

The return upper bound (799.9113 ms) passes by only 0.0887 ms — unchanged and legitimate. No threshold was relaxed and no verdict reclassified.

## Not changed

DSP equations/parameters, protocol thresholds, launcher, scored trial state, historical recordings, Trial 4 verdict.
