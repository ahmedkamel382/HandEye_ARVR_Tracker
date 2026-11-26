# Scorer v3 — Engineer 4 audit corrections (Phase 1)

**Date:** 26 Nov 2026. **File:** `dsp_filter/evaluation/check_protocol_readiness_exact.py` (v3.0).

## Corrections implemented

| Engineer 4 point | Fix in v3 |
|---|---|
| 1 Duration verdict propagation | `worst()` worst-of propagation: INCONCLUSIVE duration → INCONCLUSIVE leg/run (RUN_02_SECOND return [0.6562,0.8169]s now INCONCLUSIVE, not FAIL). End-to-end test P1. |
| 2 Malformed trace rows silently discarded | `recorder_integrity()` uses authoritative `import_vs_trace.convert_recorder` as mandatory prerequisite; malformed `n=garbage`, header/footer/overflow/complete/channel corruption all rejected with explicit reason before scoring. Tests P2, P6. |
| 3 Chronological integrity + [3,5) boundary | `prepare()` checks nonmonotonic/eligibility across all rows; `stable()` and `interruption()` both use `[a,b)` — HELD exactly at 5.000 s no longer enters baseline. Test P3. |
| 4 First-departure detection | `first_departure()` scans full chronological sequence (no eligible-only prefilter), requires 3 contiguous eligible rows, both 2nd and 3rd beyond threshold, no gap/ineligible in between. Test P4. |
| 5 Return-leg reference | Return leg is only attempted when outward has an accepted RIGHT arrival; otherwise return is INCONCLUSIVE with reason — no LEFT-reference cascade / same-timestamp artifact. |
| 6 Recorder-integrity validation | Mandatory prerequisite above; NaN first timestamp returns FAIL with reason, no unhandled ValueError. |
| 7 Regression tests | P1–P4, P6, P7 end-to-end; count text corrected; full-outcome timestamp-shift test included (v2 D4 retained in history). |

## Verdict reconciliation (v3 vs v2)

| Run | v2 | v3 |
|---|---|---|
| 20261123_06 | FAIL | FAIL |
| 07 | FAIL | FAIL |
| 08 | FAIL | FAIL |
| 09 | FAIL | FAIL |
| 10 | PASS | **PASS** (preserved) |
| 20261125_02 | INCONCLUSIVE | INCONCLUSIVE |
| 06 | FAIL | **INCONCLUSIVE** (corrected propagation) |
| 03 | INCONCLUSIVE | INCONCLUSIVE |

All original reports preserved; noDSP parameter, threshold, protocol rule or historical verdict changed. Run 10 remains the legitimate unscored H1 PASS. Trial 4 unchanged; no scored Trial 5; launcher untouched.
