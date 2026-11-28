# Scorer v5 — narrow return-reference consistency fix (28 Nov 2026)

Responding to Engineer 4's two edge cases in `e2_return_edge_cases.json`. No scorer redesign, no protocol/DSP/threshold/verdict change.

## Fix
- Replaced the boolean `right_reference_established` with an actual `right_reference` object, assigned to the accepted RIGHT arrival window **only after** the outward leg passes direction and crossing validation (and `ref = arr`).
- Wrong-direction outward movement can no longer establish the RIGHT reference; the return leg is classified INCONCLUSIVE "return leg not independently established: no qualifying RIGHT reference".
- Outward early exits (no arrival / missing crossings) leave `right_reference = None`; return analysis cannot reuse the LEFT baseline (the 4.950 s departure artifact is eliminated).
- `scorer_version` updated to **4.0** in all report dicts.
- Missing required regression fixtures now raise a RuntimeError instead of being silently skipped.

## New regressions (failing-before / passing-after)
- `v5 edge1`: synthetic wrong-direction outward with a stable arrival → return INCONCLUSIVE, reference not established.
- `v5 edge2`: synthetic outward missing-crossings early exit → return INCONCLUSIVE, no 4.95 s LEFT-baseline departure reuse.
- Existing B test now requires the RUN_03_THIRD fixture (explicit failure if absent).

## Actual results (exit codes)
- `--tests`: P1, P2, P6, P3, P4, A, B, v5 edge1, v5 edge2, C → `ALL V5 REGRESSION TESTS PASSED`, exit **0**.
- `--self-test`: `SELF-TEST run 10: PASS`, exit **0**.
- Historical verdicts unchanged: run 06–09 FAIL, run 10 PASS (baseline slope 0.00122968, brackets [495.0986, 656.1693]/[640.5350, 799.9113] ms, 0.0887 ms margin), RUN_02_SECOND INCONCLUSIVE, RUN_03_THIRD INCONCLUSIVE with explicit return reason.

## H2
Candidate `first-full-working-demo @ f29b3e5`, Debug, SHA `1AA31B9D…` accepted for local runtime qualification; scored authorization still pending Engineer 4's H1 qualification. No new executable, launcher change, scored trial, or real mouse.
