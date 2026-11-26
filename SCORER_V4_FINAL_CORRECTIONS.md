# Scorer v4 — final three targeted corrections (Engineer 4, 26 Nov 2026)

## Corrections
- **A. Adjacent timestamp gaps:** `first_departure()` now checks each adjacent interval against the existing 100 ms limit instead of only the 3-observation total span. A 190 ms adjacent gap is rejected. Regression test `A` demonstrates the old total-span check accepted it (240 ms <= 300 ms) and the corrected check rejects it.
- **B. Return-leg sequencing:** added a `right_reference_established` guard. If the outward leg has no qualifying RIGHT arrival, the return leg is classified INCONCLUSIVE with reason "return leg not independently established: no qualifying RIGHT reference" and is NOT scored using the original LEFT reference. Verified on RUN_03_THIRD.
- **C. Portability and genuine tests:** removed all 4 machine-specific absolute paths (importer path derived from `__file__`, fixture path auto-discovered / `RUN10_TRACE` env override). The timestamp-shift regression now performs a genuine +1000 s shift of `t`, `input_t`, `ms` and `cap`, verifies recorder integrity via the real `import_vs_trace.convert_recorder`, and compares the COMPLETE outcome (verdict, baseline, legs, gates, relative timings) with float tolerance 1e-9. All tests run using the genuine importer.

## No changes to verified evidence
- Run 10 re-scores to the same PASS with original brackets [495.0986, 656.1693] ms and [640.5350, 799.9113] ms (0.0887 ms margin).
- RUN_02_SECOND retains its corrected INCONCLUSIVE duration classification.
- RUN_03_THIRD return is now explicitly "not independently established".
- Runs 06–09 remain FAIL; all historical failed/inconclusive recordings preserved.

## Test results (actual, exit codes)
- `python dsp_filter/evaluation/check_protocol_readiness_exact.py --self-test` → `SELF-TEST run 10: PASS`, exit 0.
- `python dsp_filter/evaluation/check_protocol_readiness_exact.py --tests` → P1, P2, P6, P3, P4, A, B, C, `ALL V4 REGRESSION TESTS PASSED`, exit 0.
- Run-10 trace scoring → verdict PASS, exit 0 (both legs PASS).
- Genuine importer verification: every test's `recorder_integrity()` calls the authoritative `import_vs_trace.convert_recorder`; the +1000 s shifted trace passes integrity and yields an equivalent complete outcome.

## H2 candidate (unchanged, per Engineer 4 selection)
- Branch `first-full-working-demo`, source `f29b3e53f2b655b2ec80e6f407636beafca1e3aa`, Debug, exe SHA `1AA31B9D7E0F9209A8F84F3CAD89C6084DC7F9FDADF8F30DA26EFE20CD173B50`.
- Complete CTest transcript: `...\h2_candidate\out\build\windows-release\Testing\Temporary\LastTest.log` (22,702 B, all tests Passed; 8/8 CTest, 163.90 s total). Runtime DLLs and model SHA documented in `H2_CLEAN_BUILD_CANDIDATE.md` and the nested candidate archive.
- No new executable created; no launcher change; no scored trial; no real mouse.

No DSP parameter, equation, threshold, approved window, protocol rule, historical verdict, or Trial 4 result changed.
