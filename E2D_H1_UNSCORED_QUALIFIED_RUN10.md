# H1 — UNSCORED FAST hand qualification: QUALIFIED (run `E2D_UNSCORED_20261123_10`)

**Date:** 23 Nov 2026. **Classification:** UNSCORED qualification evidence only. Not a scored FAST trial; Trial 4 remains INCONCLUSIVE and unchanged.

## Result: ✅ QUALIFIED under the original protocol

Run `E2D_UNSCORED_20261123_10` passed every original gate on both legs with the exact-protocol scorer (`dsp_filter/evaluation/check_protocol_readiness_exact.py`).

- Binary: dev Release `core_engine.exe`, SHA-256 `12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A` (includes E2-B input safeguards).
- Command: `core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <run>\recording --preview --target-zone left`
- Trace: 685 frames, `COMPLETE`, `overflow=0`, all 12 cues issued, exit 0, 42 screenshots.
- Physical setup metadata recorded from operator statement.
- Protocol zero = 2.788 s (ceil rule).

### Fixed baseline `[3,5)` — PASS
25 eligible rows, SD 0.00372 (≤0.010), range 0.01433 (≤0.030), gap OK, **OLS slope 0.001230/s (≤0.005)**.

### Legs

| Gate | OUTWARD (LEFT→RIGHT) | RETURN (RIGHT→LEFT) | Protocol limit |
|---|---|---|---|
| Amplitude | 0.232 ✅ | 0.213 ✅ | ≥ 0.15 |
| Endpoint Y difference | 0.037 ✅ | 0.0015 ✅ | ≤ 0.05 |
| 10–90 duration interval | **[0.495, 0.656] s ✅** | **[0.641, 0.800] s ✅** | wholly within 0.15–0.8 s |
| Crossing intervals | 8 ✅ | 10 ✅ | ≥ 3 |
| Gaps | OK ✅ | OK ✅ | ≤ 100 ms |
| Arrival window | 25 / SD 0.00556 / slope 0.00260 ✅ | 25 / SD 0.00339 / slope 0.00303 ✅ | ≥25 / SD≤0.010 / slope≤0.005 |
| Verdict | **PASS** | **PASS** | — |

**Overall: PASS.** No threshold was relaxed; no attempt reclassified.

## Cumulative unscored attempts
| Run | Result |
|---|---|
| 06 | FAIL — baseline slope 0.006333 (Codex-confirmed) |
| 07 | FAIL — baseline OK, legs too slow |
| 08 | FAIL — baseline slope 0.006762 |
| 09 | FAIL — baseline slope 0.019543 |
| 10 | **PASS — all original gates on both legs** |

## Trial readiness statement
- H1 (tracking input qualification): **QUALIFIED** on documented physical setup with the original protocol and unchanged criteria.
- H2 (scored executable identity): **STILL BLOCKED** — no executable approved, launcher unchanged. `FAST_EXECUTABLE_COMPATIBILITY_DECISION_PACKAGE.md` holds the decision request.
- **Scored Trial 5 CANNOT begin yet** — both gates must be documented. Only H1 is now closed; H2 awaits Engineer 4/Codex approval of branch/SHA/launcher identity and confirmation the chosen build includes the input safeguards (run 10 used the build that includes them).
- All attempts preserved; Trial 4 INCONCLUSIVE and unchanged.
