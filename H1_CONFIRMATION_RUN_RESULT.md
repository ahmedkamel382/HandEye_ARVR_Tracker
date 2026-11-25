# H1 confirmation run (step 3) — `E2D_UNSCORED_20261125_01`

**Date:** 25 Nov 2026. UNSCORED. No scored trial; Trial 4 unchanged.

## Result: FAIL — baseline drift

| Item | Value |
|---|---|
| Binary | dev Release `core_engine.exe`, SHA `12B9CE6B…29A` (same identity as run 10, includes E2-B safeguards) |
| Command | `core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <run>\recording --preview --target-zone left` |
| Trace | 686 frames, `COMPLETE`, `overflow=0`, 12 cues issued, exit 0, 42 screenshots |
| Scorer | `check_protocol_readiness_exact.py` v2.0 (four audit defects fixed) |
| Protocol zero | 13.207 s |
| Baseline `[3,5)` | 25 eligible rows, SD 0.007142 (≤0.010 ✅), range 0.02126 (≤0.030 ✅), **slope 0.010565/s (>0.005 ❌)** |
| Verdict | **FAIL** — fixed baseline unstable |

## Required documentation (all recorded)
- **Camera:** USB Camera, VID_0BDA / PID_5803 (Realtek), exact OEM model unknown; MJPG requested, 640×480.
- **FPS:** 60 requested; ~12.4 delivered (see console).
- **Lighting:** steady diffuse front lighting, typical indoor; auto-exposure.
- **Distance/setup:** ~60 cm target, LEFT—10 cm—CENTER—10 cm—RIGHT, equal height/depth (operator statements in `manifest.json`).
- **Source/executable identity:** branch `engineer2/dsp-report-and-preview-overlays`, source commit `cb93e12` for run 10 binary, exe SHA `12B9CE6B…`.
- **Cue schedule:** 14 semantic cues; `cues.json` records **12 playback requests + 2 embedded STOP words** (`kind` counts), matching the previously documented 12/14 accounting.
- **Timing distinction:** cue timestamps are software issue times, not camera timestamps or measured audible onset (`clock_limit` in manifest).
- **Raw trace:** `recording/trace.txt`; `COMPLETE` marker present; full numerical report `protocol_exact_report.json`.
- **Screenshots:** 42 PNG frames in `visual/`.

## Interpretation
This confirmation attempt did not reproduce run 10's stability; the hand drifted during the fixed 2-second LEFT baseline. Run 10's independently reproduced PASS is **not** withdrawn — this additional attempt is preserved as a separate, failed attempt, exactly as Engineer 4 requires ("preserve every attempt"). It does show that H1 qualification is sensitive to the operator's stillness.

## Status
- Run 10 remains the accepted unscored H1 PASS.
- H2 executable identity remains BLOCKED; no scored Trial 5, no launcher change.
- No threshold was relaxed and no verdict was reclassified.
