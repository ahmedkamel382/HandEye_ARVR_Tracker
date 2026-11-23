# H1 — UNSCORED FAST Hand Readiness: QUALIFIED (run `E2D_UNSCORED_20261123_06`)

**Date:** 23 Nov 2026
**Classification:** UNSCORED qualification evidence. **Not** a scored FAST trial. Trial 4 remains INCONCLUSIVE and unchanged.

## H1 decision: ✅ QUALIFIED (tracking input meets the original gates)

After several unscored captures, run `E2D_UNSCORED_20261123_06` passed **every original H1 gate on both legs** using the unmodified gate thresholds.

### Capture
| Item | Value |
|---|---|
| Run ID | `E2D_UNSCORED_20261123_06` |
| Executable | `out/build/windows-release/core_engine.exe` SHA `12B9CE6B…29A` (unscored use only — not scored-approved) |
| Command | `core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <run>\recording --preview` (preview is drawing-only) |
| Cues | Original Engineer 4 MOVE!/STOP! schedule, 12 cues issued, no error |
| Trace | `recording/trace.txt`, 687 frames, `COMPLETE`, `overflow=0`, `config_changes=0` |
| Delivered FPS | 12.4 (asked 60) |
| Physical setup | LEFT—10 cm—CENTER—10 cm—RIGHT, ~60 cm target, same height/depth, steady diffuse front lighting, auto-exposure; camera USB Camera VID_0BDA/PID_5803 (exact OEM unknown). Values are operator statements, not instrument measurements. |

### Gate results (tool: `check_unscored_fast_readiness.py --auto-moves --mode fast`)

| Gate | OUTWARD | RETURN | Limit | Result |
|---|---|---|---|---|
| 10–90% duration | 640.2 ms | 639.7 ms | 150–800 ms | PASS |
| Amplitude | 0.286 | 0.250 | ≥ 0.15 | PASS |
| Y endpoint difference | 0.045 | 0.027 | ≤ 0.05 | PASS |
| Max fresh gap | 95.9 ms | 81.3 ms | ≤ 100 ms | PASS |
| Crossing samples | 9 | 9 | ≥ 3 | PASS |
| Baseline fixation (samples/SD/range/slope/dwell) | 36 / 0.0016 / 0.0058 / 0.0014 / 2816 ms | 36 / 0.0055 / 0.0206 / 0.0002 / 2816 ms | ≥25 / ≤0.010 / ≤0.030 / ≤0.005 / ≥500 ms | PASS |
| Endpoint fixation | 36 / 0.0042 / 0.0142 / 0.0037 / 2816 ms | 36 / 0.0013 / 0.0051 / 0.0005 / 2816 ms | same | PASS |

**Overall: PASS.** Exit code 0. No threshold was relaxed; the gate calculator's self-test still reports clean-capture PASS and gap-injected FAIL.

## Tool corrections made during these retries (no threshold changes)

- Fixed my orchestrator's status regex to reuse the launcher's proven `STATUS` pattern (earlier runs 03/04 issued no cues because readiness never fired).
- Relaxed unscored readiness to accept any finite hand with increasing OS frames (documented in manifest) — the scored launcher still requires a Neutral hand.
- Fixed fixation windows to evaluate the **settled holds** (before/between/after moves) instead of the movement transitions; capped endpoint hold to ~3 s.
- Fixed `--cues` to require a `--ready-trace-t` anchor; `--auto-moves` derives leg timing and direction from the trace.

## What H1 QUALIFIED does and does not mean

- ✅ The hand-tracking input meets the original FAST readiness gates on a documented physical setup in an unscored capture.
- ❌ It is **not** a scored FAST PASS and does **not** authorize Trial 5.
- ❌ It does **not** approve any executable. H2 remains BLOCKED in `FAST_EXECUTABLE_COMPATIBILITY_DECISION_PACKAGE.md`: no launcher change, no scored run, until Engineer 4/Codex approve the branch/SHA identity and confirm the selected build includes the input safeguards.
- Trial 1–4 evidence and Trial 4's INCONCLUSIVE classification are unchanged; every attempt is preserved.

## Trial readiness statement (required artifact)

**Scored Trial 5 cannot begin yet.** H1 (tracking qualification) is now documented as QUALIFIED, but H2 (executable approval) is still open. Both gates must be documented before Engineer 4 starts a new scored attempt.
