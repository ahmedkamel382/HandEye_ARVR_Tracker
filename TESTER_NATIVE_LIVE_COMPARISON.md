# Tester vs Native Hand Validation — Live Comparison Result

**DSP message item 6 (Engineers 2 + 5).** Date 23 Nov 2026.
**Classification: ✅ FIXED AND VERIFIED** for the tracking-comparison portion; real-mouse safety (item 9) remains Engineer 5's scope and was not performed.

## 1. Root cause

The HandEye Tester used `HandTracker.process_frame()` while native Core used `HandNative::process()`; malformed/non-finite payloads were accepted by the Tester path and rejected by native. Fixed in `hand_tracker.py` (`_parse()` rejects wrong counts and non-finite floats before a cursor can be published). This live test verifies the two paths now agree on real camera input.

## 2. Code changes

- `ai_pipeline/hand_kinematics/hand_tracker.py` — `_parse()` validation (commit `71b3bba`).
- `core_engine/os_integration/tester/tests/test_hand_validation_equivalence.py` — 7 stubbed equivalence tests (commit `03c0759`).
- `dsp_filter/evaluation/capture_tester_hand_live.py` — NEW log-only Tester-path camera harness (`real_mouse=False`).
- `dsp_filter/evaluation/compare_tester_native.py` — NEW offline comparison tool.

## 3. Commit

Branch `engineer2/dsp-report-and-preview-overlays`; HEAD `03c0759` plus the new uncommitted harness/compare tools (to be committed in the next commit). Local and origin refs matched at `03c0759` before these additions.

## 4. Reproduction

Before the fix, a stubbed 20+22 payload and an Inf fingertip produced `DETECTED`/`TRACKED` with X=1 in the Tester path; after the fix, stubbed tests produce LOST with no point. The live comparison below confirms both paths agree on a real static hold.

## 5. Regression tests

| Test | Result |
|---|---|
| `test_hand_input_validation.py` (hand_tracker) | 20 passed |
| `test_hand_validation_equivalence.py` (Tester path) | 7 passed |
| Live Tester vs native static-hold comparison | CONSISTENT |

## 6. Test results — live comparison

Session: `C:\Users\Doctor Laptop\Downloads\E2D_TESTER_NATIVE_COMPARE_20261123\` — static hand hold, camera 0, sequential captures, **no real mouse movement**.

| Path | Frames | States | Mean x (SD) | Mean y (SD) |
|---|---:|---|---|---|
| Tester (`HandSource → HandTracker`) | 92 | TRACKED 92 (100%) | 0.5313 (0.0015) | 0.5054 (0.0011) |
| Native (`core_engine --dsp-record-dir`) | 101 | TRACKED 101 (100%) | 0.5194 (0.0070) | 0.5284 (0.0027) |

`|mean x difference| = 0.0119`, `|mean y difference| = 0.0230`. Both paths report all fresh frames as TRACKED with no HELD/LOST during the static hold. Native console: `captured 101 | processed 101 | evicted 0 | skipped 0 | stale 0 | python errors 0`; `DSP recording: COMPLETE | stored 101 | overflow 0`.

## 7. DSP compatibility

No DSP parameter (hand 1.2/4/1, gaze 0.8/6/1.5), adapter, recorder format, timestamp, frame-ID or tracking-state contract was changed. Both captures used `--os log` / `real_mouse=False`; no mouse injection occurred.

## 8. Physical verification

Done for the tracking comparison: real camera, real hand, static hold, both code paths. **Not done**: real Windows button-release / refused-movement / stall-camera safety checks — those are Engineer 5's item 9 and require the GUI/Tester with real mouse.

## 9. Remaining limitations

- Static hold only; a live FAST sweep comparison was not repeated.
- Tester and native ran sequentially (camera cannot be shared), so small coordinate differences reflect session-to-session hand placement, not a contract mismatch.
- The native path applies its 1-Euro filter to published coordinates; comparison used the trace's raw `ix/iy` tracker input to keep the two paths comparable.

## 10. Integration status

Ready for independent review. The hand-validation fix is self-contained in `hand_tracker.py` plus tests; it does not conflict with Engineer 1's bridge validation or Engineer 5's OS integration. Not proposed for direct merge into `first-full-working-demo` without the team's review.
