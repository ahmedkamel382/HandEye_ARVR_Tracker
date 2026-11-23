# Engineer 2 — Hand Input Validation Fix and Branch Review Evidence

**Date:** 23 Nov 2026
**Branch:** `engineer2/dsp-report-and-preview-overlays`
**Commits:** `654d9d9` (overlays/config/report), `55faef2` (native E2-B validator), `71b3bba` (Tester-path input validation fix — this package)
**DSP baseline reviewed:** `origin/first-full-working-demo @ c61655f`; Engineer 1 fix `69c84c8`; E1 bridge review `240f806`
**Scope:** DSP message items 6, 7, 8 and the hand-validation portion of item 3. No DSP equation, parameter, threshold, launcher, protocol, or scored-trial change was made.

---

## Item 6 — Tester vs native hand-validation mismatch

**Status classification: 🟡 IMPLEMENTED BUT NOT FULLY VERIFIED** (automated verification complete; live Tester-vs-native and physical checks remain).

### 1. Root cause

The HandEye Tester does not use the bridge/native validator. `core_engine/os_integration/tester/tester_engine.py:281` (`HandSource.process`) calls `HandTracker.process_frame()` from `ai_pipeline/hand_kinematics/hand_tracker.py`. In that file:

- `result_to_points()` (`hand_features.py`) only checks that `hand_world_landmarks` is non-empty — no count or finiteness validation.
- `HandTracker._parse()` only checked `points is None or not result.hand_landmarks`, so a 20+22 split was accepted.
- `process_frame()` computed `cursor = np.clip(lm_xy[8], 0.0, 1.0)`, so an infinite fingertip coordinate became a valid-looking `X = 1`.

The native Core Engine rejects the same payloads, so the Tester could report tracking that the native engine would never publish.

### 2. Code changes

- `ai_pipeline/hand_kinematics/hand_tracker.py` — `_parse()` now requires exactly 21 image and 21 world landmarks, both arrays of the expected shape, and all coordinates finite (`np.isfinite`). Invalid payloads return `None`, so they take the existing no-hand HELD/LOST path and never become a fresh cursor coordinate. Both the primary VIDEO detector and ROI recovery use this single boundary.
- `ai_pipeline/hand_kinematics/tests/test_hand_input_validation.py` — new regression suite (20 tests).

No DSP, filter, freshness, recorder, launcher, or tracking-state code was changed.

### 3. Commit

`71b3bba` on `engineer2/dsp-report-and-preview-overlays`, pushed to `origin`. Local and remote refs both equal `71b3bba`.

### 4. Reproduction

Before the fix, a stubbed 20+22 result and an image fingertip of `+Inf` both produced a `DETECTED` frame with a cursor coordinate of `1.0` through `HandTracker.process_frame()`. After the fix, both produce `LOST` with `(-1, -1, 0)`; the original failing scenario is no longer reproducible.

### 5. Regression tests

`ai_pipeline/hand_kinematics/tests/test_hand_input_validation.py` covers: valid 21+21; malformed counts (20+22, 22+20, 21+20, 20+21, 21+0, 0+21, 0+0); NaN/+Inf/−Inf in image and world coordinates; invalid→valid recovery; and `process_frame()` cases asserting invalid payloads never produce a fresh coordinate.

### 6. Test results

| Test command | Result |
|---|---|
| `python -m pytest ai_pipeline/hand_kinematics/tests/test_hand_input_validation.py -q` | **20 passed** |
| `python -m pytest ai_pipeline/hand_kinematics/tests -q` | **24 passed** |
| `python -B ai_pipeline/bridge/tests/check_hand_landmark_diag.py` | ALL CHECKS PASSED, exit 0 |

Not run in this pass: native CTest after a rebuild (no build was performed), and live Tester-vs-Core comparison (requires Windows GUI and camera).

### 7. DSP compatibility

Hand filter parameters (1.2 / 4.0 / 1.0), gaze filter parameters (0.8 / 6.0 / 1.5), DSP adapter, recorder format, timestamps, frame IDs, Fresh/Held/Lost contract and acceptance thresholds were not touched. Invalid data is now rejected *before* any coordinate or DSP update; the change can only reduce invalid input reaching DSP.

### 8. Physical verification

Still required: live Tester vs native Core Engine comparison, real Windows mouse/button behavior, and a real-camera hand run confirming that valid hands still track normally. These are explicitly owned by Engineer 5 (OS integration) and by the H1 physical qualification; they were not performed here.

### 9. Remaining limitations

- No live camera or real-Windows verification.
- The fix was not rebuilt into `core_engine.exe`; native CTest results are from the previously built binaries.
- The saved-calibration and click-safety items in DSP message §9 are Engineer 5's scope.

### 10. Integration status

Ready for independent review on the development branch. **Not proposed for merge into `first-full-working-demo` yet** — merge should follow the DSP Engineer's Step 2/3 review and the outstanding physical checks.

---

## Item 7 — Diagnostic suite "31/33"

**Status: ✅ FIXED AND VERIFIED on this branch; 🟡 demo-side test file still needs a 2-line update.**

### Reproduction

The demo's test file was run against the demo's production `landmark_extractors.py` in an isolated temp directory: **31 PASS / 2 FAIL**, exit 1 — reproducing the reported result exactly. The failures were:

```
FAIL malformed result is returned unchanged (41 rows)  [(0, 3)]
FAIL label: malformed  [MALFORMED]
```

Both are outdated assertions that expect malformed payloads to pass through as 41 rows, which contradicts the approved production rejection (`pack_hand()` returns `_EMPTY`).

### Production fix on this branch

`ai_pipeline/bridge/landmark_extractors.py::_describe_hand()` now:

- reports `n = 42` only when the payload is actually published, and `n = 0` for every rejected reason (`BOTH_EMPTY`, `IMAGE_ONLY`, `WORLD_ONLY`, `MALFORMED`, `NONFINITE`);
- classifies `NONFINITE` using the **float32 representation actually stored by the native bridge**, so a float64-finite value (e.g. `1e100`) that overflows to Inf in float32 is no longer reported as `OK / n=42`;
- suppresses bbox/edge fields for `NONFINITE`, removing the misleading diagnostic values.

Production rejection behavior was not weakened: `_pack_hand()` still raises on malformed counts and the native validator still rejects non-finite input.

### Regression tests

`ai_pipeline/bridge/tests/check_hand_landmark_diag.py` now has 41 checks (42 PASS lines) covering: malformed counts → `n=0`, valid → `n=42`, non-finite image/world coordinates → `NONFINITE / n=0`, float32-overflow → `NONFINITE / n=0`, and no misleading bbox for non-finite input.

### Verification

| Command | Result |
|---|---|
| `python -B ai_pipeline/bridge/tests/check_hand_landmark_diag.py` | ALL CHECKS PASSED, exit 0 |
| `python -m pytest ai_pipeline/hand_kinematics/tests -q` | 24 passed |

### Exact remaining demo-side patch (test file only, not applied here)

```
- check("malformed result is returned unchanged (41 rows)", malformed.shape == (41, 3), str(malformed.shape))
+ check("malformed result is rejected (0 rows)", malformed.shape == (0, 3), str(malformed.shape))

- check("label: malformed", d["reason"] == "MALFORMED" and d["n"] == 41, d["reason"])
+ check("label: malformed", d["reason"] == "MALFORMED" and d["n"] == 0, d["reason"])
```

These are assertion corrections only; they align the demo's test file with its own production contract and do not weaken validation. They were not applied to `first-full-working-demo` because changes to that shared integration branch require the team's review process.

---

## Item 8 — Development branch review (Phase A)

**Status: ✅ REVIEW COMPLETE (read-only).**

Topology: our branch and the demo diverged at `24bee2c`. Demo added `69c84c8` (E1 validator), `240f806`, `fbe7bb6`, `c61655f`; our branch added `654d9d9`, `55faef2`, `71b3bba`.

| Our change | Present in demo? | Disposition |
|---|---|---|
| 21+21 + non-finite validation | Yes — E1 `69c84c8` | Mark **already covered**. Contracts differ (ours: malformed → bridge error `n=-1`; demo: malformed → `_EMPTY`/`n=0`). Do not merge competing validators. Native `handPayloadValid()` is behaviourally equivalent defense-in-depth. |
| Debug fixtures (`test_all.cpp`, bridge script) | Equivalent fixtures exist | Mark **already covered**; keep ours only as extra native non-finite coverage. |
| Preview `--target-zone`, whole-hand bbox in `tracking_types.hpp` / `main.cpp` | **No** | Genuinely missing; useful for H1 framing evidence. Needs clean build and visual verification. |
| Reduced OpenCV manifest in `vcpkg.json` | **No** (demo uses plain `opencv4`) | Genuinely different; clean build required before integration. |
| DSP progress report `.md`/`.pdf` | **No** | Documentation only. |
| Tester-path input validation (`71b3bba`) | **No** | Genuinely missing fix; this package. |

---

## Item 3 — FAST hand responsiveness readiness (H1)

**Status: 🔴 STILL OPEN.**

Trial 4 remains INCONCLUSIVE and unchanged. The completed E2-C five-run unscored qualification does **not** satisfy H1 because it used manual sweeps (no audio cue sequence), did not record physical geometry/lighting/exposure, and its diagnostic CSVs lack `capture_s`, frame IDs, detection states, and outward/return gate calculations. A NEW unscored FAST readiness capture with the documented physical setup and original criteria is still required. No scored Trial 5 is authorized.

---

## H2 — Scored executable identity

**Status: 🔴 STILL OPEN.**

`FAST_EXECUTABLE_COMPATIBILITY_DECISION_PACKAGE.md` documents the historical pin `3A3F0547…`, current Debug `C7A4CAAB…`, Release `12B9CE6B…`, and the narrow identity-adjustment recommendation. New facts to include in any final decision: baseline `c61655f`, E1 fix `240f806`, and that the historical binary reportedly found on Engineer 4's laptop predates the hand-input safeguards — so it cannot by itself satisfy "selected build includes approved input-validation safeguards". No launcher edit has been made and no executable is approved.

---

## Summary status

| Item | Status |
|---|---|
| 6 — Tester/native validation mismatch | 🟡 Implemented and automated-tested; live comparison outstanding |
| 7 — Diagnostic suite 31/33 | ✅ Fixed on this branch (41 checks pass, float32 overflow classified); 🟡 demo's 2-line test patch supplied, not applied to shared branch |
| 8 — Branch review | ✅ Read-only review complete |
| 3/H1 — Physical FAST qualification | 🔴 Still open |
| H2 — Executable approval | 🔴 Still open |

No DSP parameter, threshold, launcher, protocol, prior trial result, source under other engineers' ownership, or physical evidence was modified.
