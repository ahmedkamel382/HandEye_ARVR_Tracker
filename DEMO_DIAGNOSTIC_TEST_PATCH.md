# Ready-to-apply patch for `first-full-working-demo` — diagnostic test suite 31/33

**DSP message item 7 (Engineers 1 + 2).** This file supplies the exact two assertion corrections needed in the demo's copy of `ai_pipeline/bridge/tests/check_hand_landmark_diag.py`. It is **not applied** here because `first-full-working-demo` is the shared integration branch and changes there require the team's review process.

## Root cause

The demo's production bridge rejects malformed payloads (`pack_hand()` returns `_EMPTY`), but two test assertions still expect the old behavior (41 rows passed through). Reproduced exactly: **31 PASS / 2 FAIL**, exit 1, with the demo test file run against the demo production file in an isolated temp directory.

## Patch (test file only — no production change, no validation weakened)

```diff
--- a/ai_pipeline/bridge/tests/check_hand_landmark_diag.py
+++ b/ai_pipeline/bridge/tests/check_hand_landmark_diag.py
@@
 malformed = le._pack_hand(R(hand(20), hand(21)))
-check("malformed result is returned unchanged (41 rows)", malformed.shape == (41, 3), str(malformed.shape))
+check("malformed result is rejected (0 rows)", malformed.shape == (0, 3), str(malformed.shape))
@@
 d = describe(hand(20), hand(21))
-check("label: malformed", d["reason"] == "MALFORMED" and d["n"] == 41, d["reason"])
+check("label: malformed", d["reason"] == "MALFORMED" and d["n"] == 0, d["reason"])
```

Apply from repo root on the demo branch:

```
git apply DEMO_DIAGNOSTIC_TEST_PATCH.md      # only the diff block above
.venv\Scripts\python.exe -B ai_pipeline/bridge/tests/check_hand_landmark_diag.py
```

Expected result after applying: **33/33 PASS**, exit 0.

## Why this is safe

- It changes assertions in a test file only; production validation is untouched.
- It aligns the test with the demo's own already-integrated contract (`pack_hand()` rejects malformed counts).
- The float32-overflow diagnostic correction is already implemented on `engineer2/dsp-report-and-preview-overlays` in `ai_pipeline/bridge/landmark_extractors.py` (`_describe_hand` reports `NONFINITE / n=0` and suppresses misleading bbox fields). Cherry-pick `7d837fc` if the demo should carry that production improvement too; no threshold or DSP behavior changes.

## Verification on the development branch (already done)

| Test | Result |
|---|---|
| `check_hand_landmark_diag.py` on this branch (41 checks incl. float32 overflow) | ALL CHECKS PASSED, exit 0 |
| `pytest ai_pipeline/hand_kinematics/tests -q` | 24 passed |
| `pytest core_engine/os_integration/tester/tests/test_hand_validation_equivalence.py -q` | 7 passed |
