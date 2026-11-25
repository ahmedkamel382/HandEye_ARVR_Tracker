# How H1 (FAST hand readiness) was solved

**Date:** 23 Nov 2026
**Final result:** UNSCORED run `E2D_UNSCORED_20261123_10` → **QUALIFIED** under the original `RESPONSIVENESS_PROTOCOL.md`. Trial 4 remains INCONCLUSIVE; no scored trial was run.

---

## 1. The problem Engineer 4 reported

Codex re-scored our earlier "PASS" (run 06) and found:
- fixed LEFT baseline OLS slope **0.006333/s** (limit **0.005/s**);
- the 10–90% duration calculation was wrong — it reported point values (~640 ms) that did not establish the original **0.8 s upper bound**.

So H1 was NOT solved by run 06, and the earlier claim of "all gates passed" was withdrawn.

## 2. Root cause of the false PASS

Our first calculator (`check_unscored_fast_readiness.py`) used **approximations**:
| Aspect | Wrong (old tool) | Correct (original protocol) |
|---|---|---|
| Time origin | raw trace time | `ceil(first stored timestamp × 1000)/1000` |
| Baseline | window relative to detected move | **fixed `[3,5)` protocol seconds**, never substituted |
| Eligible row | count 42, finite x/y | + Neutral state 0, `Filtered` status 1, published, finite raw+filtered X/Y |
| 10–90 duration | point estimate | **interval** `[raw90_low−raw10_high, raw90_high−raw10_low]`; PASS needs upper bound ≤0.8 s |
| Verdict | point thresholds | conservative PASS / FAIL / INCONCLUSIVE |

## 3. What was fixed in the tooling

Created `dsp_filter/evaluation/check_protocol_readiness_exact.py`, implementing the original protocol exactly:
- fixed zero and fixed `[3,5)` baseline with full stable-window gates (≥25 rows, gap ≤100 ms, SD ≤0.010, range ≤0.030, **slope ≤0.005/s**);
- first departure (3 consecutive eligible rows), first arrival (2 s stable window, mean shift ≥0.15);
- endpoint Y difference ≤0.05;
- 10–90 crossings from the candidate search window and **interval-based** duration;
- ≥3 observed intervals spanning 10–90, gaps ≤100 ms;
- self-test with a synthetic protocol-clean trace (still PASS after all fixes).

Two bugs found and corrected during the campaign:
1. baseline window misplacement (run 06 re-score exposed it);
2. 10–90 crossings were searched from the departure row, excluding the 10% crossing — fixed to search from the end of the reference window.

## 4. The physical campaign (5 unscored attempts)

| Run | Baseline slope | Legs | Result |
|---|---|---|---|
| 06 | 0.006333 ❌ | not scored | FAIL (matches Codex) |
| 07 | 0.002496 ✅ | outward 0.96–1.14 s ❌, return 0.64–0.80 s INCONCLUSIVE | FAIL — moved too slowly |
| 08 | 0.006762 ❌ | fast but not scored | FAIL — hand drifted at LEFT |
| 09 | 0.019543 ❌ | not scored | FAIL — hand drifted more |
| 10 | **0.001230 ✅** | both legs PASS | **QUALIFIED** |

## 5. Why run 10 passed

Two independent conditions were finally met in the same run:
1. **Stillness at LEFT during protocol seconds 3–5** — slope 0.001230/s, well under 0.005. Operator held the hand completely motionless through the countdown and the first five seconds (this was the recurring failure in runs 06/08/09).
2. **Fast but trackable sweeps** — 10–90 intervals [0.495, 0.656] s outward and [0.641, 0.800] s return, both wholly inside 0.15–0.8 s; 8 and 10 crossing intervals; hand continuously visible (no Held/Lost, no gaps >100 ms); endpoint Y differences 0.037 and 0.0015 (≤0.05).

## 6. Run 10 evidence
- Binary: dev Release `core_engine.exe`, SHA-256 `12B9CE6B…29A` (includes the E2-B input-validation safeguards).
- Command: `core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <run>\recording --preview --target-zone left`.
- Trace: 685 frames, `COMPLETE`, `overflow=0`, all 12 cues issued, exit 0, 42 screenshots.
- Physical setup metadata recorded (10+10 cm marks, ~60 cm, same height/depth, lighting, camera VID_0BDA/PID_5803).
- Full report: `E2D_H1_UNSCORED_QUALIFIED_RUN10.md`; per-run report JSON: `E2D_UNSCORED_20261123_10\protocol_exact_report.json`.

## 7. What H1 does NOT solve
- **H2 (executable identity) is still blocked.** No executable is approved; the launcher is unchanged. `FAST_EXECUTABLE_COMPATIBILITY_DECISION_PACKAGE.md` holds the answerable decision request.
- **Scored Trial 5 cannot start** until H2 is approved by Engineer 4/Codex. Run 10 is unscored evidence only.
- Trial 1–4 classifications, including Trial 4 INCONCLUSIVE, are unchanged.
