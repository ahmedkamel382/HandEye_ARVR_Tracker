# E2-D — UNSCORED FAST Readiness Capture Kit (H1)

**Status: NOT EXECUTED.** No camera was opened, no physical movement was captured, and no H1 result is claimed. This kit is prepared so a human operator with the physical setup can run it reproducibly.

**Purpose:** satisfy DSP Engineer item H1 — new **unscored** FAST readiness evidence using the original cue schedule, physical geometry, raw trace fields and original gates. This is **not** a scored trial and does not authorize Trial 5. Trial 4 remains INCONCLUSIVE and unchanged.

---

## 1. What is provided

| File | Role |
|---|---|
| `dsp_filter/evaluation/run_unscored_fast_qualification.py` | Reuses the unmodified Engineer 4 launcher's exact cue plan, audio synthesis and playback helpers; starts the native engine directly with `--dsp-record-dir`; writes manifest, cue log, console log; never overwrites existing files |
| `dsp_filter/evaluation/check_unscored_fast_readiness.py` | Parses the native `E4REC` trace and evaluates every original H1 gate; emits PASS / FAIL / INCONCLUSIVE; has a validated `--self-test` |

The scored launcher (`run_cued_responsiveness.py`) was **not modified**. Its branch/SHA preflight is deliberately bypassed for this unscored run; the scored identity decision remains BLOCKED and is handled separately in `FAST_EXECUTABLE_COMPATIBILITY_DECISION_PACKAGE.md`.

## 2. Exact cue schedule (unchanged, reused from the launcher)

`fast` mode, 55-second engine run:

| Time (s) | Cue |
|---:|---|
| 0 | Hold left. Keep still. |
| 5 | Get ready to move right. |
| 8 / 9 / 10 | Three / Two / One |
| 11 | MOVE! Go right. |
| 11.5 | STOP! Hold right. |
| 19.5 | Get ready to move left. |
| 22.5 / 23.5 / 24.5 | Three / Two / One |
| 25.5 | MOVE! Go left. |
| 26.0 | STOP! Hold left. |
| 34 | Finished. Keep holding left. |

## 3. Physical setup that must be recorded by the operator

Before running, mark and measure:

- LEFT — 10 cm — CENTER — 10 cm — RIGHT; ~20 cm one-way, ~40 cm round trip
- both markers at the **same height** and **same depth**
- ~60 cm hand-to-camera distance
- whole hand inside the safe margins at both marks; front-facing palm
- camera make/model, capture mode/codec, direct USB vs hub, fixed mount
- lighting, exposure, focus (and whether stable)

Fill these into `manifest.json` → `physical_setup` before evaluating gates. Blank values are not invented by the tool.

## 4. Procedure

All commands run from the project root using the project venv (`.venv\Scripts\python.exe`).

**Step 1 — prepare audio and manifest (no camera, no speaker):**
```
python dsp_filter/evaluation/run_unscored_fast_qualification.py ^
  --run-id E2D_UNSCORED_YYYYMMDD_01 ^
  --out-dir "C:\Users\Doctor Laptop\Downloads" ^
  --mode fast ^
  --exe out\build\windows-release\core_engine.exe ^
  --prepare-only
```

**Step 2 — fill in `physical_setup` in the generated `manifest.json`.**

**Step 3 — real unscored capture (opens camera 0 and plays audio):**
```
python dsp_filter/evaluation/run_unscored_fast_qualification.py ^
  --run-id E2D_UNSCORED_YYYYMMDD_01 ^
  --out-dir "C:\Users\Doctor Laptop\Downloads" ^
  --mode fast ^
  --exe out\build\windows-release\core_engine.exe
```
The operator positions the hand at LEFT before the countdown, starts on MOVE, reaches RIGHT on STOP, and returns on the second MOVE. Every attempt is preserved; re-run with a new `--run-id` for another attempt.

Artifacts: `manifest.json`, `audio/`, `cues.json`, `console.txt`, `MANIFEST_STATUS.json`, `recording/trace.txt` (E4DSP numerical trace).

**Step 4 — evaluate gates:**
```
python dsp_filter/evaluation/check_unscored_fast_readiness.py ^
  --trace "...\E2D_UNSCORED_YYYYMMDD_01\recording\trace.txt" ^
  --cues "...\E2D_UNSCORED_YYYYMMDD_01\cues.json" ^
  --mode fast ^
  --out "...\E2D_UNSCORED_YYYYMMDD_01\readiness_report.json"
```
Exit 0 = PASS; exit 1 = FAIL or INCONCLUSIVE (read the JSON for per-leg reasons).

**Step 5 — validation of the tools themselves (no camera):**
```
python dsp_filter/evaluation/check_unscored_fast_readiness.py --self-test
```
Validated result: clean synthetic capture → PASS; injected >100 ms gap → FAIL.

## 5. Original gates enforced (no threshold is relaxed)

| Gate | Threshold |
|---|---|
| Raw 10–90% duration | 150–800 ms |
| Raw horizontal amplitude | ≥ 0.15 |
| Endpoint vertical difference | ≤ 0.05 |
| Fresh-data gap | ≤ 100 ms |
| Fresh time intervals spanning FAST 10–90% | ≥ 3 |
| Stable fixation | ≥ 25 eligible samples, SD ≤ 0.010, range ≤ 0.030, slope ≤ 0.005/s, dwell ≥ 500 ms |

Both directions (LEFT→RIGHT and RIGHT→LEFT) are evaluated independently.

## 6. STOP conditions

- Any gate fails → report `NOT QUALIFIED`; do **not** relax criteria.
- Cue deadline missed or engine exits early → preserve the attempt and mark inconclusive.
- Physical metadata missing → the result is unscored evidence only, not H1 closure.
- No executable is approved → do **not** start a scored Trial 5. Trial 4 stays INCONCLUSIVE.

## 7. Result template

```
Run ID:            E2D_UNSCORED_YYYYMMDD_01
Physical setup:    <complete / incomplete>
LEFT_TO_RIGHT:     PASS / FAIL / INCONCLUSIVE  (<failing gates>)
RIGHT_TO_LEFT:     PASS / FAIL / INCONCLUSIVE  (<failing gates>)
Overall H1:        QUALIFIED / NOT QUALIFIED
Trial 5 readiness: <written statement with reasons>
```

## 8. Current status

🟡 **KIT READY — NOT EXECUTED.** The tools are prepared and the gate calculator is validated with synthetic traces; the audio-preparation path was exercised without camera or speaker. The physical capture, physical metadata, and Trial 5 readiness statement have not been produced and are not claimed.
