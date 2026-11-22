# Hand Tracking / Core Engine Integration — Technical Progress and Resolution Report

**Audience:** DSP Engineer (Engineer 4), Engineer 1 — Core Engine Integration, Engineer 4 — FAST Protocol / Readiness Review<br>
**Prepared by:** Engineer 2 — Hand Tracking / project owner<br>
**Date:** 22 November 2026<br>
**Repository:** `HandEye_ARVR_Tracker`, branch `first-full-working-demo`<br>
**Nature:** Read-only evidence audit and progress report. No source, configuration, tests, builds, or camera sessions were performed for this report.

---

## 1. Executive Summary

The team has completed the native Core Engine build, runtime health check, and an unscored five-test native qualification series. The diagnostic bridge now records the information the DSP Engineer asked for: per-frame image/world landmark counts, a reason code, normalized bounding box, and four edge margins. The deployed hand model identity is pinned by SHA-256, and the reduced OpenCV/vcpkg configuration is installed and reproducible at the pinned vcpkg baseline.

However, the work is **not complete**:

- Phase 2 results are **unscored qualification evidence**, not an official FAST Trial 4 PASS and not proof of DSP/native integration acceptance.
- The malformed/non-finite landmark work (E2-B) has been investigated but **not implemented**. Current code still accepts a 42-element payload based on count alone and clamps invalid coordinates at three separate layers.
- The physical FAST movement qualification (E2-C) has a documented protocol basis, but several details remain **UNKNOWN**, and **no physical E2-C test has been performed**.
- The Phase 4 evidence package exists only as source evidence; the matrix, version record, parity table, fixture results, and experiment manifest have **not yet been produced**.
- The official FAST trials remain **INCONCLUSIVE** per `dsp_filter/README.md:592`.

This report answers every question found in the original DSP-originated reports, separates verified achievement from planned work, and assigns the remaining actions to Engineer 1, Engineer 4, and the project owner.

---

## 2. Project Scope and Original DSP Report

### 2.1 Candidate source documents

| Document | Role |
|---|---|
| `FAST-Hand-Tracking-Investigation.pdf` (18 Nov 2026, “Engineer 4 to Engineer 2”) | **Primary original DSP report.** Contains the four numbered “Questions for Engineer 2”, the decision request, the capture-setup agreement, and the evidence/reproducibility section. |
| `FAST-Hand-Tracking-Investigation-Review.pdf` (18 Nov 2026, “Review and DSP-Relevant Analysis”) | Companion DSP-relevant review with five decision questions and the explicit “what the DSP engineer should not do / retain” guidance. |
| `Engineer_2_Hand_Tracking_Action_Report.pdf` (22 Nov 2026, “Cross-team action report”) | Follow-up action request defining E2-A, E2-B, E2-C, the Engineer 4 contract, and the requested deliverable. Used as the authoritative task decomposition for the later phases. |
| `FAST-Hand-Tracking-Investigation-Response.pdf` | Engineer 2’s earlier response to Engineer 4, not the original DSP request. |
| `HAND_KINEMATICS_FOR_DSP_ENGINEER.md` | Engineer 2 to DSP contract documentation, not the DSP’s question set. |

The primary source used for the direct-answer section is the 18 November Engineer 4 report, supplemented by the 22 November action report where the DSP contract and deliverables were restated. No question from these documents is intentionally omitted.

### 2.2 Original problems identified by the DSP-side reports

1. FAST Trials 1–3 were inconclusive: nine processed frames produced zero landmark rows and became HELD; no frame reached LOST.
2. The numerical traces could not explain **why** those frames were empty (detector miss vs bridge discard vs partial list).
3. Trial 2 had continuous tracking but failed the first-direction rule; Trial 1 failed speed/endpoint checks; Trial 3 failed baseline drift.
4. The DSP was shown not to be defective: all 2,972 accepted X/Y pairs replayed within `1e-12` normalized units. The 256.321 ms gap correctly triggered the existing 250 ms filter reset.
5. Before Trial 4, the team needed a repeatable, unscored capture setup: fixed lighting/camera geometry, 20 cm marks, whole-hand margin checks, and separately measured delivered vs processed FPS.
6. The bridge needed a minimal diagnostic: image/world list presence, combined count, normalized hand extent, and timestamps, collectible without video or scoring changes.
7. Later action report added explicit E2-B (malformed/non-finite payloads) and E2-C (physical 20 cm FAST sweep) work.

---

## 3. Direct Answers to the DSP Engineer’s Questions

| Original question / requirement | Our investigation | Current answer | Resolution status | Evidence | Remaining work | Responsible engineer |
|---|---|---|---|---|---|---|
| **18-Nov Q1:** “Does the integrated extractor and six-miss grace behavior described on page 2 match the intended hand contract? Was VIDEO mode in effect for these runs? No hand ROI argument was passed; the extractor also supports an environment override.” | Inspected extractor defaults, bridge, `HandNative`, and Phase 1/2 diagnostic CSVs. | The integrated extractor requires both image and world landmark lists before packing 42 rows; native grace is six misses; current qualified runs used VIDEO mode. The original Trial 1–4 ROI configuration was **never recorded**, so the report cannot confirm VIDEO/ROI for those trials from artifacts alone. | **Partially Solved** | `ai_pipeline/bridge/landmark_extractors.py:192-195,240-248`; `core_engine/include/hand_math.hpp:58,177-200`; Phase 1/2 CSVs all `mode=VIDEO`; earlier audit reports state ROI was not recorded. | Recover original launch commands or state ROI as permanently unknown; do not infer from later runs. | Engineer 1 + Engineer 4 |
| **18-Nov Q2:** “Can existing diagnostics distinguish a missing image-landmark list from a missing world-landmark list? Can they distinguish a detector miss from an inter-frame tracking miss for the listed frames? Identify what was actually logged.” | Inspected diagnostic schema and reason logic; inspected Phase 1/2 CSVs. | Yes, the implemented diagnostic distinguishes `BOTH_EMPTY`, `IMAGE_ONLY`, `WORLD_ONLY`, `MALFORMED`, and `OK`, with `img_n`, `world_n`, `n`, bbox, and edge margins. It **cannot** distinguish a MediaPipe detector miss from an inter-frame VIDEO tracking miss; that distinction is not exposed by the detector/bridge. | **Partially Solved** | `ai_pipeline/bridge/landmark_extractors.py:68-71,78-131,180-195`; 15-column headers in `phase1_core_runtime_diag.csv` and five Phase 2 CSVs. | If the distinction is required, define a new probe with Engineer 1; do not claim the current CSV provides it. | Engineer 1 / Engineer 2 |
| **18-Nov Q3:** “What hand size, distance, lighting and frame-edge margin work reliably for this camera and open-hand pose during a 0.5 s sweep? Can existing tools verify the whole-hand margin at both marks before a scored run?” | Analyzed all five final Phase 2 CSVs and native console warnings. | Edge margin is now measurable and passed in all final runs (`>=0.10`). Camera delivered 12.4–12.5 FPS and processed 12–13 FPS, with a low-light/exposure warning. Physical distance and lighting were **not measured**; only normalized margins and console FPS were recorded. | **Partially Solved** | `phase2_core_{left_v4,center_v2,right_v1,slow_v3,fast_v8}_diag.csv`; native output warning “below 60 FPS … low light …”; report section 6. | Record physical distance, lighting, exposure, and marker geometry in a trial manifest before scored work. | Engineer 2 + Engineer 4 |
| **18-Nov Q4:** “If the cause remains unknown, what is the smallest useful observation for a separately approved setup check …? Who owns it, and can it be collected without saving video or slowing the scored run?” | Verified implemented diagnostic writer and its activation. | The minimal observation was implemented: per-frame list counts, reason, bbox, edge margins, mode, and timestamp. It is opt-in via `HAND_LANDMARK_DIAG=1`, writes a CSV, does not save video, and does not change tracking. Ownership of the extractor file was Engineer 1’s; Engineer 2 implemented and qualified the diagnostic with the bridge owner’s file-level authority per earlier approvals. | **Solved** | `ai_pipeline/bridge/landmark_extractors.py:30-32,68-71,134-177`; Phase 1 runtime CSV (151 rows); Phase 2 CSVs. | Keep the diagnostic off in scored runs unless explicitly approved; archive outputs with a manifest. | Engineer 1 / Engineer 2 |
| **18-Nov decision:** agree repeatable setup; retain original criteria; keep recorder, camera 0, hand-only dry run, DSP settings unchanged; preserve inconclusive attempts; SLOW RIGHT to LEFT separate. | Reviewed Phase 2 launch commands and native settings. | Phase 2 used a fixed native setup (camera 0, `--os log`, no DSP parameter changes), preserved all attempts, and did not revise FAST criteria. The 20 cm physical geometry and official Trial 4 repeat have not been executed. | **Partially Solved** | Phase 2 launch procedure recorded in session logs; `dsp_filter/README.md:592`; no Trial 4 artifacts found. | Engineer 4 must confirm geometry and schedule any Trial 4. | Engineer 4 |
| **22-Nov E2-A:** confirm deployed model, capture-vs-processed FPS, stable exposure/margin, 42-row detection on LEFT/CENTER/RIGHT/slow/fast; report misses with timestamp/bbox/exposure notes; never revise scored criteria. | Verified model hash, build, diagnostic CSVs, and native FPS output. | Model SHA verified; delivered/processed FPS measured; all five final CSVs have 42 rows on every row, no misses, and margins >=0.10. Exposure notes are only a console warning, not a saved field. | **Partially Solved** | SHA `FBC2A300…CDE1`; Phase 2 CSVs; native console output. | Save exposure/camera mode in a manifest; do not treat unscored qualification as scored. | Engineer 2 |
| **22-Nov E2-B:** agree exact 21+21 contract; define validity and zero/partial path; verify classifiers do not process malformed/nonfinite; fixture tests; never silently clamp NaN/Inf. | Inspected Python packer, bridge, `HandNative`, freshness gate, DSP adapter, DSP filter, and tests. | The need is confirmed, but the fix is **not implemented**. 20+22 still packs to 42; NaN/Inf and out-of-range values can be clamped into valid-looking points at three layers. No integration fixtures exist. | **Investigated — Not Solved** | `landmark_extractors.py:180-195`; `py_bridge.hpp:52-69`; `hand_math.hpp:177-220,218`; `freshness_gate.hpp:94`; `dsp_adapter.hpp:52-59`; `check_hand_landmark_diag.py:71`. | Engineer 1 must implement contract validation and fixtures. | Engineer 1 / Joint |
| **22-Nov E2-C:** predeclare camera/marker geometry; unscored full-hand, fixed-depth 20 cm sweep; report eligible frames, amplitude, Y drift, gap, 42-row completeness; do not score old invalid legs. | Recovered audit text and `RESPONSIVENESS_PROTOCOL.md`. | Protocol elements are documented, but “20 cm total travel” vs two marks 20 cm apart is unresolved, and no physical test has been performed. | **Blocked / Not Started** | `Engineer_2_Hand_Tracking_Action_Report.pdf:2`; `dsp_filter/evaluation/RESPONSIVENESS_PROTOCOL.md:23-41,119-180,230-251`. | Engineer 4 clarification and user physical execution required. | Engineer 4 / Engineer 2 |
| **Contract:** fresh DETECTED/RECOVERED observations only; HELD cached; LOST clears filter. | Inspected freshness gate, `HandNative`, OS bridge, DSP adapter, and existing tests. | Implemented in the current source: TRACKED updates filter, HELD replays cached filtered point with no filter update, LOST resets. Tests exist in `test_all.cpp` and `test_dsp_adapter.cpp`. | **Solved** (implementation present; tests not re-run in this read-only task) | `core_engine/include/freshness_gate.hpp:48-67`; `tracking_types.hpp:49-50`; `core_engine/tests/unit/test_all.cpp:156-190`; `test_dsp_adapter.cpp:157-185`. | Re-run unit tests during Phase 3. | Engineer 1 |
| **Contract:** six-frame grace and >250 ms filter gap reset are distinct. | Inspected native grace and DSP timing constants. | Confirmed distinct: native grace is six missed frames; DSP `reset_gap_seconds` is 0.25 s. Trial 3’s 256.321 ms gap correctly caused reinitialization. | **Solved** | `hand_math.hpp:58`; `dsp_filter/include/one_euro_filter.hpp:13-17`; `FAST-Hand-Tracking-Investigation.pdf:2`. | None for this item. | Engineer 1 / Engineer 4 |
| **Contract:** report native Core status separately from Python reference; distinguish model miss from bridge discard using flags, counts, bbox, margins. | Inspected diagnostic fields and E4DSP recorder schema. | The diagnostic CSV can separate empty result types and provides bbox/edges. The E4DSP recorder schema carries `landmarks`, `state`, `tracking_state`, `missed_frames`, `published_tracking`. Phase 2 evidence did **not** include a paired E4DSP recorder trace, so native-vs-Python parity is not demonstrated for those runs. | **Partially Solved** | `landmark_extractors.py:68-71`; `dsp_filter/evaluation/README.md:320-328`; Phase 2 CSVs. | Execute and archive a recorder session if required by Engineer 4. | Engineer 1 / Engineer 4 |
| **Contract:** keep DSP thresholds unchanged; return model/capture verdict supported by diagnostics, not a presumption of model defect. | Inspected source and Phase 2 commands; no DSP parameter changes were made. | DSP parameters were not changed. Current verdict is that Phase 2 establishes capture/framing qualification only; it does not establish a DSP defect or a DSP pass. | **Solved** | `dsp_filter/README.md:588-594`; Phase 2 native output showed default hand filter `1.2/4/1`; no DSP source edits in git status. | None for this item. | Engineer 4 |
| **Deliverable:** qualification matrix, model SHA/version, native-vs-Python parity table, fixture tests with Engineer 1. | Audited repository for these artifacts. | None of the four deliverables has been produced as a Phase 4 package. SHA is known; version is not. Parity tooling exists but no current measurement table exists. | **Not Started / Blocked** | Report sections 9 and 10; missing artifacts in repository root. | Assemble Phase 4 after Phase 3. | Engineer 2 / Joint |
| **Review memo decision:** “Are the DSP equations and parameters supported by these recordings?” | Read review memo and replay results. | Yes. Replay matched published output within `1e-12` normalized units. | **Solved** | `FAST-Hand-Tracking-Investigation-Review.pdf:1`; `FAST-Hand-Tracking-Investigation.pdf:2`. | None. | Engineer 4 |
| **Review memo decision:** “Did the DSP receive repeated/stale samples during HELD?” | Read source and review memo. | No. All nine HELD frames had zero landmark rows and no DSP/adapter call; cached filtered point was republished. | **Solved** | `FAST-Hand-Tracking-Investigation-Review.pdf:1-2`; `freshness_gate.hpp:53-62`. | None. | Engineer 1 / Engineer 4 |
| **Review memo decision:** “Is Trial 2 a clean reference trial?” | Read report. | No. Continuous tracking, but first-direction rule failed and no admissible RIGHT reference existed for return. | **Solved** | `FAST-Hand-Tracking-Investigation.pdf:1`; review memo page 1. | None. | Engineer 4 |
| **Review memo decision:** “Is a production DSP change justified?” | Read report and current source. | No. No DSP source change is justified by current evidence. | **Solved** | `FAST-Hand-Tracking-Investigation-Review.pdf:1`; `dsp_filter/README.md:592`. | None. | Engineer 4 |
| **Review memo decision:** “What is the next step?” | Read report and review memo. | Qualify setup in unscored runs, then repeat Trial 4 with original criteria. Phase 2 completed the unscored native qualification; official Trial 4 has not been run. | **Partially Solved** | Phase 2 CSVs; no Trial 4 artifacts found. | Engineer 4 must schedule and approve Trial 4. | Engineer 4 |

---

## 4. Problem Resolution Matrix

Status definitions: **SOLVED** = implementation/correction complete with verification evidence; **PARTIALLY SOLVED** = some work complete but important evidence missing; **INVESTIGATED** = cause/action understood but fix not implemented; **NOT STARTED** = no meaningful implementation or verification; **BLOCKED** = requires external decision/dependency.

| ID | Original Problem | What We Did | Current Status | Evidence | Remaining Work |
|---|---|---|---|---|---|
| P1 | Native Core build prerequisites missing (vcpkg, OpenCV, pybind11) | Bootstrapped pinned shallow vcpkg; installed reduced OpenCV; built Core | **SOLVED** | `C:\vcpkg` HEAD `40f3c709…`; `vcpkg.json` 7 features; `out/build/windows-release/core_engine.exe`, `OpenCVConfig.cmake` | None |
| P2 | Deployed model identity uncertain | Computed SHA-256 of `assets/hand_landmarker.task` and dist copy | **SOLVED** | SHA `FBC2A300…CDE1`, size 7,819,105 B, both copies identical | Record model **version** if required |
| P3 | Runtime configuration (`VIDEO`, ROI off, diagnostics) unverified | Native live run with diagnostic CSV; source inspection | **Partially Solved** | `phase1_core_runtime_diag.csv` (151 rows, `VIDEO`); `[ai] ready` no ROI suffix; source defaults | Persist launch/health-check logs; original Trial 1–4 ROI remains unknown |
| P4 | Health check passed but not persisted | Ran `core_engine.exe --check`, exit 0, `ALL CHECKS PASSED (1 warning)` | **Partially Solved** | Session record only; no saved log file in repo | Save health-check log in trial package |
| P5 | Empty landmark cause ambiguous | Implemented/qualified diagnostic CSV with reason and bbox; ran native sessions | **Solved** for unscored qualification | `landmark_extractors.py:68-71,134-177`; five Phase 2 CSVs | Does not distinguish detector miss from inter-frame tracking miss |
| P6 | Camera delivered vs processed FPS | Native console reported delivered 12.4–12.5 FPS and processed ~12–13 FPS | **Partially Solved** | Phase 2 console output; low-light warning | Persist FPS evidence; improve lighting/camera for scored work |
| P7 | Unscored qualification evidence missing | Completed LEFT/CENTER/RIGHT/SLOW/FAST native runs | **Solved** (unscored only) | Five final CSVs, all margins >=0.10, no zero rows | Not a Trial 4 pass; no DSP acceptance claim |
| P8 | Preview could not show hand position/target zone | Added `--target-zone` and actual magenta bbox overlay | **Partially Solved** | `main.cpp:125,346-382,526-546,696,725`; build exit 0 | No saved live visual verification of alignment |
| P9 | 20+22 payload accepted as 42 | Cause confirmed; no fix | **Investigated** | `landmark_extractors.py:180-195`; `hand_math.hpp:179` | Engineer 1 implementation + fixtures |
| P10 | NaN/Inf become fresh pointer | Cause confirmed; native clamps at three layers | **Investigated** | `hand_math.hpp:218`; `freshness_gate.hpp:94`; `dsp_adapter.hpp:59` | Engineer 1 fix + fixture tests |
| P11 | Invalid input must never be fresh | Freshness states exist, but invalid `n=42` bypasses them | **Partially Solved** | `freshness_gate.hpp:48-67`; `tracking_types.hpp:49-50` | E2-B implementation required |
| P12 | Silent clamping of invalid coordinates | Three clamping layers identified | **Investigated** | as above; `dsp_filter/evaluation/README.md:327` | Decide which clamps remain for valid points only |
| P13 | E2-C physical 20 cm protocol unresolved | Protocol recovered; conflicts documented | **Blocked** | audit report p.2; `RESPONSIVENESS_PROTOCOL.md:23-41,119-180,230-251` | Engineer 4 clarification |
| P14 | Phase 4 evidence package | No matrix/manifest/parity table/fixture report produced | **Not Started** | repository root lacks these artifacts | Assemble after Phase 3 |
| P15 | Official FAST Trial 4 | Not run | **Blocked** | `dsp_filter/README.md:592` | Engineer 4 scheduling and approval |

---

## 5. Phase 1 — Environment and Native Core Build

### Completed and verified

- **vcpkg:** shallow checkout at exact pinned baseline `40f3c709db80acf154ac4b17a1f83c564ebd022e`; `vcpkg.exe version 2026-07-13-bf04c909169fdbb30821c02c6eb01f1cd1295d05`.
- **Reduced OpenCV manifest:** `vcpkg.json` uses `default-features: false` with `highgui`, `win32ui`, `msmf`, `dshow`, `thread`, `fs`, `jpeg`. Installed OpenCV 4.12.0 at `out/build/windows-release/vcpkg_installed/x64-windows`.
- **CMake configuration:** `windows-release` preset; CMakeCache confirms Ninja, `x64-windows`, `.venv` Python 3.11.9, pybind11 3.1.0.
- **Build:** `cmake --build out\build\windows-release --target core_engine` succeeded, producing `out/build/windows-release/core_engine.exe`.
- **Model:** `assets/hand_landmarker.task` SHA-256 `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`; identical dist copy.
- **Runtime diagnostic:** `phase1_core_runtime_diag.csv` — 151 rows, 15 columns, all `mode=VIDEO`, written by native Core with `[hand-diag] writing …` in the session record.
- **Handshake:** `hand_roi=0` inferred from source default and absence of ROI suffix, plus env `HAND_RUNNING_MODE=VIDEO` in launch.

### Partially verified / limitations

- No persisted console log of `core_engine.exe --check`; the `ALL CHECKS PASSED (1 warning)` result exists only in the session record.
- Camera warning: delivered 12.4 FPS vs requested 60; low-light/exposure was suggested by the engine but not measured.
- Original FAST Trial 1–4 ROI and launch command were never recorded; later evidence cannot prove their configuration.

---

## 6. Phase 2 — Native Core Qualification

| Test | CSV | Rows | Schema | `mode` | `n=42` | `reason` | Edge minima | Result |
|---|---|---|---|---|---|---|---|---|
| LEFT | `phase2_core_left_v4_diag.csv` | 151 | 15-col exact | VIDEO 151 | 151 | OK 151 | l .1548 r .6124 t .3370 b .1578 | PASS |
| CENTER | `phase2_core_center_v2_diag.csv` | 151 | 15-col exact | VIDEO 151 | 151 | OK 151 | l .4121 r .3690 t .3262 b .1243 | PASS |
| RIGHT | `phase2_core_right_v1_diag.csv` | 151 | 15-col exact | VIDEO 151 | 151 | OK 151 | l .6436 r .1622 t .3552 b .1353 | PASS |
| SLOW | `phase2_core_slow_v3_diag.csv` | 152 | 15-col exact | VIDEO 152 | 152 | OK 152 | l .1059 r .1164 t .3315 b .2324 | PASS |
| FAST | `phase2_core_fast_v8_diag.csv` | 152 | 15-col exact | VIDEO 152 | 152 | OK 152 | l .1142 r .1491 t .3006 b .2888 | PASS |

All five have zero zero-row or malformed observations and every edge margin >= 0.10.

### What this demonstrates

- A locked camera and controlled full-hand framing can produce stable 42-row detection at the native Core bridge.
- The diagnostic CSV is capable of recording exactly the bbox/edge/miss information the DSP Engineer requested.
- VIDEO mode and the diagnostic path operate correctly in a native session.

### What this does NOT demonstrate

- It is not an official FAST Trial 4 PASS.
- It is not DSP acceptance: no E4DSP recorder trace was produced for these runs, and no paired raw/filtered responsiveness analysis was performed.
- It is not E2-B fixture evidence: no malformed or non-finite payloads were injected.
- It is not E2-C: no 20 cm physical path, depth control, endpoint stability, or physical travel measurement was performed.

### Native origin caveat

Persisted shell logs directly contain `[hand-diag] writing …` lines for the final FAST attempts (`fast_v6`, `fast_v7`, `fast_v8`). For `left_v4`, `center_v2`, `right_v1`, `slow_v3`, and `fast_v8` the session transcript and file metadata support native origin, but equally strong persisted console evidence is not available for every file. No Python tracker was launched in those sessions; the launch command was always `core_engine.exe`.

---

## 7. Target Zone and Hand Bounding Box Improvements

### Why the first overlay was insufficient

The initial `--target-zone left|center|right` overlay showed only:

- a green 10% safe frame,
- a yellow selected third,
- the existing red index-fingertip dot.

The DSP asked for **whole-hand margin**, and Phase 2 failures showed that the fingertip can be inside a zone while the full hand bbox crosses the frame edge. The static zone could not show actual detected hand extent.

### What was implemented

`core_engine/src/main.cpp` and `core_engine/include/tracking_types.hpp` were modified:

- `Config::target_zone` and `--target-zone` parsing/validation/usage (`main.cpp:125,696,725`).
- Static safe frame and target third drawing (`main.cpp:346-369`).
- Actual bbox fields in `ModuleOut` (`tracking_types.hpp:53-57`): `bbox_l/t/r/b`, `bbox_valid`.
- Bbox calculated from the current frame’s 21 image landmarks only when `n == kHandPacked` (`main.cpp:526-546`), with finite/range validation and positive width/height; no clamping.
- Magenta outline and `bbox cx=…` label drawn before the red dot (`main.cpp:373-382`), so the fingertip remains on top.
- Bbox validity is independent of `detected()`, preventing stale boxes during HELD dropouts.

### Verification status

- Build succeeded (`BUILD_EXIT=0`), only `main.cpp.obj` rebuilt and `core_engine.exe` relinked.
- CLI checks confirmed valid zones accepted and invalid zone rejected.
- **Live visual alignment was not verified** because no camera session was run after that implementation for the purpose of visual validation. Phase 2 runs after the change used the CSV evidence; the bbox was available in the preview but no saved screenshot or visual record exists.

Do not confuse the successful build with real-camera visual validation.

---

## 8. Phase 3 — Integration Tests (E2-B and E2-C)

### E2-B readiness

| Layer | Current behavior | Status |
|---|---|---|
| `landmark_extractors.py:180-195` | Packs any non-empty image and world lists; 20+22 becomes 42 | **MISSING validation** |
| `py_bridge.hpp:52-69` | Validates only `(N,3)` shape; no content validation | **EXISTING / insufficient** |
| `hand_math.hpp:177-220` | Rejects `n != 42`; no finite/range check; clamps at `:218` | **MISSING / CLAMPING** |
| `freshness_gate.hpp:48-67,94` | TRACKED/HELD/LOST implemented; clamps at `:94` | **EXISTING / CLAMPING** |
| `dsp_adapter.hpp:52-59` | Rejects non-finite but clamps out-of-range at `:59` | **EXISTING / CLAMPING** |
| `dsp_filter/src/one_euro_filter.cpp:85-90,168-176` | Rejects non-finite/out-of-range, resets | **EXISTING** |
| `core_engine/tests/unit/test_all.cpp:106-154` | `Packed` helper and bridge-error `n=10` test exist | **EXISTING / insufficient** |
| `ai_pipeline/bridge/tests/check_hand_landmark_diag.py:71` | `MALFORMED` label check only | **EXISTING / insufficient** |

No integration fixtures exist for 20+22, NaN/±Inf, out-of-range finite coordinates, invalid-to-valid, or valid-to-invalid while retained. E2-B is therefore **Investigated — Not Solved**.

### E2-C readiness

Confirmed from `dsp_filter/evaluation/RESPONSIVENESS_PROTOCOL.md`:

- two marks about 20 cm apart, same height/depth;
- slow approximately 3 s, fast approximately 0.5 s;
- 55 s engine run, hand-only, `--filter dsp --os log --no-gaze`;
- eligible row: count 42, fresh Tracked, Neutral, Filtered, published, finite raw/filtered X/Y;
- stable windows >=25 samples, gap <=100 ms, raw X SD <=0.01, range <=0.03, slope <=0.005/s;
- amplitude >=0.15 normalized, endpoint mean Y difference <=0.05;
- FAST raw 10–90% within 0.15–0.8 s; added 50%/90% delay <=200 ms; settling <=250 ms;
- verdict PASS/FAIL/INCONCLUSIVE by uncertainty bounds.

Unresolved:

- audit says “20 cm **total travel**” while protocol describes marks ~20 cm apart (one-way 20 cm, total path 40 cm);
- number of repetitions beyond “report all attempts”;
- audio-guided launcher vs manual sweep;
- camera/depth verification method;
- E2-C launcher hardcodes `out/build/windows-debug/core_engine.exe` at `run_cued_responsiveness.py:702`, while the built executable is `windows-release`.

**No physical E2-C test has been performed.** E2-C is **Blocked / Not Started**.

---

## 9. Phase 4 — Results Handoff

| Deliverable | Status | Evidence / missing item |
|---|---|---|
| 4.1 Qualification Matrix | **NOT STARTED** | Data exists in this report; no standalone matrix file |
| 4.2 Model SHA / version | **PARTIALLY COMPLETE** | SHA `FBC2A300…` verified; version string not found in any available document |
| 4.3 Native-vs-Python Parity Table | **NOT STARTED** | `core_engine/tests/parity/run_parity.py` and `parity_cases.csv` exist; no current measured table |
| 4.4 Joint Fixture Test Results | **NOT STARTED** | Depends on Phase 3 E2-B fixtures; none executed |
| 4.5 Diagnostics CSV + Experiment Manifest | **PARTIALLY COMPLETE** | CSVs exist; no manifest tying test names, configs, model, build ID, environment, and results together |

Engineer 4 must review the completed package and decide whether the system is ready for an official FAST Trial 4. Phase 4 completion is not itself a Trial 4 pass.

---

## 10. What We Have Actually Achieved

### 10.1 Implemented and verified

- Reproducible native Core build with pinned vcpkg and reduced OpenCV features.
- Verified model SHA and local model copies.
- Native runtime diagnostic CSV with the exact fields DSP requested.
- Five unscored native qualification runs with valid 42-row data and edge margins >=0.10.
- Buildable target-zone and actual-bbox preview overlay.

### 10.2 Implemented but not fully verified

- `--target-zone` and bbox overlays: build and CLI verified; live visual alignment not saved as evidence.
- `hand_roi=0` and VIDEO mode: source and Phase 1/2 evidence support it; original Trial 1–4 configuration remains unrecorded.
- Health check: reported pass, but no persisted log artifact.

### 10.3 Investigated but not implemented

- E2-B: malformed 20+22, NaN/±Inf, invalid-as-fresh, and silent clamping causes are identified, but no fix or fixture exists.
- E2-C: protocol recovered, but physical procedure conflict and launcher path blocker remain.

### 10.4 Not yet started

- Phase 3 fixture implementation and execution.
- Physical E2-C test.
- Phase 4 deliverables.
- Official FAST Trial 4.

### Distinction of evidence levels

| Level | Current status |
|---|---|
| Build success | Achieved |
| Runtime success | Achieved for diagnostic runs; health-check log not persisted |
| Qualification success | Achieved for unscored Phase 2 only |
| Integration test success | Not achieved; Phase 3 not implemented |
| DSP acceptance | Not achieved; FAST remains inconclusive |
| Official FAST Trial 4 readiness | Not achieved |

---

## 11. Remaining Problems and Risks

1. **E2-B not implemented:** current code can accept malformed/non-finite 42-element payloads as fresh.
2. **Three clamping layers:** `hand_math.hpp:218`, `freshness_gate.hpp:94`, `dsp_adapter.hpp:59` can sanitize invalid data into plausible values.
3. **No fixtures:** current tests do not cover the DSP's malformed/non-finite cases.
4. **E2-C ambiguity:** 20 cm total path vs one-way 20 cm marks; no physical test.
5. **Launcher path mismatch:** audio-guided launcher expects debug executable; release build exists.
6. **Evidence provenance:** several native-origin proofs live only in session records, not persisted logs.
7. **Model version missing:** SHA is known, version is not.
8. **No Phase 4 package:** qualification matrix, parity table, manifest, and fixture report absent.

---

## 12. Responsibilities and Next Actions

### Engineer 1 — Core Engine Integration

- Confirm the exact 21+21 payload contract and rejection boundary.
- Implement rejection of 20+22 and other malformed counts.
- Reject NaN/±Inf and out-of-range finite landmarks before cursor/classifier use.
- Resolve which clamping layers remain for valid normalized points only.
- Ensure invalid input cannot become fresh TRACKED; preserve TRACKED/HELD/LOST and six-miss grace.
- Add native fixtures in `core_engine/tests/unit/test_all.cpp` / `test_dsp_adapter.cpp`.
- Agree/implement any new invalid-reason enum or reporting field if needed.
- Build test targets and run unit/integration tests.

### Engineer 4 — DSP Protocol and Readiness Review

- Confirm E2-C interpretation: one-way 20 cm marks vs 20 cm total physical travel.
- Confirm audio-guided launcher vs manual stopwatch; number of repetitions; depth verification method.
- Confirm required recorder evidence (E4DSP trace, manifest, console, paired CSV).
- Fix or approve workaround for launcher’s hardcoded `out/build/windows-debug/core_engine.exe`.
- Confirm final acceptance criteria and Phase 4 package format.
- Review Phase 4 package and decide official FAST Trial 4 readiness.

### Engineer 2 — Project Owner / Hand Tracking

- Preserve and catalog Phase 1/2 evidence.
- Save native console logs and build/executable identity for future packages.
- Coordinate physical E2-C sessions; do not perform them until protocol is clarified.
- Record model version/source if obtainable.
- Produce parity evidence using existing `run_parity.py` tooling when Phase 3 is ready.
- Assemble the Phase 4 qualification matrix and manifest after Phase 3 fixtures pass.

---

## 13. Final Readiness Assessment

**Are we ready for Phase 3 implementation?** No. We are ready to **begin** Phase 3 implementation only after Engineer 1 and Engineer 4 confirm the open contract/protocol decisions. The technical gaps are understood, but no Phase 3 fix or fixture has been implemented.

**Are we ready for official FAST Trial 4?** No. Phase 2 is unscored qualification only; DSP acceptance is not established; E2-B/E2-C are incomplete; Phase 4 evidence package is not assembled.

**Most important completed achievements:** native Core build, model SHA verification, diagnostic bridge, and five clean unscored qualification CSVs.

**Most important unresolved blockers:** E2-B implementation/fixtures, E2-C protocol ambiguity and physical test, missing Phase 4 package, and missing persisted health-check/native-origin logs for some artifacts.

**Answer to the requested final question:** If this report is sent today, the DSP Engineer, Engineer 1, and Engineer 4 will understand what was achieved, what was only investigated, what is unresolved, and what each must do next — provided they read the direct-answer and responsibility sections. The project is **not complete** and must not be represented as Trial-4-ready.

---

## 14. Evidence and Artifact Index

| Artifact | Path / reference |
|---|---|
| Primary DSP report | `C:\Users\Doctor Laptop\Downloads\FAST-Hand-Tracking-Investigation.pdf` |
| DSP-relevant review | `C:\Users\Doctor Laptop\Downloads\FAST-Hand-Tracking-Investigation-Review.pdf` |
| Follow-up action report | `C:\Users\Doctor Laptop\Downloads\Engineer_2_Hand_Tracking_Action_Report.pdf` |
| E2-C protocol | `dsp_filter/evaluation/RESPONSIVENESS_PROTOCOL.md` |
| DSP status | `dsp_filter/README.md:588-594` |
| Model asset | `assets/hand_landmarker.task`, SHA `FBC2A300…CDE1` |
| vcpkg manifest | `vcpkg.json` (7 reduced OpenCV features) |
| vcpkg checkout | `C:\vcpkg`, HEAD `40f3c709…` |
| CMake cache | `out/build/windows-release/CMakeCache.txt` |
| Executable | `out/build/windows-release/core_engine.exe` |
| Phase 1 native CSV | `C:\Users\Doctor Laptop\Downloads\phase1_core_runtime_diag.csv` |
| Phase 2 CSVs | `C:\Users\Doctor Laptop\Downloads\phase2_core_left_v4_diag.csv`, `phase2_core_center_v2_diag.csv`, `phase2_core_right_v1_diag.csv`, `phase2_core_slow_v3_diag.csv`, `phase2_core_fast_v8_diag.csv` |
| Diagnostic implementation | `ai_pipeline/bridge/landmark_extractors.py:68-71,78-131,134-177,180-195` |
| Bridge | `core_engine/include/py_bridge.hpp:52-69` |
| Native hand logic | `core_engine/include/hand_math.hpp:27-28,58,177-220` |
| Freshness gate | `core_engine/include/freshness_gate.hpp:48-67,94` |
| DSP adapter | `core_engine/include/dsp_adapter.hpp:52-59` |
| Tracking result | `core_engine/include/tracking_types.hpp:42-58` |
| Preview changes | `core_engine/src/main.cpp:36,125,346-382,526-556,696,725` |
| Unit fixtures | `core_engine/tests/unit/test_all.cpp:106-154`, `core_engine/tests/unit/test_dsp_adapter.cpp:99,132,157-185` |
| Parity tools | `core_engine/tests/parity/run_parity.py`, `core_engine/generated/hand_forest/parity_cases.csv` |
| Recorder/manifest tools | `dsp_filter/evaluation/check_recorder_pipeline.py:71`, `dsp_filter/evaluation/run_cued_responsiveness.py:365-395,684-708` |
