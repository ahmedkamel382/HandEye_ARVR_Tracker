# Phase 2 — Evidence mapping, H2 provenance and integration note (Engineer 4 points 8–10)

**Date:** 26 Nov 2026. All facts read from local repository and run folders.

## 1. Complete H1 run mapping and capture order (point 8)

All 16 unscored run folders exist locally under `C:\Users\Doctor Laptop\Downloads\`, each with `recording/trace.txt`, `manifest.json`, `cues.json`, `console.txt`, `MANIFEST_STATUS.json`, `recording/COMPLETE`, and a `protocol_exact_report.json` (re-scored with the current v3 scorer). Capture order is by run ID sequence; later report re-scoring changed some folder mtimes, so mtimes are NOT used as capture timestamps.

### 23 Nov 2026 campaign (accepted run is 10)
| Run | Protocol zero (s) | Cue events | v3 verdict | In supplied archives? |
|---|---:|---|---|---|
| E2D_UNSCORED_20261123_01 | 0.0 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_02 | 2.131 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_03 | 0.0 | 0 (no cues) | FAIL | No — local only |
| E2D_UNSCORED_20261123_04 | 0.0 | 0 (no cues) | FAIL | No — local only |
| E2D_UNSCORED_20261123_05 | 2.135 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_06 | 2.168 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_07 | 2.066 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_08 | 0.0 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261123_09 | 0.0 | 14 | FAIL | No — local only |
| **E2D_UNSCORED_20261123_10** | 2.788 | 14 | **PASS** | **Yes — run-10 evidence zip** |

### 25 Nov 2026 confirmation campaign
| Run | Protocol zero (s) | Cue events | v3 verdict | In supplied archives? |
|---|---:|---|---|---|
| E2D_UNSCORED_20261125_01 | 13.207 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261125_02 | 2.299 | 14 | INCONCLUSIVE | **Yes — as RUN_01_BEST** |
| E2D_UNSCORED_20261125_03 | 2.144 | 14 | INCONCLUSIVE | **Yes — as RUN_03_THIRD** |
| E2D_UNSCORED_20261125_04 | 2.072 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261125_05 | 2.725 | 14 | FAIL | No — local only |
| E2D_UNSCORED_20261125_06 | 2.745 | 14 | INCONCLUSIVE (was FAIL under v2) | **Yes — as RUN_02_SECOND** |

Best-3 archive name mapping (as delivered, original IDs stripped from that zip):
- `RUN_01_BEST` = `E2D_UNSCORED_20261125_02`
- `RUN_02_SECOND` = `E2D_UNSCORED_20261125_06`
- `RUN_03_THIRD` = `E2D_UNSCORED_20261125_03`

All failed and inconclusive attempts are preserved locally and none was reclassified as a PASS. Cue accounting for all 25 Nov runs: **12 playback requests + 2 embedded STOP words = 14 semantic cue events**; cue times are software issue times, not measured physical audible onset (documented in each manifest's `clock_limit`).

## 2. H2 executable provenance clarification (point 9)

This is the critical distinction Engineer 4 asked for — **repository checkout identity ≠ executable build identity**:

| Aspect | Value |
|---|---|
| Executable build time (file mtime) | **2026-11-22 16:05:18** |
| Executable SHA-256 | `12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A` |
| Repository commit at 14:24 that day | `654d9d9` (included reduced OpenCV config + preview overlays) |
| Native E2-B validator commit | `55faef2` at **22:01** on 22 Nov — **after** the executable was built |
| Tester input-validation commit | `71b3bba` at **10:07** on 23 Nov — **after** the executable was built |
| Run 10 manifest "source" field | `cb93e12` at 14:03 on 23 Nov — this is the **checkout identity recorded at qualification preparation**, not the commit the exe was built from |

**Conclusion:** the executable predates the committed E2-B and Tester safeguards. Earlier documents that described `12B9CE6B…` as "after E2-B" or associated it with commit `cb93e12` conflated the checkout identity with the build identity. The claim that the run-10 executable "includes the approved input-validation safeguards" is **not established by commit history** and must be treated as unverified unless Engineer 4 independently confirms it (e.g. by rebuilding from a specified clean commit and comparing behavior). This does not change run 10's numerical H1 PASS, but it directly affects H2 authorization.

Other H2 limitations already documented in `H2_EXECUTABLE_CANDIDATE_DOSSIER.md`: two pre-existing local source modifications, no certified clean-build log, no durable executable archive, Release-for-Debug substitution, and the reduced-OpenCV configuration in our branch history (known PNG-sequence regression). All remain open.

## 3. Integration note (point 10)

- PR #3 is merged into `origin/first-full-working-demo` at `f29b3e5` (verified locally via `git fetch`; parent `2201255`).
- **Do not merge** `engineer2/dsp-report-and-preview-overlays` into `first-full-working-demo`: it contains overlapping validation changes (now superseded by merged PR #3) and the excluded reduced-OpenCV `vcpkg.json` (13 added lines vs demo). Our branch should contribute only the unique tools/reports (readiness scorer, capture harness, H1/H2 evidence), selectively and after review.
- H2 candidate selection and launcher changes remain Engineer 4's decision. No launcher edit, scored trial, or real-mouse activation was performed.

No DSP parameter, equation, threshold, protocol rule, or Trial 4 verdict was changed. Run 10 remains the legitimate historical unscored PASS.
