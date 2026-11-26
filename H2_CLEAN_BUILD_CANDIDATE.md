# Phase 3 — Clean logged H2 candidate build from integrated demo

**Date:** 26 Nov 2026. **Status:** candidate prepared, NOT approved by Engineer 4.

## Why this build
Engineer 4 point 9 required distinguishing repository checkout identity from executable build identity. The earlier candidate (`12B9CE6B…`) was built 2026-11-22 16:05, **before** the committed E2-B (22:01) and Tester (23 Nov) safeguards, and its manifest's `cb93e12` was a checkout identity, not the build commit. To provide unambiguous provenance, I built a fresh candidate from the **integrated demo with PR #3 merged**.

## Build provenance (fully logged)
| Item | Value |
|---|---|
| Source worktree | isolated `h2_candidate` worktree at **`f29b3e53f2b655b2ec80e6f407636beafca1e3aa`** (`first-full-working-demo`, PR #3 merged) |
| Build type | **Debug** (`CMAKE_BUILD_TYPE:STRING=Debug` in CMakeCache.txt; the configure command did not set Release) |
| Generator / compiler | Ninja; MSVC `cl.exe` 14.51.36231; Windows SDK 10.0.26100.0 |
| vcpkg | `2026-07-13-bf04c909169fdbb30821c02c6eb01f1cd1295d05`; baseline `40f3c709db80acf154ac4b17a1f83c564ebd022e`; triplet x64-windows |
| OpenCV | broad config (demo `vcpkg.json`: `opencv4`, no reduced features — reduced-OpenCV regression excluded) |
| Configure / total build | 66.1 s / 93.8 s (cached dependencies reused) |
| Full build log | `C:\Users\Doctor Laptop\AppData\Local\Temp\opencode\dsp_audit\h2_build_log.txt` (35,222 bytes) — copied into the archive as `BUILD_LOG.txt` |
| CTest | **8/8 passed** (163.90 s) on this built binary |
| Executable SHA-256 | **`1AA31B9D7E0F9209A8F84F3CAD89C6084DC7F9FDADF8F30DA26EFE20CD173B50`**, size 7,923,200 |
| Model SHA-256 | `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1` |
| Runtime DLLs | debug variants (`opencv_core4d.dll` etc.), hashes in `CANDIDATE_MANIFEST.txt` |

## Durable archive
**`C:\Users\Doctor Laptop\Downloads\H2_CANDIDATE_F29B3E5_CLEAN_BUILD.zip`** (16,397,707 bytes, 21 files): `core_engine.exe`, 18 runtime DLLs, `hand_landmarker.task`, `BUILD_LOG.txt`, `CANDIDATE_MANIFEST.txt`. This satisfies "provide the actual executable or an accessible evidence location" and closes the "no certified build log / no durable archive" gaps for this candidate.

## Important notes
- This candidate is **Debug**, not Release. That may be preferable because the historical pin was Debug, but Engineer 4 must choose; the earlier `12B9CE6B…` Release candidate is NOT superseded by this document.
- CTest was invoked with `-C Release`, but the compiled binaries are Debug; all 8 tests passed on the actual Debug binaries. CTest's config label did not change the built code.
- Safeguards: PR #3 is merged into `f29b3e5` (21+21 validation, nonfinite rejection, Tester `_ValidatingDetector`, gaze head-pose validation) — so this candidate unambiguously includes the committed safeguards.
- No launcher edit, no scored trial, no DSP/threshold/protocol change. Trial 4 unchanged.

## Next
Engineer 4 must (a) approve or reject this candidate, (b) authorize the exact launcher identity change, and (c) independently repeat H1 qualification on this candidate before scored Trial 5.
