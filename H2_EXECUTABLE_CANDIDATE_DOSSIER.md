# H2 — Scored FAST executable candidate dossier (Engineer 2 proposal, NOT approved)

**Date:** 25 Nov 2026. **Status:** 🔴 H2 BLOCKED — this is one exact candidate for Engineer 4 / Codex review. Nothing here authorizes a scored trial, a launcher change, or a merge.

## 1. Proposed candidate identity

| Field | Value |
|---|---|
| Source commit (exact source used for run 10) | `cb93e127324009a71960a0b4ca69261e297c20ec` |
| Current branch HEAD | `0fa85968c2ae98e752a1eedd9d606ab446359c0e` |
| Branch identity | `engineer2/dsp-report-and-preview-overlays` |
| Commits after run-10 source commit | `393e834` (add exact scorer), `0fa8596` (fix four scorer defects) — **tools/docs/reports only; no production source change** |
| Build type / path | Release, `out\build\windows-release\core_engine.exe` |
| Executable SHA-256 (verified 25 Nov) | `12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A` |
| Executable size | 4,583,936 bytes |
| Executable location | `C:\Users\Doctor Laptop\Downloads\HandEye_ARVR_Tracker\out\build\windows-release\core_engine.exe` (accessible on this machine; not a durable archive) |
| Model SHA-256 | `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`, 7,819,105 bytes |

## 2. Clean working-tree confirmation — ⚠️ NOT CLEAN

`git status --short` still shows two **pre-existing user modifications**:
- `ai_pipeline/hand_kinematics/hand_data_collector.py`
- `ai_pipeline/hand_kinematics/test_hand_tracker.py`

These were present before our work and were never touched. They must be committed/stashed by their owner before a scored build is authorized, or explicitly ruled irrelevant by Engineer 4. **This is a genuine H2 prerequisite.**

## 3. Build configuration, compiler and dependencies

| Item | Value |
|---|---|
| Preset / generator | `windows-release` / Ninja |
| Compiler | MSVC `cl.exe` 19.51.36248 (`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`) |
| Windows SDK | 10.0.26100.0 |
| vcpkg baseline | `40f3c709db80acf154ac4b17a1f83c564ebd022e` (shallow checkout at `C:\vcpkg`) |
| vcpkg version | `2026-07-13-bf04c909169fdbb30821c02c6eb01f1cd1295d05` |
| vcpkg triplet | `x64-windows` |
| OpenCV | 4.12.0 (broad feature set in the demo baseline; **note:** our development branch's reduced `vcpkg.json` is a known regression and is **excluded** from PR #6 and must not be used) |
| Python | `.venv\Scripts\python.exe` (project venv), Python 3.11.x |
| DSP | `CORE_HAVE_DSP_FILTER=1` (Engineer 4's `dsp_filter`, unchanged) |

## 4. Clean-build command (exact)

```bat
cd C:\Users\Doctor Laptop\Downloads\HandEye_ARVR_Tracker
git checkout cb93e127324009a71960a0b4ca69261e297c20ec   REM in a clean worktree/branch, not this one
cmake --preset windows-release
cmake --build out\build\windows-release --target core_engine
```

**Complete build logs:** not available as a single archived clean-build log for this exact commit. Earlier session logs exist in the OpenCode shell-output directory but are not a certified clean-build record. Engineer 4 must require a fresh logged clean build before authorization. The exact-head PR #6 build we performed (`pr6_clean_build`) is a different source commit and is **not** the H2 candidate.

## 5. Runtime DLL and model hashes (Release executable directory)

| DLL | SHA-256 |
|---|---|
| `jpeg62.dll` | `A6A8EBB5FA1E89E27349DEA861064CD39EA572B99E11743FECFDDE831013E924` |
| `opencv_core4.dll` | `097EDA8360F50E27EACAF458CFAF234B792A39F8CBB150A6A5F3998688019FE1` |
| `opencv_highgui4.dll` | `8D610F2756FAE94A12FEDA1E9FCA218B220DDDE908FCB94CF9EFCD756019756E` |
| `opencv_imgcodecs4.dll` | `6507F96766B66ECFC6DFFDF7DAA1B0059F4616320E605F3970766AB43F7E027C` |
| `opencv_imgproc4.dll` | `14515F51E7508E7F3DE4C54D095BA576BEAD0BBD01C95D76A667136120CBA754` |
| `opencv_videoio4.dll` | `1F42FA364ACE8D78E0B38C20D49027A4454F6234075C4388FAE99EC42802A0A5` |
| `python3.dll` | `DB84BC052CFB121FE4DB36242BA5F1D2C031B600EF5D8D752CF25B7C02B6BAC6` |
| `python311.dll` | `0817A2A657A24C0D5FBB60DF56960F42FC66B3039D522EC952DAB83E2D869364` |
| `z.dll` | `09BAEF852417247407EE5BF7DC734D669A496295F8AB83243C7137C1963059E4` |

## 6. Required safeguards present

- E2-B input validation: Python 21+21 / finite landmark packing and native `n==42` + all-126-float finite check (commits `55faef2` in our branch; PR #3 preserves the equivalent demo contract).
- Tester path validation: `hand_tracker.py::_parse()` rejects malformed/nonfinite coordinates (commit `71b3bba`).
- Input validation before publication/DSP: ineligible rows never become fresh coordinates; DSP parameters untouched (hand `1.2,4,1`; gaze `0.8,6,1.5`).
- Run 10 used exactly this Release executable (source commit `cb93e12`, SHA `12B9CE6B…`) and its manifest records the command, cues, physical setup and full trace.

## 7. Exact proposed launch command (matching run 10 minus unscored preview flags)

```bat
core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <NEW_SCORED_TRIAL_DIR>\recording
```

For strict scored use, drop `--preview --target-zone left` (they were preview-only and drawing-only in run 10). The audio cue schedule and manifest must come from Engineer 4's unchanged launcher, not from our unscored harness.

## 8. Proposed launcher identity change — PROPOSAL ONLY, NOT APPLIED

```diff
- EXPECTED_BRANCH = "codex/engineer4-dsp-recorder"
+ EXPECTED_BRANCH = "engineer2/dsp-report-and-preview-overlays"

- EXPECTED_EXE_SHA256 = "3A3F05472A441E225006A276D03297A38135F13827E83F1B13CA278E46167200"
+ EXPECTED_EXE_SHA256 = "12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A"
```
The executable path (`out/build/windows-debug/core_engine.exe`) would need review too, because the candidate is a **Release** executable in `out\build\windows-release\` — the launcher currently points at Debug. Engineer 4 explicitly warned against silent Debug→Release substitution. The launcher was **not modified**.

## 9. Can H1 qualification transfer to another executable identity?

**Not by default.** `RESPONSIVENESS_PROTOCOL.md:21` says to "Remain on `codex/engineer4-dsp-recorder` and use the verified executable." Run 10 was recorded with branch `engineer2/dsp-report-and-preview-overlays` / Release `12B9CE6B…`. Therefore:
- H1 run 10 is valid **unscored qualification evidence for the input path**, and the DSP Engineer has independently accepted it;
- it does **not** automatically qualify a different executable identity for scored use;
- Engineer 4 must either (a) explicitly authorize this candidate identity (branch/SHA/path/safeguards) for the scored protocol, or (b) require a repeated unscored H1 qualification using the exact proposed scored executable/configuration. Our recommendation: if authorization is not explicit, repeat H1 on the exact candidate before Trial 5.

## 10. Remaining risks / limitations

- No clean working tree (two pre-existing user files); no certified clean-build log for the candidate commit; executable only exists locally, not in a durable archive.
- Release-for-Debug substitution requires explicit Engineer 4 authorization.
- Historical binary `3A3F0547…` predates the input safeguards and was not recovered locally.
- Trial 5 is not authorized; no scored recording is authorized by this dossier.
