# FAST Executable Compatibility Decision Package

**Purpose:** provide DSP Engineer / Codex with an answerable decision on the executable for the next scored FAST trial.  
**Authority:** Engineer 2 independent technical review. This document does **not** grant scored-trial approval.  
**Updated:** 23 Nov 2026 — added the current source/build identity, the proposed launcher diff, the `c61655f` baseline, and the safeguard-provenance issue raised in the DSP Engineer's 23 Nov message.  
**Current Engineer 2 branch:** `engineer2/dsp-report-and-preview-overlays` @ `71b3bba` (local and `origin` refs equal).

---

## 1. Historical executable recovery

**Decision: `NOT FOUND LOCALLY`.**

| Item | Value |
|---|---|
| Target historical executable SHA-256 | `3A3F05472A441E225006A276D03297A38135F13827E83F1B13CA278E46167200` |
| Current Debug SHA-256 | `C7A4CAAB4C060CDEFEFB7BC8E9312B6564B81974C343E7F899F0A6EFFC583D79` |
| Current Release SHA-256 | `12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A` |

The completed local search found:

- The target historical SHA-256 appears only in documentation, the scored launcher, and a test constant — not as a matching executable.
- No matching executable exists in local files, Git history, or available archives.
- Commit `6354a143964d7f4aa8f1957bed0a2eed59e37341` exists locally, but no executable artifact is attached to that commit.
- The historical branch exists locally at `4f61ab24...` (`origin/codex/engineer4-dsp-recorder`), but contains no matching binary.
- The only current executables are:
  - `out/build/windows-debug/core_engine.exe` with SHA-256 `C7A4CAAB...D79`;
  - `out/build/windows-release/core_engine.exe` with SHA-256 `12B9CE6B...9A`.

Recovering the exact historical executable from current local evidence is therefore not feasible.

**New information from the DSP Engineer's 23 Nov 2026 message:** Codex reportedly found the historical binary on **Engineer 4's own laptop**, but that binary **predates the latest hand-input safeguards**. This does not change the `NOT FOUND LOCALLY` finding for this environment (that laptop is outside the searched environment), and it does **not** make the binary acceptable: the DSP Engineer now requires the selected build to include the approved input-validation safeguards. Whether that older binary can be used is a written Engineer 4 / Codex decision, not an Engineer 2 assumption.

---

## 2. Current source and build identity for approval

### Source baseline

| Item | Value |
|---|---|
| Candidate branch | `engineer2/dsp-report-and-preview-overlays` |
| Candidate HEAD | `71b3bba` (commits `654d9d9` → `55faef2` → `71b3bba`) |
| Verified DSP baseline | `origin/first-full-working-demo @ c61655f` |
| Engineer 1 validator already in demo | `69c84c8` (`240f806` removes conflict markers from that cherry-pick) |
| Launcher branch pin (unchanged) | `codex/engineer4-dsp-recorder` |

Engineer 1's `69c84c8` already implements 21+21 / non-finite validation in the demo. The candidate branch adds equivalent native defense-in-depth (`handPayloadValid()`) plus the Tester-path fix in `71b3bba`. **No competing validator was merged into the demo.**

### Executable identity (measured 22 Nov 2026, before commit `71b3bba`)

| Build | Path | SHA-256 |
|---|---|---|
| Debug (recommended candidate type) | `out/build/windows-debug/core_engine.exe` | `C7A4CAAB4C060CDEFEFB7BC8E9312B6564B81974C343E7F899F0A6EFFC583D79` |
| Release | `out/build/windows-release/core_engine.exe` | `12B9CE6B53490BB0494C99D9E3763ED74C52670209AB9627C6D80E65A056AB9A` |

**Important:** these hashes predate commit `71b3bba`. A rebuild from the current source will produce a different SHA. Do not assume the values above match a newly built executable — the approved SHA must be recalculated from the exact artifact archived with the trial.

### Complete scored launch command (fixed by the launcher, unchanged)

```
core_engine.exe --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 55 --dsp-record-dir <record_dir>
```

Recorder branch/commit pin in the original sessions: `codex/engineer4-dsp-recorder @ 6354a143964d7f4aa8f1957bed0a2eed59e37341`; original protocol SHA-256 `DFEF9DA999D513C3C04E8D2E873D0A468B07BAF3237F8CE1A2A4CDDEA2E03FE9`.

### Exact proposed launcher identity diff — PROPOSAL ONLY, NOT APPLIED

```diff
  # dsp_filter/evaluation/run_cued_responsiveness.py
- EXPECTED_BRANCH = "codex/engineer4-dsp-recorder"
+ EXPECTED_BRANCH = "engineer2/dsp-report-and-preview-overlays"

- EXPECTED_EXE_SHA256 = "3A3F05472A441E225006A276D03297A38135F13827E83F1B13CA278E46167200"
+ EXPECTED_EXE_SHA256 = "<SHA-256 of the rebuilt, approved executable>"
```

The executable path at `run_cued_responsiveness.py:702` (`out/build/windows-debug/core_engine.exe`) is deliberately left unchanged if Debug is approved. **No launcher file was modified while preparing this package.**

---

## 3. OpenCode independent recommendation

**Recommendation: `OPTION C — REVIEW A NARROWLY SCOPED LAUNCHER IDENTITY ADJUSTMENT`.**

- **Option A — recover the original executable:** not locally feasible based on the completed search.
- **Option B — silently approve a current executable:** unsafe. It would substitute a new branch, source baseline, and executable hash without explicit scored-protocol review.
- **Option C — narrowly scoped launcher identity adjustment:** safest technically supported path. It preserves the original scored protocol while making the executable identity change explicit and reviewable.

If a current executable is approved, **Debug is the recommended candidate for external review** because it preserves the historical Debug build type and executable path (`out\build\windows-debug\core_engine.exe`). Release would introduce an additional Debug-to-Release substitution.

**Debug is NOT approved yet. Release is NOT approved.**

This is OpenCode's technical recommendation, **not** DSP Engineer approval.

---

## 4. Exact external decision required

Written decisions are required from the DSP Engineer / Codex:

1. **Acceptance that the historical executable is unavailable.**  
   Confirm that the completed local search is sufficient and that Option A is closed.

2. **Approval or rejection of the proposed current branch.**  
   Current branch: `engineer2/dsp-report-and-preview-overlays`.  
   Launcher currently expects: `codex/engineer4-dsp-recorder`.

3. **Approval or rejection of the exact Debug executable SHA-256.**  
   Candidate: `C7A4CAAB4C060CDEFEFB7BC8E9312B6564B81974C343E7F899F0A6EFFC583D79`.

4. **Authorization for the exact launcher identity changes.**  
   Authorize or reject changing:
   - `EXPECTED_BRANCH`;
   - `EXPECTED_EXE_SHA256`.  
   If Debug is approved, the executable path can remain `out/build/windows-debug/core_engine.exe`.

5. **Confirmation that nothing else changes.**  
   Confirm that the original scored protocol, DSP equations, DSP parameters, timing, thresholds, and acceptance criteria remain unchanged.

6. **Decision on source baseline.**  
   The E2-B and Tester-path changes are now committed: `55faef2` and `71b3bba` on `engineer2/dsp-report-and-preview-overlays`. Confirm whether this branch/HEAD is an acceptable scored source baseline, or whether the approved build must instead come from `first-full-working-demo @ c61655f` (which already contains Engineer 1's validator `69c84c8`).

7. **Decision on the Engineer 4 laptop binary.**  
   The reported historical binary predates the hand-input safeguards. Confirm explicitly whether it is rejected on that basis, or accepted with stated limitations. Do not treat its recovery alone as satisfying the safeguard requirement.

8. **Rebuild and re-hash requirement.**  
   The hashes in section 2 were measured before commit `71b3bba`. If a current build is approved, require a reproducible rebuild from the approved commit, a fresh SHA-256 of the exact artifact, and archival of that artifact with the trial. Insert that fresh SHA into the proposed `EXPECTED_EXE_SHA256` diff in section 2.

9. **Authorization for the exact proposed diff.**  
   Approve or reject the two-line `EXPECTED_BRANCH` / `EXPECTED_EXE_SHA256` change shown in section 2. No other launcher line is proposed for change.

**No launcher changes are authorized until this approval is received.**

---

## 5. Remaining FAST readiness blockers

### Executable and launcher

1. Executable / branch / SHA approval.
2. Launcher identity authorization.

### Before a scored trial

3. New Trial ID. Trial 4 must remain INCONCLUSIVE and unchanged.
4. Required physical setup metadata for the new trial.

### During the scored trial

5. Original DSP numerical trace from `--dsp-record-dir`.
6. Recorder integrity verification.
7. Paired-data import.
8. Cue/timing log.
9. Console summary.
10. Configuration manifest.
11. Preservation of all attempts, including failed or inconclusive attempts.

### After the scored trial

12. Independent DSP mathematical replay.
13. Separate evaluation of the outward and return legs under the original protocol.
14. Final acceptance decision under the original criteria.

---

## 6. Final status

- **No executable is currently approved.**
- **No new scored FAST trial is authorized.**
- **Trial 4 remains INCONCLUSIVE and unchanged.**
- **Unscored Phase 4 remains `CLOSED WITH DOCUMENTED LIMITATIONS`.**
- **No further Engineer 2 investigation is required before requesting the external decision.**  
  The remaining work is a reproducible rebuild + fresh SHA-256 of the approved artifact, which must be done and recorded by the party preparing the scored trial; it is not an open-ended investigation.

No scored recording is authorized by this document.
