# Measuring live hand and gaze coordinates

For the separate slow and fast hand experiments, follow the
[audio-guided responsiveness protocol](RESPONSIVENESS_PROTOCOL.md). Use
`run_cued_responsiveness.py rehearse --session SESSION_DIRECTORY` to hear a
prepared sequence and practice both directions with the camera closed.

`import_vs_trace.py` and `evaluate_holds.py` compare the input and output of the
C++ DSP on **the same observation** using the project's Python 3.11 standard
library. Production code has no dependency on either tool. The first deliverable
from a run is the paired CSV. Trial labels are separate so
that the same observations can be scored again under the same protocol.

## Capture one tracker with the optional recorder

The recorder is based on working-demo commit `6354a14`, including `839dcd6`'s
precise capture clock and filter-outcome handling. It is disabled unless
`--dsp-record-dir NEW_DIRECTORY` is supplied. It requires `--filter dsp`,
`--os log`, and exactly one tracker (`--no-gaze` for hand, `--no-hand` for gaze).
It cannot run with `--check`, forest self-tests, disabled filtering or real-mouse
output. Keep the tray paused. Do not change filter settings during a recording.

Agree on the channel, participant task, predeclared label windows and build first.
**A camera trial requires the user's explicit approval.** These instructions are
a reproducible procedure, not permission to launch a camera during code review.
Run the approved trial without the debugger or tracepoints, using the known
working runtime environment. Record the source commit, build configuration,
startup filter line, camera mode, channel and trial ID outside Git. There is no
new live quality result until a complete approved recording has been collected
and scored.

The parent data folder must already exist outside the repository. The recorder
atomically reserves a **new** run directory and refuses to reuse an existing
directory, including an empty one. Example commands from the repository root,
after approval; use one command for the chosen channel and a new name each run:

~~~powershell
$dataRoot = 'D:\HandEyeEvaluation' # An existing local folder outside Git
$run = Join-Path $dataRoot ('hand-center-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
.\out\build\windows-debug\core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 30 --dsp-record-dir $run
if ($LASTEXITCODE -ne 0) { throw 'Retain the run and console summary; investigate before scoring.' }

# Separate approved gaze trial, after choosing its task and label windows:
$run = Join-Path $dataRoot ('gaze-center-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
.\out\build\windows-debug\core_engine.exe --camera 0 --filter dsp --os log --no-hand --gaze-filter 0.8,6,1.5 --seconds 30 --dsp-record-dir $run
if ($LASTEXITCODE -ne 0) { throw 'Retain the run and console summary; investigate before scoring.' }
~~~

Before capture, 4096 frame slots are preallocated and the directory receives an
`INCOMPLETE` marker. Each submitted frame contributes one numerical row for the
selected channel, including Held, Lost and calibration observations. Append does
no allocation, formatting, locking, clock query or file I/O. At capacity tracking
continues but the recording becomes incomplete; use shorter trials if needed.
The recorder captures no camera images or video. It records the point actually
passed to `TrayApp::submit_frame`, which is receiver submission rather than a
mouse-action acknowledgement.

After orderly shutdown joins the capture/AI threads, drains results, stops the
OS bridge and releases the camera, it writes `trace.partial`, publishes
`trace.txt`, and finally publishes `COMPLETE` and removes `INCOMPLETE` on success.
Never repair a failed run by deleting markers or editing the trace. Import
requires all of the following:

- `COMPLETE` is present and `INCOMPLETE` is absent beside `trace.txt`.
- The v2 header and footer are present; the footer has equal positive submitted,
  attempted and stored counts, at most 4096, `complete=1`, `overflow=0`, and
  `config_changes=0`.
- Every stored observation passes the timestamp, detector, adapter and
  publication consistency checks.

Overflow, truncation, write failure, empty capture, count mismatch or a filter
configuration change invalidates the run. A marker alone is insufficient. Keep
the entire run directory, console shutdown summary and labels together; do not
copy only `trace.txt` into a folder without its completion evidence.

### Import and score a completed recorder run

The importer auto-detects `E4REC` v2. It uses the footer's submission count;
optional `--submitted N` must agree with it. No camera, debugger, inference or
OS action is involved in these offline commands:

~~~powershell
$python = '.\.venv\Scripts\python.exe'
$tools = '.\dsp_filter\evaluation'
$run = Read-Host 'Path to the completed local trial folder'
$channel = 'hand' # Change to gaze for a gaze recording
$paired = & $python -B "$tools\import_vs_trace.py" "$run\trace.txt" --channel $channel
if ($LASTEXITCODE -ne 0) { throw 'Import failed. Retain the trace and completion markers unchanged.' }
New-Item -ItemType File -Path "$run\paired.csv" -Value ($paired -join [Environment]::NewLine) -ErrorAction Stop | Out-Null
$coverage = & $python -B "$tools\evaluate_holds.py" "$run\paired.csv"
if ($LASTEXITCODE -ne 0) { throw 'Coverage evaluation failed.' }
New-Item -ItemType File -Path "$run\coverage.json" -Value ($coverage -join [Environment]::NewLine) -ErrorAction Stop | Out-Null
$report = & $python -B "$tools\evaluate_holds.py" "$run\paired.csv" --holds "$run\holds.csv"
if ($LASTEXITCODE -ne 0) { throw 'Hold evaluation failed.' }
New-Item -ItemType File -Path "$run\holds-report.json" -Value ($report -join [Environment]::NewLine) -ErrorAction Stop | Out-Null
& $python -B -m unittest discover -s $tools -p 'test_*.py' -v
~~~

These commands refuse existing analysis outputs; choose fresh filenames when
repeating an analysis. Prepare `holds.csv` from
the predeclared physically stationary intervals described below. Run the
unlabelled coverage command first; a complete recording can still contain too
few eligible neutral samples for a jitter result.

### Recorder v2 clocks, columns and outcomes

The 26-column CSV preserves the original 12 columns and the three hand freshness
columns, then appends 11 recorder fields. Both hand and gaze recorder runs use
this schema. Legacy 12- and 15-column debugger files remain supported unchanged.

| Columns | v2 meaning |
| --- | --- |
| `frame_id,channel,observation_ms,capture_ms` | Submitted frame identity and selected channel; `observation_ms` retains MediaPipe's strictly increasing integer milliseconds, while `capture_ms` is the truncated host capture time. Neither integer is the v2 DSP clock. |
| `landmarks,state,published` | Actual packed detector return count, upstream gesture/calibration state, and receiver-bound point present (0/1). Fresh hand has 42 rows; usable gaze has 484. Detector errors use −1. |
| `raw_x,raw_y,filtered_x,filtered_y,status` | Actual clamped DSP input when an update ran, receiver-bound output when published, and actual DSP status or a bridge event (`Held`, `HeldNoCache`, `Lost`, `Bypassed`). Missing coordinates stay empty. |
| `tracking_state,missed_frames,published_tracking` | Upstream freshness (`Unavailable` for gaze), hand miss count and actual receiver-bound freshness. Gesture state stays separate. |
| `dsp_timestamp_seconds` | Actual precise `capture_s` passed to DSP and the OS payload, serialized with round-trip precision; not MediaPipe milliseconds divided by 1000. |
| `adapter_called,dsp_called,adapter_outcome` | Adapter call and underlying DSP update flags (0/1), and `NotCalled`, `Ok`, `Rejected` or `Invalid`. A nonfinite guard can call the adapter without updating DSP. |
| `reset_count,filter_generation` | Observed adapter reset calls in this frame, including its nonfinite guard, and the filter-configuration generation. Library-internal invalid/numerical resets are also identified by status/outcome. An accepted recording remains at generation 1. |
| `tracker_ok,tracker_x,tracker_y` | Upstream result before the adapter clamp or publication gate, including sentinels needed for diagnostics. |
| `dsp_x,dsp_y` | Actual optional DSP return, empty when absent. This distinguishes an accepted result from a cached publication. |

`capture_s` comes from the original host `steady_clock` timestamp immediately
after the camera read, relative to pipeline start. MediaPipe milliseconds may be
bumped by 1 ms to stay strictly increasing. The importer retains both values and
pairs records by frame; it does not reconstruct precise time from milliseconds.
For v2, cadence, wash-in and labels use `dsp_timestamp_seconds * 1000`, and the
JSON report identifies `measurement_clock` as `precise_capture_seconds`.
Hardware exposure time is unavailable.

| Observation | Verified v2 publication behavior |
| --- | --- |
| Accepted `Initialized`, `Filtered` or `ReinitializedAfterGap` | Outcome `Ok`; actual DSP return equals the published point, marked Tracked. Initialization output equals actual input. |
| Fresh input with `RejectedTimestamp` | Outcome `Rejected`; DSP history is retained. A previous cached point is published as Held, otherwise there is no point and publication is Lost. The status remains `RejectedTimestamp`. |
| Invalid/numerical DSP result or nonfinite adapter bypass | Outcome `Invalid`; no point is published and cache/history are cleared. A bypass has no DSP update and status `Bypassed`. |
| Upstream hand Held | Neither adapter nor DSP update runs. The gate publishes its cached filtered point as Held, or no point as Lost (`HeldNoCache`). |
| Lost or gaze calibration | No adapter update or point. Established loss resets the adapter once; calibration remains identifiable by state −1. The next usable observation initializes directly. |

The importer rejects stale per-frame diagnostics, inconsistent actual input or
DSP/publication output, invalid cache reuse, impossible status/outcome pairs and
malformed or incomplete records. It validates the contract; it does not perform
an independent numerical replay of the filter equations. The scorer revalidates
v2 row metadata before calculating metrics.

No record includes DSP execution duration. Use the standalone Release
`dsp_trace_recorder_benchmark.exe` for the measured cost of enabling instrumentation
on a synthetic adapter/gate workload. Its preallocation and post-run serialization
are outside the timed region. That result does not establish live cadence impact,
worst individual call latency or camera-to-cursor delay. Compare approved live
recorder-enabled/disabled trials separately before making those claims.

### Verify the integration with synthetic video

`check_recorder_pipeline.py` drives the built executable through the actual
`main` and `OsBridge` recorder path using the repository's existing synthetic
video and fake-AI test fixtures. It always selects `--video --filter dsp --os log`
with one channel per run and never opens a webcam. Run it with the established
project environment, which supplies the existing fixture dependencies, including
OpenCV and the stand-in model trainer. Existing readable saved tray settings are
required; the verifier refuses first-run settings creation rather than launching
with a missing settings file:

~~~powershell
.\.venv\Scripts\python.exe -B .\dsp_filter\evaluation\check_recorder_pipeline.py --exe 'D:\GitHub\HandEye_ARVR_Tracker\out\build\windows-debug\core_engine.exe'
# Optional: choose an existing external parent for a newly created artifact folder.
.\.venv\Scripts\python.exe -B .\dsp_filter\evaluation\check_recorder_pipeline.py --exe 'D:\GitHub\HandEye_ARVR_Tracker\out\build\windows-debug\core_engine.exe' --output-parent 'D:\HandEyeEvaluation'
~~~

Use one command; supply the absolute path of the executable under review. The
default output parent is the system temporary directory. The harness saves the
synthetic video/model, exact commands, console logs, recorder output, paired CSV,
unlabelled coverage JSON and validation report in a new directory. It checks
real shutdown/submission counts, frame-pool return, zero Python errors, successful
CLI import of actual recorded input/output pairs and unchanged saved tray settings.
Settings are verified before and after every executable run, including failed
runs; a change stops verification and is never silently restored. The real tray
may append its ordinary LocalAppData log outside the artifact directory.
It also attempts the same output directory again and requires rejection with the
original trace hash preserved. It does not create stationary labels or report
live jitter, response delay or camera performance.

### Verified recorder checks, 2026-11-18

Root Debug and Release each passed all seven CTest suites (115.69 and 64.58
seconds respectively), including 13 recorder cases with 4,520 checks. The offline
Python suite passed 75 tests in 1.005 seconds, including 13 mocked verifier-guard
cases. Final synthetic-video runs through the real executable and CLI importer
retained every submitted observation:

| Build / channel | Submitted and imported rows | Lost | Initialized | Filtered | Held |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debug / hand | 324 | 83 | 1 | 234 | 6 |
| Release / hand | 327 | 83 | 1 | 237 | 6 |
| Debug / gaze | 329 | 60 | 1 | 268 | 0 |
| Release / gaze | 329 | 60 | 1 | 268 | 0 |

Each run captured 329 synthetic input frames. Debug hand discarded 5 stale
frames before DSP and Release hand discarded 2; gaze discarded none. These are
upstream discards, not missing recorder observations. Every run had zero Python
errors, returned all 8/8 pool slots, preserved saved tray settings, and rejected
output-directory reuse without changing the original trace hash. These runs
verify integration and complete pairing; their throughput does not measure live
FPS, recorder overhead or stationary jitter. Final console/test logs are retained
outside Git as `debug-recorder-pipeline-final.log`,
`release-recorder-pipeline-final.log` and `python-tests-final.log`.

On the i7-1255U/MSVC 19.51 Release build, 31 alternating 4,096-frame benchmark pairs
measured mean enabled/disabled costs of 282.9810/221.0654 ns per frame, a paired
increase of 61.9156 ns (28.0078%); output checksums matched exactly. This is the
synthetic adapter/gate-and-recorder workload, excluding AI-thread metadata/queue
cost, startup/preallocation, serialization, disk I/O, camera and OS calls. Batch
averages do not establish live FPS, worst-call time or end-to-end latency. See
the [full build and benchmark results](../README.md#recorder-verification-2026-11-18).

## Historical capture in Visual Studio 2026

The procedure and tracepoint expressions below describe the older `3ea3ce3`
clock/adapter contract and preserve reproducibility of existing captures. They
are not the current recorder instructions; the legacy importer assumes the old
milliseconds-derived DSP time. Source line numbers are historical and must not
be reused on the precise-clock base without revising the trace contract.

**Capture contract at `3ea3ce3`.** Hand tracking distinguishes Tracked, Held
and Lost. Use the complete `E4HAND … E4DSP P hand` message on every publication.
This records freshness even when the core reuses a cached point or publishes
an empty point. Keep all three kinds of observation in the capture.

1. Agree on the trial with the participant first. Use one tracker process, the existing Debug
   executable and matching native symbols. Choose one channel per run so the adapter instance
   is unambiguous. Record the commit, build, startup filter line, camera mode, trial ID, task,
   and whether tracepoints are active. Store numerical output in a local trial folder outside Git.
2. From the repository root in a VS Developer PowerShell, open the executable for debugging.
   For the hand pilot use the first command below. For a separate gaze pilot use the second.
   Use the known working runtime environment. Leave the OS stage in logging mode and the tray
   paused. The filter flags below override saved parameters for this process only.

   ~~~powershell
   devenv /DebugExe .\out\build\windows-debug\core_engine.exe --camera 0 --filter dsp --os log --no-gaze --hand-filter 1.2,4,1 --seconds 30
   devenv /DebugExe .\out\build\windows-debug\core_engine.exe --camera 0 --filter dsp --os log --no-hand --gaze-filter 0.8,6,1.5 --seconds 30
   ~~~

3. Before starting execution, open the source files in the native debugger. In the left
   gutter, add the three tracepoints below for your chosen channel. Under breakpoint
   **Actions**, enter each message as one line and select **Continue execution**. Disable
   hexadecimal display. Use **View → Output → Debug**. Clear the output before each run.
   Launch with F5 only when the participant is ready. Verify the executable, PDB and bound
   sources match. Leave the points enabled through the final submitted frame.
   [Visual Studio tracepoints](https://learn.microsoft.com/en-us/visualstudio/debugger/using-tracepoints)
   · [Starting an executable under the debugger](https://learn.microsoft.com/en-us/visualstudio/ide/reference/debugexe-devenv-exe)

**Detector N** — at the assignment after the landmarker call, where `n` is already set.
Choose the corresponding message and line in `core_engine/src/main.cpp`.

~~~text
Line 454, r.hand = (n >= 0) ? handn.process(hand_pts.data(), n) : hand_failed;
E4DSP N hand frame={r.frame_id} ms={ts} cap={pkt.ts_ms} n={n} other={cfg.use_gaze}

Line 432, r.gaze = ...
E4DSP N gaze frame={r.frame_id} ms={ts} cap={pkt.ts_ms} n={n} other={cfg.use_hand}
~~~

**Adapter A** — at `if (r.point)`, line 33 of `core_engine/include/dsp_adapter.hpp`.
Both channels use this message.

~~~text
E4DSP A t={t_sec} rx={rx} ry={ry} status={(int)r.status} has={r.point._Has_value} dx={r.point._Has_value ? r.point._Value.x : 0.0} dy={r.point._Has_value ? r.point._Value.y : 0.0}
~~~

**Publication P** — at `tray_->submit_frame(f)`, line 99 of `core_engine/include/os_bridge.hpp`.
Select one message. The guards read optional coordinates only when a point exists.

~~~text
E4HAND trk={r.hand.tracking_state._Has_value ? (int)r.hand.tracking_state._Value : -1} missed={r.hand.missed_frames} out_trk={(int)f.hand.tracking} E4DSP P hand frame={r.frame_id} ms={r.ts_ms} ok={r.hand.ok} x={(double)r.hand.x} y={(double)r.hand.y} s={r.hand.state} pub={f.hand.point._Has_value} fx={f.hand.point._Has_value ? f.hand.point._Value.x : 0.0} fy={f.hand.point._Has_value ? f.hand.point._Value.y : 0.0}

E4DSP P gaze frame={r.frame_id} ms={r.ts_ms} ok={r.gaze.ok} x={(double)r.gaze.x} y={(double)r.gaze.y} s={r.gaze.state} pub={f.gaze.point._Has_value} fx={f.gaze.point._Has_value ? f.gaze.point._Value.x : 0.0} fy={f.gaze.point._Has_value ? f.gaze.point._Value.y : 0.0}
~~~

For hand, `trk` is the upstream freshness, `missed` is the upstream miss counter,
and `out_trk` is the freshness forwarded to Engineer 5. Both enums use 0 for
Tracked, 1 for Held and 2 for Lost. The trace uses −1 for an unset upstream
optional. The gesture integer is still recorded separately as `s`. Read all
fields from this one publication tracepoint. When a cached point is available,
expect Held / Held, a published point, and no A record. After a cache clear,
Held / Lost with an empty publication and no A is possible. Every fresh finite
hand observation that enters DSP needs its own A record at the exact observation
time, even if X/Y happen to match the preceding frame.

The optional member names above are specific to this MSVC Debug build. Check the pilot for
unresolved expressions, truncated messages, and full-precision doubles (up to 17 significant
digits). Follow the named statements if line numbers change. Save the complete Debug output
as text and the console shutdown summary separately. Remove the trial's tracepoints afterwards.
The offline tests cover both publication formats. A real debugger pilot validated
28 paired hand records at approximately 5.36 Hz. Its scheduling overhead and
state 1 (pinch) observations make it unsuitable for the neutral stationary jitter
protocol. This validated capture/pairing, not live jitter or recorder overhead.

### Convert a historical debugger capture

Example PowerShell, from the repository root. Substitute the saved paths and the actual
shutdown count for that run. The importer can also read the copied `os stage: N frames submitted`
summary directly from the trace text. Use new output filenames for each trial.

~~~powershell
$python = '.\.venv\Scripts\python.exe'
$tools = '.\dsp_filter\evaluation'
$run = Read-Host 'Path to the local trial folder'
$submitted = Read-Host 'OS frames submitted at shutdown'
& $python -B "$tools\import_vs_trace.py" "$run\debug.txt" --channel hand --submitted $submitted | Set-Content -Encoding utf8 "$run\paired.csv"
if ($LASTEXITCODE -ne 0) { throw 'Trace import failed. Review the error and retain the original trace.' }
& $python -B "$tools\evaluate_holds.py" "$run\paired.csv"
& $python -B "$tools\evaluate_holds.py" "$run\paired.csv" --holds "$run\holds.csv"
& $python -B -m unittest discover -s $tools -p 'test_*.py' -v
~~~

Use `--channel gaze` for a gaze run. N and P join by frame ID and corrected observation
milliseconds. A joins by exact binary64 observation seconds within the same run.
Frame IDs can skip. Every submitted frame must have a detector and publication record.
Duplicate records, ambiguous times, unmatched calls, input mismatches and accepted DSP
results that disagree with the published point fail the import. A hand recording with
the freshness prefix produces 15 CSV columns. Older unprefixed hand recordings and gaze
recordings use the original 12. Use a consistent publication message throughout a run.

| Columns | Interpretation |
| --- | --- |
| `frame_id,channel,observation_ms,capture_ms` | Per-run frame ID, channel, historical DSP time in milliseconds, original host timestamp in milliseconds immediately after the camera read. Sensor exposure time is unavailable. This older core supplied `observation_ms / 1000.0` to DSP. |
| `landmarks,state,published` | Packed detector row count, upstream state, downstream point present (0/1). Hand is fresh at 42 and can be retained at 0. Gaze is fresh at 484. A Python call error yields a negative count. Gaze state −1 identifies calibration. |
| `raw_x,raw_y,filtered_x,filtered_y,status` | Clamped adapter input and OS-bound output for the same frame. On accepted DSP statuses, the importer verifies the output against the actual DSP return. Rejections can publish a fallback. The raw fields are empty whenever no numerical input entered DSP. |
| `tracking_state,missed_frames,published_tracking` | Three additional hand columns for prefixed recordings. They preserve upstream freshness, the upstream miss counter, and the freshness delivered to Engineer 5. Gesture `state` remains a separate field. |

| Upstream hand / publication | What the importer verifies and retains |
| --- | --- |
| Tracked / Tracked | For a fresh finite point, the detector, adapter input/status and published X/Y must match the frame. The gate caches the published X/Y. |
| Held / Held | No adapter update. The published X/Y must equal the last cached publication. The row keeps those output coordinates, empty raw fields and status `Held`. |
| Held / Lost | No adapter update or published point. The row keeps empty coordinates and status `HeldNoCache`. A later Held publication requires a new cached point. |
| Lost / Lost | No adapter update or published point. The row keeps empty coordinates and status `Lost`. Loss clears the cache used for comparison. |

These three Held/Lost row statuses describe bridge events. Older captures use the
original pairing rules, including an actual adapter call if the old engine filtered a
retained hand point. An absent point in an unprefixed recording carries `Missing`.
Gaze calibration retains its state −1 on an empty publication and is excluded as
calibration in the hold report.

## Label and score stationary holds

A label file describes **physically stationary** intervals on the measurement clock:
precise capture seconds multiplied by 1000 for v2, historical observation milliseconds
for legacy debugger traces. Use a new
trial name for each hold and nonoverlapping half-open intervals `[start_ms, end_ms)`.
For example, when the participant is stationary throughout the run, predeclare this interior
window **before capture** and check afterwards that the recording covers it:

~~~csv
trial,channel,start_ms,end_ms,target_x,target_y
hand_center_1,hand,10000,22000,,
~~~

The unlabelled command reports coverage, cadence and statuses only. With labels, each
hold reports row count, used count, exclusions by primary reason, and intervals for all
observations and for the included samples. Effective rate is `(N−1)/elapsed_seconds`.
The span can bridge interruptions. `within_run_covered_ms` sums only adjacent eligible
intervals in uninterrupted runs. Keep the trace, labels, commands, and JSON report together.
The output includes the input file hashes and wash-in/gap settings.

For each hold, scoring uses fresh neutral published points with status `Filtered` and
four finite normalized coordinates. Held, HeldNoCache, Lost, rejected, invalid and
calibration observations stay in frame counts and sampling intervals but contribute
no coordinates to either jitter estimate. State 1 (hand pinch) is nonneutral and
is excluded even if its DSP status is `Filtered`.
The default 500 ms wash-in starts at the first clean
observation in a hold and restarts after an exclusion or gap above 250 ms. These are
analysis settings (`--wash-in-ms`, `--max-gap-ms`). Predeclare them across comparisons.
At least two included pairs are needed to calculate dispersion. For each stream separately:

~~~text
RMS_2D = sqrt(population_variance(X) + population_variance(Y))
reduction_percent = 100 * (1 - RMS_2D_filtered / RMS_2D_raw)
~~~

Both streams use the same rows and their own hold mean. Zero raw dispersion yields a null
reduction. Negative reduction is reported as measured. If an independent target is known
in DSP coordinates, fill in both target columns to obtain signed mean bias and its length.
A screen crosshair alone does not establish a reference in normalized gaze-angle space.
Small RMS at a clipped boundary or a biased mean can coexist with poor accuracy.

## Trial sequence and evidence to keep

| Trial | Participant action | Evidence and interpretation |
| --- | --- | --- |
| Cadence pilot | Hold a relaxed, visible index finger stationary for the run. Repeat later with a steady head and one gaze target. | Compare observation rate, reset count, eligible duration and gaps with a matching approved dry run under the same channel/build with recording disabled. Retain the ordinary two-channel rate separately. Run recorder trials without tracepoints; a slow debugger trace measures only that operating condition. |
| Hand jitter | Support the wrist and hold at a physical reference away from the image boundary. Repeat at least three holds, separately labeled. | Report paired RMS, reduction, sample count, actual intervals and bias if calibrated. Fresh identical coordinates are valid. Retained coordinates during a detector miss are excluded. |
| Gaze jitter | Complete blink calibration, then fixate with head steady. Repeat center and off-center targets in separate holds. | Require complete face packets and neutral, post-calibration output. Report each target separately to avoid counting between-target movement as noise. |
| Responsiveness | For each channel separately, move between two fixed references horizontally and vertically. Repeat slowly, then quickly. Include a stop and a reversal with a settled hold at each end. | Mark direction and motion periods separately from stationary labels. Use the actual raw trajectory as the input reference. Record frame brackets for raw and filtered 50% and 90% crossings on the movement axis. Use pre/post raw hold means as common endpoints. Report settling against a declared endpoint band and dwell, and observed overshoot. A reversal needs its own leg. |
| Gaze loss and return | Establish a fresh calibrated gaze, cover the lens with an opaque card for several seconds, then uncover and look toward a different target. Repeat a short and a long loss. | Observe a usable packet disappearing, an absent published point, one established reset and a fresh `Initialized` return with output equal to input. Check actual Engineer 5 receipt. Mere absence of a gesture or state −1 is insufficient. |

Declare target, label windows, timing comparisons, and acceptable delay **before reviewing the
results**. Report each repeat, including failed or incomplete holds. A gap between two sampled
positions limits a crossing to that interval. At 15 FPS a typical interval is about 67 ms.
Compare raw and filtered crossing brackets and report their uncertainty. Missing crossings,
poorly separated plateaus or gaps spanning the transition cannot establish a rise time.
Arbitrary motion is unsuitable for a step-response claim.

The following extra tracepoints belong to the **historical** functional gaze-loss
procedure, separate from the three legacy measurement markers. They are not needed
for a v2 numerical capture; exact receiver processing remains a separate functional
check because a recorder row certifies submission only:

~~~text
os_bridge.hpp:175, on if (had), in the gaze-only convert function:
E4RESET t={t} established={had}

os_controller.cpp:65, first timestamp read in OsController::update:
E4OS t={frame.timestamp_seconds} gaze={frame.gaze.point._Has_value} x={frame.gaze.point._Has_value ? frame.gaze.point._Value.x : 0.0} y={frame.gaze.point._Has_value ? frame.gaze.point._Value.y : 0.0}
~~~

Join the gaze reset and controller records by exact observation time. The first missing
point after an established track should have `established=true`. Subsequent losses should
have it false. On return, check the same-frame A/P/OS coordinates and the `Initialized`
status. A zero-row gaze packet means the face/pose packet is unusable. To distinguish actual
absence of face landmarks from missing pose, inspect `result.face_landmarks` and
`result.facial_transformation_matrixes` at `ai_pipeline/bridge/landmark_extractors.py`'s
empty-packet branch during a separate functional trial with Python debugging available.
Record only presence and time. Native packet data alone cannot distinguish the two causes.

For subsequent tuning, retain **all** chronological rows. In replay of the current base,
advance hand DSP only for observed Tracked inputs, reproduce Held publications from
the cache, retain history and cached Held publication on a rejected update, publish
nothing after an invalid update, and reset on established loss. Use the precise
recorded DSP timestamp for v2. Gaze calibration publishes no point and
does not seed the filter. Replay the current configuration first and check it against the
accepted live outputs. Compare candidate configurations on that identical complete input,
including loss resets, then score the same hold and motion windows. The raw paired signal
is the DSP-off baseline. Reserve independent repeats for confirmation. The tools in this
directory currently calculate stationary metrics. Movement scoring and parameter comparison
are later tasks once labeled data exists.

The debugger changes thread scheduling. Avoid using tracepoint-to-tracepoint wall time to
estimate the cost of a DSP call. Use the existing Release DSP microbenchmark for compute
cost and the separate recorder benchmark for instrumentation cost. The optional recorder
provides the approved implementation path for a future measurement without tracepoints;
it does not by itself establish normal live cadence or validate final parameters.
