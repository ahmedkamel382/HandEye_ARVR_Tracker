# Engineer 4 coordinate stabilization

This module implements a C++17, allocation-free update path for canonical/current
Casiez-style 1 Euro filtering. It stabilizes numerical tracking observations after
AI inference and before downstream coordinate mapping or action execution.

The Windows core links this library and uses it for hand and gaze when
`--filter dsp` is selected. The recorder work is based on the working demo at
`6354a14`, including the precise capture clock and filter-outcome handling from
`839dcd6`. Live dry-run traces at the earlier `d455492` verified accepted output
through Engineer 5's controller. The outstanding experiments are controlled live
jitter, movement response, and confirmed gaze loss followed by recovery.
[Evaluation procedure](evaluation/README.md)

## Ownership and signal flow

```text
Camera capture -> Python MediaPipe landmark extraction -> native HandNative / GazeNative
    -> TrackingResult (frame ID, time, hand and gaze ModuleOut)
    -> OsBridge::update -> convert -> each channel's FreshnessGate
    -> fresh hand or usable gaze -> channel's DspFilterAdapter -> TrackingFilter2D::update
    -> FramePayload (optional point and state per channel) -> TrayApp::submit_frame -> OsController::update
```

The library under `include/` and `src/` uses only the C++17 standard library.
Each channel owns separate X and Y filter histories. The bridge carries the
module state alongside the coordinates. Engineer 5 maps them and decides which
source and gesture actions to use.

## Public API

Include `one_euro_filter.hpp` and link the `dsp_filter` static target.
All public symbols are in namespace `dsp`.

```cpp
struct FilterConfig {
    double min_cutoff_hz;
    double beta;
    double derivative_cutoff_hz;
};
struct TimingConfig {
    double min_dt_seconds = 0.000001;
    double reset_gap_seconds = 0.25;
};
struct Point2D { double x; double y; };
struct ScalarResult { std::optional<double> value; FilterStatus status; };
struct TrackingResult2D { std::optional<Point2D> point; FilterStatus status; };

OneEuroFilter1D(FilterConfig config, TimingConfig timing = {});
ScalarResult update(double value, double timestamp_seconds) noexcept;
void reset() noexcept;

TrackingFilter2D(FilterConfig x_config, FilterConfig y_config,
                 TimingConfig timing = {});
TrackingResult2D update(Point2D point, double timestamp_seconds,
                        bool tracking_valid) noexcept;
void reset() noexcept;
```

The last six declarations summarize member functions of their corresponding
classes, not free functions. Constructors validate configuration and may throw
`std::invalid_argument`; updates and resets do not throw. Configuration is copied
by value. Reset preserves configuration. To change parameters, construct a new
instance at a controlled boundary; its next observation initializes a new track.

`FilterStatus` values are `Initialized`, `Filtered`, `ReinitializedAfterGap`,
`TrackingLost`, `InvalidInput`, `RejectedTimestamp`, and `NumericalError`.
Only the first three accompany a usable output. Always check the optional output.
An absent point is never a fabricated `(0,0)` position or a fresh stale position.

## Coordinate and validity contracts

`OneEuroFilter1D` accepts generic finite scalar signals in consistent units, such
as normalized coordinates or the 100/500-valued synthetic tests.

`TrackingFilter2D` enforces the current project's finite normalized `[0,1]` X/Y
domain, inclusively. It does not clamp invalid input into the valid range.

- **Hand.** The default point is the index fingertip in the full mirrored image.
  Optional anchors use the index knuckle or the midpoint between the thumb and
  index tips. On `3ea3ce3`, each fresh hand detection is marked `Tracked`.
  After an established detection the first six consecutive misses are `Held`.
  The upstream tracker repeats the last coordinates and gesture. The bridge
  skips DSP and republishes the cached output. With an empty cache it publishes
  no point and marks the output `Lost`. The seventh miss publishes upstream
  `(-1,-1,0)` with freshness `Lost` and clears established hand filter history.
- **Gaze.** The native tracker adds head pose to the predicted eye-in-socket yaw
  and pitch. It publishes `x=clamp(0.5+yaw_deg/60,0,1)` and
  `y=clamp(0.5+pitch_deg/60,0,1)`. These are normalized angular coordinates
  before screen mapping. The native path applies no positional EMA. During blink
  calibration, including the completing frame, it returns `(-1,-1,-1)` with
  invalid coordinates, gesture state −1 and NaN angles.
  The bridge forwards the state with an empty point and skips the DSP update.
  An unusable face/pose packet yields `(-1,-1,0)`. The first usable observation
  after calibration seeds the gaze filter directly.
- **Validity and freshness.** The bridge's `ModuleOut::detected()` tests
  `ok && x >= 0 && y >= 0`. Neutral state zero and position `(0,0)` are both
  legitimate. State, detector freshness, track identity and confidence are not
  inputs to the filter. Core uses hand freshness and position validity to decide
  when to call it. The hand trace records the upstream and outgoing freshness
  alongside the detector count, including when coordinates happen to repeat.
- **Library contract.** A direct call with false validity resets both axes and
  returns no point. A direct call with invalid normalized coordinates, including
  `(-1,-1)`, also resets both axes. The core adapter has its own policies below.

## Canonical mathematics

For current raw value `x`, previous FILTERED value `previous_filtered_value`,
previous filtered derivative `previous_filtered_derivative`, and accepted time:

```text
dt = timestamp_seconds - previous_timestamp
derivative = (x - previous_filtered_value) / dt
alpha(f, dt) = 1 / (1 + 1 / (2*pi*f*dt))
alpha_d = alpha(derivative_cutoff_hz, dt)
filtered_derivative = alpha_d * derivative
                    + (1-alpha_d) * previous_filtered_derivative
cutoff = min_cutoff_hz + beta * abs(filtered_derivative)
alpha_x = alpha(cutoff, dt)
filtered_value = alpha_x * x + (1-alpha_x) * previous_filtered_value
```

The derivative uses **current raw minus previous filtered**, not previous raw.
Previous raw input is not stored. Each axis owns its filtered position, filtered
derivative, timestamp, and initialized flag. This follows the convention in the
[authors' C++ reference](https://github.com/casiez/OneEuroFilter/blob/56126d84fd9107b4a8942deb5785a854730f404c/cpp/OneEuroFilter.cc).
The implementation here is written from the approved equations, not copied
third-party source code.

At first sample: output equals input, derivative state is zero, and timestamp is
recorded. There is no initial division or artificial interpolation from zero.

The implementation uses mathematically equivalent guarded alpha and convex
interpolation forms. It uses a portable `constexpr` pi value, not `M_PI`.

## Timing, reset, and numerical-error policy

DSP receives explicit `double` seconds. In `core_engine/src/main.cpp`, capture
records a host `steady_clock` time immediately after a successful camera read.
Since `839dcd6`, `frameTimes` derives `TrackingResult::capture_s` directly from
that original `captured_at` time relative to pipeline start. `OsBridge::update`
passes this precise value to both DSP channels and the OS payload. It is neither
truncated to whole milliseconds nor bumped to satisfy MediaPipe. Separately,
`TrackingResult::ts_ms` is the greater of the whole capture milliseconds and the
preceding MediaPipe timestamp plus 1 ms. Only results lacking `capture_s` use
`ts_ms / 1000.0` as a compatibility fallback. Hardware exposure time and AI
completion time are outside this contract; buffered camera reads can still have
nearly identical host timestamps. Tests supply their own timestamps, including
irregular intervals.

The initial timing policies are configurable engineering choices, not measured
optima: `min_dt_seconds=1e-6`, `reset_gap_seconds=0.25`.

| Condition | Actual behavior |
| --- | --- |
| First valid observation | Seed position directly, derivative zero |
| Nonfinite timestamp | No output; `RejectedTimestamp`; preserve history |
| `dt <= 0` or nonfinite computed dt | Same rejection; preserve history |
| `0 < dt < min_dt_seconds` | Same rejection; do not advance timestamp |
| `dt == min_dt_seconds` | Accept normally |
| `min_dt_seconds <= dt <= reset_gap_seconds` | Canonical update using actual dt |
| `dt > reset_gap_seconds` | Seed from new position; `ReinitializedAfterGap` |
| Explicit `reset()` | Clear history; retain parameters |
| False tracking validity | Reset both axes; `TrackingLost`; no point |
| Invalid normalized coordinate | Reset both axes; `InvalidInput`; no point |
| Unrecoverable numerical calculation | Reset affected scalar/both wrapper axes; `NumericalError`; no output |

After any reset/loss, the first valid observation becomes the output directly.
There is no blending with a lost track. Even a single reported loss resets the
track because upstream supplies neither confidence nor identity. A visible
relocation on reacquisition can reflect the new observation; this does not
guarantee an accurate first sample.

Precedence is explicit: the tracking wrapper handles false validity and invalid
coordinates before timing. Thus a missing track resets even if its timestamp is
bad. For the generic scalar filter, a nonfinite timestamp is rejected first;
otherwise a nonfinite scalar resets it. Timing rejection of valid coordinates
never changes accepted history.

Configuration requires finite positive cutoffs, finite nonnegative beta, finite
positive timing limits, and reset gap greater than minimum dt. The implementation
checks representable time constants, differences, derivatives, adaptive cutoff,
alpha, and outputs. Extremely small/large but finite values that cannot be safely
processed produce an explicit numerical error, not NaN/Inf output. New state is
committed only after validation. The 2D wrapper calculates on two fixed-size
stack candidates, committing both together or neither; a numerical failure
clears both axes. An alpha rounded to exactly one returns the raw value directly
to avoid cancellation of very small observations against very large history.

If no observations arrive, the filter cannot independently detect a timeout.
Core must stop using stale results and discard out-of-order queued observations.
The gap check operates on the next call. Reset on source/coordinate-system changes
is also the caller's responsibility.

## Configuration and empirical tuning

**INITIAL TUNING VALUES ONLY - not final or scientifically validated.**

| Profile constant | min_cutoff_hz | beta | derivative_cutoff_hz |
| --- | ---: | ---: | ---: |
| `kInitialHandConfig` | 1.2 | 4.0 | 1.0 |
| `kInitialGazeConfig` | 0.8 | 6.0 | 1.5 |

Lower minimum cutoff generally reduces stationary jitter while increasing lag.
Beta controls adaptation to movement. Derivative cutoff controls how quickly the
estimated motion changes. These effects interact, so a higher beta can also
increase sensitivity to noisy apparent movement. Beta depends on signal units:
normalized-coordinate values must not be reused unchanged for pixel signals.

Tune hand and gaze separately using representative sessions:

1. Record stationary fixation/hover and slow movement with valid timestamps.
2. Start with beta zero and adjust minimum cutoff for the jitter/slow-motion tradeoff.
3. Increase beta while measuring rapid target changes and direction reversals.
4. Adjust derivative cutoff while checking noise-driven adaptation and response.
5. Repeat at actual observed rates and lighting/distance conditions; test both axes.
6. Compare RMS jitter, bias, movement error, rise/settling time, and user targeting.
7. Keep separate held-out sessions for evaluation; do not tune only to one trace.

The Windows bridge reads saved filter settings and then applies any
`--hand-filter` and `--gaze-filter` overrides. Record the active parameters
for each session. In-session changes rebuild both instances and clear history.
Real-world tuning requires labeled holds and movement trials on identical input
sequences for each candidate.

## Integrated adapter behavior

`OsBridge::update` sends both channels through `convert` and separate freshness
gates. Gaze has no upstream freshness enum and uses position detection. A hand
hold leaves the filter's timestamp and both axis histories unchanged. If the next
fresh hand observation is more than the
configured gap after the last accepted update, it reinitializes even during the
grace period. For either channel, the first loss of an established track calls
the adapter's `reset()`, clears its history and publishes no point. Either channel's
loss also clears its gate's publication cache. This path bypasses `update(..., false)`,
so a live loss has no returned DSP `TrackingLost` status. The next valid
observation initializes directly.

For a detected point, `DspFilterAdapter::update` clamps finite coordinates
to the unit square and calls `f_.update({rx, ry}, t_sec, true)`. A present result
returns outcome `Ok` and becomes the published and cached point. Timestamp
rejection returns `Rejected` while preserving DSP history; the gate republishes
its cached point marked `Held`, or publishes no point when the cache is empty.
A nonfinite input, invalid input or numerical error returns `Invalid`, clears
filter history and causes the gate to clear its cache and publish no point.
The adapter's compatibility X/Y fallback values are never published as an
accepted result. Its nonfinite-input guard resets without calling the DSP update.
The bridge applies a final clamp. The outgoing payload contains optional
positions, gesture state integers and publication freshness. It omits the DSP
status, so a point in the payload alone cannot establish that DSP accepted the
observation. The optional recorder captures the actual status and outcome.
[Adapter](../core_engine/include/dsp_adapter.hpp) · [OS bridge](../core_engine/include/os_bridge.hpp)

The ordinary console line reads raw module outputs from the latest
`TrackingResult` and rounds them to two decimal places. To compare DSP on/off,
pair the adapter input and accepted output for the **same observation**. Exclude
calibration, retained detections and reset transients from steady-state jitter.

One owning update thread per filter instance is required. The filter update has
no mutex, clock, I/O, logging, heap allocation or dynamic history. Concurrent
updates to the same instance require external coordination.

## Optional numerical recorder and review scope

Recording is disabled by default. Opt in with `--dsp-record-dir NEW_DIRECTORY`,
`--filter dsp`, `--os log`, and exactly one tracker: `--no-gaze` for hand or
`--no-hand` for gaze. The parent directory must already exist; the run directory
must not exist, even if empty. The recorder reserves 4096 numerical frame slots
before capture. It records one selected-channel row after each actual
`TrayApp::submit_frame` call, including Held, Lost and calibration observations.
This establishes submission to the receiver, not an acknowledgement of mouse
movement or an action executed by Windows. It records no camera images or video.

The processing path copies numerical fields into preallocated memory; it performs
no recorder allocation, formatting, locking, clock query or disk I/O. Startup
reserves the directory and writes `INCOMPLETE`. After capture and AI threads join,
remaining results are submitted, the OS bridge stops, and the camera is released;
only then are records serialized. A usable run requires `COMPLETE`, no
`INCOMPLETE`, and a valid `E4REC END` footer with equal positive submitted,
attempted and stored counts, zero overflow and zero configuration changes.
Overflow, truncated output, write failure or a filter change invalidates the
run. Overflow does not stop tracking. Preserve failed runs for diagnosis.

The v2 importer produces 26 CSV columns while retaining the older 12-column and
15-column debugger formats. It keeps precise `dsp_timestamp_seconds` separately
from MediaPipe milliseconds, actual DSP input/optional output/status, adapter
outcome/reset count, upstream tracker data and downstream publication/freshness.
Recorder rows contain no DSP execution duration. The separate Release recorder
benchmark compares enabled/disabled synthetic adapter-and-gate workloads;
live cadence impact and end-to-end latency still need measurement.
[Capture, import and scoring workflow](evaluation/README.md)

Only four core files contain recorder instrumentation for teammate review:

| Core file | Recorder addition |
| --- | --- |
| `core_engine/include/tracking_types.hpp` | Optional actual hand/gaze detector counts carried with the existing result. |
| `core_engine/src/main.cpp` | Opt-in CLI validation, detector-count capture, recorder lifetime, and shutdown save. |
| `core_engine/include/dsp_adapter.hpp` | Optional observation slot for actual update input, status, optional result, outcome and resets. |
| `core_engine/include/os_bridge.hpp` | Clear slots each frame, attach them to the selected filters, track configuration generation and append after submission. |

The precise clock, freshness gates and `Rejected`/`Invalid` behavior already
belong to the working-demo base. Recorder storage, tests, benchmark and offline
tools remain under `dsp_filter/`. Review the changes on
`codex/engineer4-dsp-recorder`; these instructions do not merge the branch.

## Build and test independently in Visual Studio

1. Open the installed Visual Studio **Developer PowerShell**, configured for x64,
   or use its **x64 Native Tools Command Prompt** from the Start menu.
2. Change to the repository root. Confirm `git branch --show-current` reports
   `codex/engineer4-dsp-recorder` for recorder review; do not switch or merge
   branches as part of these checks.
3. Configure/build/test Debug:

```powershell
cd D:\GitHub\HandEye_ARVR_Tracker
cmake -S dsp_filter -B out/build/dsp-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build out/build/dsp-debug
ctest --test-dir out/build/dsp-debug --output-on-failure
.\out\build\dsp-debug\dsp_filter_tests.exe
```

4. Configure/build/test Release and run the optional benchmark:

```powershell
cmake -S dsp_filter -B out/build/dsp-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DDSP_BUILD_BENCHMARKS=ON
cmake --build out/build/dsp-release
ctest --test-dir out/build/dsp-release --output-on-failure
.\out\build\dsp-release\dsp_filter_tests.exe
.\out\build\dsp-release\dsp_filter_benchmark.exe
.\out\build\dsp-release\dsp_trace_recorder_benchmark.exe
```

The CMake binary bundled with this machine's Visual Studio is under
`D:/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/`.
Developer shells expose CMake, Ninja, and compiler environment paths. Ordinary
PowerShell on this machine did not expose them. If tools are absent, stop and
check the installed Visual Studio components rather than changing project code.
Build products stay in ignored `out/build/` directories.

The test executable prints each individual PASS/FAIL, reference error, quality
metrics, and long-run counts; failure returns a nonzero exit code. Checks remain
active under Release `NDEBUG`. CTest registers this executable as one suite.
No GoogleTest, Catch2, or additional package is required.

### Historical DSP build matrix

All **60 individual tests** passed in each configuration below. The benchmark
is separate from the correctness suite and is not a CTest timing assertion.

| Configuration | Configure/build | CTest |
| --- | --- | --- |
| DSP-only Debug | PASS | PASS (60 checks in one executable) |
| DSP-only Release | PASS | PASS (60 checks in one executable) |
| Root project Debug, including existing core executable | PASS | PASS |
| Root project Release, including existing core executable | PASS | PASS |

The current root CMake registers the DSP directory, links `dsp_filter`
privately into `core_engine` and defines `CORE_HAVE_DSP_FILTER` for that target
when the DSP target exists. Standalone builds avoid the tracking and OS
dependencies. The earlier root Release verification reused
the existing installation at `out/build/windows-debug/vcpkg_installed`, which
already contains both Debug and Release libraries. Manifest installation was
disabled, so these builds did not install/update packages. Exact root commands
used from the repository root in an x64 developer environment:

```powershell
cmake -S . -B out/build/windows-debug -DBUILD_TESTING=ON -DVCPKG_MANIFEST_INSTALL=OFF
cmake --build out/build/windows-debug
ctest --test-dir out/build/windows-debug --output-on-failure

cmake --preset windows-release `
  "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=D:/Microsoft Visual Studio/18/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  "-DVCPKG_INSTALLED_DIR:PATH=D:/GitHub/HandEye_ARVR_Tracker/out/build/windows-debug/vcpkg_installed" `
  "-DOpenCV_DIR:PATH=D:/GitHub/HandEye_ARVR_Tracker/out/build/windows-debug/vcpkg_installed/x64-windows/share/opencv4" `
  -DVCPKG_MANIFEST_INSTALL=OFF -DBUILD_TESTING=ON
cmake --build out/build/windows-release
ctest --test-dir out/build/windows-release --output-on-failure
```

These explicit Release overrides reuse this machine's existing toolchain and
installed packages; they do not edit `CMakePresets.json`. On another machine,
the corresponding already-installed paths must be supplied by the core owner.

### Recorder verification, 2026-11-18

Both root builds passed all seven CTest suites: Debug in 115.69 seconds and
Release in 64.58 seconds. Each build passed the 60 DSP tests, 13 recorder cases
with 4,520 checks, 76 OS tests, 20 core tests with 347 checks, 13 end-to-end checks,
and 9 adapter tests with 47 checks. Parity covered 3,000 gaze and 3,000 hand cases
with zero mismatches. These are synthetic/deterministic checks, not live quality
measurements.

The offline Python suite passed 75 tests in 1.005 seconds, including 13 mocked
verifier-guard cases. The real-executable synthetic recorder/importer runs also
passed: Debug retained/imported 324 hand and 329 gaze submissions; Release
retained/imported 327 hand and 329 gaze submissions. Each run captured 329 input
frames. The 5 Debug hand and 2 Release hand stale frames were discarded before
DSP, so they are not missing recorder rows; neither gaze run discarded stale
frames. Every submitted frame was retained and imported. All four runs reported
zero Python errors, all 8/8 pool slots returned, unchanged saved tray settings,
and refusal to reuse the existing output directory with its trace hash preserved.
Synthetic pipeline cadence is not a recorder-overhead or live-FPS measurement.
[Per-status integration results](evaluation/README.md#verified-recorder-checks-2026-11-18)

On the Intel Core i7-1255U (10 cores, 12 logical processors), AMD64 Windows 11 Pro
10.0.26200, MSVC 19.51.36260.0 Release build, the recorder benchmark ran 31
alternating enabled/disabled pairs of 4,096 frames, after two warm-up batches per
mode. Output checksums matched exactly.

| Recorder mode | Mean ns/frame | Median batch mean ns/frame | p95 batch mean ns/frame | Maximum batch mean ns/frame |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 221.0654 | 188.2568 | 331.5186 | 343.5059 |
| Enabled | 282.9810 | 256.0303 | 400.9277 | 430.5664 |

The paired mean increase was 61.9156 ns/frame (28.0078%). This measures synthetic
adapter/gate updates plus recorder observation and record-copy work. It excludes
AI-thread detector metadata/queue cost, startup/preallocation, serialization,
disk I/O, camera and OS calls. These are machine-specific batch averages, not
worst individual call times or a live FPS/latency result. The separate DSP-only
benchmark rerun measured means of 87.3406 ns/scalar update, 204.2164 ns/2D update,
and 370.4211 ns/hand-plus-gaze update, with zero rejected updates. Earlier
measurements below are retained as separate historical runs.

## Deterministic validation and measurement definitions

The offline evaluation tools have a separate Python unittest suite. It covers
same-frame pairing, DSP-to-publication equality, hand cache reuse and loss,
older recordings, exclusion counts, timing gaps, target bias and analytical
2D jitter values.
Run it from the repository root with
`.\.venv\Scripts\python.exe -B -m unittest discover -s dsp_filter/evaluation -p "test_*.py"`.
These tests are independent of CTest. With the sibling core headers present,
CTest also registers `dsp_trace_recorder_tests`, which exercises the actual
adapter and freshness gate without a camera, Python inference or OS actions.
The separate `evaluation/check_recorder_pipeline.py --exe ABSOLUTE_EXE_PATH`
check runs the real executable and recorder on existing synthetic-video/fake-AI
fixtures, verifies completed imports and file-preservation behavior, and keeps
its artifacts outside Git. It opens no webcam and runs the OS bridge in logging
mode. Existing readable tray settings are required and verified before/after
every run; they are never silently restored. The tray may append its ordinary
LocalAppData log outside the experiment folder. See the
[reproduction commands](evaluation/README.md#verify-the-integration-with-synthetic-video).

The analytical fixture uses `min_cutoff=derivative_cutoff=5/pi`, `beta=1/(2*pi)`:

| Timestamp (s) | Raw | Independent expected output |
| ---: | ---: | ---: |
| 0.0 | 0 | 0 |
| 0.1 | 1 | 3/5 |
| 0.2 | 1 | 41/49 |
| 0.3 | 0 | 1640/4299 |

Tolerance is `abs(actual-expected) <= 1e-12 + 1e-12*abs(expected)`. The constants
were derived algebraically before implementation. The previous-raw error would
give `37/45` at the third sample and explicitly fails the test. Maximum measured
absolute reference error was `1.110223e-16` in both standalone configurations.
Beta-zero tests also compare against an independently evaluated closed-form
residual product using actual timestamp differences.

Two additional regressions use fixed analytical constants and the same strict
tolerance. `adaptive_irregular_dt_canonical_reference` uses
`min_cutoff=5/(2*pi)`, `derivative_cutoff=5/pi`, `beta=1/(2*pi)` and checks
`(time,input)=(0,0),(0.1,1),(0.3,1)` against `0,1/2,13/16`.
`nondefault_min_dt_preserves_history` sets `min_dt_seconds=0.01` and rejects
observations at 0.005 and 0.015 seconds, before and after motion. Accepted samples
at 0, 0.01, 0.02, and 0.04 seconds are checked against `0,3/5,41/49,1230/4789`
using `min_cutoff=derivative_cutoff=50/pi`, `beta=1/(2*pi)`. This verifies the
configured boundary, larger intervals, and preservation of filtered position,
derivative, and accepted timestamp across rejection.

The suite covers first sample, constants, ascending/descending steps, ramps,
stationary noise, adaptation, reversal, 30/60 FPS and irregular timing, dropped
frames, invalid/tiny/gap timestamp boundaries, NaN/Inf, normalized boundaries,
sentinels, reset/loss/reacquisition, atomic axes, independent streams/configs,
overflow, cancellation, recovery, and one million mixed observations.

Noise uses `std::mt19937(42)` and explicit mapping
`2*(integer/4294967296.0)-1` to avoid distribution-library differences. Run 1,200
samples at 60 Hz and discard the first 120 (2 seconds), leaving 1,080 measurements.
Standard deviation uses population variance around each signal's own mean. RMS
jitter uses squared error around the known target. Reduction is
`100*(1-filtered_RMS/raw_RMS)`; zero raw RMS is reported as N/A. Also report bias
through means and maximum absolute target error.

Step times are sample-quantized, measured from the first changed input observation
(not the preceding baseline sample). A reported zero means the threshold was
crossed on that first observation, not zero end-to-end latency. Settling means
remaining within 5% of step amplitude for the rest of the 600-sample observation
window. No continuous-time interpolation is used. Ramp lag is an approximation
from mean position error divided by known nonzero velocity.

### Measured synthetic quality, 2026-09-27

| Test | Raw mean | Filtered mean | Raw stddev | Filtered stddev | Raw RMS | Filtered RMS | RMS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar 500 +/- 5, cutoff 1, beta 0 | 500.051086363 | 500.057896777 | 2.89990473124 | 0.668107367154 | 2.90035467949 | 0.670611281409 | 76.8783% |
| Hand initial, 0.5 +/- 0.005 | 0.500051086363 | 0.500056154636 | 0.00289990473129 | 0.000789115155653 | 0.00290035467949 | 0.000791110657601 | 72.7237% |
| Gaze initial, same noise | 0.500051086363 | 0.500055239657 | 0.00289990473129 | 0.000744215531588 | 0.00290035467949 | 0.000746262807022 | 74.2699% |

| Step at 60 Hz | t10 (ms) | t50 (ms) | t90 (ms) | 10-90 rise (ms) | 5% settling (ms) | Maximum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar 0 to 100 / 100 to 0; config (1, 0.1, 1) | 0 | 0 | 16.667 | 16.667 | 16.667 | 14.1628159391 |
| Hand initial 0.2 to 0.8 | 0 | 0 | 33.333 | 33.333 | 33.333 | 0.234823620844 |
| Gaze initial 0.2 to 0.8 | 0 | 0 | 16.667 | 16.667 | 33.333 | 0.144422157252 |

For the hand step, beta zero with the same minimum cutoff had t90=316.667 ms.
Hand ramps at +/-0.2 normalized units/second had RMS error 0.008080382216 and
approximate lag 40.402 ms after a one-second warm-up (181 measured samples).
This nonzero lag is a limitation to assess during real targeting.

The million-observation test produced 999,556 valid outputs, 245 deliberate
losses, 199 deliberately invalid positions, 333 additional rejected timestamp
calls, and 98 gap reinitializations. It also included explicit resets, varying
intervals, stationary periods, noise, motion, and reversals. No invalid output,
crash, or recovery failure occurred.

### Historical Release microbenchmark

Machine: Intel Core i7-1255U, AMD64 Windows; MSVC 19.51.36260.0; Release `/O2`.
Each workload uses 10,000 warm-up calls followed by 100 batches of 10,000 measured
calls (1,000,000 total). Signals are precomputed; timestamps are explicit.
`steady_clock` surrounds each batch, and printed checksums consume all outputs.
Results include loop, timestamp, and output-checksum handling overhead.

| Workload | Mean ns/update | Mean us/update | Slowest batch mean ns/update | Total measured ms |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 74.0180 | 0.074018 | 127.26 | 74.0180 |
| Tracking 2D | 135.3950 | 0.135395 | 283.25 | 135.3950 |
| Hand plus gaze (four axes) | 283.5689 | 0.283569 | 557.06 | 283.5689 |

These are one observed run, not guaranteed limits. A slowest batch average is
not worst single-call latency. Scheduling, thermals, hardware, and compiler
settings affect measurements. Computation time does not equal filter response
delay or full webcam-to-cursor latency. There are no benchmark timing assertions
in correctness tests. The benchmark reported zero rejected updates.

## External canonical fixture status

**PENDING - external Casiez ground-truth dataset not included because its
applicable redistribution license could not be confidently established.**

Per Bassel's explicit decision, neither `groundTruth.csv` nor a derivative copy
is downloaded, embedded, or included in this module. There is no third-party
test-data file, no dataset checksum to report, no `.gitignore` change, and no
network access during configure/build/test. The independently derived analytical
test remains mandatory and passes; this external validation is a separately
documented nonblocking pending item.

## Live verification and open measurements

The following evidence comes from earlier webcam dry runs of the integrated
Windows executable at `d455492`, included in development merge `66485b0`.
It establishes behavior of that earlier base, not validation of the current
precise-clock recorder integration.

| Evidence | Observed result |
| --- | --- |
| Independent numerical replay of accepted live observations | Across two traces, 517 accepted hand/gaze updates matched replay to a maximum absolute coordinate difference of approximately `1.11e-16`. |
| Engineer 5 handoff | In one trace, both accepted channel outputs matched the actual controller input for 100 paired frames (200 channel outputs). The OS ran in logging mode. |
| Live hand loss and recovery | Three completed intervals showed published loss, a channel reset and direct initialization from the first returning hand point. |
| Live gaze loss and recovery | The face detector continued to report landmarks during the attempted trial. This case still needs a confirmed missing-face interval and a returning observation. |

Debugger tracepoints altered the processed frame intervals. The prior traces
verify numerical behavior under their observed timestamps. They contain no
labeled stationary or controlled movement intervals from which to report live
jitter reduction or responsiveness. In Bassel's separate 60-second dry run,
900 processed frames reached the OS submission boundary with zero Python errors.
That throughput summary contains no paired raw and filtered positions.

A later debugger pilot validated import of 28 real paired hand records at
approximately 5.36 Hz. Debugger overhead altered cadence, and hand state 1
(pinch) made that run unsuitable for the Neutral stationary protocol.

### Live recorder results

The following webcam experiments ran outside the debugger with OS logging.
Hand used `1.2,4,1` and gaze used `0.8,6,1.5`. Stationary trials paired the
raw DSP input and filtered output from each eligible observation in the fixed
5–25 s window, using the evaluator's 500 ms wash-in. Each signal's 2D RMS
jitter is its dispersion about its own mean. Reduction is
`100 × (1 − filtered RMS / raw RMS)`. This measures steadiness at the DSP
boundary. Absolute pointing accuracy requires a separate reference.

| Live experiment | Result under its predeclared criteria |
| --- | --- |
| Stationary hand | **PASS.** The Neutral open-hand trial measured **4.1626%** RMS jitter reduction. |
| Stationary gaze | **PASS, Trial 2.** RMS jitter reduction was **21.3619%**. Trial 1 measured **17.3995%** and missed the 18 s coverage requirement with 17.888472 s. |
| Horizontal SLOW hand | **PASS in both directions, Trial 2.** Trial 1 passed LEFT → RIGHT. Its return was inconclusive because the movement duration fell outside the SLOW category. |
| Vertical SLOW hand | **PASS in both directions, Trial 1.** The largest added-delay upper bounds were 144.9 ms for crossing and 113.9 ms for settling, within the respective 200 ms and 250 ms limits. |
| Horizontal FAST hand | **INCONCLUSIVE after four trials.** In Trial 4 both directions exceeded the endpoint Y-alignment limit. The return also had a 224.9 ms gap between fresh detections. All four trials passed independent DSP replay. |
| Scored horizontal gaze responsiveness | **ON HOLD.** Upstream raw gaze fixation did not meet the stability limits during unscored qualification. Independent DSP replay passed all 1,269 accepted updates in the fixation-only experiment. |
| Tracking-loss recovery and combined live end-to-end validation | **INCOMPLETE.** Earlier live traces verified three hand recoveries. Confirmed live gaze loss and return, and validation of both channels together through the complete system, remain outstanding. |

Movement delays are measured relative to the upstream observation using actual
frame timestamps. The results apply to these recorded trials and settings.
[Evaluation procedure](evaluation/README.md)

The [audio-guided launcher](evaluation/run_cued_responsiveness.py) is specific
to this Windows hand responsiveness experiment series. Its preflight requires
the recorder development branch and the verified Debug executable's SHA-256.
Live recording also requires manual `RECORD` confirmation and `--os log`.
Use the core executable for the general team demo.
