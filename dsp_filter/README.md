# Engineer 4 coordinate stabilization

This module implements a C++17, allocation-free update path for canonical/current
Casiez-style 1 Euro filtering. It stabilizes numerical tracking observations after
AI inference and before downstream coordinate mapping or action execution.

**DSP MODULE COMPLETE - AWAITING UPSTREAM/DOWNSTREAM INTEGRATION**

The current core executable only captures and displays frames. It does not yet
receive Python AI results. This module is independently built and tested; no
runtime bridge, webcam logic, AI classification, GUI, or OS integration is added.

## Ownership and signal flow

```text
Future core receives actual hand/gaze (X, Y, State) results and frame timestamps
    -> core determines coordinate usability
    -> hand TrackingFilter2D: independent X and Y OneEuroFilter1D instances
    -> gaze TrackingFilter2D: independent X and Y OneEuroFilter1D instances
    -> optional stabilized X/Y, with explicit DSP status
    -> core / Engineer 5 mapping and execution

Original hand and gaze State integers bypass DSP unchanged.
```

Production headers live directly in `include/`; implementations live directly in
`src/`. Both original `.gitkeep` files remain. No third-party runtime dependency
is required, including Python, OpenCV, MediaPipe, Pybind11, or Windows APIs.

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

- Hand runtime returns mirrored-image normalized index-fingertip X/Y and State.
- Gaze runtime returns normalized eye-relative X/Y ratios and State. These are
  not yet calibrated screen coordinates and are not the same physical signal as
  hand positions.
- Missing hand/face tracking currently returns `(-1.0, -1.0, 0)`.
- Core must translate tracking usability into explicit `tracking_valid`.
- `State == 0` is not a validity test; both neutral and missing results use it.
- `(0,0)` is a legitimate position. `(-1,-1)` is rejected even if a caller
  incorrectly passes `tracking_valid=true`.
- Invalid/nonfinite/out-of-range coordinates or false validity clear both axes
  and return no position. No Z dimension is invented.
- Pinch/fist, calibration, wink, closure, pause, and cursor-source decisions stay
  in the upstream/core/OS layers. DSP accepts no State integer.

Numeric validity does not prove observation accuracy. The current upstream API
does not expose confidence or track identity. An in-range outlier may look like
rapid motion to this filter; no undocumented outlier classifier is included.

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

Timestamps are explicit `double` seconds on one monotonic observation timeline.
DSP never queries the clock or assumes a frame rate. Engineer 1 should associate
a `steady_clock` timestamp with each captured frame and preserve it through
inference. Acquisition time and inference-completion time are different.
Tests can supply synthetic timestamps directly. A finite arbitrary time origin
is permitted; seconds since application start are a practical choice.

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

The synthetic results below establish implementation behavior, not real-world
profile optimality. Real-world tuning is pending valid integrated/recorded data.

## Integration example (documentation only)

These objects should live across observations, not be reconstructed every frame:

```cpp
dsp::TrackingFilter2D hand(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
dsp::TrackingFilter2D gaze(dsp::kInitialGazeConfig, dsp::kInitialGazeConfig);

// At the future core's actual result-consumption point:
auto filtered_hand = hand.update({hand_x, hand_y}, frame_time_seconds, hand_usable);
auto filtered_gaze = gaze.update({gaze_x, gaze_y}, frame_time_seconds, gaze_usable);
// Forward only present .point values to downstream mapping.
// Carry original hand_state and gaze_state separately and unchanged.
```

The variable names represent values supplied by the future bridge, not functions
or structures implemented here. No fake production observations are created.
Core owns RGB/BGR/mirroring, calibration/closure gating, timestamps, and routing.
Engineer 5 owns mapping and action execution. Update once per new observation,
not repeatedly on a stale result at the display-loop frequency.

One owning update thread per instance is required. The module has no mutex,
clock, I/O, logging, per-update heap allocation, or dynamic history. Concurrent
updates to the same instance require external coordination.

## Build and test independently in Visual Studio

1. Open the installed Visual Studio **Developer PowerShell**, configured for x64,
   or use its **x64 Native Tools Command Prompt** from the Start menu.
2. Change to the repository root. Confirm `git branch --show-current` reports
   `bassel-development`; the following commands do not change branches.
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

### Verified build matrix

All **60 individual tests** passed in each configuration below. The benchmark
is separate from the correctness suite and is not a CTest timing assertion.

| Configuration | Configure/build | CTest |
| --- | --- | --- |
| DSP-only Debug | PASS | PASS (60 checks in one executable) |
| DSP-only Release | PASS | PASS (60 checks in one executable) |
| Root project Debug, including existing core executable | PASS | PASS |
| Root project Release, including existing core executable | PASS | PASS |

Root registration adds only `include(CTest)` and `add_subdirectory(dsp_filter)`.
The core is not linked to DSP until it actually consumes AI results. No global
compiler flag or dependency/preset/source file was changed. Root Release reused
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
No compiler/linker warnings were emitted. CMake's Release discovery tried POSIX
pthread probes, reported them unavailable on Windows, and then successfully
reported `Found Threads: TRUE`; this was not a failed build or test. An initial
manual compiler-launch quoting error was corrected by loading the Visual Studio
environment into the child PowerShell process, without changing project files.

## Deterministic validation and measurement definitions

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

### Release microbenchmark

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

## Team-document consistency and integration limits

The teammate hand/gaze documents require numerical X/Y stabilization with State
preserved separately. This module honors those boundaries. Gaze calibration and
OS gating remain upstream/downstream responsibilities. The report's broader 3D
and multithreaded architecture is not evidence of a current Z output or completed
runtime bridge. Treelite/Pybind11 choices remain outside Engineer 4 ownership.

Known upstream issues are preserved: the expected hand model asset is missing;
the gaze runner omits its required timestamp/RGB preparation; saved classifier
feature names use 15-frame features while current source uses 10. No collector,
trainer, model, inference file, asset, setup, dependency, or core source is fixed
as part of this module. The current gaze collector's feature CSV also does not
provide the runtime gaze X/Y trajectories needed for real profile tuning.

No measured user study, real-world profile validation, full-system latency claim,
or fully integrated cursor-control claim is made.
