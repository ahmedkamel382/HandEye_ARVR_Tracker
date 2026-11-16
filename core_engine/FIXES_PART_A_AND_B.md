# HandEye AR/VR Tracker: Fixes in Part A and Part B

Combined set: 9 files under `core_engine/` (see `combined_AB_files.zip`). Copy them over the project root.

| Part | Tracker issue | Engineer | Fix |
|------|---------------|----------|-----|
| A | #5 DSP adapter rejection behavior | Engineer 1 | A substitute value is never published as a real tracking result |
| B | #6 Capture timestamp precision | Engineer 1 | DSP and OS stages use the exact capture time, not truncated milliseconds |

---

## Part A: Fix #5, DSP adapter rejection behavior

**Problem.** When the DSP filter rejected an observation (duplicate or non-increasing timestamp) or could not use it
(non-finite input, numerical error), the adapter handed back a substitute value (the last output, or the raw clamped
input). The engine could not tell that apart from a real smoothed point, so the substitute could be published to the
OS stage as if it were a measurement.

**What changed**

| File | Change |
|------|--------|
| `euro_builtin.hpp` | New `enum class FilterOutcome { Ok, Rejected, Invalid }`. `IPointFilter` gets a new `update(x, y, t)` that returns the outcome. Its default calls `filter()` and returns `Ok`, so filters that cannot fail (built-in 1-Euro) need no change. |
| `dsp_adapter.hpp` | `update()` now reports what happened. **Ok**: smoothed point. **Rejected**: duplicate/non-increasing timestamp, history intact. **Invalid**: non-finite input, InvalidInput or NumericalError. The filter is reset and there is no valid output. For non-finite input the adapter now also resets the filter and clears its cached last output (before, it just left the values alone). The old `filter()` stays and gives the same x/y values as before; the outcome is dropped. |
| `freshness_gate.hpp` | The gate calls `update()`. **Rejected**: replays the cached filtered point flagged HELD (with no cache, nothing is published), and the filter history is untouched. **Invalid**: publishes nothing, clears the cache, and the next observation starts a new track. Two new counters: `rejected` and `invalid`. |
| `os_bridge.hpp` | Hand and gaze now share one `convert(...)` path through a `FreshnessGate` each (`hand_gate_`, `gaze_gate_`). The old separate gaze path is removed. Gaze therefore gets the same outcome handling: no substitute is published, and the gate's cache and filter are reset together when the filter is rebuilt. |
| `test_dsp_adapter.cpp` | New tests: `update()` reports Ok/Rejected/Invalid, and the gate handles adapter rejection and invalid input with Eng 4's real filter. The tests check that a duplicate timestamp leaves no trace in the filter history. |
| `test_all.cpp` | New test: gate filter outcomes (Rejected / Invalid) using a scripted filter, covering hand and gaze. |

**Behavior to be aware of.** On a Rejected observation the OS stage now receives the previous point flagged
`Held`, not the substitute. On an Invalid observation there is no point at all (same as a lost track).

---

## Part B: Fix #6, Capture timestamp precision

**Problem.** The engine had a single timestamp, `ts_ms`: integer milliseconds, forced strictly increasing because
MediaPipe VIDEO mode needs that. The DSP filter and OS stage used it as well, so they got truncated milliseconds and
artificial +1 ms bumps. At 60 fps a 16.67 ms frame gap became a whole 16 or 17 ms, and frames in one camera burst
got fake spacing.

**What changed**

| File | Change |
|------|--------|
| `frame_clock.hpp` (new) | `frameTimes(pipeline_start, captured_at, last_ts_ms)` returns two clocks: `ts_ms` (integer ms, strictly increasing, MediaPipe only) and `capture_s` (seconds from the original capture time point, nanosecond resolution, no truncation, no bump). Header-only, so it is unit-testable. |
| `tracking_types.hpp` | `TrackingResult` gets `capture_s` (default `-1.0` = not provided). |
| `main.cpp` | Includes `frame_clock.hpp`, computes both timestamps with `frameTimes()`, and fills `r.capture_s`. MediaPipe still gets `ts_ms`. |
| `os_bridge.hpp` | The DSP and OS time base is `capture_s`. It falls back to `ts_ms / 1000` only when a result does not carry `capture_s` (< 0). |
| `test_all.cpp` | New test: `capture_s` is exact (not truncated), `ts_ms` is strictly increasing even for frames in the same millisecond, and the MediaPipe bump never leaks into `capture_s`. It includes a 20,000-frame random-timing run. |

**Known limit (from the code comments).** `capture_s` is the moment `cv::VideoCapture::read()` returns, not the sensor
exposure time. After a camera stall, buffered frames are read back to back and get nearly identical stamps. That is a
limit of the camera driver interface, not of this code.

---

## Files touched by both parts

| File | Resolution |
|------|------------|
| `os_bridge.hpp` | Part B's copy already contained all of Part A's changes plus the `capture_s` change, so B's copy is used. |
| `test_all.cpp` | Part B's copy already contained Part A's new test plus the clock test, so B's copy is used. |

No conflicts between the two parts.

## Verification

Run on Linux against your original project with the 9 merged files on top:

- `test_all.cpp`: 20 tests, 347 checks, 0 failed.
- `test_dsp_adapter.cpp`: 9 tests, 47 checks, 0 failed (built with Eng 4's `one_euro_filter.cpp`).
- **Not compiled here:** `os_bridge.hpp` and `main.cpp` (Windows-only). Please do a full Windows build and run
  `ctest --output-on-failure`.
- **Still open:** tracker items #1-#4 and #7-#9 are not covered by these two fixes. #8 (failed mouse-button
  release) is still the High-priority item to fix before unrestricted mouse control.
