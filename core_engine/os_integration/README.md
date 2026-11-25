# Engineer 5 — OS integration & GUI

This module turns stabilized tracking output into real Windows mouse input,
and gives the user a tray app with a calibration wizard. It is Phase 4 of the
frame's journey:

```text
C++ core (Eng 1) grabs frame
  -> Python AI (Eng 2 hands, Eng 3 eyes) returns (x, y, state)
  -> DSP 1-Euro filter (Eng 4) returns stabilized x, y
  -> OS integration (Eng 5, this folder)
       map x, y to the screen          (ScreenMapper)
       turn states into clicks         (GestureInterpreter)
       keep clicks on target           (CursorLock)
       pick hand or gaze               (SourceSelector)
       fire SendInput()                (win32::InputInjector)
  -> the mouse moves / clicks
```

Everything lives in `core_engine/os_integration/`. Engineer 5 changes only this
folder; the teammates' code is used, never edited.

**In the engine today.** On `first-full-working-demo` the root `CMakeLists.txt`
adds this folder and links `os_integration_tray` into `core_engine.exe`.
`core_engine/include/os_bridge.hpp` (Engineer 1) hosts the `TrayApp`, submits
one `FramePayload` per frame with `TrayApp::submit_frame()` and rebuilds the DSP
filters when the tray's Smoothing menu changes (`TrayOptions::on_settings_changed`).

## What is in the folder

| Path | What it is |
| --- | --- |
| `include/os_integration/`, `src/` | The library. Portable C++17 core plus a small Win32 layer. |
| `tools/os_demo.cpp` | Phase 1 deliverable: proves the program controls the Windows cursor. |
| `app/` | `handeye_tray.exe`: system-tray app, settings, and calibration wizard. |
| `capi/` | `os_integration_capi.dll`: a plain-C interface to `OsController` (used by the tester). |
| `tester/` | **HandEye Tester**: `HandEyeTester.exe` + a test-bench UI for every part of the system. |
| `tests/` | 120 library and tray tests, 10 C-interface tests and a latency benchmark (no GoogleTest needed). |

### Library files

| Header | Responsibility |
| --- | --- |
| `types.hpp` | The hand-off contract: `FramePayload`, state enums, points, actions. |
| `screen_mapper.hpp` | Normalized camera space → screen pixels → SendInput's 0..65535. |
| `gesture_interpreter.hpp` | Debounces per-frame states and emits edge-triggered actions. |
| `cursor_stabilizer.hpp` | Click lock (anchor/drag), position history, hand/gaze source selection. |
| `os_controller.hpp` | **The one entry point**: `update(FramePayload)` once per frame. |
| `input_sink.hpp` | Output interface, plus a recording sink for tests and dry runs. |
| `win32_input.hpp` | `SendInput` injector, monitor enumeration, DPI awareness (Windows only). |
| `settings.hpp` | Every tunable value, with validation, INI load/save and smoothing presets. |
| `calibration.hpp` | Maths for the wizard: region fitting and blink/wink timing. |
| `physical_input_guard.hpp` | Gives the real mouse priority over the virtual one. |

## Input contract (what Engineer 1 hands over)

```cpp
struct FramePayload {
    double timestamp_seconds;   // monotonic, the same clock the DSP filters use
    TrackerSample hand;         // { optional<NormalizedPoint> point; int state; TrackerState tracking; }
    TrackerSample gaze;
};
```

* `tracking` says how fresh the sample is (added with the engine's freshness
  work; it defaults to `Tracked`, so `{point, state}` still compiles):
  * `Tracked`: observed this frame.
  * `Held`: a short dropout; `point` is the last smoothed point, replayed so the
    cursor does not stutter. A Held frame keeps the cursor and keeps a gesture
    that is already confirmed (a drag survives), but it can never start or
    confirm a gesture, and calibration ignores it (`fresh_point()`).
  * `Lost`: no usable position; any point is ignored.

* `point` is the **stabilized** DSP output. Leave it empty when the tracker lost
  its target or DSP returned no value. Python's `(-1, -1)` sentinel, NaN and
  anything outside `[0, 1]` are also treated as "no point".
* `state` is the **raw** integer from Python. States bypass DSP, as
  `dsp_filter/README.md` specifies.
  * Hand (`hand_tracker.py`): 0 neutral, 1 pinch, 2 fist.
  * Gaze (`gaze_intent.py`): −1 calibrating, 0 neutral, 1 left wink, 2 right wink, 3 sustained closure.
  * Any other integer is treated as neutral and counted in `FrameReport::unknown_states`.
* Frames with a timestamp that isn't finite, or isn't newer than the previous
  frame, are rejected and nothing changes.

## Gesture → action map (defaults)

| Gesture (confirmed after debouncing) | Action |
| --- | --- |
| Pinch starts / ends | Left button down / up, so pinch-and-move is a **drag** |
| Fist (held 0.30 s) | Right click. Configurable: none, middle click, toggle pause. |
| Left wink (held 0.25 s) | Left click |
| Right wink (held 0.25 s) | Right click |
| Eyes closed (held 1.0 s) | Toggle pause. Works while paused, so control can resume hands-free. |
| Gaze state −1 (calibrating) | Gaze cursor holds still, as `gaze_intent.py` requests |

## How it works

### 1. Mapping to the screen (`ScreenMapper`)

1. **Region → unit square.** An `InputRegion` is the part of camera space that
   should cover the whole screen: `u = (x − x_min) / (x_max − x_min)`, clamped
   to [0, 1].
   * The default hand region is the central 70% of the frame. Small, relaxed
     hand movements then reach every screen edge, which counters "gorilla arm".
   * For gaze, the calibration wizard measures the band of the yaw/pitch
     coordinates (`0.5 + degrees / 60`) the user actually moves in.
   * `x_min > x_max` is allowed and flips that axis. Calibration produces this
     automatically when an input runs opposite to the screen.
2. **Unit → pixel:** `px = left + round(u · (width − 1))` inside the target
   monitor. Monitors to the left of or above the primary have negative
   coordinates, which is handled.
3. **Pixel → SendInput.** Absolute mouse coordinates are not pixels: they run
   0..65535 across the virtual desktop (`MOUSEEVENTF_VIRTUALDESK`). Windows
   converts back with `pixel = floor(abs · width / 65536)`, so we send
   `abs = ceil(offset · 65536 / width)`, the smallest value that lands exactly
   on the requested pixel.
   * A test checks this round trip for **every pixel** of 16 widths from 1 to
     65536 (484,161 pixels).
   * On real hardware, `os_demo trace` measured **0.00 px error** on a
     two-monitor 3840×1080 desktop.

### 2. Gestures (`StateDebouncer`, `GestureInterpreter`)

The Random Forest and the pinch heuristic classify each frame on its own, so a
single wrong frame is common. A state is **confirmed** only after it has
persisted for its onset time. For example, a wink must last 0.25 s, which rules
out a reflexive blink, and closing the eyes to pause needs 1.0 s. It also
works in the other direction: neutral must persist for the release time before
a held pinch lets go.

* **Glitch tolerance:** a different state lasting up to 0.05 s (one frame at
  30 FPS) is ignored, so one flicker doesn't restart the timer.
* **Stalled stream:** if frames stop for more than 0.25 s, partial evidence is
  discarded.
* **Edge-triggered actions:** an action fires on the transition into a
  confirmed state, never repeatedly while the state is held.

### 3. Clicks that land where you pointed (`CursorLock`)

Pinching pulls the index fingertip (the hand cursor) toward the thumb, and a
wink corrupts the iris ratio (the gaze cursor). Without compensation, every
click would land tens of pixels away from its target. So:

* **Snap back.** As soon as a click gesture *starts forming*, the cursor
  returns to where it was 0.10 s earlier. A 64-entry position history per
  channel makes this possible.
* **Hold.** The cursor stays there until the gesture ends, plus 0.15 s so
  double-clicks line up.
* **Escape by moving.** Moving more than 30 px either
  * starts a **drag**, if the pinch holds the button (the cursor follows with a
    constant offset, so it doesn't jump), or
  * releases the lock, if no button is held (the user was just moving).
* **Gaze never escapes** a lock, because eye "movement" during a wink is noise.

### 4. Choosing hand or gaze (`SourceSelector`)

The default is **hand preferred**: the hand drives the cursor while it is
visible, and gaze takes over when it isn't. A switch only happens after the
new choice has been stable for 0.2 s, so a flickering hand detection can't
make the cursor jump back and forth. *Hand only* and *gaze only* are also
available.

### 5. Safety

| Risk | Protection |
| --- | --- |
| The program takes over the mouse unexpectedly | It starts **paused**. Ctrl+Alt+P, the tray menu or a 1 s eye closure toggles control. |
| A cursor move is refused | It is not counted as sent. The target stays owed, also if tracking is lost: it is retried, and no press or click is sent until Windows accepts it. A drop waits up to 0.3 s for its move, then releases where the cursor is. |
| The tester changes mode or output | It releases held buttons before loading anything, keeps retrying refused releases (camera mode included) and releases after 0.5 s without camera frames. A watchdog thread does this, and handles Esc and Stop, even while a camera read is blocked. |
| A saved Activate turns the real mouse on | The tester accepts Activate only for the configuration that is already running; Start, Stop and any option change cancel it. |
| A gesture turns the real mouse on | In the tester's Real mouse mode, eye closure (and a fist set to `toggle_pause`) can pause the output but never resume it; only Activate or Space does. Simulation and the tray keep hands-free resume. |
| A broken settings file loses calibration | A file with errors is never saved over (tester and tray) or reloaded. Saves edit the file as it is at that moment and keep `<file>.bak`. A UTF-8 byte-order mark is accepted. |
| A button stays stuck down | Buttons are released on pause, inhibit, settings change, stale input (0.5 s without frames), controller destruction and app exit. A release Windows **refuses** keeps the button tracked as held and is retried every frame and every 250 ms by the tray, with a warning, until Windows accepts it; a click whose release was dropped is detected (`ClickResult::PressedOnly`) and released the same way (team tracker item #8). |
| Fighting the user's real mouse | A low-level hook detects **physical** mouse input (not injected input) and yields control for 1.5 s. |
| Clicks during calibration | The controller is inhibited while the wizard runs. |
| Windows blocks the input | `SendInput` refusals are detected and reported (UIPI: an admin window has focus). |
| Two instances fight over the cursor | A single-instance mutex prevents it. |

### 6. Calibration wizard (`calibration.hpp`, `app/calibration_wizard.cpp`)

**Hand range / gaze range**

1. Five targets appear in turn: the centre, then the corners at 12% and 88%.
2. For each target, the wizard records about 1.2 s of input after a 0.9 s
   settle time.
3. It fits `input = offset + slope · screen` per axis, using least squares on
   the per-target **medians** so outlier frames don't matter.
4. The input at screen = 0 and screen = 1 is exactly the new `InputRegion`.

The fit is rejected, with a plain-language reason, when there is too little
movement, too few tracked frames, or an average error above 12% of the screen.
The average error is `sqrt(mean(error_u^2 + error_v^2))` over the targets, in
screen fractions. The tray log lists every target (screen position, sample
count, median input, error) and the fitted region and error, for accepted and
rejected fits. Every raw sample is saved beside the log as
`calibration_<hand|gaze>_<date>_<time>.csv` (`target_u,target_v,x,y`, full
precision), from which `fit_input_region` reproduces the fit exactly.

**Blink & wink timing**

This is the "Look here / Blink now" step from the plan.

1. It waits until `gaze_intent.py` has finished its own 30-frame baseline
   calibration.
2. It records natural blinks, then deliberate left winks, then deliberate right
   winks.
3. It picks a wink threshold between the longest blink the model mistook for a
   wink and the user's typical deliberate wink.
4. It sets the pause (eyes-closed) time well above the user's natural blinks.

**Filter sensitivity.** The tray's *Smoothing* menu scales Engineer 4's minimum
cutoff: ×2 Responsive, ×1 Balanced (their defaults), ×0.5 Smooth. The settings
store the values in `[hand_filter]` / `[gaze_filter]`, ready for the core loop
to build its `dsp::FilterConfig`.

## Build and test

These use the Visual Studio 2022 **x64 Native Tools / Developer** prompt, from
the repository root. No vcpkg, OpenCV, Python or other team module is needed to build this module.

```powershell
cmake -S core_engine/os_integration -B out/build/os-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build out/build/os-debug
ctest --test-dir out/build/os-debug --output-on-failure

cmake -S core_engine/os_integration -B out/build/os-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DOSI_BUILD_BENCHMARKS=ON
cmake --build out/build/os-release
.\out\build\os-release\os_integration_tests.exe
.\out\build\os-release\os_integration_benchmark.exe
```

| CMake option | Default | Meaning |
| --- | --- | --- |
| `OSI_BUILD_APPS` | ON | Build `os_demo` and `handeye_tray` (Windows only). |
| `OSI_BUILD_TESTER` | ON | Build `HandEyeTester.exe` and `os_integration_capi.dll` (Windows only). |
| `OSI_BUILD_BENCHMARKS` | OFF | Per-frame latency benchmark. |

Warnings (`/W4 /permissive-`) apply only to Engineer 5 targets.

### Verified on the development machine

| Check | Result |
| --- | --- |
| Debug and Release builds, MSVC 19.44 `/W4 /permissive-` | 0 warnings |
| `os_integration_tests` | 120 / 120 passed |
| `os_integration_capi_tests` | 10 / 10 passed |
| Tester logic (`pytest core_engine/os_integration/tester/tests`) | 137 / 137 passed |
| `os_integration_benchmark` (Release, 1M frames) | 66 ns per frame p50 |
| `os_demo trace` (real cursor, 2 monitors) | 0.00 px max error |
| `os_demo selftest` | Left, right and middle click and drag all arrived at the exact pixel |
| HandEye Tester, live camera | Index finger, hand and full-system modes ran at 30 fps with the teammates' trackers |

## Running it

### Phase 1 demo: prove the OS can be controlled

```powershell
.\out\build\os-release\os_demo.exe list                       # monitors
.\out\build\os-release\os_demo.exe trace --seconds 5          # cursor draws a circle; reports accuracy and latency
.\out\build\os-release\os_demo.exe trace --pattern square --screen all
.\out\build\os-release\os_demo.exe selftest                   # clicks only inside its own test window
```

Pressing Esc or moving the real mouse stops `trace` and `selftest`.

### The tray app

1. Start it: `.\out\build\os-release\handeye_tray.exe --dry-run --simulate`
   * An icon appears in the notification area.
   * Colours: grey ring = waiting for data, amber = paused, green = in control,
     blue = yielding to the real mouse, purple = calibrating.
   * `--dry-run` logs clicks instead of performing them. This is a safe first run.
   * `--simulate` feeds a synthetic hand cursor (a slow circle that never
     clicks), standing in for the core engine until it sends real frames.
2. Right-click the icon to see the menu: cursor source, screen, smoothing, and
   **Calibrate** (hand range, gaze range, blink & wink timing). You can also
   launch with `--calibrate hand|gaze|eyes`. Calibration needs real tracking
   frames from the core engine.
3. Press **Ctrl+Alt+P** to take or release control.

Settings live in `%APPDATA%\HandEyeTracker\os_integration.ini`, which can be
opened and reloaded from the menu. The log is at
`%LOCALAPPDATA%\HandEyeTracker\handeye_tray.log`.

### HandEye Tester

A test bench for every part of the system, one part at a time. **Double-click
`HandEyeTester.exe`** (built next to `os_integration_capi.dll`, for example in
`out\build\windows-release\core_engine\os_integration\`). It needs the
project's `.venv` (`setup.bat`).

| Test | What runs |
| --- | --- |
| Camera check | Only the webcam: picture, frame rate, brightness |
| Index finger only | Engineer 2's `IndexFingerTracker`: the cursor follows the fingertip, no clicks |
| Hand + gestures | Engineer 2's `HandTracker`: cursor, pinch (left button / drag), fist (right click) |
| Gaze pointer | Engineer 3's `GazeIntent`: cursor from yaw/pitch (same mapping as the engine), wink clicks |
| Blinks & winks only | Engineer 3's eye states only; the cursor does not move |
| Full system | Hand and gaze together, with the engine's rules |

* The trackers run **unchanged**; their output goes through Engineer 4's 1-Euro
  filter (optional) into **Engineer 5's real `OsController`** via
  `os_integration_capi.dll`.
* **Simulation** (default) never touches the mouse: a map shows where the
  cursor would go. **Real mouse** starts paused; Esc pauses instantly and
  moving the real mouse takes over for 1.5 s.
* Live camera overlays (pointer, trail, hand box, active zone, eye and iris
  points, gaze arrow), a Live data tab (every tracker and controller value and
  counter), live **Tuning** sliders (zones, gesture times, smoothing; saving to
  your settings is optional), a 10-target **accuracy test**, **CSV recording**
  (Documents\HandEyeTests) and a **Run checks** button.
* **Calibration is preserved.** Start never rewrites settings. A slider edits
  only its own value (a range slider: one axis, about its own centre and in its
  own direction). Save applies just the edited values to the file as it is at
  that moment, so a calibration saved by the tray meanwhile survives. A file with
  errors is left untouched: correct it, or use Reset and then Save.
* **Comparing with core_engine.** Low-light enhancement, lost-hand recovery and
  IndexFingerTracker's own smoothing default to off, because the engine has no
  such stages. The screen starts from `screen_target` and the camera is asked
  for MJPG like the engine. A label lists any option that differs, unsaved
  tuning included. The tester runs the
  Python trackers, so its timings are **not** core_engine's latency.
* **Smoothing** follows `freshness_gate.hpp`: a rejected timestamp replays the
  last filtered point as Held, an invalid input publishes nothing.
* Run it without the launcher: `.venv\Scripts\python.exe core_engine\os_integration\tester\handeye_tester.py`.
  Log: `%LOCALAPPDATA%\HandEyeTracker\tester.log`.

## Integration with the engine

Already done by Engineer 1 on `first-full-working-demo` (`core_engine/include/os_bridge.hpp`):
the engine hosts a `TrayApp`, calls `submit_frame()` once per processed frame,
`request_exit()` at shutdown, and rebuilds its DSP filters from
`TrayOptions::on_settings_changed`. Anything that changes `FramePayload` or the
`TrayApp` API must be agreed with Engineer 1, because `os_bridge.hpp` is the
only file that knows both sides.

Other tools can drive the controller without C++ through `capi/osi_capi.h`
(create, update, apply settings, drain recorded events, smoothing filter); see
`tester/osi_capi.py` for a complete ctypes binding.

## Known limitations and next steps

* **Uncalibrated gaze defaults & live verification (audit finding #1):** on
  `exp/gaze-e3-e5-joint-candidate`, the gaze models, the fixed vertical Y sign
  (`y = 0.5 - pitch / 60`) and the default `gaze_region`
  (`[0.25, 0.58, 0.75, 0.82]`) align all four gaze directions with the screen
  axes and keep a straight-ahead gaze inside the active region. Run the tray's
  Calibrate > Gaze range wizard for a per-user fit; it saves
  `gaze_calibrated = true`. Live cued fixation and wizard verification remain
  pending before merge to `first-full-working-demo`.
* **Settings saved before the Y-sign fix** are converted on load. A file is from
  before the fix when it has neither `settings_version` nor `gaze_calibrated`.
  Then an old placeholder (`0.3 0.3 0.7 0.7`, `0.25 0.25 0.75 0.75` or
  `0.25 0.60 0.75 0.80`) becomes the current default with
  `gaze_calibrated = false`, and any other gaze region gets y -> 1 - y (the same
  pixel for the same pitch) with `gaze_calibrated = true`. New files are written
  as `settings_version = 2`. An older build reading a version 2 file would map
  gaze upside down, so do not mix builds from before and after this change.
* **Files saved by experimental builds from `8f98cfa` up to `7070bd7`** (22 to
  23 Nov 2026, development machines only) already use the corrected direction
  but carry neither marker, so they would be converted a second time (gaze
  upside down). Before the first start of a current build: back up
  `%APPDATA%\HandEyeTracker\os_integration.ini`, then either add
  `settings_version = 2` under `[app]` or run the gaze calibration again
  (`settings_migration_window_files_are_repaired_by_a_version_line`).
* **Gaze accuracy:** the estimate from `gaze_intent.py` is coarse, so gaze
  pointing is best for large targets. Hand pointing is the precise mode.
* **Elevated windows:** Windows blocks injected input into apps running as
  administrator unless the tray app is also elevated. This is reported, not
  bypassed.
* **Live tuning still to do:** gaze range and direction and the wink timing
  must be checked with a person in front of the camera (the tester's Gaze and
  Winks modes, then the tray wizard). The defaults are starting values.
* **Calibration ownership (team decision, 22 Nov 2026):** this module's wizard
  and `os_integration.ini` are the only runtime calibration. Engineer 3's
  `run_gaze.py` stays a standalone diagnostic.
* **Scrolling:** there is no scroll gesture yet. Adding one means a new
  `ActionType` plus a `MOUSEEVENTF_WHEEL` input.
