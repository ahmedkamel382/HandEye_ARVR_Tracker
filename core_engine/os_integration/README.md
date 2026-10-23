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

Everything lives in `core_engine/os_integration/`. Nothing outside this folder
was modified: not the teammates' Python, not `dsp_filter/`, not
`core_engine/src/main.cpp`, and not the root `CMakeLists.txt`.

## What is in the folder

| Path | What it is |
| --- | --- |
| `include/os_integration/`, `src/` | The library. Portable C++17 core plus a small Win32 layer. |
| `tools/os_demo.cpp` | Phase 1 deliverable: proves the program controls the Windows cursor. |
| `app/` | `handeye_tray.exe`: system-tray app, settings, and calibration wizard. |
| `tests/` | 71 self-contained tests and a latency benchmark (no GoogleTest needed). |

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
    TrackerSample hand;         // { optional<NormalizedPoint> point; int state; }
    TrackerSample gaze;
};
```

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
   * For gaze, the calibration wizard measures the narrow band the iris ratio
     actually moves in.
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
| A button stays stuck down | Buttons are released on pause, inhibit, settings change, stale input (0.5 s without frames), controller destruction and app exit. |
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
| `OSI_BUILD_BENCHMARKS` | OFF | Per-frame latency benchmark. |

Warnings (`/W4 /permissive-`) apply only to Engineer 5 targets.

### Verified on the development machine

| Check | Result |
| --- | --- |
| Debug and Release builds, MSVC 19.44 `/W4` | 0 warnings |
| `os_integration_tests` | 71 / 71 passed (Debug and Release, clean build folders) |
| `os_integration_benchmark` (Release, 1M frames) | 64 ns per frame p50, 69 ns p99 |
| `os_demo trace` (real cursor, 2 monitors) | 0.00 px max error over 179 frames; about 0.2 ms per frame including SendInput (Debug) |
| `os_demo selftest` | Left, right and middle click and drag all arrived at the exact pixel |
| Tray app with `--simulate --dry-run` | Frames → controller → mouse events logged; clean exit verified |

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

## Integration guide for Engineer 1 (Phases 3–4)

The core loop owns frames, timestamps, Pybind11 and the DSP filters. It hands
this module one `FramePayload` per new observation:

```cpp
#include "os_integration/os_controller.hpp"
#include "os_integration/win32_input.hpp"

osi::win32::enable_per_monitor_dpi_awareness();            // once, at startup
const osi::Settings settings = osi::load_settings_file(path).settings;
osi::win32::InputInjector injector;
osi::OsController os(settings, osi::win32::query_screen_geometry(settings.mapping.screen_target), injector);
os.set_active(true);                                        // or leave paused for the user

// per frame, after Python + DSP:
osi::FramePayload frame;
frame.timestamp_seconds = t;
if (filtered_hand.point) frame.hand.point = osi::NormalizedPoint{filtered_hand.point->x, filtered_hand.point->y};
frame.hand.state = hand_state;          // raw Python integer
if (filtered_gaze.point) frame.gaze.point = osi::NormalizedPoint{filtered_gaze.point->x, filtered_gaze.point->y};
frame.gaze.state = gaze_state;
const osi::FrameReport report = os.update(frame);   // ~64 ns + SendInput
```

**To keep the tray app, settings and wizard as well,** the core hosts a
`TrayApp` instead of a bare `OsController`. It runs the tray on its own thread
and submits each frame; `app/tray_main.cpp` (`--simulate`) shows exactly this:

```cpp
#include "tray_app.hpp"

osi::app::TrayOptions options;               // settings/log paths, dry run, ...
options.on_settings_changed = [](const osi::Settings& s) {
    // rebuild dsp::TrackingFilter2D from s.hand_filter / s.gaze_filter
};
osi::app::TrayApp tray(GetModuleHandleW(nullptr), options);
std::thread ui([&] { tray.run(); });           // tray icon, menu, wizard

// per frame, from the core loop thread (thread-safe):
tray.submit_frame(frame);

// on shutdown:
tray.request_exit();
ui.join();
```

To add this module to the root build, one line goes in the root
`CMakeLists.txt` after `add_subdirectory(dsp_filter)`, then link
`os_integration_win32` (controller only) or `os_integration_tray` (with the
tray app). That file belongs to Engineer 1, so this line was not added here:

```cmake
add_subdirectory(core_engine/os_integration)
```

## Known limitations and next steps

* **Gaze accuracy:** the iris ratio from `gaze_intent.py` is coarse, so gaze
  pointing is best for large targets. Hand pointing is the precise mode.
* **Hand model:** `assets/hand_landmarker.task` must be added to the repo (or
  downloaded) before hand tracking runs.
* **Elevated windows:** Windows blocks injected input into apps running as
  administrator unless the tray app is also elevated. This is reported, not
  bypassed.
* **Live tracking:** real camera frames reach this module only once the core
  loop (Phases 2 and 3) is connected. Until then, `os_demo` and `--simulate`
  demonstrate the OS side.
* **Scrolling:** there is no scroll gesture yet. Adding one means a new
  `ActionType` plus a `MOUSEEVENTF_WHEEL` input.
