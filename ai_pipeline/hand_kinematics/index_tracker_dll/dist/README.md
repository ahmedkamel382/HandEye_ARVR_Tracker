# Index Finger Tracker DLL

C-compatible x64 DLL for the existing `IndexFingerTracker`. It exposes landmark 8
as normalized pointer coordinates and preserves the Python tracker's mapping,
mirroring, EMA smoothing, tracking-loss, and invalid-result behavior.

## Quick Test

Run the complete build and test workflow from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_pipeline\hand_kinematics\index_tracker_dll\scripts\test.ps1
```

The runner checks the environment, builds `Release` with the existing
`scripts/build.ps1`, verifies the build artifacts, and tests the DLL through its
exported C ABI. It does not install packages, download assets, modify Git, move
the mouse, or create OS clicks. It builds into `build/bin/Release` and does not
restage `dist/` during a test run.

By default it runs the deterministic mock API tests and a real
Python/MediaPipe initialization and no-hand-loss test. A hand image is optional
because no test image is committed:

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_pipeline\hand_kinematics\index_tracker_dll\scripts\test.ps1 `
  -HandImage "C:\path\to\hand.jpg"
```

The image must contain a visible hand. Supported formats are the image formats
supported by Windows Imaging Component, such as JPEG, PNG, and BMP.

The live-camera stage is opt-in and uses the selected camera index:

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_pipeline\hand_kinematics\index_tracker_dll\scripts\test.ps1 `
  -Camera -CameraIndex 0 -CameraSeconds 10
```

By default the camera stage is headless: it does not create a window or send
pointer/click events. The camera is opened only for this stage and is released
in a `finally` block. The probe auto-stops after `-CameraSeconds` (1-120
seconds). A requested camera that cannot be opened or does not produce a
tracked hand is reported as `FAIL`, not silently skipped.

Add `-CameraPreview` to show a live, mirrored OpenCV window with a green
landmark-8 marker, `TRACKED`/`LOST` status, and FPS. The marker follows the
DLL's `camera_x`/`camera_y` output. The raw camera frame is passed to the DLL;
only the preview copy is mirrored. Press `q` or `Esc` to close the window early.
The camera, window, and tracker are released on exit. Neither mode moves the
system mouse or generates clicks. `-CameraPreview` also enables the camera stage
if `-Camera` is omitted.

For a 20-second webcam test with live preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_pipeline\hand_kinematics\index_tracker_dll\scripts\test.ps1 `
  -Camera -CameraIndex 0 -CameraSeconds 20 -CameraPreview
```

### Requirements

- Windows x64 and PowerShell 5.1 or newer.
- Visual Studio C++ x64 build tools and CMake. The build script can use the CMake
  bundled with Visual Studio.
- Python 3.11 x64 with `numpy`, `opencv-python` (or OpenCV with the `cv2`
  module), and `mediapipe`. The repository's `.venv` is used when present.
- The runtime files under `dist/python/` and the model at
  `dist/model/hand_landmarker.task`.

Use an explicit Python when the repository `.venv` is not the desired runtime:

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_pipeline\hand_kinematics\index_tracker_dll\scripts\test.ps1 `
  -PythonExecutable "C:\path\to\python.exe"
```

Use `-SkipBuild` only when the existing `build/bin/Release` output is known to be
current. The runner uses that build output directly; it does not test a stale
copy from another directory.

### Stages and results

The runner reports these stages separately:

- `Environment preflight`: Windows/PowerShell, repository layout, Python and
  imports, runtime assets, and (unless `-SkipBuild`) CMake and VS C++ tools.
- `DLL build` and `Build artifacts`: Release build, DLL/import library, and test
  executables. Build errors stop the run.
- `Exported API smoke tests (mock backend)`: deterministic DLL load, symbols,
  config/ABI validation, valid and lost results, strides, argument errors,
  repeated calls, and shutdown.
- `MediaPipe initialization and no-hand loss`: actual embedded Python/MediaPipe
  initialization, no-hand LOST results, and cleanup.
- `Landmark tracking / loss / reacquisition`: only when `-HandImage` is supplied;
  validates landmark 8, normalized ranges, padded frames, loss reset, and
  reacquisition.
- `Live camera test`: only when `-Camera`, `-CameraPreview`, or `-CameraIndex` is
  supplied. Add `-CameraPreview` for the live OpenCV window.

`PASS` means the stage ran and its assertions passed. `FAIL` means it ran and
failed, or a requested stage could not run; the exit code is nonzero. `SKIP`
means an optional stage was intentionally not run and does not fail the run.
All required stages must pass for exit code `0`.

### Common setup errors

- `Python not found`: create the repository environment with
  `py -3.11 -m venv .venv` and install the packages listed above, or pass
  `-PythonExecutable`.
- `missing Python package ...`: install `numpy`, `opencv-python`, and
  `mediapipe` into the selected Python. The runner does not install them.
- `Visual Studio C++ x64 build tools not found` or `CMake not found`: install
  the Visual Studio Desktop development with C++ workload and CMake.
- `missing ...dist\...`: restore the tracked `dist/python` files and
  `dist/model/hand_landmarker.task`.
- `cannot open camera`: close other camera applications and try another
  `-CameraIndex`.
- `cannot create preview window`: run from an interactive Windows desktop
  session with an OpenCV build that supports GUI windows; headless sessions can
  still run without `-CameraPreview`.
- `no hand was tracked`: use a well-lit image or camera frame with one visible
  hand; real MediaPipe results vary with lighting, image quality, and hardware.

Real-backend checks depend on MediaPipe and the local image/camera. The runner
uses fresh `build/bin/Release` output and validates the current machine only;
teammates may still need to install the tools and Python packages above.

## 1. Existing eye-tracking integration: what was found

The repository has no C++ eye-tracker interface DLL. The three files under
`ai_pipeline/gaze_intent/models/` are Treelite/TL2cgen model libraries:

- `gesture_model.dll` (wink/eye-state Random Forest)
- `gaze_pitch.dll`
- `gaze_yaw.dll`

They are x64 PE DLLs with only `KERNEL32.dll` as an import dependency. Their
exports are model-inference functions:

```text
get_leaf_output_type
get_num_class
get_num_feature
get_num_target
get_threshold_type
predict
```

`ai_pipeline/gaze_intent/gaze_intent.py` loads them from Python with
`tl2cgen.Predictor`; MediaPipe feature extraction, calibration, smoothing, and
the `GazeIntent.process_frame` contract stay in Python. The root CMake project
embeds Python through `pybind11::embed`, and the C++ core calls the Python
modules. There is therefore no frame-processing eye ABI to copy.

## 2. Chosen architecture

The index tracker uses a native x64 DLL with a C ABI that embeds CPython and
imports the existing `index_tracker.py`. The C++ caller never handles Python
objects. The DLL:

1. Initializes CPython if the host has not already done so.
2. Adds the configured Python source and site-packages directories.
3. Imports `index_tracker.IndexFingerTracker`.
4. Copies each BGR frame into a NumPy array and calls `track()`.
5. Returns a plain C result and keeps the GIL/host runtime private.

This reuses the project's Python/MediaPipe deployment model and preserves the
validated tracker behavior. A native C++ MediaPipe port would require a separate
C++ MediaPipe runtime, a native model loader, and a reimplementation of the
tracker contract, with a high risk of behavior drift.

Trade-offs:

- Runtime requires Python 3.11 x64, NumPy, OpenCV, MediaPipe, the model file,
  and the Python tracker source.
- Processing is synchronous and holds the GIL while the tracker runs.
- The interpreter remains initialized until process exit; it is not finalized
  by `hand_index_destroy`.
- This is a C ABI wrapper around Python, not a dependency-free native tracker.

No additional smoothing, filtering, prediction, deadband, gesture logic, or OS
cursor control was added.

## 3. Exported API

Header: `include/hand_index_tracker.h`

```c
const char*     hand_index_version(void);
uint32_t        hand_index_abi_version(void);
const char*     hand_index_error_string(hand_index_error error);

void            hand_index_config_default(hand_index_config* config);
void            hand_index_result_init(hand_index_result* result);

hand_index_error hand_index_create(
    const hand_index_config* config,
    hand_index_tracker** out_tracker,
    char* error_message,
    size_t error_message_capacity);

hand_index_error hand_index_process_bgr(
    hand_index_tracker* tracker,
    const uint8_t* bgr_data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    hand_index_result* out_result,
    char* error_message,
    size_t error_message_capacity);

void            hand_index_destroy(hand_index_tracker* tracker);
```

All functions use the C calling convention (`__cdecl` on x86; the x64 ABI is
unified). Strings are UTF-8. The DLL exports only the `hand_index_*` API.

`hand_index_error` values:

| Value | Meaning |
|---|---|
| `HAND_INDEX_OK` | Success |
| `HAND_INDEX_ERROR_INVALID_ARGUMENT` | Bad pointer, dimension, stride, config, or ABI size |
| `HAND_INDEX_ERROR_PYTHON` | Python/import/processing failure |
| `HAND_INDEX_ERROR_MODEL` | Missing or unloadable MediaPipe model |
| `HAND_INDEX_ERROR_CLOSED` | Python runtime unavailable or tracker closed |
| `HAND_INDEX_ERROR_INTERNAL` | Allocation/internal failure |

## 4. Result and coordinate conventions

`hand_index_result` contains:

- `valid`: `1` only when landmark 8 was observed on this frame.
- `status`: `HAND_INDEX_STATUS_TRACKED` or `HAND_INDEX_STATUS_LOST`.
- `landmark_index`: always `8`.
- `x`, `y`: normalized `[0,1]`, region-mapped, EMA-smoothed pointer coordinates.
- `camera_x`, `camera_y`: normalized, clamped landmark-8 coordinates in the
  internally mirrored frame, before region mapping and before EMA smoothing.

Rules:

- X increases left-to-right and Y increases top-to-bottom in the internally
  mirrored frame.
- The Python tracker mirrors the incoming BGR frame internally. Pass the raw,
  unmirrored camera frame. Do not pass a frame that was already flipped for
  display, or the tracker will see a double-mirrored frame.
- Region mapping is applied once inside the tracker. Do not pre-map coordinates
  or later apply the same region again.
- Tracking loss returns `valid=0` with `x=y=camera_x=camera_y=-1.0f`. Coordinates
  from the previous valid frame are never returned as valid.
- A frame with no hand is `HAND_INDEX_OK` plus a LOST result, not an error.
- `stride_bytes=0` means tightly packed `width * 3` BGR8. Otherwise stride must
  be at least `width * 3`.
- The DLL copies the caller's frame for the duration of the call and retains no
  pointer to it.

## 5. Build

Prerequisites:

- Windows x64
- Visual Studio C++ build tools with the x64 toolset
- CMake
- Python 3.11 x64 with NumPy, OpenCV, MediaPipe, and the model

The repository's existing environment is used by default:

```powershell
cd C:\path\to\HandEye_ARVR_Tracker\ai_pipeline\hand_kinematics\index_tracker_dll
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

The script configures CMake, builds `Release`, verifies exports, runs the mock
and real smoke tests, and stages `dist/`. To also run the valid-hand real-backend
integration test, pass a hand image:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 `
  -HandImage "C:\path\to\hand.jpg"
```

No test image is committed to the repository. The integration test accepts a
user-supplied JPEG, PNG, BMP, or other Windows Imaging Component format.

Manual build:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DPython3_EXECUTABLE="..\..\..\.venv\Scripts\python.exe"
& $cmake --build build --config Release --parallel
```

Generated files:

```text
build/bin/Release/hand_index_tracker.dll
build/lib/Release/hand_index_tracker.lib
build/bin/Release/hand_index_example.exe
build/bin/Release/hand_index_smoke.exe
build/bin/Release/hand_index_real_backend_integration.exe
```

## 6. Runtime deployment

The C++ application needs:

1. `hand_index_tracker.dll`.
2. `hand_index_tracker.lib` at link time and `hand_index_tracker.h` at compile
   time.
3. Python 3.11 x64 with `numpy`, `opencv-python`/`opencv-contrib-python`, and
   `mediapipe`. The tested environment has MediaPipe `1.0.0`, NumPy `2.4.6`, and
   OpenCV `5.0.0.93`.
4. `index_tracker.py` and `hand_features.py`.
5. `hand_landmarker.task`.
6. `python311.dll` available to the process. Normally add the Python 3.11
   installation directory to `PATH`. If placing it beside the executable,
   ensure Python can still find its standard library (for example, set
   `PYTHONHOME` to the Python installation).
7. The Microsoft Visual C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`,
   `VCRUNTIME140_1.dll`).

Use `python_dir`, `site_packages_dir`, and `model_path` in
`hand_index_config` so the DLL does not depend on the process working directory
or environment variables. Relative paths are resolved against the current
working directory, so absolute paths are recommended.

The staged `dist/` layout is:

```text
dist/
  bin/hand_index_tracker.dll
  bin/hand_index_example.exe
  bin/hand_index_smoke.exe
  bin/hand_index_real_backend_integration.exe
  lib/hand_index_tracker.lib
  include/hand_index_tracker.h
  python/index_tracker.py
  python/hand_features.py
  model/hand_landmarker.task
  examples/index_tracker_example.cpp
  README.md
```

## 7. Minimal C++ integration

```cpp
#include "hand_index_tracker.h"
#include <opencv2/opencv.hpp>

hand_index_config config;
hand_index_config_default(&config);
config.python_dir = "C:\\path\\to\\dist\\python";
config.site_packages_dir = "C:\\path\\to\\.venv\\Lib\\site-packages";
config.model_path = "C:\\path\\to\\dist\\model\\hand_landmarker.task";

char error[1024] = {0};
hand_index_tracker* tracker = nullptr;
if (hand_index_create(&config, &tracker, error, sizeof(error)) != HAND_INDEX_OK) {
    // log error
}

cv::VideoCapture camera(0);
cv::Mat frame;  // raw BGR camera frame, not the display-mirrored copy
camera >> frame;

hand_index_result result;
hand_index_result_init(&result);
const hand_index_error status = hand_index_process_bgr(
    tracker,
    frame.data,
    frame.cols,
    frame.rows,
    static_cast<int32_t>(frame.step),
    &result,
    error,
    sizeof(error));

if (status == HAND_INDEX_OK && result.valid) {
    // result.x, result.y are normalized pointer coordinates
} else if (status == HAND_INDEX_OK) {
    // no hand: do not use the previous coordinates
}

hand_index_destroy(tracker);
```

`examples/index_tracker_example.cpp` is a complete no-camera example. Run it
with:

```powershell
.\build\bin\Release\hand_index_example.exe `
  ".\dist\python" `
  "..\..\..\.venv\Lib\site-packages" `
  ".\dist\model\hand_landmarker.task"
```

## 8. Tests and verification status

Verified in this environment:

- CMake configure with MSVC 19.51 and Python 3.11.9 x64.
- Clean Release build of the DLL, import library, smoke test, example, and
  real-backend integration test.
- `dumpbin /exports` shows all 8 `hand_index_*` exports.
- C++ smoke test links against the import library and calls the public API.
- Mock test validates valid results, config pass-through, stride-0 and padded
  frames, loss without stale coordinates, invalid arguments, old ABI sizes,
  repeated calls, and destroy.
- Real test initializes the actual Python environment and MediaPipe model, then
  submits three no-hand frames and verifies LOST results.
- `scripts/test.ps1` rebuilt and tested the release DLL, and the optional
  live-camera probe opened/released the camera and produced TRACKED results
  through the real DLL C ABI on this machine.
- `hand_index_real_backend_integration.exe` decoded a user-supplied image with
  Windows Imaging Component, linked and loaded the compiled DLL, and ran the
  actual embedded Python/MediaPipe backend. It verified:
  - a valid TRACKED landmark-8 result with finite normalized coordinates;
  - equivalence of `x/y` and `camera_x/camera_y` with full region and alpha 1;
  - a packed frame and a padded-stride frame produced agreeing results;
  - a blank frame returned LOST with all coordinates reset to `-1`;
  - the hand was reacquired after loss and returned to the original position.
- Existing Python checks pass: `check_index_tracker.py`, `check_tracker_logic.py`,
  `check_script_smoke.py`, `check_tracking_observability.py`,
  `check_preprocess.py`, `check_compare_benchmarks.py`, and
  `check_evaluate_hand_baseline.py`.

Run the integration test manually with the compiled DLL and real paths:

```powershell
$env:PATH = "<python-base>;$env:PATH"
.\build\bin\Release\hand_index_real_backend_integration.exe `
  "C:\path\to\hand.jpg"
```

Optional overrides:

```text
--python-dir <dir>       directory containing index_tracker.py
--site-packages <dir>    Python site-packages directory
--model <file>           hand_landmarker.task path
```

The test is real-backend-only. It has no mock mode and fails if the image cannot
be decoded or does not produce a hand detection.

Not verified here:

- Deployment on a machine different from this build machine.
- MediaPipe/OpenCV versions other than the installed environment.
- Video-file decoding; the integration test currently validates still images.
- Concurrent `hand_index_destroy` with an in-flight `hand_index_process_bgr`
  call. The caller must not destroy an instance while another thread is using it.

## 9. Status checklist

| Item | Status |
|---|---|
| Source complete | Yes |
| Build complete (x64 DLL + import library) | Yes |
| DLL load and API invocation | Yes |
| Valid-result path tested at DLL boundary | Yes, with deterministic mock |
| Real Python/MediaPipe initialization tested | Yes |
| Real valid-hand still-image test through the DLL | Yes |
| Real-camera valid-hand test | Yes, with optional `-Camera` test on this machine |
| Integration guide and example | Yes |
