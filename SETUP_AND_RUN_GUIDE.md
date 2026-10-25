# HandEye AR/VR Tracker - Setup and Run Guide (Windows)

This guide takes a fresh Windows PC from nothing to a running tracker. Follow the steps in order.
Read section 1 first: it explains why the program has to be built on each computer.

---------------------------------------------------------------------------------------------------

## 0. What went wrong on the first attempt (the screenshot)

The photo shows the engine starting correctly (camera opened, "Starts PAUSED" printed) and then stopping
with exit code 2 because the embedded Python could not start. The line, as far as it can be read from the
photo, is:

    DLL load failed while importing _ctypes: Access is denied.

Notes:

* The many `INFO ... opencv_core_parallel ... FAILED` lines are NORMAL. OpenCV is probing optional plug-ins.
  Ignore them.
* The camera part worked. The problem is Python.
* "Access is denied" means Windows refused to let the program read a file of the Python installation.
  The usual causes, most likely first: (1) Python was installed from the Microsoft Store, (2) antivirus or
  Windows security blocked a Python file, (3) Python lives in a folder this user cannot read.
  The fix is in section 6, problem A. The short version: install Python 3.11 from python.org for your own user,
  delete `.venv` and `out`, run `setup.bat`, build again.
* The friend also ran the Debug build from Visual Studio. Use the Release build from the command line instead
  (section 4): it is the one that is fast enough for tracking.

---------------------------------------------------------------------------------------------------

## 1. Two things to understand before you start

1. **The program is built on the machine that runs it.** The build bakes the location of this computer's
   Python and its virtual environment into the executable. Do NOT copy `core_engine.exe` or the `out` folder
   to another PC and expect it to work. Every PC clones the repo and builds it.
2. **Nothing is installed globally by the project.** Python packages go into a `.venv` folder inside the repo,
   and OpenCV is built by vcpkg inside `out\`. To start over, delete `.venv` and `out` and repeat the steps.

---------------------------------------------------------------------------------------------------

## 2. Requirements

| Item | Needed | Notes |
|---|---|---|
| Windows | 10 or 11, 64-bit | The mouse control and tray app are Windows only. |
| Webcam | Any USB or built-in webcam | 720p/480p is fine. Good, even light makes it faster and steadier. |
| CPU / RAM | Any recent laptop, 8 GB RAM or more | No GPU needed. |
| Disk space | About 15 GB free | OpenCV is compiled from source by vcpkg (about 10 GB while building). |
| Internet | Needed once | vcpkg and pip download packages. |
| Time | 30 to 90 minutes the first time | Mostly the OpenCV build. Later builds take a minute or two. |
| Git | Any recent version | https://git-scm.com/download/win |
| Visual Studio 2022 | Community edition is fine | With the workload "Desktop development with C++" (it includes CMake and Ninja). |
| vcpkg | Latest | Installed in step 3.3. |
| Python | **3.11 exactly**, from python.org | Not the Microsoft Store version. `setup.bat` can install it. |
| Visual C++ Redistributable | 2015-2022 x64 | Already present if Visual Studio is installed. |

---------------------------------------------------------------------------------------------------

## 3. One-time setup

### 3.1 Install Visual Studio 2022
Start the Visual Studio Installer, choose **Desktop development with C++**, and make sure these are ticked in
the right-hand panel: *MSVC v143 build tools*, *Windows 10/11 SDK*, *C++ CMake tools for Windows*.
Afterwards you will use the **x64 Native Tools Command Prompt for VS 2022** (Start menu, search "x64 Native").
All build commands below are typed there.

### 3.2 Install Git
Install with the default options.

### 3.3 Install vcpkg and tell Windows where it is
In the x64 Native Tools prompt:

```
cd /d C:\
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
setx VCPKG_ROOT C:\vcpkg
```

Then **close the prompt and open a new x64 Native Tools prompt** (and restart Visual Studio if it was open).
Check it:

```
echo %VCPKG_ROOT%
```

It must print `C:\vcpkg`. (Do not use `--depth 1` when cloning: the project pins a specific vcpkg commit.)

### 3.4 Get the project
```
cd /d C:\
git clone https://github.com/ahmedkamel382/HandEye_ARVR_Tracker.git
cd /d C:\HandEye_ARVR_Tracker
git checkout first-full-working-demo
```
Use a SHORT path without spaces (like `C:\HandEye_ARVR_Tracker`). Avoid OneDrive, Desktop and Documents folders:
OneDrive sync and long paths cause random build and "Access is denied" problems.

Check that these files exist after cloning (if one is missing, see problem C in section 6):

```
dir assets
dir ai_pipeline\gaze_intent\models
```
Expected:
* `assets\face_landmarker.task`, `assets\hand_landmarker.task`, `assets\hand_gesture_rf.pkl`
* `ai_pipeline\gaze_intent\models\gesture_model.dll`, `gaze_pitch.dll`, `gaze_yaw.dll`

### 3.5 Python and the virtual environment
Check that no Microsoft Store Python is involved:

```
where python
py -0p
```
If a path contains `WindowsApps`, that is the Store Python: uninstall it (Settings > Apps) first, or make sure
step 3.5 uses a python.org installation. Then run:

```
cd /d C:\HandEye_ARVR_Tracker
setup.bat
```

`setup.bat` does this: looks for Python 3.11 (installs 3.11.9 for the current user with winget if missing, then
asks you to close the window and run it again), creates `.venv`, and installs `pybind11` and everything in
`requirements.txt` (MediaPipe, OpenCV, NumPy, scikit-learn, ...). It downloads a lot (the file also lists
torch, which is only for model training); let it finish until it prints "Setup completed successfully".

Verify:
```
.venv\Scripts\python.exe -c "import sys, ctypes, numpy, mediapipe, cv2, sklearn; print(sys.base_prefix)"
```
It must print a folder like `C:\Users\<you>\AppData\Local\Programs\Python\Python311` with no error.
If it fails with "Access is denied", go to section 6, problem A.

---------------------------------------------------------------------------------------------------

## 4. Build

In the x64 Native Tools prompt:

```
cd /d C:\HandEye_ARVR_Tracker
cmake --preset windows-release
cmake --build out\build\windows-release
```

* The first `cmake --preset ...` is the long one: vcpkg downloads and compiles OpenCV. Do not close the window.
* Warnings about "forest" or "gaze model ... missing" at the end of configure mean a file is missing: fix it
  before continuing (section 6, problem C).
* Always use **windows-release** to run the tracker. The `windows-debug` preset is for stepping through code
  in Visual Studio and is much slower.
* If you change Python version, move the repo, or reinstall Python: delete `out` and `.venv` and start again
  from step 3.5.

---------------------------------------------------------------------------------------------------

## 5. Run

```
cd /d C:\HandEye_ARVR_Tracker\out\build\windows-release
```

**Step 1 - health check (touches nothing):**
```
core_engine --check
```
It tests Python, the packages, the gaze models, the hand model, the MediaPipe files and the camera, and ends
with `ALL CHECKS PASSED` or a list of what to fix with a hint for each item.

**Step 2 - dry run with a preview window (the mouse is NOT moved):**
```
core_engine --preview
```
A window shows the camera with the tracked points. A tray icon appears near the clock (click the `^` arrow if it
is hidden). Mouse actions are only logged to `%LOCALAPPDATA%\HandEyeTracker\handeye_tray.log`.
The engine starts PAUSED: press **Ctrl+Alt+P** (or use the tray menu) to activate it.

**Step 3 - real mouse control:**
```
core_engine --control --start-active
```

**Quit:** tray icon > Exit, or Ctrl+C in the window, or F9, or `q` in the preview window.

### Using it
* Keep your face in view and BOTH EYES OPEN for the first second: the gaze module calibrates for 30 frames.
* Move your hand with the index finger extended to move the cursor.
* Pinch (thumb + index) = left button down; release = up. Moving while pinched drags.
* Fist held for 0.3 s = right click.
* Left wink = left click, right wink = right click, eyes closed for 1 s = pause/resume.
* Ctrl+Alt+P = pause/resume. Moving the real mouse gives it priority for 1.5 s.
* Right-click the tray icon > **Calibrate** to fit the hand and gaze ranges to your screen and your camera
  position. Do this once, especially for gaze.
* Settings are saved in `%APPDATA%\HandEyeTracker\os_integration.ini`.

### Tests (optional)
```
cd /d C:\HandEye_ARVR_Tracker\out\build\windows-release
ctest -C Release --output-on-failure
```
Close the engine first (only one instance may run at a time).

---------------------------------------------------------------------------------------------------

## 6. Troubleshooting

### A. `DLL load failed while importing _ctypes: Access is denied`
Python could not be started, because Windows blocked a file in the Python installation.
1. Find out which Python the build uses:
   ```
   .venv\Scripts\python.exe -c "import sys; print(sys.base_prefix)"
   ```
2. Test that Python on its own (replace the path with the one printed):
   ```
   "C:\path\printed\above\python.exe" -c "import ctypes; print('ctypes ok')"
   ```
3. If the path contains `WindowsApps` (Microsoft Store Python): uninstall it, install Python 3.11.9 from
   https://www.python.org/downloads/ ("Install for current user" is fine), then delete `.venv` and `out`,
   run `setup.bat`, and build again (sections 3.5 and 4).
4. If step 2 itself fails, or the path is fine but it still says "Access is denied": your antivirus or
   Windows Security (Protection history) is blocking a file. Add the repo folder and the Python folder to the
   exclusions, restore anything it quarantined, and retry. On a university or company PC, ask IT or use a
   personal PC.
5. Make sure the repo is not inside OneDrive and that you run the program as a normal user (not another user's
   account).

### B. `The Python installation this program was built with is not there any more`
Python was moved, upgraded or uninstalled after the build. Delete `out` (and `.venv` if Python changed) and
repeat from section 3.5.

### C. `Gaze model could not be loaded ... gesture_model.dll` / MediaPipe or hand model missing
The model files are not in your copy. Check section 3.4's `dir` commands. If they are missing after a
`git clone`, the repository owner has to commit them (see section 7), or copy the files from the project owner
into the folders named there. The engine needs:
`assets\face_landmarker.task`, `assets\hand_landmarker.task`, `assets\hand_gesture_rf.pkl`,
`ai_pipeline\gaze_intent\models\gesture_model.dll`, `gaze_pitch.dll`, `gaze_yaw.dll`.

### D. `cmake --preset ...` fails immediately
* `CMAKE_TOOLCHAIN_FILE ... $env{VCPKG_ROOT}` or "vcpkg.cmake not found": `VCPKG_ROOT` is not set. Redo
  step 3.3 and open a NEW prompt (and restart Visual Studio).
* `Ninja` or `cl.exe` not found: you are not in the x64 Native Tools prompt, or the C++ workload is missing.
* "Could NOT find Python3": run `setup.bat` first; the preset looks for `.venv\Scripts\python.exe`.
* `No module named pybind11`: `setup.bat` did not finish; run it again.
* A vcpkg error about the baseline commit: the vcpkg clone is too old or shallow. Run `git -C C:\vcpkg pull`
  (or re-clone without `--depth`).

### E. `ModuleNotFoundError: No module named ...` when running
The package is missing from `.venv`:
```
cd /d C:\HandEye_ARVR_Tracker
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
then run `core_engine --check`.

### F. Camera problems
* "Cannot open camera" / no frames: close Zoom, Teams, Chrome tabs or anything else using the webcam; in
  Windows Settings > Privacy & security > Camera, allow desktop apps to use the camera; try `--camera 1`.
* "Camera delivers 15-30 FPS": that is the webcam (low light slows its exposure). Add light. The engine still
  works, it just tracks at the camera's rate.

### G. It runs but nothing moves
* The engine starts paused: press Ctrl+Alt+P, or start with `--start-active`.
* You are in dry run: use `--control` for the real mouse.
* Gaze direction or range feels wrong: tray icon > Calibrate.
* Windows ignores injected input into windows that run as administrator: run the engine from an
  administrator prompt if you need to control those.

### H. Visual Studio opens it but build/run fails
Use the command-line steps in sections 4 and 5. If you prefer Visual Studio: File > Open > Folder, choose the
`windows-release` configuration at the top, set `VCPKG_ROOT` first (section 3.3), and start Visual Studio AFTER
setting it.

---------------------------------------------------------------------------------------------------

## 7. Notes for the repository owner

* `.gitignore` currently contains `*.dll` and `*.csv`. That means the three gaze models
  (`ai_pipeline/gaze_intent/models/*.dll`) and `core_engine/generated/hand_forest/parity_cases.csv` are NOT
  pushed to GitHub, and anyone cloning the repo gets a project that cannot start. Add these two lines at the
  END of `.gitignore`:
  ```
  !ai_pipeline/gaze_intent/models/*.dll
  !core_engine/generated/hand_forest/parity_cases.csv
  ```
  then commit them:
  ```
  git add .gitignore ai_pipeline/gaze_intent/models core_engine/generated
  git status
  git commit -m "Track the gaze model DLLs and the hand forest parity cases"
  git push
  ```
  `git status` should list the three `.dll` files as new. (Check that `assets/*.task` and
  `assets/hand_gesture_rf.pkl` are tracked too: they are not ignored, so they should already be.)
* Never send the `.exe` or the `out` folder to a teammate (see section 1).
* For development use `windows-debug`; for running the tracker use `windows-release`.
