# Change summary: gaze sync + hand freshness (TRACKED / HELD / LOST)

Base: `New folder (4)` archive. Apply with `git apply freshness_and_gaze_sync_COMBINED.patch` (or unzip `all_changed_files.zip` over the repo).

## A. Gaze sync with gaze_intent.py
| Requirement | Where |
|---|---|
| No-face: reset_tracking_state (clear histories; unfinished calibration restarts), x=y=-1, state 0, four angles NaN | `gaze_math.hpp` `process`, `resetTrackingState()`, `markInvalid()` |
| Calibration: accumulate only when raw_l > 1e-6 && raw_r > 1e-6; no (0.5,0.5); x=y=-1, state -1, NaN angles | `gaze_math.hpp` |
| `ear()` h > 1e-12; `irisRatio()` len_sq < 1e-12 | `gaze_math.hpp` (`kEarDistEps`, `kIrisBaselineEps`) |
| NaN == NaN in the parity check | `tests/parity/run_parity.py` (`same()`) |
| Tests updated/added | `tests/unit/test_all.cpp`: calibration/no-face assertions, zero-EAR guard, face-loss reset |

## B. Hand freshness (CORE_ENGINE_INTEGRATION_REPORT)
| Report section | Implementation | Tests |
|---|---|---|
| 8.1 contract | `tracking_types.hpp`: `TrackingState`, `ModuleOut::tracking_state` (optional, hand only), `missed_frames`; `detected()` unchanged | transitions test |
| 8.2 HandNative | `hand_math.hpp`: `has_track_`; TRACKED / HELD (misses 1..6) / LOST (7+, startup, error); error does not mutate state; `main.cpp`: Python failure -> LOST | `hand: TRACKED / HELD / LOST transitions`; 236k-frame differential vs old x,y,state: identical |
| 8.3 bridge gate | `freshness_gate.hpp` + `os_bridge.hpp::convertHand`: only TRACKED updates the filter; HELD republishes cached FILTERED point; LOST resets and publishes nothing; gaze path untouched | spy-filter tests (IDs 1-9), chain test == observation-only filter, same with Eng 4's adapter |
| 8.4 DSP | no change; HELD is not `tracking_valid=false` | adapter test: 0 errors, 0 rejected timestamps |
| 8.5 OS | `types.hpp`: `osi::TrackerState`, `TrackerSample::tracking` (default Tracked), `fresh_point()`; `os_controller.cpp`: HELD cannot start/confirm gestures, keeps confirmed ones, LOST ignores points; `tray_app.cpp`: wizard samples via `fresh_point()` | 5 new tests in `test_os_integration.cpp` |
| 8.6 compatibility | fields appended with defaults; `{point, state}` still compiles; gaze, `detected()`, gesture encoding unchanged | gaze tests, ROI/preview unchanged |

### Decisions taken (the report's recommendations; change them in one place each)
1. HELD keeps the cursor at the cached filtered point (`freshness_gate.hpp`, `os_controller.cpp`).
2. HELD cannot start or confirm a gesture; an already confirmed one continues (`os_controller.cpp`, debouncer fed `confirmed_hand()`).
3. Calibration accepts TRACKED only (`tray_app.cpp`, `fresh_point()`).
4. The first miss after a real observation is HELD (`HandNative::process`).

## Verified here (Linux, g++ -Wall -Wextra -Wpedantic, clean tree from the original archive + patch)
core unit 18 tests / 309 checks, DSP adapter 7 / 29, OS integration 74 passed, OsBridge Windows branch against mocked Windows headers.

## NOT verified here: needs your Windows machine
`cmake --build` (MSVC /W4), `ctest -C Release` incl. `gaze_hand_parity` (needs the Treelite DLLs, MediaPipe and the updated gaze_intent.py), `e2e`, `tray_app.cpp` (2-line change), a live run.
