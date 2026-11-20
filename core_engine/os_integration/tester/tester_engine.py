"""Background engine of the HandEye tester.

One worker thread owns the camera, the teammates' trackers (used exactly as
they are, never modified) and Engineer 5's OsController (through osi_capi).
The UI only reads snapshots, so a slow tracker never freezes the window.
"""

from __future__ import annotations

import csv
import ctypes
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

import osi_capi

REPO = Path(__file__).resolve().parents[3]
AI_ROOT = REPO / "ai_pipeline"


# ----------------------------------------------------------------------------- modes

@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    description: str
    hand: Optional[str]   # None, "index" or "gesture"
    gaze: Optional[str]   # None, "pointer" or "states"
    owner: str


MODES: Dict[str, Mode] = {m.key: m for m in [
    Mode("camera", "Camera check", "Only the webcam: picture, frame rate and lighting. Nothing is tracked.",
         None, None, "Engineer 1 (camera)"),
    Mode("index", "Index finger only", "The cursor follows your index fingertip. No gestures, no clicks. "
         "Uses Engineer 2's IndexFingerTracker.", "index", None, "Engineer 2"),
    Mode("hand", "Hand + gestures", "Index fingertip moves the cursor, pinch = left button (hold to drag), "
         "fist = right click. Uses Engineer 2's HandTracker.", "gesture", None, "Engineer 2"),
    Mode("gaze", "Gaze pointer", "The cursor follows where you look; winks click, eyes closed 1 s pauses. "
         "Uses Engineer 3's GazeIntent.", None, "pointer", "Engineer 3"),
    Mode("winks", "Blinks & winks only", "Only the eye states: left/right wink clicks and the eyes-closed pause. "
         "The cursor does not move. Uses Engineer 3's GazeIntent.", None, "states", "Engineer 3"),
    Mode("full", "Full system", "Hand and gaze together, like the real engine: the hand drives the cursor "
         "when visible, gaze otherwise.", "gesture", "pointer", "Engineers 2 + 3"),
]}

HAND_STATES = {0: "neutral", 1: "pinch", 2: "fist"}
EYE_STATES = {-1: "calibrating", 0: "neutral", 1: "left wink", 2: "right wink", 3: "eyes closed"}
TRACKING_NAMES = {osi_capi.TRACKED: "tracked", osi_capi.HELD: "held", osi_capi.LOST: "lost"}


# ----------------------------------------------------------------------------- settings text

class IniDoc:
    """Minimal editor for os_integration.ini text that keeps comments and order."""

    def __init__(self, text: str):
        self.lines: List[str] = text.splitlines()

    def _find(self, section: str, key: str) -> Tuple[int, int]:
        current, section_end = None, -1
        for i, raw in enumerate(self.lines):
            line = raw.split("#", 1)[0].split(";", 1)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                if current == section:
                    section_end = i
                    break
                current = line[1:-1].strip()
                continue
            if current == section and "=" in line and line.split("=", 1)[0].strip() == key:
                return i, -1
        if current == section and section_end < 0:
            section_end = len(self.lines)
        return -1, section_end

    def get(self, section: str, key: str) -> Optional[str]:
        index, _ = self._find(section, key)
        if index < 0:
            return None
        return self.lines[index].split("=", 1)[1].split("#", 1)[0].strip()

    def get_float(self, section: str, key: str, default: float) -> float:
        try:
            return float(self.get(section, key) or default)
        except ValueError:
            return default

    def set(self, section: str, key: str, value) -> None:
        text = format_value(value)
        index, section_end = self._find(section, key)
        if index >= 0:
            self.lines[index] = f"{key} = {text}"
        elif section_end >= 0:
            self.lines.insert(section_end, f"{key} = {text}")
        else:
            self.lines += ["", f"[{section}]", f"{key} = {text}"]

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return " ".join(format_value(float(v)) for v in value)
    return str(value)


def user_settings_path() -> Path:
    return Path(os.environ.get("APPDATA", str(Path.home()))) / "HandEyeTracker" / "os_integration.ini"


# ----------------------------------------------------------------------------- configuration

@dataclass
class TesterConfig:
    mode: str = "index"
    camera_index: int = 0
    width: int = 640
    height: int = 480
    real_mouse: bool = False
    all_monitors: bool = False
    smoothing: bool = True
    clicks: bool = True
    lowlight: bool = True        # HandTracker option
    roi_recovery: bool = True    # HandTracker option
    index_alpha: float = 0.35    # IndexFingerTracker smoothing
    show_landmarks: bool = True
    show_zone: bool = True
    show_trail: bool = True


# ----------------------------------------------------------------------------- observations

@dataclass
class HandObs:
    point: Optional[Tuple[float, float]] = None     # pointer, normalised, mirrored frame
    camera_point: Optional[Tuple[float, float]] = None
    state: int = 0
    tracking: int = osi_capi.LOST
    status: str = "off"
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    ms: float = 0.0


@dataclass
class GazeObs:
    point: Optional[Tuple[float, float]] = None
    state: int = 0
    yaw: float = float("nan")
    pitch: float = float("nan")
    head_yaw: float = float("nan")
    head_pitch: float = float("nan")
    landmarks: object = None
    face: bool = False
    ms: float = 0.0


@dataclass
class Counters:
    left_down: int = 0
    left_up: int = 0
    left_click: int = 0
    right_click: int = 0
    middle_click: int = 0
    pauses: int = 0
    frames: int = 0
    refused: int = 0


@dataclass
class Snapshot:
    running: bool = False
    frame: Optional[np.ndarray] = None   # RGB, mirrored, with overlays
    mode: str = "index"
    fps: float = 0.0
    loop_ms: float = 0.0
    brightness: float = 0.0
    camera_size: Tuple[int, int] = (0, 0)
    hand: HandObs = field(default_factory=HandObs)
    gaze: GazeObs = field(default_factory=GazeObs)
    report: osi_capi.Report = field(default_factory=osi_capi.Report)
    target_rect: Tuple[int, int, int, int] = (0, 0, 1920, 1080)
    counters: Counters = field(default_factory=Counters)
    tracked_pct: float = 0.0
    held_pct: float = 0.0
    lost_pct: float = 0.0
    yielding: bool = False
    recording: Optional[str] = None
    recorded_rows: int = 0
    smoothing_available: bool = False
    status: str = "Stopped"
    error: Optional[str] = None


# ----------------------------------------------------------------------------- trackers

def prepare_ai_imports() -> None:
    """Make the teammates' modules importable without changing them."""
    os.environ.setdefault("AI_PIPELINE_ROOT", str(AI_ROOT))
    for sub in ("hand_kinematics", "gaze_intent"):
        path = str(AI_ROOT / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    # tl2cgen (gaze models) calls os.add_dll_directory(<venv>\Library\bin), which fails if the
    # folder does not exist. The CI creates it too; it lives inside the virtual environment.
    try:
        (Path(sys.prefix) / "Library" / "bin").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


class HandSource:
    def __init__(self, kind: str, config: TesterConfig):
        prepare_ai_imports()
        self.kind = kind
        if kind == "index":
            from index_tracker import IndexFingerTracker
            self.tracker = IndexFingerTracker(smoothing_alpha=max(0.01, min(1.0, config.index_alpha)))
        else:
            from hand_tracker import HandTracker
            self.tracker = HandTracker(enable_lowlight=config.lowlight, enable_roi_recovery=config.roi_recovery)

    def process(self, bgr: np.ndarray) -> HandObs:
        start = time.perf_counter()
        obs = HandObs()
        if self.kind == "index":
            r = self.tracker.track(bgr)
            obs.status = r.status
            if r.valid and r.x is not None:
                obs.point = (float(r.x), float(r.y))
                obs.camera_point = (float(r.camera_x), float(r.camera_y))
                obs.tracking = osi_capi.TRACKED
        else:
            x, y, state = self.tracker.process_frame(bgr)
            obs.status = str(getattr(self.tracker, "status", ""))
            obs.tracking = {"DETECTED": osi_capi.TRACKED, "RECOVERED_ROI": osi_capi.TRACKED,
                            "HELD": osi_capi.HELD}.get(obs.status, osi_capi.LOST)
            if x >= 0 and y >= 0 and obs.tracking != osi_capi.LOST:
                obs.point = (float(x), float(y))
                obs.camera_point = obs.point
                obs.state = int(state)
            bbox = getattr(self.tracker, "_last_bbox", None)   # read-only, for drawing
            obs.bbox = tuple(bbox) if bbox else None
            obs.confidence = float(getattr(self.tracker, "last_confidence", 0.0))
        obs.ms = (time.perf_counter() - start) * 1000.0
        return obs

    def close(self) -> None:
        for name in ("close", "shutdown"):
            fn = getattr(self.tracker, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return


class GazeSource:
    def __init__(self):
        prepare_ai_imports()
        import gaze_intent
        self.module = gaze_intent
        self.tracker = gaze_intent.GazeIntent()
        self._start = time.perf_counter()
        self._last_ms = -1

    def process(self, rgb_mirrored: np.ndarray) -> GazeObs:
        start = time.perf_counter()
        ts = max(int((start - self._start) * 1000.0), self._last_ms + 1)   # strictly increasing for MediaPipe
        self._last_ms = ts
        yaw, pitch, head_yaw, head_pitch, state, landmarks = self.tracker.process_frame(rgb_mirrored, ts)
        obs = GazeObs(state=int(state), yaw=yaw, pitch=pitch, head_yaw=head_yaw, head_pitch=head_pitch,
                      landmarks=landmarks, face=landmarks is not None)
        if self.module.is_valid_output(yaw, pitch):
            # Same mapping as the engine (GazeNative::setPointer): 0.5 + angle / 60, clamped.
            obs.point = (self.module.angle_to_norm_coord(yaw), self.module.angle_to_norm_coord(pitch))
        obs.ms = (time.perf_counter() - start) * 1000.0
        return obs

    def close(self) -> None:
        try:
            self.tracker.shutdown()
        except Exception:
            pass


# ----------------------------------------------------------------------------- engine

class Smoother:
    """Engine-like smoothing gate: only fresh points enter the filter."""

    def __init__(self, filt: Optional[osi_capi.SmoothingFilter]):
        self.filter = filt
        self.last: Optional[Tuple[float, float]] = None

    def apply(self, point, tracking: int, t: float):
        if self.filter is None:
            return point
        if tracking == osi_capi.TRACKED and point is not None:
            self.last = self.filter.apply(point, t) or point
            return self.last
        if tracking == osi_capi.HELD:
            return self.last
        if self.last is not None:
            self.filter.apply(None, t)   # reset
        self.last = None
        return None

    def close(self):
        if self.filter:
            self.filter.close()


class TesterEngine:
    YIELD_SECONDS = 1.5
    YIELD_TOLERANCE_PX = 4

    def __init__(self, library: osi_capi.OsiLibrary, log: Callable[[str], None]):
        self.lib = library
        self._log = log
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._config = TesterConfig()
        self._config_dirty = True
        self._settings_ini = self._initial_settings()
        self._settings_dirty = True
        self._want_active: Optional[bool] = None
        self._snapshot = Snapshot()
        self._events: Deque[Tuple[float, str]] = deque(maxlen=500)
        self._record_request: Optional[Optional[Path]] = None
        self.settings_error: Optional[str] = None

    # ---- settings -----------------------------------------------------------------
    def _initial_settings(self) -> str:
        path = user_settings_path()
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                pass
        return self.lib.default_settings_ini()

    def settings_text(self) -> str:
        with self._lock:
            return self._settings_ini

    def set_settings_text(self, text: str) -> None:
        with self._lock:
            self._settings_ini = text
            self._settings_dirty = True

    def reset_settings(self) -> None:
        self.set_settings_text(self.lib.default_settings_ini())

    # ---- control ------------------------------------------------------------------
    @property
    def config(self) -> TesterConfig:
        with self._lock:
            return replace(self._config)

    def configure(self, **changes) -> None:
        with self._lock:
            self._config = replace(self._config, **changes)
            self._config_dirty = True

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._want_active = active

    def start_recording(self, path: Path) -> None:
        with self._lock:
            self._record_request = path

    def stop_recording(self) -> None:
        with self._lock:
            self._record_request = Path()   # empty path = stop

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def drain_events(self) -> List[Tuple[float, str]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def event(self, text: str) -> None:
        with self._lock:
            self._events.append((time.time(), text))
        self._log(text)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._config_dirty = True
            self._settings_dirty = True
        self._thread = threading.Thread(target=self._run, name="tester-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ---- worker -------------------------------------------------------------------
    def _publish(self, **fields) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, **fields)

    def _run(self) -> None:
        import cv2
        state = _WorkerState()
        try:
            self._publish(running=True, status="Starting camera...", error=None)
            while not self._stop.is_set():
                with self._lock:
                    config_dirty, self._config_dirty = self._config_dirty, False
                    settings_dirty, self._settings_dirty = self._settings_dirty, False
                    config, settings = replace(self._config), self._settings_ini
                    want_active, self._want_active = self._want_active, None
                    record_request, self._record_request = self._record_request, None
                if config_dirty:
                    self._apply_config(state, config, settings, cv2)
                    settings_dirty = False
                if settings_dirty and state.controller:
                    error = state.controller.apply_settings(settings)
                    self.settings_error = error
                    if error:
                        self.event(f"Settings not applied: {error}")
                    else:
                        state.zone = IniDoc(settings)
                        state.rebuild_smoothers(self.lib, config, settings)
                        self.event("Settings applied")
                if want_active is not None and state.controller:
                    state.controller.set_active(want_active)
                    if not want_active:
                        state.controller.release_all()
                    self.event("Output ACTIVE" if want_active else "Output paused")
                if record_request is not None:
                    self._handle_recording(state, record_request, config)
                if not self._step(state, config, cv2):
                    break
        except Exception as error:   # report instead of dying silently
            self._log(f"engine error: {error!r}")
            self._publish(error=str(error), status="Stopped (error)")
        finally:
            state.close()
            self._publish(running=False, frame=None, status="Stopped")

    def _apply_config(self, state: "_WorkerState", config: TesterConfig, settings: str, cv2) -> None:
        mode = MODES[config.mode]
        old = state.config
        # Camera
        if state.cap is None or old is None or (config.camera_index, config.width, config.height) != (
                old.camera_index, old.width, old.height):
            if state.cap is not None:
                state.cap.release()
            self._publish(status=f"Opening camera {config.camera_index}...")
            cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {config.camera_index}. Close other apps using it "
                                   f"(Zoom, Teams, the engine) or pick another camera.")
            state.cap = cap
            self.event(f"Camera {config.camera_index} opened")
        # Trackers (reloaded only when their options change)
        hand_key = (mode.hand, config.lowlight, config.roi_recovery, round(config.index_alpha, 3)) if mode.hand else None
        if hand_key != state.hand_key:
            if state.hand:
                state.hand.close()
            state.hand = None
            if mode.hand:
                self._publish(status="Loading hand tracker...")
                state.hand = HandSource(mode.hand, config)
                self.event(f"Hand tracker loaded ({'IndexFingerTracker' if mode.hand == 'index' else 'HandTracker'})")
            state.hand_key = hand_key
        if bool(mode.gaze) != (state.gaze is not None):
            if state.gaze:
                state.gaze.close()
                state.gaze = None
            if mode.gaze:
                self._publish(status="Loading gaze tracker...")
                state.gaze = GazeSource()
                self.event("Gaze tracker loaded (GazeIntent) - keep both eyes open for the first second")
        # Controller (recreated when the output changes)
        if state.controller is None or old is None or (config.real_mouse, config.all_monitors) != (
                old.real_mouse, old.all_monitors):
            was_active = state.controller is not None and state.last_report.active and not config.real_mouse
            if state.controller:
                state.controller.close()
            try:
                state.controller = self.lib.controller(settings, config.all_monitors, config.real_mouse)
            except ValueError as error:
                # Never stop over settings: say why and carry on with the defaults.
                self.event(f"These settings could not be used ({error}); using the defaults instead")
                settings = self.lib.default_settings_ini()
                with self._lock:
                    self._settings_ini = settings
                state.controller = self.lib.controller(settings, config.all_monitors, config.real_mouse)
            self.settings_error = None
            state.zone = IniDoc(settings)
            # Real mouse always starts paused; the simulation starts active.
            state.controller.set_active(not config.real_mouse or was_active)
            state.expected_cursor = None
            target, _ = state.controller.geometry()
            self._publish(target_rect=target)
            self.event("Output: REAL MOUSE (paused - press Activate)" if config.real_mouse
                       else "Output: simulation (the real mouse is never touched)")
        state.rebuild_smoothers(self.lib, config, settings)
        self._publish(smoothing_available=state.smoothing_available)
        if old is None or old.mode != config.mode:
            self.event(f"Mode: {mode.title}")
            state.reset_stats()
        state.config = config

    def _handle_recording(self, state: "_WorkerState", request: Path, config: TesterConfig) -> None:
        if state.recorder:
            state.recorder.close()
            self.event(f"Recording saved: {state.recorder.path} ({state.recorder.rows} frames)")
            state.recorder = None
            self._publish(recording=None)
        if str(request) not in ("", "."):
            state.recorder = CsvRecorder(request)
            self.event(f"Recording to {request}")
            self._publish(recording=str(request), recorded_rows=0)

    def _step(self, state: "_WorkerState", config: TesterConfig, cv2) -> bool:
        ok, bgr = state.cap.read()
        if not ok or bgr is None:
            state.failures += 1
            if state.failures > 60:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.01)
            return True
        state.failures = 0
        now = time.perf_counter()
        t = now - state.t0
        loop_start = now
        mode = MODES[config.mode]

        mirrored = cv2.flip(bgr, 1)
        hand = state.hand.process(bgr) if state.hand else HandObs()     # trackers mirror internally
        gaze = state.gaze.process(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)) if state.gaze else GazeObs()

        # Inputs for Engineer 5's OsController
        hand_in = None
        if mode.hand:
            point = state.hand_smoother.apply(hand.point, hand.tracking, t)
            hand_state = hand.state if (config.clicks and mode.hand == "gesture") else 0
            hand_in = osi_capi.TrackerInput(point, hand_state,
                                            hand.tracking if point is not None else osi_capi.LOST)
        gaze_in = None
        if mode.gaze:
            gaze_tracking = osi_capi.TRACKED if gaze.point is not None else osi_capi.LOST
            point = state.gaze_smoother.apply(gaze.point, gaze_tracking, t) if mode.gaze == "pointer" else None
            eye_state = gaze.state if (config.clicks or gaze.state == -1) else 0
            gaze_in = osi_capi.TrackerInput(point, eye_state, osi_capi.TRACKED if point else osi_capi.LOST)

        controller = state.controller
        yielding = False
        if config.real_mouse and controller:
            yielding = self._physical_mouse_check(state, now)
            if self._escape_pressed() and state.last_report.active:
                controller.set_active(False)
                controller.release_all()
                self.event("Esc pressed: output paused")
        report = controller.update(t, hand_in, gaze_in) if (controller and mode.key != "camera") else osi_capi.Report(
            active=state.last_report.active)
        if config.real_mouse and report.moved and report.cursor:
            state.expected_cursor = report.cursor
        state.last_report = report
        if controller:
            controller.drain_events()   # bounded buffer; the report already lists the actions
        self._count(state, report, hand, gaze, mode)

        # Overlays and statistics
        frame = draw_overlays(cv2, mirrored, config, mode, hand, gaze, report, state)
        dt = now - state.last_frame if state.last_frame else 0.0
        state.last_frame = now
        if dt > 0:
            state.fps = 0.9 * state.fps + 0.1 * (1.0 / dt) if state.fps else 1.0 / dt
        brightness = float(np.mean(bgr[::8, ::8]))
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        if state.recorder:
            state.recorder.write(t, mode.key, hand, gaze, report, state.fps)
        tracked, held, lost = state.tracking_pcts()
        self._publish(
            running=True, frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode=mode.key, fps=state.fps,
            loop_ms=loop_ms, brightness=brightness, camera_size=(bgr.shape[1], bgr.shape[0]), hand=hand, gaze=gaze,
            report=report, counters=replace(state.counters), tracked_pct=tracked, held_pct=held, lost_pct=lost,
            yielding=yielding, recorded_rows=state.recorder.rows if state.recorder else 0,
            status="Running", error=None)
        return True

    def _physical_mouse_check(self, state: "_WorkerState", now: float) -> bool:
        """Yield to the real mouse for 1.5 s when the user moves it."""
        controller = state.controller
        actual = self.lib.cursor_position()
        if state.yield_until and now < state.yield_until:
            return True
        if state.yield_until:
            state.yield_until = 0.0
            controller.set_inhibited(False)
            state.expected_cursor = None
            self.event("Physical mouse released: tester output resumed")
        expected = state.expected_cursor
        if expected and actual and max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])) > \
                self.YIELD_TOLERANCE_PX:
            state.yield_until = now + self.YIELD_SECONDS
            controller.set_inhibited(True)
            self.event("Real mouse moved: tester output yields for 1.5 s")
            return True
        return False

    @staticmethod
    def _escape_pressed() -> bool:
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
        except Exception:
            return False

    def _count(self, state: "_WorkerState", report: osi_capi.Report, hand: HandObs, gaze: GazeObs, mode: Mode):
        c = state.counters
        c.frames += 1
        for action, button in report.actions:
            if action == osi_capi.ACTION_DOWN and button == "left":
                c.left_down += 1
                self.event("Left button DOWN (pinch)")
            elif action == osi_capi.ACTION_UP and button == "left":
                c.left_up += 1
                self.event("Left button UP")
            elif action == osi_capi.ACTION_CLICK:
                setattr(c, f"{button}_click", getattr(c, f"{button}_click") + 1)
                self.event(f"{button.capitalize()} CLICK")
        if report.pause_toggled:
            c.pauses += 1
            self.event("Eyes closed: output " + ("resumed" if report.active else "paused"))
        if report.injection_failed:
            c.refused += 1
        if report.release_pending and not state.release_warned:
            self.event("WARNING: Windows refused a button release - retrying (tracker item #8)")
        state.release_warned = report.release_pending
        if mode.hand:
            state.track_history.append(hand.tracking)
        elif mode.gaze:
            state.track_history.append(osi_capi.TRACKED if gaze.face else osi_capi.LOST)


class _WorkerState:
    def __init__(self):
        self.config: Optional[TesterConfig] = None
        self.cap = None
        self.hand: Optional[HandSource] = None
        self.hand_key = None
        self.gaze: Optional[GazeSource] = None
        self.controller: Optional[osi_capi.Controller] = None
        self.hand_smoother = Smoother(None)
        self.gaze_smoother = Smoother(None)
        self.smoothing_available = False
        self.zone = IniDoc("")
        self.last_report = osi_capi.Report()
        self.expected_cursor = None
        self.yield_until = 0.0
        self.recorder: Optional[CsvRecorder] = None
        self.t0 = time.perf_counter()
        self.last_frame = 0.0
        self.fps = 0.0
        self.failures = 0
        self.release_warned = False
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=40)
        self.counters = Counters()
        self.track_history: Deque[int] = deque(maxlen=150)

    def reset_stats(self):
        self.counters = Counters()
        self.track_history.clear()
        self.trail.clear()

    def tracking_pcts(self) -> Tuple[float, float, float]:
        n = len(self.track_history)
        if not n:
            return 0.0, 0.0, 0.0
        return tuple(100.0 * sum(1 for v in self.track_history if v == k) / n
                     for k in (osi_capi.TRACKED, osi_capi.HELD, osi_capi.LOST))

    def rebuild_smoothers(self, lib: osi_capi.OsiLibrary, config: TesterConfig, settings: str):
        self.hand_smoother.close()
        self.gaze_smoother.close()
        doc = IniDoc(settings)
        probe = lib.smoothing_filter(1.0, 0.0, 1.0)
        self.smoothing_available = probe is not None
        if probe:
            probe.close()
        if config.smoothing and self.smoothing_available:
            def make(section, d):
                return lib.smoothing_filter(doc.get_float(section, "min_cutoff_hz", d[0]),
                                            doc.get_float(section, "beta", d[1]),
                                            doc.get_float(section, "derivative_cutoff_hz", d[2]))
            self.hand_smoother = Smoother(make("hand_filter", (1.2, 4.0, 1.0)))
            self.gaze_smoother = Smoother(make("gaze_filter", (0.8, 6.0, 1.5)))
        else:
            self.hand_smoother = Smoother(None)
            self.gaze_smoother = Smoother(None)

    def close(self):
        if self.recorder:
            self.recorder.close()
        if self.controller:
            self.controller.close()   # releases any held button
        for source in (self.hand, self.gaze):
            if source:
                source.close()
        self.hand_smoother.close()
        self.gaze_smoother.close()
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------- recording

class CsvRecorder:
    COLUMNS = ["t", "mode", "hand_x", "hand_y", "hand_state", "hand_tracking", "hand_status", "hand_ms",
               "gaze_x", "gaze_y", "gaze_state", "yaw_deg", "pitch_deg", "gaze_ms",
               "cursor_x", "cursor_y", "source", "actions", "active", "fps"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self.rows = 0

    def write(self, t, mode, hand: HandObs, gaze: GazeObs, report: osi_capi.Report, fps: float):
        hp, gp, cur = hand.point, gaze.point, report.cursor
        actions = ";".join(f"{['down', 'up', 'click', 'pause'][a]}:{b}" for a, b in report.actions)
        self._writer.writerow([
            f"{t:.4f}", mode, _fmt(hp[0] if hp else None), _fmt(hp[1] if hp else None), hand.state,
            TRACKING_NAMES.get(hand.tracking, ""), hand.status, f"{hand.ms:.1f}",
            _fmt(gp[0] if gp else None), _fmt(gp[1] if gp else None), gaze.state, _fmt(gaze.yaw),
            _fmt(gaze.pitch), f"{gaze.ms:.1f}", cur[0] if cur else "", cur[1] if cur else "", report.source,
            actions, int(report.active), f"{fps:.1f}"])
        self.rows += 1

    def close(self):
        self._file.close()


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.5f}"


# ----------------------------------------------------------------------------- drawing

ACCENT = (255, 194, 76)      # BGR
GOOD = (80, 200, 120)
WARN = (60, 170, 255)
BAD = (90, 90, 255)
WHITE = (240, 240, 240)
EYE_POINTS = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
IRIS_POINTS = list(range(468, 478))


def draw_overlays(cv2, frame, config: TesterConfig, mode: Mode, hand: HandObs, gaze: GazeObs,
                  report: osi_capi.Report, state: _WorkerState):
    out = frame.copy()
    h, w = out.shape[:2]

    def px(p):
        return int(p[0] * (w - 1)), int(p[1] * (h - 1))

    if mode.hand:
        if config.show_zone and mode.hand == "gesture":
            region = _region(state.zone.get("mapping", "hand_region"), (0.15, 0.15, 0.85, 0.85))
            x0, y0 = px((min(region[0], region[2]), min(region[1], region[3])))
            x1, y1 = px((max(region[0], region[2]), max(region[1], region[3])))
            cv2.rectangle(out, (x0, y0), (x1, y1), (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(out, "active zone = whole screen", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (160, 160, 160), 1, cv2.LINE_AA)
        if config.show_landmarks and hand.bbox:
            bx0, by0, bx1, by1 = hand.bbox
            cv2.rectangle(out, px((bx0, by0)), px((bx1, by1)), ACCENT, 1, cv2.LINE_AA)
        dot = hand.camera_point or hand.point
        if dot:
            if config.show_trail:
                state.trail.append(dot)
            colour = {osi_capi.TRACKED: GOOD, osi_capi.HELD: WARN}.get(hand.tracking, BAD)
            if hand.state == 1:
                colour = (255, 255, 0)
            elif hand.state == 2:
                colour = (255, 0, 255)
            cv2.circle(out, px(dot), 10, colour, 2, cv2.LINE_AA)
            cv2.circle(out, px(dot), 3, colour, -1, cv2.LINE_AA)
        if config.show_trail and len(state.trail) > 1:
            points = [px(p) for p in state.trail]
            for i in range(1, len(points)):
                cv2.line(out, points[i - 1], points[i], ACCENT, 1 + i * 2 // len(points), cv2.LINE_AA)
    if mode.gaze and config.show_landmarks and gaze.landmarks is not None:
        lm = gaze.landmarks
        try:
            for i in EYE_POINTS:
                cv2.circle(out, px((lm[i].x, lm[i].y)), 1, (200, 200, 200), -1, cv2.LINE_AA)
            for i in IRIS_POINTS:
                cv2.circle(out, px((lm[i].x, lm[i].y)), 1, ACCENT, -1, cv2.LINE_AA)
            if gaze.point:
                cx = (lm[468].x + lm[473].x) / 2
                cy = (lm[468].y + lm[473].y) / 2
                ex = cx + (gaze.point[0] - 0.5) * 0.4
                ey = cy + (gaze.point[1] - 0.5) * 0.4
                cv2.arrowedLine(out, px((cx, cy)), px((ex, ey)), ACCENT, 2, cv2.LINE_AA, tipLength=0.25)
        except (IndexError, AttributeError, TypeError):
            pass
    # Heads-up line
    label = f"{mode.title}  |  {state.fps:4.1f} fps"
    cv2.rectangle(out, (0, 0), (w, 26), (20, 20, 20), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if mode.key != "camera":
        flag = "LIVE MOUSE" if config.real_mouse else "SIMULATION"
        on = "ACTIVE" if report.active else "PAUSED"
        colour = BAD if (config.real_mouse and report.active) else (GOOD if report.active else WARN)
        text = f"{flag} - {on}"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(out, text, (w - size[0] - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


def _region(text: Optional[str], default):
    try:
        values = [float(v) for v in (text or "").replace(",", " ").split()]
        return tuple(values) if len(values) == 4 else default
    except ValueError:
        return default
