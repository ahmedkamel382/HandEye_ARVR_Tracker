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
from types import SimpleNamespace
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
    Mode("full", "Full system", "Hand and gaze together, with the engine's rules: the hand drives the cursor "
         "when visible, gaze otherwise.", "gesture", "pointer", "Engineers 2 + 3"),
]}

HAND_STATES = {0: "neutral", 1: "pinch", 2: "fist"}
HAND_FIST, EYES_CLOSED = 2, 3
EYE_STATES = {-1: "calibrating", 0: "neutral", 1: "left wink", 2: "right wink", 3: "eyes closed"}
TRACKING_NAMES = {osi_capi.TRACKED: "tracked", osi_capi.HELD: "held", osi_capi.LOST: "lost"}


# ----------------------------------------------------------------------------- settings text

class IniDoc:
    """Minimal editor for os_integration.ini text that keeps comments and order."""

    def __init__(self, text: str):
        self.lines: List[str] = text.splitlines()

    @staticmethod
    def _split_comment(raw: str) -> Tuple[str, str]:
        # Same rule as settings.cpp: the first '#' or ';' starts a comment.
        starts = [i for i in (raw.find("#"), raw.find(";")) if i >= 0]
        cut = min(starts) if starts else len(raw)
        return raw[:cut], raw[cut:]

    def _find(self, section: str, key: str) -> Tuple[int, int]:
        """(line of the key, end of the section). A repeated key: the last wins, as in settings.cpp."""
        current, found, section_end = None, -1, -1
        for i, raw in enumerate(self.lines):
            line = self._split_comment(raw)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                if current == section:
                    section_end = i
                current = line[1:-1].strip()
            elif current == section and "=" in line and line.split("=", 1)[0].strip() == key:
                found = i
        if current == section:
            section_end = len(self.lines)
        return found, section_end

    def get(self, section: str, key: str) -> Optional[str]:
        index, _ = self._find(section, key)
        if index < 0:
            return None
        return self._split_comment(self.lines[index])[0].split("=", 1)[1].strip()

    def get_float(self, section: str, key: str, default: float) -> float:
        try:
            return float(self.get(section, key) or default)
        except ValueError:
            return default

    def set(self, section: str, key: str, value) -> None:
        text = format_value(value)
        index, section_end = self._find(section, key)
        if index >= 0:
            comment = self._split_comment(self.lines[index])[1]
            self.lines[index] = f"{key} = {text}" + (f"  {comment}" if comment else "")
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


def read_settings(path: Path) -> str:
    """The file's text without a byte-order mark. Raises OSError or UnicodeDecodeError."""
    return path.read_text(encoding="utf-8-sig")


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
    lowlight: bool = False       # HandTracker option; core_engine has no such stage
    roi_recovery: bool = False   # HandTracker option; core_engine has no such stage
    index_alpha: float = 1.0     # IndexFingerTracker's own smoothing; 1 = off, so the 1-Euro filter is the only one
    show_landmarks: bool = True
    show_zone: bool = True
    show_trail: bool = True


ENGINE_MATCH = dict(lowlight=False, roi_recovery=False, index_alpha=1.0, smoothing=True, clicks=True)


def engine_differences(config: TesterConfig, settings: Optional[str] = None) -> List[str]:
    """How this configuration differs from core_engine, for the parity label."""
    mode = MODES[config.mode]
    found = []
    if mode.hand == "gesture":
        if config.lowlight:
            found.append("low-light enhancement is on")
        if config.roi_recovery:
            found.append("lost-hand recovery is on")
    if mode.hand == "index":
        found.append("core_engine does not use IndexFingerTracker")
        if config.index_alpha < 1.0 and config.smoothing:
            found.append("two smoothing filters are stacked")
    if mode.gaze == "states":
        found.append("the gaze cursor is off in this test")
    if mode.key != "camera" and not config.smoothing:
        found.append("smoothing is off")
    if (mode.hand == "gesture" or mode.gaze) and not config.clicks:
        found.append("clicks are off")
    if settings is not None and mode.key != "camera" and saved_all_monitors(settings) != config.all_monitors:
        found.append("the screen differs from screen_target in the settings")
    return found


def saved_all_monitors(settings: str) -> bool:
    return IniDoc(settings).get("mapping", "screen_target") == "all_monitors"


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
    invalid_input: int = 0          # landmark results the engine would reject too


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
    for sub in ("hand_kinematics", "gaze_intent", "bridge"):
        path = str(AI_ROOT / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    # tl2cgen (gaze models) calls os.add_dll_directory(<venv>\Library\bin), which fails if the
    # folder does not exist. The CI creates it too; it lives inside the virtual environment.
    try:
        (Path(sys.prefix) / "Library" / "bin").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


_NO_HAND = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
_NO_FACE = SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[], face_blendshapes=[])


class _ValidatingDetector:
    """Sits between MediaPipe and a teammate's tracker, which stays unchanged. A result core_engine would not
    accept reaches the tracker as "nothing detected", exactly what the engine's tracker receives in that case."""

    def __init__(self, inner, accept: Callable[[object], bool], nothing):
        self._inner, self._accept, self._nothing = inner, accept, nothing
        self.rejected = 0

    def _check(self, result):
        if self._accept(result):
            return result
        self.rejected += 1
        return self._nothing

    def detect_for_video(self, image, timestamp_ms):
        return self._check(self._inner.detect_for_video(image, timestamp_ms))

    def detect(self, image):
        return self._check(self._inner.detect(image))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _hand_accepted(result) -> bool:
    """ai_pipeline/bridge/landmark_extractors.pack_hand, the rule core_engine uses: exactly 21 image and 21 world
    landmarks, all finite. No hand at all passes through unchanged."""
    from landmark_extractors import pack_hand
    image = result.hand_landmarks[0] if getattr(result, "hand_landmarks", None) else None
    world = result.hand_world_landmarks[0] if getattr(result, "hand_world_landmarks", None) else None
    if image is None and world is None:
        return True
    return pack_hand(image, world)[1]["valid"]


def _face_accepted(result) -> bool:
    """Use the same packed face and pose representation as the native bridge."""
    from landmark_extractors import pack_face
    faces = getattr(result, "face_landmarks", None)
    if not faces:
        return True
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if matrices is None or len(matrices) == 0 or np.shape(matrices[0]) != (4, 4):
        return False
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return pack_face(faces[0], matrices[0])[1]["valid"]
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def _guard(tracker, names, accept, nothing) -> List[_ValidatingDetector]:
    guards = []
    for name in names:
        inner = getattr(tracker, name, None)
        if inner is not None:
            guard = _ValidatingDetector(inner, accept, nothing)
            setattr(tracker, name, guard)
            guards.append(guard)
    if not guards:
        raise RuntimeError(f"{type(tracker).__name__} has no detector to validate: update the tester")
    return guards


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
        self._guards = _guard(self.tracker, ("detector", "roi_detector", "_detector"), _hand_accepted, _NO_HAND)

    @property
    def rejected(self) -> int:
        return sum(g.rejected for g in self._guards)

    def process(self, bgr: np.ndarray) -> HandObs:
        start = time.perf_counter()
        obs = HandObs()
        if self.kind == "index":
            r = self.tracker.track(bgr)
            obs.status = r.status
            if r.valid and r.x is not None and r.y is not None:
                rx, ry = float(r.x), float(r.y)
                if math.isfinite(rx) and math.isfinite(ry):
                    obs.point = (rx, ry)
                    cx = float(r.camera_x) if r.camera_x is not None else rx
                    cy = float(r.camera_y) if r.camera_y is not None else ry
                    obs.camera_point = (cx, cy) if (math.isfinite(cx) and math.isfinite(cy)) else obs.point
                    obs.tracking = osi_capi.TRACKED
        else:
            x, y, state = self.tracker.process_frame(bgr)
            obs.status = str(getattr(self.tracker, "status", ""))
            obs.tracking = {"DETECTED": osi_capi.TRACKED, "RECOVERED_ROI": osi_capi.TRACKED,
                            "HELD": osi_capi.HELD}.get(obs.status, osi_capi.LOST)
            fx, fy = float(x), float(y)
            if not (math.isfinite(fx) and math.isfinite(fy)):
                obs.tracking = osi_capi.LOST
            elif fx >= 0 and fy >= 0 and obs.tracking != osi_capi.LOST:
                obs.point = (fx, fy)
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
        self._guards = _guard(self.tracker, ("detector",), _face_accepted, _NO_FACE)
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
            # Same mapping as the engine (GazeNative::setPointer): X = 0.5 + yaw / 60, Y = 0.5 - pitch / 60, clamped.
            obs.point = (
                self.module.angle_to_norm_coord(yaw),
                self.module.angle_to_norm_coord(pitch, invert=True),
            )
        obs.ms = (time.perf_counter() - start) * 1000.0
        return obs

    @property
    def rejected(self) -> int:
        return sum(g.rejected for g in self._guards)

    def close(self) -> None:
        try:
            self.tracker.shutdown()
        except Exception:
            pass


# ----------------------------------------------------------------------------- engine

class Smoother:
    """The same rules as core_engine/include/freshness_gate.hpp: a raw point never stands in for a filtered one."""

    def __init__(self, filt: Optional[osi_capi.SmoothingFilter]):
        self.filter = filt
        self.last: Optional[Tuple[float, float]] = None     # last published fresh point
        self._has_history = False

    def apply(self, point, tracking: int, t: float) -> Tuple[Optional[Tuple[float, float]], int]:
        """(point to publish, TRACKED / HELD / LOST)."""
        # freshness_gate.hpp, rule for rule. ModuleOut::detected() is x >= 0 && y >= 0: False for NaN and -Inf
        # (they end tracking even on a HELD sample), True for +Inf. A HELD sample that counts as detected is not a
        # new observation, so its coordinates are not used: the cached point is replayed (also for +Inf). A TRACKED
        # sample's coordinates must be finite; a non-finite one ends tracking.
        detected = point is not None and point[0] >= 0 and point[1] >= 0
        if not detected or tracking not in (osi_capi.TRACKED, osi_capi.HELD):
            return self._lose()
        if tracking == osi_capi.HELD:
            return self._replay()
        if self.filter is None:
            if not (math.isfinite(point[0]) and math.isfinite(point[1])):
                return self._lose()
            output = point
        else:
            status, output = self.filter.update(point, t)
            if status == osi_capi.FILTER_REJECTED:
                return self._replay()                       # not a new measurement; history intact
            if status != osi_capi.FILTER_OK:
                self.last, self._has_history = None, False  # the filter reset itself
                return None, osi_capi.LOST
        if not (math.isfinite(output[0]) and math.isfinite(output[1])):
            return self._lose()     # core's defensive check after the filter; FILTER_OK always gives a finite point
        self.last = (min(1.0, max(0.0, output[0])), min(1.0, max(0.0, output[1])))
        self._has_history = True
        return self.last, osi_capi.TRACKED

    def _replay(self):
        return (self.last, osi_capi.HELD) if self.last is not None else (None, osi_capi.LOST)

    def _lose(self):
        if self._has_history and self.filter is not None:
            self.filter.reset()
        self.last, self._has_history = None, False
        return None, osi_capi.LOST

    def close(self):
        if self.filter:
            self.filter.close()


class TesterEngine:
    YIELD_SECONDS = 1.5
    YIELD_TOLERANCE_PX = 4
    STALE_SECONDS = 0.5           # release buttons when camera frames stop, like the tray app
    WATCHDOG_SECONDS = 0.03       # Esc, Stop and stalls must not wait for a camera read

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
        self._ready = False               # a frame has run with the current configuration
        self._snapshot = Snapshot()
        self._events: Deque[Tuple[float, str]] = deque(maxlen=500)
        self._record_request: Optional[Optional[Path]] = None
        self.settings_error: Optional[str] = None

    # ---- settings -----------------------------------------------------------------
    def _initial_settings(self) -> str:
        path = user_settings_path()
        self.saved_settings_error: Optional[str] = None
        if path.is_file():
            try:
                text = read_settings(path)
                problem = self.check_settings(text)
                if problem is None:
                    return text
            except (OSError, UnicodeDecodeError) as error:
                problem = str(error)
            self.saved_settings_error = problem
            self._log(f"Your saved settings could not be used ({problem}); using the defaults instead")
        return self.lib.default_settings_ini()

    def check_settings(self, text: str) -> Optional[str]:
        """None when OsController accepts these settings, else the reason."""
        try:
            self.lib.controller(text, False, False).close()
            return None
        except ValueError as error:
            return str(error)

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
            config = replace(self._config, **changes)
            if _changes_control(self._config, config):
                # An Activate was meant for what was on screen when it was pressed, nothing later.
                self._ready = False
                if self._want_active:
                    self._want_active = None
            self._config = config
            self._config_dirty = True

    def set_active(self, active: bool) -> None:
        with self._lock:
            allowed = self._ready or not active     # pausing is always allowed
            if allowed:
                self._want_active = active
        if not allowed:
            self.event("Activate ignored: wait until the test is running, then press it again")

    def _mark_ready(self) -> None:
        with self._lock:
            if not self._config_dirty and not self._stop.is_set():
                self._ready = True

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
            self._ready, self._want_active, self._record_request = False, None, None
            self._snapshot = replace(self._snapshot, report=osi_capi.Report(), recording=None, frame=None,
                                     status="Starting camera...")
        self._thread = threading.Thread(target=self._run, name="tester-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            self._ready, self._want_active = False, None
        if self._thread:
            self._thread.join(timeout=timeout)
            # A worker that is still shutting down still owns the mouse buttons: never start a second one.
            if not self._thread.is_alive():
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
        finished = threading.Event()
        watchdog = threading.Thread(target=self._watchdog, args=(state, finished), name="tester-watchdog", daemon=True)
        watchdog.start()
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
                    if self._apply_config(state, config, settings, cv2):
                        settings_dirty = False      # a new controller already has these settings
                if settings_dirty and state.controller:
                    error = state.controller.apply_settings(settings)
                    self.settings_error = error
                    if error:
                        self.event(f"Settings not applied: {error}")
                    else:
                        state.zone = IniDoc(settings)
                        state.rebuild_smoothers(self.lib, config, settings)
                        self.event("Settings applied")
                if want_active is not None and state.controller and not self._stop.is_set():
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
            finished.set()
            watchdog.join(timeout=1.0)
            with self._lock:
                self._ready = False
            if not state.close():
                self.event("WARNING: stopped with a mouse button still down: Windows refused every release. "
                           "Click the physical mouse to free it.")
            self._publish(running=False, frame=None, status="Stopped", recording=None, recorded_rows=0)

    def _watchdog(self, state: "_WorkerState", finished: threading.Event) -> None:
        """Safety that cannot wait for the worker, which may be blocked in a camera read or a tracker load."""
        escape_reported = False
        while not finished.wait(self.WATCHDOG_SECONDS):
            try:
                controller, config = state.controller, state.config
                stopping = self._stop.is_set()
                escape = bool(config and config.real_mouse and self._escape_pressed())
                if controller and (stopping or escape):
                    was_active = state.last_report.active
                    controller.set_active(False)
                    controller.release_all()
                    if escape and was_active and not escape_reported:
                        escape_reported = True
                        self.event("Esc pressed: output paused")
                if not escape:
                    escape_reported = False
                if stopping or time.perf_counter() - state.heartbeat > self.STALE_SECONDS:
                    for held in ([controller] if controller else []) + list(state.retired):
                        held.release_all()
            except Exception as error:      # the watchdog must survive anything
                self._log(f"watchdog: {error!r}")

    def _apply_config(self, state: "_WorkerState", config: TesterConfig, settings: str, cv2) -> bool:
        """Returns True when it created a new controller."""
        mode = MODES[config.mode]
        old = state.config
        created = False
        # Safety first: opening a camera or loading a tracker takes seconds, so release before either.
        if old is not None and state.controller and _changes_control(old, config):
            self._release_buttons(state, f"switched to {mode.title}" if old.mode != config.mode else "options changed")
            state.expected_cursor = None
            # Also drops a move still owed from the previous test. _step lifts it unless this is camera mode.
            state.controller.set_inhibited(True)
        # Camera
        if state.cap is None or old is None or (config.camera_index, config.width, config.height) != (
                old.camera_index, old.width, old.height):
            if state.cap is not None:
                state.cap.release()
            self._publish(status=f"Opening camera {config.camera_index}...")
            cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.camera_index)
            # The same requests, in the same order, as core_engine's openSource().
            mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cap, "get"):
                try:
                    raw_fc = cap.get(cv2.CAP_PROP_FOURCC)
                    fc_int = int(round(raw_fc)) if math.isfinite(raw_fc) and raw_fc > 0 else 0
                    if fc_int != mjpg:
                        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                        cap.set(cv2.CAP_PROP_FPS, 60)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {config.camera_index}. Close other apps using it "
                                   f"(Zoom, Teams, the engine) or pick another camera.")
            state.cap = cap
            cam_info = ""
            if hasattr(cap, "get"):
                try:
                    fc_raw = cap.get(cv2.CAP_PROP_FOURCC)
                    fc_int = int(round(fc_raw)) if math.isfinite(fc_raw) and fc_raw > 0 else 0
                    fc_bytes = fc_int.to_bytes(4, "little", signed=False) if fc_int else b""
                    fc_str = fc_bytes.decode("ascii") if len(fc_bytes) == 4 and all(32 <= b <= 126 for b in fc_bytes) else (
                        f"0x{fc_int:08X}" if fc_int else "NONE")
                    neg_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                    neg_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    neg_fps = cap.get(cv2.CAP_PROP_FPS)
                    auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    cam_info = f" ({neg_w}x{neg_h} FOURCC={fc_str} reported={neg_fps:.1f} FPS auto_exp={auto_exp:.3g})"
                except Exception:
                    cam_info = ""
            self.event(f"Camera {config.camera_index} opened{cam_info}")
        # Trackers (reloaded only when their options change)
        hand_key = (mode.hand, config.lowlight, config.roi_recovery, round(config.index_alpha, 3)) if mode.hand else None
        if hand_key != state.hand_key:
            if state.hand:
                state.rejected_seen -= _rejected(state.hand)
                state.hand.close()
            state.hand = None
            if mode.hand:
                self._publish(status="Loading hand tracker...")
                state.hand = HandSource(mode.hand, config)
                self.event(f"Hand tracker loaded ({'IndexFingerTracker' if mode.hand == 'index' else 'HandTracker'})")
            state.hand_key = hand_key
        if bool(mode.gaze) != (state.gaze is not None):
            if state.gaze:
                state.rejected_seen -= _rejected(state.gaze)
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
                state.retire_controller()
            created = True
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
            active = not config.real_mouse or was_active
            state.controller.set_active(active)
            state.last_report = osi_capi.Report(active=active)
            state.expected_cursor = None
            target, _ = state.controller.geometry()
            self._publish(target_rect=target, report=state.last_report)
            self.event("Output: REAL MOUSE (paused - press Activate)" if config.real_mouse
                       else "Output: simulation (the real mouse is never touched)")
        state.rebuild_smoothers(self.lib, config, settings)
        if state.controller and hasattr(state.controller, "set_clicks_enabled"):
            state.controller.set_clicks_enabled(config.clicks)
        self._publish(smoothing_available=state.smoothing_available)
        if old is None or old.mode != config.mode:
            self.event(f"Mode: {mode.title}")
            state.reset_stats()
            if mode.gaze == "pointer" and gaze_is_uncalibrated(settings, self.lib.default_settings_ini()):
                self.event("Gaze is not calibrated: the default range is a placeholder, so the cursor can stick "
                           "to an edge or move the wrong way")
        state.config = config
        return created

    def _release_buttons(self, state: "_WorkerState", why: str) -> None:
        controller = state.controller
        controller.drain_events()
        released = controller.release_all()
        attempted = any(e.kind == osi_capi.EVENT_UP for e in controller.drain_events())
        if not released:
            self.event(f"WARNING: {why}, but Windows refused the button release - retrying")
        elif attempted:
            self.event(f"Released the held button ({why})")

    def _service_releases(self, state: "_WorkerState", stale: bool = False) -> None:
        """Retries that must not depend on frames or on which controller is current."""
        for controller in list(state.retired):
            if controller.release_all():
                controller.close()
                state.retired.remove(controller)
                self.event("Released the button held by the previous output")
        if stale and state.controller:
            state.controller.release_all()

    def _handle_recording(self, state: "_WorkerState", request: Path, config: TesterConfig) -> None:
        if state.recorder:
            recorder, state.recorder = state.recorder, None
            try:
                recorder.close()
                self.event(f"Recording saved: {recorder.path} ({recorder.rows} frames)")
            except OSError as error:
                self.event(f"Recording could not be finished: {error}")
            self._publish(recording=None)
        if str(request) not in ("", "."):
            try:
                state.recorder = CsvRecorder(request)
            except OSError as error:
                self.event(f"Could not record to {request}: {error}")
                return
            self.event(f"Recording to {request}")
            self._publish(recording=str(request), recorded_rows=0)

    def _step(self, state: "_WorkerState", config: TesterConfig, cv2) -> bool:
        ok, bgr = state.cap.read()
        if not ok or bgr is None:
            state.failures += 1
            stale = time.perf_counter() - state.last_frame > self.STALE_SECONDS
            self._service_releases(state, stale=stale or state.last_report.release_pending)
            if state.failures > 60:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.01)
            return True
        state.failures = 0
        self._service_releases(state)
        now = time.perf_counter()
        t = now - state.t0
        loop_start = now
        mode = MODES[config.mode]

        mirrored = cv2.flip(bgr, 1)
        hand = state.hand.process(bgr) if state.hand else HandObs()     # trackers mirror internally
        gaze = state.gaze.process(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)) if state.gaze else GazeObs()

        # Inputs for Engineer 5's OsController: always pass raw states so wink/blink cursor lock and
        # closure-pause stay active even when config.clicks is False (OsController suppresses clicks cleanly).
        # Real mouse: only Activate (button or Space) switches output on. A gesture that toggles pause may still pause,
        # but while output is paused it is withheld, so it can never switch the real mouse on by itself.
        no_resume = config.real_mouse and not state.last_report.active
        hand_in = None
        if mode.hand:
            point, tracking = state.hand_smoother.apply(hand.point, hand.tracking, t)
            hand_state = hand.state if mode.hand == "gesture" else 0
            if no_resume and hand_state == HAND_FIST and state.zone.get("gestures", "fist_action") == "toggle_pause":
                hand_state = 0
            hand_in = osi_capi.TrackerInput(point, hand_state, tracking)
        gaze_in = None
        if mode.gaze:
            gaze_tracking = osi_capi.TRACKED if gaze.point is not None else osi_capi.LOST
            point, tracking = (state.gaze_smoother.apply(gaze.point, gaze_tracking, t) if mode.gaze == "pointer"
                               else (None, osi_capi.LOST))
            eye_state = gaze.state
            if no_resume and eye_state == EYES_CLOSED:
                eye_state = 0
            gaze_in = osi_capi.TrackerInput(point, eye_state, tracking)

        controller = state.controller
        yielding = False
        camera_only = mode.key == "camera"
        if config.real_mouse and controller:
            yielding = not camera_only and self._physical_mouse_check(state, now)
        if controller:
            # Camera mode still updates, so refused releases keep being retried; it just cannot produce output.
            controller.set_inhibited(camera_only or yielding)
            report = controller.update(t, hand_in, gaze_in)
            if camera_only:
                report.cursor = None
        else:
            report = osi_capi.Report(active=state.last_report.active)
        report.release_pending = report.release_pending or bool(state.retired)
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
            try:
                state.recorder.write(t, mode.key, hand, gaze, report, state.fps)
            except OSError as error:
                self.event(f"Recording stopped, the file could not be written: {error}")
                self._handle_recording(state, Path(), config)
        tracked, held, lost = state.tracking_pcts()
        self._publish(
            running=True, frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode=mode.key, fps=state.fps,
            loop_ms=loop_ms, brightness=brightness, camera_size=(bgr.shape[1], bgr.shape[0]), hand=hand, gaze=gaze,
            report=report, counters=replace(state.counters), tracked_pct=tracked, held_pct=held, lost_pct=lost,
            yielding=yielding, recorded_rows=state.recorder.rows if state.recorder else 0,
            status="Running", error=None)
        state.heartbeat = time.perf_counter()
        self._mark_ready()
        return True

    def _physical_mouse_check(self, state: "_WorkerState", now: float) -> bool:
        """Yield to the real mouse for 1.5 s when the user moves it."""
        actual = self.lib.cursor_position()
        if state.yield_until and now < state.yield_until:
            return True
        if state.yield_until:
            state.yield_until = 0.0
            state.expected_cursor = None
            self.event("Physical mouse released: tester output resumed")
        if not state.last_report.active:
            state.expected_cursor = None    # paused: moving the mouse (say, to press Activate) is no conflict
        expected = state.expected_cursor
        if expected and actual and max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])) > \
                self.YIELD_TOLERANCE_PX:
            state.yield_until = now + self.YIELD_SECONDS
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
        elif state.release_warned and not report.release_pending:
            self.event("Windows accepted the button release")
        state.release_warned = report.release_pending
        rejected = _rejected(state.hand, state.gaze)
        if rejected > state.rejected_seen:
            if c.invalid_input == 0:
                self.event("Rejected invalid landmarks from MediaPipe, as core_engine does (counted in Live data)")
            c.invalid_input += rejected - state.rejected_seen
        state.rejected_seen = rejected
        if mode.hand:
            state.track_history.append(hand.tracking)
        elif mode.gaze:
            state.track_history.append(osi_capi.TRACKED if gaze.face else osi_capi.LOST)


def _rejected(*sources) -> int:
    return sum(getattr(source, "rejected", 0) for source in sources if source is not None)


def _changes_control(old: TesterConfig, new: TesterConfig) -> bool:
    """Anything but a drawing option interrupts what the buttons are doing."""
    drawing = dict(show_landmarks=new.show_landmarks, show_zone=new.show_zone, show_trail=new.show_trail)
    return replace(old, **drawing) != new


LEGACY_PLACEHOLDER_GAZE_REGIONS = (
    (0.3, 0.3, 0.7, 0.7),
    (0.25, 0.25, 0.75, 0.75),
    (0.25, 0.60, 0.75, 0.80),
)


def gaze_is_uncalibrated(settings: str, defaults: str) -> bool:
    doc = IniDoc(settings)
    fallback = _region(IniDoc(defaults).get("mapping", "gaze_region"), None)
    region = _region(doc.get("mapping", "gaze_region"), fallback)
    cal = (doc.get("mapping", "gaze_calibrated") or "").strip().lower()
    if region in LEGACY_PLACEHOLDER_GAZE_REGIONS:
        return True
    if cal in ("true", "1", "yes", "on"):
        return False
    if region == fallback:
        return True
    # Custom non-placeholder region (e.g. legacy INI or manual edit): treat as calibrated.
    return False


class _WorkerState:
    RELEASE_ATTEMPTS = 50         # x 20 ms when stopping
    RETIRE_ATTEMPTS = 10          # x 20 ms when the output changes

    def __init__(self):
        self.config: Optional[TesterConfig] = None
        self.cap = None
        self.hand: Optional[HandSource] = None
        self.hand_key = None
        self.gaze: Optional[GazeSource] = None
        self.controller: Optional[osi_capi.Controller] = None
        self.retired: List[osi_capi.Controller] = []    # replaced, but Windows still refuses a release
        self.hand_smoother = Smoother(None)
        self.gaze_smoother = Smoother(None)
        self.smoothing_available = False
        self.zone = IniDoc("")
        self.last_report = osi_capi.Report()
        self.expected_cursor = None
        self.yield_until = 0.0
        self.recorder: Optional[CsvRecorder] = None
        self.t0 = time.perf_counter()
        self.heartbeat = self.t0        # last completed frame, for the watchdog
        self.last_frame = 0.0
        self.fps = 0.0
        self.failures = 0
        self.release_warned = False
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=40)
        self.counters = Counters()
        self.rejected_seen = 0
        self.track_history: Deque[int] = deque(maxlen=150)

    def reset_stats(self):
        self.counters = Counters()
        self.rejected_seen = _rejected(self.hand, self.gaze)
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

    @staticmethod
    def _release_with_retry(controller, attempts: int) -> bool:
        for attempt in range(attempts):
            if controller.release_all():
                return True
            if attempt + 1 < attempts:
                time.sleep(0.02)
        return False

    def retire_controller(self) -> None:
        controller, self.controller = self.controller, None
        controller.set_active(False)
        if self._release_with_retry(controller, self.RETIRE_ATTEMPTS):
            controller.close()
        else:
            self.retired.append(controller)     # destroying it would forget the button it holds

    def close(self) -> bool:
        """False when a button is still down because Windows refused every release."""
        released = True
        controllers = self.retired + ([self.controller] if self.controller else [])
        self.retired, self.controller = [], None
        for controller in controllers:          # before anything that could fail
            released = self._release_with_retry(controller, self.RELEASE_ATTEMPTS) and released
            controller.close()
        if self.recorder:
            self.recorder.close()
        for source in (self.hand, self.gaze):
            if source:
                source.close()
        self.hand_smoother.close()
        self.gaze_smoother.close()
        if self.cap is not None:
            self.cap.release()
        return released


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
        if config.show_zone:            # index mode maps through the zone too
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
