"""ctypes binding for os_integration_capi.dll (Engineer 5's C interface).

The tester drives the real osi::OsController through this module, so what you
see in the tester is exactly what the engine's OS stage does.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

DLL_NAME = "os_integration_capi.dll"
EXPECTED_VERSION = 2
MAX_ACTIONS = 8

OUTPUT_RECORD = 0
OUTPUT_MOUSE = 1

TRACKED, HELD, LOST = 0, 1, 2
FILTER_INVALID, FILTER_OK, FILTER_REJECTED = 0, 1, 2

ACTION_DOWN, ACTION_UP, ACTION_CLICK, ACTION_TOGGLE_PAUSE = 0, 1, 2, 3
EVENT_MOVE, EVENT_DOWN, EVENT_UP, EVENT_CLICK = 0, 1, 2, 3
BUTTON_NAMES = ("left", "right", "middle")
SOURCE_NAMES = ("none", "hand", "gaze")


class _Tracker(ctypes.Structure):
    _fields_ = [("has_point", ctypes.c_int32), ("x", ctypes.c_double), ("y", ctypes.c_double),
                ("state", ctypes.c_int32), ("tracking", ctypes.c_int32)]


class _Report(ctypes.Structure):
    _fields_ = [("processed", ctypes.c_int32), ("source", ctypes.c_int32),
                ("has_cursor", ctypes.c_int32), ("cursor_x", ctypes.c_int32), ("cursor_y", ctypes.c_int32),
                ("moved", ctypes.c_int32), ("action_count", ctypes.c_int32),
                ("action_type", ctypes.c_int32 * MAX_ACTIONS), ("action_button", ctypes.c_int32 * MAX_ACTIONS),
                ("pause_toggled", ctypes.c_int32), ("active", ctypes.c_int32), ("inhibited", ctypes.c_int32),
                ("injection_failed", ctypes.c_int32), ("release_pending", ctypes.c_int32),
                ("gaze_calibrating", ctypes.c_int32), ("hand_held", ctypes.c_int32),
                ("unknown_states", ctypes.c_int32), ("confirmed_hand", ctypes.c_int32),
                ("confirmed_eye", ctypes.c_int32)]


class _Event(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("button", ctypes.c_int32), ("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int32), ("top", ctypes.c_int32), ("width", ctypes.c_int32),
                ("height", ctypes.c_int32)]


@dataclass
class TrackerInput:
    """One tracker's output for one frame, as OsController expects it."""
    point: Optional[Tuple[float, float]] = None
    state: int = 0
    tracking: int = TRACKED


@dataclass
class Report:
    processed: bool = False
    source: str = "none"
    cursor: Optional[Tuple[int, int]] = None
    moved: bool = False
    actions: List[Tuple[int, str]] = field(default_factory=list)  # (ACTION_*, button name)
    pause_toggled: bool = False
    active: bool = False
    inhibited: bool = False
    injection_failed: bool = False
    release_pending: bool = False
    gaze_calibrating: bool = False
    hand_held: bool = False
    unknown_states: int = 0
    confirmed_hand: int = 0
    confirmed_eye: int = 0


@dataclass
class Event:
    kind: int
    button: str
    x: int
    y: int


def find_dll(start: Optional[Path] = None) -> Optional[Path]:
    """OSI_CAPI_DLL if set (the launcher sets it), else the newest build under out/build."""
    env = os.environ.get("OSI_CAPI_DLL")
    if env and Path(env).is_file():
        return Path(env)
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        build = parent / "out" / "build"
        if build.is_dir():
            found = sorted(build.rglob(DLL_NAME), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                return found[0]
    return None


class OsiLibrary:
    def __init__(self, path: Optional[Path] = None):
        path = path or find_dll()
        if path is None:
            raise FileNotFoundError(
                f"{DLL_NAME} not found. Build core_engine/os_integration (it is built next to HandEyeTester.exe).")
        self.path = Path(path)
        self._dll = ctypes.CDLL(str(self.path))
        d = self._dll
        d.osi_version.restype = ctypes.c_int32
        d.osi_default_settings_ini.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        d.osi_default_settings_ini.restype = ctypes.c_int32
        d.osi_create.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
        d.osi_create.restype = ctypes.c_void_p
        d.osi_destroy.argtypes = [ctypes.c_void_p]
        d.osi_apply_settings_ini.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
        d.osi_apply_settings_ini.restype = ctypes.c_int32
        d.osi_set_active.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        d.osi_set_inhibited.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        d.osi_release_all.argtypes = [ctypes.c_void_p]
        d.osi_release_all.restype = ctypes.c_int32
        d.osi_update.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(_Tracker),
                                 ctypes.POINTER(_Tracker), ctypes.POINTER(_Report)]
        d.osi_update.restype = ctypes.c_int32
        d.osi_drain_events.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Event), ctypes.c_int32]
        d.osi_drain_events.restype = ctypes.c_int32
        d.osi_screen_geometry.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rect), ctypes.POINTER(_Rect)]
        d.osi_screen_geometry.restype = ctypes.c_int32
        d.osi_cursor_position.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        d.osi_cursor_position.restype = ctypes.c_int32
        d.osi_filter_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        d.osi_filter_create.restype = ctypes.c_void_p
        d.osi_filter_destroy.argtypes = [ctypes.c_void_p]
        d.osi_filter_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int32]
        d.osi_filter_apply.restype = ctypes.c_int32
        self.version = int(d.osi_version())
        if self.version != EXPECTED_VERSION:
            raise RuntimeError(f"{DLL_NAME} version {self.version}, tester expects {EXPECTED_VERSION}: rebuild it")
        # Bound after the version check: an older DLL does not export these.
        d.osi_filter_update.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double), ctypes.c_double]
        d.osi_filter_update.restype = ctypes.c_int32
        d.osi_filter_reset.argtypes = [ctypes.c_void_p]

    def default_settings_ini(self) -> str:
        size = self._dll.osi_default_settings_ini(None, 0)
        buffer = ctypes.create_string_buffer(size + 1)
        self._dll.osi_default_settings_ini(buffer, size + 1)
        return buffer.value.decode("utf-8")

    def cursor_position(self) -> Optional[Tuple[int, int]]:
        x, y = ctypes.c_int32(), ctypes.c_int32()
        if self._dll.osi_cursor_position(ctypes.byref(x), ctypes.byref(y)):
            return x.value, y.value
        return None

    def controller(self, settings_ini: str, all_monitors: bool, real_mouse: bool) -> "Controller":
        return Controller(self, settings_ini, all_monitors, real_mouse)

    def smoothing_filter(self, min_cutoff: float, beta: float, d_cutoff: float) -> Optional["SmoothingFilter"]:
        handle = self._dll.osi_filter_create(min_cutoff, beta, d_cutoff)
        return SmoothingFilter(self, handle) if handle else None


class Controller:
    """One osi::OsController. The lock lets the tester's watchdog release buttons from its own thread."""

    def __init__(self, lib: OsiLibrary, settings_ini: str, all_monitors: bool, real_mouse: bool):
        self._lib = lib
        self._d = lib._dll
        self._lock = threading.RLock()
        self._h = None
        error = ctypes.create_string_buffer(512)
        self._h = self._d.osi_create(settings_ini.encode("utf-8"), 1 if all_monitors else 0,
                                     OUTPUT_MOUSE if real_mouse else OUTPUT_RECORD, error, len(error))
        if not self._h:
            raise ValueError(error.value.decode("utf-8", "replace") or "controller could not be created")
        self.real_mouse = real_mouse
        self._report = _Report()
        self._events = (_Event * 256)()

    def close(self) -> None:
        with self._lock:
            if self._h:
                self._d.osi_destroy(self._h)  # releases any held button
                self._h = None

    def __del__(self):
        self.close()

    def apply_settings(self, settings_ini: str) -> Optional[str]:
        """None on success, else the reason (the controller is then unchanged)."""
        error = ctypes.create_string_buffer(512)
        with self._lock:
            applied = self._d.osi_apply_settings_ini(self._h, settings_ini.encode("utf-8"), error, len(error))
        if applied:
            return None
        return error.value.decode("utf-8", "replace")

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._d.osi_set_active(self._h, 1 if active else 0)

    def set_inhibited(self, inhibited: bool) -> None:
        with self._lock:
            self._d.osi_set_inhibited(self._h, 1 if inhibited else 0)

    def release_all(self) -> bool:
        with self._lock:
            return bool(self._d.osi_release_all(self._h))

    @staticmethod
    def _tracker(value: Optional[TrackerInput]) -> _Tracker:
        t = _Tracker()
        if value is None:
            t.tracking = LOST
            return t
        if value.point is not None:
            t.has_point, t.x, t.y = 1, float(value.point[0]), float(value.point[1])
        t.state = int(value.state)
        t.tracking = int(value.tracking)
        return t

    def update(self, timestamp: float, hand: Optional[TrackerInput], gaze: Optional[TrackerInput]) -> Report:
        h, g = self._tracker(hand), self._tracker(gaze)
        r = self._report
        with self._lock:
            self._d.osi_update(self._h, timestamp, ctypes.byref(h), ctypes.byref(g), ctypes.byref(r))
        return Report(
            processed=bool(r.processed), source=SOURCE_NAMES[r.source] if 0 <= r.source < 3 else "none",
            cursor=(r.cursor_x, r.cursor_y) if r.has_cursor else None, moved=bool(r.moved),
            actions=[(r.action_type[i], BUTTON_NAMES[r.action_button[i] % 3]) for i in range(r.action_count)],
            pause_toggled=bool(r.pause_toggled), active=bool(r.active), inhibited=bool(r.inhibited),
            injection_failed=bool(r.injection_failed), release_pending=bool(r.release_pending),
            gaze_calibrating=bool(r.gaze_calibrating), hand_held=bool(r.hand_held),
            unknown_states=r.unknown_states, confirmed_hand=r.confirmed_hand, confirmed_eye=r.confirmed_eye)

    def drain_events(self) -> List[Event]:
        with self._lock:
            n = self._d.osi_drain_events(self._h, self._events, len(self._events))
        return [Event(e.kind, BUTTON_NAMES[e.button % 3], e.x, e.y) for e in self._events[:n]]

    def geometry(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        target, desktop = _Rect(), _Rect()
        with self._lock:
            self._d.osi_screen_geometry(self._h, ctypes.byref(target), ctypes.byref(desktop))
        return ((target.left, target.top, target.width, target.height),
                (desktop.left, desktop.top, desktop.width, desktop.height))


class SmoothingFilter:
    """Engineer 4's 1-Euro TrackingFilter2D, used read-only through the C interface."""

    def __init__(self, lib: OsiLibrary, handle):
        self._d = lib._dll
        self._h = handle

    def apply(self, point: Optional[Tuple[float, float]], timestamp: float) -> Optional[Tuple[float, float]]:
        x = ctypes.c_double(point[0] if point else -1.0)
        y = ctypes.c_double(point[1] if point else -1.0)
        if self._d.osi_filter_apply(self._h, ctypes.byref(x), ctypes.byref(y), timestamp, 1 if point else 0):
            return x.value, y.value
        return None

    def update(self, point: Tuple[float, float], timestamp: float) -> Tuple[int, Optional[Tuple[float, float]]]:
        """(FILTER_*, smoothed point). The point is None unless the status is FILTER_OK."""
        x, y = ctypes.c_double(point[0]), ctypes.c_double(point[1])
        status = self._d.osi_filter_update(self._h, ctypes.byref(x), ctypes.byref(y), timestamp)
        return status, (x.value, y.value) if status == FILTER_OK else None

    def reset(self) -> None:
        self._d.osi_filter_reset(self._h)

    def close(self) -> None:
        if self._h:
            self._d.osi_filter_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()
