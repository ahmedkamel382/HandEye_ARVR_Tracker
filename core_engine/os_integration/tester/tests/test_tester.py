"""Tests for the HandEye tester logic. No camera, no real mouse.

Run from the repository root:
    .venv\\Scripts\\python.exe -m pytest core_engine\\os_integration\\tester\\tests -q
Needs a built os_integration_capi.dll (out/build/...); those tests skip without it.
"""

import csv
import math
import sys
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osi_capi  # noqa: E402
import tester_app  # noqa: E402
import tester_engine  # noqa: E402
from tester_app import AccuracyTest, _region  # noqa: E402
from tester_engine import (MODES, CsvRecorder, GazeObs, HandObs, IniDoc, Smoother, TesterConfig,  # noqa: E402
                           TesterEngine, engine_differences, format_value, gaze_is_uncalibrated)

TesterConfig.__test__ = TesterEngine.__test__ = False  # not test classes, despite the names


@pytest.fixture(autouse=True)
def private_settings(tmp_path, monkeypatch):
    """No test may read or write the developer's real os_integration.ini."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


@pytest.fixture(scope="module")
def lib():
    path = osi_capi.find_dll()
    if path is None:
        pytest.skip("os_integration_capi.dll not built")
    return osi_capi.OsiLibrary(path)


# ----------------------------------------------------------------------------- settings text

def test_ini_get_set_keeps_comments_and_order():
    doc = IniDoc("# top\n[mapping]\nhand_region = 0.1 0.1 0.9 0.9 # note\n\n[gestures]\nwink_onset_seconds = 0.25\n")
    assert doc.get("mapping", "hand_region") == "0.1 0.1 0.9 0.9"
    doc.set("gestures", "wink_onset_seconds", 0.3)
    doc.set("gestures", "fist_action", "none")            # new key goes into its section
    doc.set("app", "start_active", True)                   # new section is appended
    text = doc.text()
    assert text.startswith("# top\n[mapping]")
    assert "wink_onset_seconds = 0.3\nfist_action = none" in text
    assert text.rstrip().endswith("[app]\nstart_active = true")
    assert IniDoc(text).get_float("gestures", "wink_onset_seconds", 0) == 0.3


def test_format_value():
    assert format_value(True) == "true"
    assert format_value((0.15, 0.15, 0.85, 0.85)) == "0.15 0.15 0.85 0.85"
    assert format_value(0.1 + 0.2) == "0.3"


def test_region_parsing():
    assert _region("0.1 0.2 0.3 0.4", None) == (0.1, 0.2, 0.3, 0.4)
    assert _region("bad", (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert _region(None, (1, 2, 3, 4)) == (1, 2, 3, 4)


def test_modes_are_consistent():
    assert {"camera", "index", "hand", "gaze", "winks", "full"} <= set(MODES)
    assert MODES["index"].hand == "index" and MODES["index"].gaze is None
    assert MODES["full"].hand == "gesture" and MODES["full"].gaze == "pointer"
    assert MODES["winks"].gaze == "states"


# ----------------------------------------------------------------------------- accuracy test

def test_accuracy_test_click_mode():
    test = AccuracyTest()
    test.start(dwell=False)
    for _ in range(AccuracyTest.TARGETS):
        target = test.current(1920, 1080)
        test.update((target[0] + 500, target[1]), clicked=True)   # miss
        test.update((target[0] + 5, target[1]), clicked=True)     # hit
    assert not test.running
    assert len(test.results) == AccuracyTest.TARGETS
    assert "10 hits, 10 misses" in test.summary()


def test_accuracy_test_dwell_mode(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("tester_app.time.perf_counter", lambda: clock[0])
    test = AccuracyTest()
    test.start(dwell=True)
    target = test.current(1000, 800)
    test.update(target, clicked=False)
    clock[0] += 0.5
    test.update(target, clicked=False)
    assert len(test.results) == 0
    clock[0] += 0.6
    test.update(target, clicked=False)
    assert len(test.results) == 1


# ----------------------------------------------------------------------------- smoothing gate

class FakeFilter:
    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.calls = []

    def update(self, point, t):
        self.calls.append(point)
        status = self.statuses.pop(0) if self.statuses else osi_capi.FILTER_OK
        return status, (point[0] / 2, point[1] / 2) if status == osi_capi.FILTER_OK else None

    def reset(self):
        self.calls.append(None)

    def close(self):
        pass


def test_smoother_only_filters_fresh_points():
    f = FakeFilter()
    s = Smoother(f)
    assert s.apply((0.2, 0.4), osi_capi.TRACKED, 0.0) == ((0.1, 0.2), osi_capi.TRACKED)
    assert s.apply((9, 9), osi_capi.HELD, 0.1) == ((0.1, 0.2), osi_capi.HELD)   # held: cached point, no filter call
    assert s.apply(None, osi_capi.LOST, 0.2) == (None, osi_capi.LOST)           # lost: reset
    assert f.calls == [(0.2, 0.4), None]
    assert Smoother(None).apply((0.5, 0.5), osi_capi.TRACKED, 0) == ((0.5, 0.5), osi_capi.TRACKED)


# ----------------------------------------------------------------------------- recording

def test_csv_recorder(tmp_path):
    path = tmp_path / "session.csv"
    rec = CsvRecorder(path)
    report = osi_capi.Report(cursor=(10, 20), source="hand", actions=[(osi_capi.ACTION_CLICK, "right")], active=True)
    rec.write(0.5, "hand", HandObs(point=(0.25, 0.75), state=1, tracking=osi_capi.TRACKED, status="DETECTED"),
              GazeObs(), report, 30.0)
    rec.close()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["hand_x"] == "0.25000" and rows[0]["actions"] == "click:right"
    assert rows[0]["yaw_deg"] == "" and rows[0]["cursor_x"] == "10"


# ----------------------------------------------------------------------------- the real library

def test_library_controller_round_trip(lib):
    ini = lib.default_settings_ini()
    assert "[gestures]" in ini
    c = lib.controller(ini, all_monitors=False, real_mouse=False)
    try:
        r = c.update(0.0, osi_capi.TrackerInput((0.5, 0.5)), None)
        assert r.processed and not r.active and r.cursor is not None
        c.set_active(True)
        actions = []
        for i in range(10):
            r = c.update((i + 1) / 32, osi_capi.TrackerInput((0.5, 0.5), state=1), None)
            actions += r.actions
        assert (osi_capi.ACTION_DOWN, "left") in actions and r.confirmed_hand == 1
        events = c.drain_events()
        assert any(e.kind == osi_capi.EVENT_DOWN for e in events)
        assert c.apply_settings("[mapping]\nhand_region = 1 2\n") is not None   # rejected, with a reason
        assert c.apply_settings(ini) is None
        target, desktop = c.geometry()
        assert target[2] > 0 and desktop[2] > 0
        assert c.release_all()
    finally:
        c.close()


def test_library_rejects_bad_settings(lib):
    with pytest.raises(ValueError):
        lib.controller("[gestures]\nwink_onset_seconds = soon\n", False, False)


def test_library_smoothing_filter(lib):
    f = lib.smoothing_filter(1.2, 4.0, 1.0)
    if f is None:
        pytest.skip("built without dsp_filter")
    assert f.apply((0.2, 0.3), 0.0) == (0.2, 0.3)
    x, _ = f.apply((0.8, 0.3), 0.02)
    assert 0.2 < x < 0.8
    assert f.apply(None, 0.04) is None
    f.close()


def test_library_cursor_position(lib):
    pos = lib.cursor_position()
    assert pos is None or (isinstance(pos[0], int) and isinstance(pos[1], int))


# ============================================================ audit of 22 Nov 2026

TRACKED, HELD, LOST = osi_capi.TRACKED, osi_capi.HELD, osi_capi.LOST


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"no display for Tk: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def make_app(lib, tk_root, monkeypatch):
    monkeypatch.setattr(tester_app.TesterApp, "tick", lambda self: None)
    windows = []

    def make(**regions):
        doc = IniDoc(lib.default_settings_ini())
        for key, value in regions.items():
            doc.set("mapping", key, value)
        if regions:
            path = tester_engine.user_settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(doc.text(), encoding="utf-8")
        window = tk.Toplevel(tk_root)
        window.withdraw()
        windows.append(window)
        app = tester_app.TesterApp(window, lib, lambda text: None)
        monkeypatch.setattr(app.engine, "start", lambda: None)
        return app

    yield make
    for window in windows:
        if window.winfo_exists():
            window.destroy()


def regions_of(text):
    doc = IniDoc(text)
    return doc.get("mapping", "hand_region"), doc.get("mapping", "gaze_region")


# ---------------------------------------------------------------- #2 calibration survives

@pytest.mark.parametrize("hand, gaze", [
    ("0.1 0.22 0.7 0.95", "0.35 0.32 0.65 0.10"),                   # the action plan's example
    ("0.15 0.15 0.85 0.85", "0.20 0.50 0.80 0.10"),                 # the cross-team report's example
    ("0.91 0.8 0.13 0.04", "0.7123456789 0.3301234567 0.2876543211 -0.0612345678"),
    ("0.02 0.01 0.99 0.98", "0.49 0.21 0.52 0.18"),                 # outside the sliders' comfortable range
])
def test_calibrated_regions_survive_load_start_mode_change_readback(make_app, hand, gaze):
    app = make_app(hand_region=hand, gaze_region=gaze)
    before = app.engine.settings_text()
    app.toggle_run()
    for mode in MODES:
        app.mode_var.set(mode)
        app._mode_changed()
    app.root.update()
    assert app.engine.settings_text() == before
    assert app._tuning_ini() == before
    assert regions_of(before) == (hand, gaze)


@pytest.mark.parametrize("slider, key, axis, value, span", [
    ("zone_x", "hand_region", 0, 0.5, 0.5), ("zone_y", "hand_region", 1, 0.6, 0.6),
    ("gaze_x", "gaze_region", 0, 6.0, 0.2), ("gaze_y", "gaze_region", 1, 9.0, 0.3),
])
def test_tuning_resizes_one_axis_keeping_centre_and_direction(make_app, slider, key, axis, value, span):
    regions = {"hand_region": "0.81 0.22 0.11 0.95", "gaze_region": "0.7123456789 0.32 0.3 0.10"}
    app = make_app(**regions)
    old = [float(v) for v in regions[key].split()]
    app.sliders[slider].set(value)
    doc = IniDoc(app._tuning_ini())
    new = [float(v) for v in doc.get("mapping", key).split()]
    low, high = new[axis], new[axis + 2]
    assert abs(high - low) == pytest.approx(span, abs=1e-5)
    assert (low + high) / 2 == pytest.approx((old[axis] + old[axis + 2]) / 2, abs=1e-5)
    assert (high < low) == (old[axis + 2] < old[axis])
    written, kept = regions[key].split(), doc.get("mapping", key).split()
    assert (written[1 - axis], written[3 - axis]) == (kept[1 - axis], kept[3 - axis])
    other = "gaze_region" if key == "hand_region" else "hand_region"
    assert doc.get("mapping", other) == regions[other]


def test_editing_another_control_leaves_regions_untouched(make_app):
    app = make_app(hand_region="0.1 0.22 0.7 0.95", gaze_region="0.35 0.32 0.65 0.10")
    app.sliders["wink"].set(0.4)
    app._push_tuning()
    text = app.engine.settings_text()
    assert regions_of(text) == ("0.1 0.22 0.7 0.95", "0.35 0.32 0.65 0.10")
    assert IniDoc(text).get("gestures", "wink_onset_seconds") == "0.4"


def test_moving_a_slider_back_restores_the_exact_text(make_app):
    app = make_app(gaze_region="0.7123456789 0.33 0.2876543211 -0.06")
    before = app.engine.settings_text()
    loaded = app.sliders["gaze_x"].get()
    app.sliders["gaze_x"].set(20.0)
    app._push_tuning()
    app.sliders["gaze_x"].set(loaded)
    app._push_tuning()
    assert app.engine.settings_text() == before


def test_start_and_tuning_never_write_the_settings_file(make_app):
    app = make_app(gaze_region="0.35 0.32 0.65 0.10")
    path = tester_engine.user_settings_path()
    before = path.read_bytes()
    app.sliders["wink"].set(0.4)
    app._push_tuning()
    app.toggle_run()
    assert path.read_bytes() == before


def test_save_keeps_a_calibration_made_while_the_tester_was_open(make_app, monkeypatch):
    app = make_app(gaze_region="0.3 0.3 0.7 0.7")
    path = tester_engine.user_settings_path()
    doc = IniDoc(path.read_text(encoding="utf-8"))
    doc.set("mapping", "gaze_region", "0.71 0.33 0.29 0.08")        # the tray wizard saves
    path.write_text(doc.text(), encoding="utf-8")
    app.sliders["wink"].set(0.4)
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    app.save_user_settings()
    saved = IniDoc(path.read_text(encoding="utf-8"))
    assert saved.get("mapping", "gaze_region") == "0.71 0.33 0.29 0.08"
    assert saved.get("gestures", "wink_onset_seconds") == "0.4"


def test_reset_then_save_writes_the_defaults(make_app, lib, monkeypatch):
    app = make_app(gaze_region="0.71 0.33 0.29 0.08")
    app.reset_tuning()
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    app.save_user_settings()
    saved = tester_engine.user_settings_path().read_text(encoding="utf-8")
    assert regions_of(saved) == regions_of(lib.default_settings_ini())


def test_invalid_saved_settings_fall_back_before_the_sliders_load(make_app, lib):
    app = make_app(gaze_region="0.5 0.1 0.505 0.9")                 # span too small
    assert app.engine.settings_text() == lib.default_settings_ini()
    assert app._pending_log and "could not be used" in app._pending_log[0]


def test_ini_reads_comments_and_repeats_like_the_engine():
    doc = IniDoc("[hand_filter]\nbeta = 2\nbeta = 9 ; tuned\n")
    assert doc.get_float("hand_filter", "beta", 4.0) == 9.0
    doc.set("hand_filter", "beta", 7.0)
    assert doc.text() == "[hand_filter]\nbeta = 2\nbeta = 7  ; tuned\n"


# ---------------------------------------------------------------- #3 buttons on mode change

class FakeCamera:
    working = True

    def read(self):
        return (True, np.zeros((48, 64, 3), np.uint8)) if self.working else (False, None)

    def release(self):
        pass


class FakeHand:
    def __init__(self, *args):
        pass

    def process(self, frame):
        return HandObs(point=(0.5, 0.5), state=1, tracking=TRACKED, status="DETECTED")

    def close(self):
        pass


class FakeGaze:
    def __init__(self, *args):
        pass

    def process(self, frame):
        return GazeObs()

    def close(self):
        pass


class RecordingController:
    """The real controller in record mode, remembering every event."""

    def __init__(self, controller):
        self.inner, self.events, self.updates = controller, [], 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def drain_events(self):
        new = self.inner.drain_events()
        self.events += new
        return new

    def update(self, *args):
        self.updates += 1
        return self.inner.update(*args)

    def close(self):
        self.drain_events()                                         # destroying the controller discards them
        self.inner.close()

    def count(self, kind):
        self.drain_events()
        return sum(1 for e in self.events if e.kind == kind)


class RefusingController:
    """Holds a button and refuses to release it until told otherwise."""

    def __init__(self):
        self.held, self.accept, self.closed, self.inhibited, self.updates, self.attempts = True, False, False, False, 0, 0

    def release_all(self):
        self.attempts += 1
        self.held = self.held and not self.accept
        return not self.held

    def update(self, t, hand, gaze):
        self.updates += 1
        if self.held:
            self.release_all()                                      # OsController retries pending releases
        return osi_capi.Report(processed=True, active=True, inhibited=self.inhibited, release_pending=self.held)

    def set_inhibited(self, value):
        self.inhibited = value

    def set_active(self, value):
        pass

    def drain_events(self):
        return []

    def close(self):
        self.closed = True


@pytest.fixture
def worker(lib, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(tester_engine, "HandSource", FakeHand)
    monkeypatch.setattr(tester_engine, "GazeSource", FakeGaze)
    states = []

    def make(mode, controller=None):
        log = []
        engine = TesterEngine(lib, log.append)
        state = tester_engine._WorkerState()
        state.t0 = time.perf_counter() - 10.0                       # room for setup frames
        state.cap = FakeCamera()
        state.controller = controller or RecordingController(lib.controller(lib.default_settings_ini(), False, False))
        state.controller.set_active(True)
        state.config = TesterConfig(mode=mode)
        engine._apply_config(state, state.config, engine.settings_text(), cv2)
        states.append(state)
        return engine, state, log, cv2

    yield make
    for state in states:
        state.RELEASE_ATTEMPTS = 1
        state.close()


@pytest.mark.parametrize("old_mode, new_mode", [(a, b) for a in MODES for b in MODES if a != b])
def test_every_mode_change_releases_a_held_button(worker, old_mode, new_mode):
    engine, state, log, cv2 = worker(old_mode)
    controller = state.controller
    pinch = osi_capi.TrackerInput((0.5, 0.5), 1)
    for i in range(8):
        controller.inner.update(i / 32, pinch, None)
    assert controller.count(osi_capi.EVENT_DOWN) == 1, "setup: no button held"
    engine._apply_config(state, replace(state.config, mode=new_mode), engine.settings_text(), cv2)
    assert controller.count(osi_capi.EVENT_UP) == 1
    assert any("Released the held button" in line for line in log)


def test_camera_mode_releases_and_keeps_updating_without_output(worker):
    engine, state, log, cv2 = worker("hand")
    for _ in range(10):
        time.sleep(0.02)
        engine._step(state, state.config, cv2)
    controller = state.controller
    assert controller.count(osi_capi.EVENT_DOWN) == 1, "setup: no pinch"
    camera = replace(state.config, mode="camera")
    engine._apply_config(state, camera, engine.settings_text(), cv2)
    assert controller.count(osi_capi.EVENT_UP) == 1
    updates, events = controller.updates, len(controller.events)
    for _ in range(3):
        time.sleep(0.002)
        engine._step(state, camera, cv2)
    assert controller.updates == updates + 3
    assert len(controller.events) == events, "camera mode produced mouse output"
    assert state.last_report.inhibited and state.last_report.cursor is None


def test_a_continued_pinch_is_not_pressed_again_after_a_mode_change(worker):
    engine, state, log, cv2 = worker("hand")
    for _ in range(10):
        time.sleep(0.02)
        engine._step(state, state.config, cv2)
    full = replace(state.config, mode="full")
    engine._apply_config(state, full, engine.settings_text(), cv2)
    for _ in range(8):
        time.sleep(0.02)
        engine._step(state, full, cv2)
    assert state.controller.count(osi_capi.EVENT_DOWN) == 1


@pytest.mark.parametrize("new_mode", ["camera", "index", "full", "gaze", "winks"])
def test_refused_release_on_mode_change_is_logged_and_retried(worker, new_mode):
    controller = RefusingController()
    engine, state, log, cv2 = worker("hand", controller)
    config = replace(state.config, mode=new_mode)
    engine._apply_config(state, config, engine.settings_text(), cv2)
    assert any("retrying" in line for line in log)
    for _ in range(3):
        engine._step(state, config, cv2)
    assert controller.updates == 3 and controller.held
    controller.accept = True
    engine._step(state, config, cv2)
    assert not controller.held


def test_buttons_are_released_before_a_slow_tracker_loads(worker, monkeypatch):
    engine, state, log, cv2 = worker("hand")
    order = []

    class SlowHand(FakeHand):
        def __init__(self, *args):
            order.append("load")

    monkeypatch.setattr(tester_engine, "HandSource", SlowHand)
    monkeypatch.setattr(engine, "_release_buttons", lambda *args: order.append("release"))
    engine._apply_config(state, replace(state.config, mode="index"), engine.settings_text(), cv2)
    assert order == ["release", "load"]


def test_replaced_output_keeps_retrying_a_refused_release(worker):
    old = RefusingController()
    engine, state, log, cv2 = worker("hand", old)
    state.RETIRE_ATTEMPTS = 1
    config = replace(state.config, all_monitors=True)               # any option that replaces the controller
    assert engine._apply_config(state, config, engine.settings_text(), cv2) is True
    assert state.retired == [old] and not old.closed
    engine._step(state, config, cv2)
    assert state.last_report.release_pending
    old.accept = True
    engine._step(state, config, cv2)
    assert old.closed and not state.retired


def test_stalled_camera_releases_held_buttons(worker):
    controller = RefusingController()
    controller.accept = True
    engine, state, log, cv2 = worker("hand", controller)
    state.cap.working = False
    state.last_frame = time.perf_counter() - 1.0
    engine._step(state, state.config, cv2)
    assert not controller.held


def test_yielding_to_the_physical_mouse_inhibits_output(worker, monkeypatch):
    controller = RefusingController()                               # a fake: no real mouse output is possible
    engine, state, log, cv2 = worker("hand", controller)
    monkeypatch.setattr(engine, "_escape_pressed", lambda: False)
    monkeypatch.setattr(engine, "_physical_mouse_check", lambda *args: True)
    engine._step(state, replace(state.config, real_mouse=True), cv2)
    assert controller.inhibited


def test_stopping_reports_a_button_windows_would_not_release(worker):
    controller = RefusingController()
    engine, state, log, cv2 = worker("hand", controller)
    state.RELEASE_ATTEMPTS = 3
    assert state.close() is False
    assert controller.attempts >= 3 and controller.closed


def test_stop_timeout_never_starts_a_second_worker(lib):
    engine = TesterEngine(lib, lambda text: None)
    release = threading.Event()
    blocked = threading.Thread(target=release.wait, daemon=True)
    blocked.start()
    engine._thread = blocked
    try:
        engine.stop(timeout=0.01)
        engine.start()
        assert engine.running and engine._thread is blocked
    finally:
        release.set()
        blocked.join(2)


# ---------------------------------------------------------------- #4 smoothing gate

@pytest.fixture
def dsp_filter(lib):
    filt = lib.smoothing_filter(1.2, 4.0, 1.0)
    if filt is None:
        pytest.skip("built without dsp_filter")
    yield filt
    filt.close()


@pytest.mark.parametrize("stale_time", [1.0, 0.5])
def test_stale_timestamp_replays_the_filtered_point(dsp_filter, stale_time):
    gate = Smoother(dsp_filter)
    assert gate.apply((0.2, 0.3), TRACKED, 1.0) == ((0.2, 0.3), TRACKED)
    assert gate.apply((0.9, 0.8), TRACKED, stale_time) == ((0.2, 0.3), HELD)
    assert gate.apply((0.9, 0.8), HELD, 1.1) == ((0.2, 0.3), HELD)


def test_rejection_without_a_cache_publishes_nothing():
    gate = Smoother(FakeFilter(osi_capi.FILTER_REJECTED))
    assert gate.apply((0.9, 0.8), TRACKED, 0.0) == (None, LOST)


def test_invalid_filter_result_clears_the_cache():
    gate = Smoother(FakeFilter(osi_capi.FILTER_OK, osi_capi.FILTER_INVALID))
    gate.apply((0.2, 0.4), TRACKED, 0.0)
    assert gate.apply((0.9, 0.8), TRACKED, 0.1) == (None, LOST)
    assert gate.apply((0.9, 0.8), HELD, 0.2) == (None, LOST)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("axis", [0, 1])
def test_non_finite_points_are_never_published(dsp_filter, bad, axis):
    point = (bad, 0.4) if axis == 0 else (0.4, bad)
    for gate in (Smoother(dsp_filter), Smoother(None)):
        assert gate.apply(point, TRACKED, 1.0) == (None, LOST)
    assert dsp_filter.update(point, 2.0) == (osi_capi.FILTER_INVALID, None)


def test_held_without_smoothing_replays_the_last_observation():
    gate = Smoother(None)
    assert gate.apply((0.9, 0.8), HELD, 0.0) == (None, LOST)
    gate.apply((0.2, 0.3), TRACKED, 0.1)
    assert gate.apply((0.9, 0.8), HELD, 0.2) == ((0.2, 0.3), HELD)


# ---------------------------------------------------------------- #1 hint and #5 parity

def test_defaults_match_core_engine_options():
    config = TesterConfig(mode="hand")
    assert (config.lowlight, config.roi_recovery, config.index_alpha, config.smoothing) == (False, False, 1.0, True)
    assert engine_differences(config) == []
    assert engine_differences(replace(config, mode="full")) == []
    assert len(engine_differences(replace(config, lowlight=True, roi_recovery=True))) == 2


def test_index_mode_labels_double_smoothing():
    single = engine_differences(TesterConfig(mode="index"))
    assert not any("two smoothing" in text for text in single)
    stacked = engine_differences(TesterConfig(mode="index", index_alpha=0.35))
    assert any("two smoothing" in text for text in stacked)


def test_match_engine_button_restores_parity(make_app):
    app = make_app()
    app.mode_var.set("hand")
    app.lowlight_var.set(True)
    app.roi_var.set(True)
    app._options_changed()
    assert "Differs" in app.parity_label.cget("text")
    app.match_engine()
    assert engine_differences(app.engine.config) == []
    assert "match core_engine" in app.parity_label.cget("text")


def test_uncalibrated_gaze_is_detected(lib):
    defaults = lib.default_settings_ini()
    assert gaze_is_uncalibrated(defaults, defaults)
    assert gaze_is_uncalibrated("", defaults)
    doc = IniDoc(defaults)
    doc.set("mapping", "gaze_region", "0.71 0.33 0.29 0.08")
    assert not gaze_is_uncalibrated(doc.text(), defaults)
    for legacy in ("0.3 0.3 0.7 0.7", "0.25 0.25 0.75 0.75", "0.25 0.60 0.75 0.80"):
        legacy_doc = IniDoc(defaults)
        legacy_doc.set("mapping", "gaze_region", legacy)
        assert gaze_is_uncalibrated(legacy_doc.text(), defaults)
    explicit_cal = IniDoc(defaults)
    explicit_cal.set("mapping", "gaze_calibrated", "true")
    assert not gaze_is_uncalibrated(explicit_cal.text(), defaults)


# ---------------------------------------------------------------- #7 Run checks

@pytest.mark.parametrize("close_first", [True, False])
def test_run_checks_dialog_can_be_closed_early(make_app, tk_root, monkeypatch, close_first):
    finished = threading.Event()
    threads = []

    def slow_checks(*args, **kwargs):
        threads.append(threading.current_thread())
        time.sleep(0.15)
        finished.set()
        return [(True, "fake check", "")]

    monkeypatch.setattr(tester_app, "system_checks", slow_checks)
    errors = []
    monkeypatch.setattr(tk_root, "report_callback_exception", lambda *info: errors.append(info[1]))
    app = make_app()
    monkeypatch.setattr(app.root, "report_callback_exception", lambda *info: errors.append(info[1]), raising=False)
    app.run_checks()
    dialog = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    text = next(w for w in dialog.winfo_children() if isinstance(w, tk.Text))
    if close_first:
        dialog.destroy()
    deadline = time.time() + 0.8
    while time.time() < deadline:
        tk_root.update()
        time.sleep(0.01)
    assert finished.is_set() and not errors
    assert threads[0] is not threading.main_thread()
    if not close_first:
        assert "ALL CHECKS PASSED" in text.get("1.0", "end")


# ---------------------------------------------------------------- Space shortcut

@pytest.mark.parametrize("widget, active, toggles", [
    ("run_button", False, 0),       # Space presses Start / Stop; it must not also switch the mouse on
    ("active_button", False, 0),    # the button toggles by itself
    ("active_button", True, 0),     # toggling twice would switch the mouse back on
    ("run_button", True, 1),        # pausing is always allowed
    ("root", False, 1),
    ("root", True, 1),
])
def test_space_cannot_activate_by_accident(make_app, monkeypatch, widget, active, toggles):
    app = make_app()
    calls = []
    monkeypatch.setattr(app, "toggle_active", lambda: calls.append(1))
    monkeypatch.setattr(TesterEngine, "running", property(lambda self: True))
    snapshot = replace(app.engine.snapshot(), report=osi_capi.Report(active=active))
    monkeypatch.setattr(app.engine, "snapshot", lambda: snapshot)
    app._space_pressed(type("Event", (), {"widget": getattr(app, widget)})())
    assert len(calls) == toggles


# ============================================================ R1-R8

def write_user_settings(text, encoding="utf-8"):
    path = tester_engine.user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def calibrated(lib, **values):
    doc = IniDoc(lib.default_settings_ini())
    doc.set("mapping", "gaze_region", "0.71 0.33 0.29 0.08")
    for key, value in values.items():
        doc.set("mapping", key, value)
    return doc.text()


# ---------------------------------------------------------------- R1

@pytest.mark.parametrize("damage", ["bad line", "not utf-8"])
def test_save_never_overwrites_a_file_that_failed_to_load(make_app, lib, monkeypatch, damage):
    text = calibrated(lib)
    if damage == "bad line":
        path = write_user_settings(text + "this is not a setting\n")
    else:
        path = write_user_settings(text + "# caf\u00e9\n", "cp1252")
    before = path.read_bytes()
    app = make_app()
    app.sliders["wink"].set(0.4)
    errors = []
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(tester_app.messagebox, "showerror", lambda *a, **k: errors.append(a))
    app.save_user_settings()
    assert path.read_bytes() == before and errors


def test_file_broken_after_start_is_not_overwritten(make_app, lib, monkeypatch):
    path = write_user_settings(calibrated(lib))
    app = make_app()
    path.write_text(calibrated(lib) + "broken line\n", encoding="utf-8")
    before = path.read_bytes()
    app.sliders["wink"].set(0.4)
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(tester_app.messagebox, "showerror", lambda *a, **k: None)
    app.save_user_settings()
    assert path.read_bytes() == before


def test_byte_order_mark_keeps_calibration(make_app, lib, monkeypatch):
    path = write_user_settings(calibrated(lib), "utf-8-sig")
    app = make_app()
    app.sliders["wink"].set(0.4)
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    app.save_user_settings()
    saved = IniDoc(path.read_text(encoding="utf-8-sig"))
    assert saved.get("mapping", "gaze_region") == "0.71 0.33 0.29 0.08"
    assert saved.get("gestures", "wink_onset_seconds") == "0.4"


def test_save_keeps_a_backup_also_after_reset(make_app, lib, monkeypatch):
    path = write_user_settings(calibrated(lib) + "broken line\n")
    before = path.read_bytes()
    app = make_app()
    app.reset_tuning()                                              # the explicit way out of a broken file
    monkeypatch.setattr(tester_app.messagebox, "askyesno", lambda *a, **k: True)
    app.save_user_settings()
    assert path.with_name(path.name + ".bak").read_bytes() == before


# ---------------------------------------------------------------- whole worker, fake camera

class SlowCamera(FakeCamera):
    """Stands in for cv2.VideoCapture. `gate` blocks reads while it is clear."""
    gate = None
    calls = []

    def __init__(self, *args):
        pass

    def isOpened(self):
        return True

    def set(self, key, value):
        self.calls.append((key, value))
        return True

    def read(self):
        self.gate.wait(5)
        time.sleep(0.005)
        return True, np.zeros((48, 64, 3), np.uint8)


@pytest.fixture
def running_engine(lib, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    gate = threading.Event()
    gate.set()
    camera = type("Camera", (SlowCamera,), {"gate": gate, "calls": []})
    monkeypatch.setattr(cv2, "VideoCapture", camera)
    monkeypatch.setattr(tester_engine, "HandSource", FakeHand)
    monkeypatch.setattr(tester_engine, "GazeSource", FakeGaze)
    monkeypatch.setattr(TesterEngine, "_escape_pressed", staticmethod(lambda: False))
    monkeypatch.setattr(TesterEngine, "_physical_mouse_check", lambda *args: False)
    create = lib.controller
    made = []

    def record_only(settings, monitors, real_mouse):                # never the real mouse
        controller = RecordingController(create(settings, monitors, False))
        controller.mouse_requested = real_mouse
        made.append(controller)
        return controller

    monkeypatch.setattr(lib, "controller", record_only)
    engines = []

    def make():
        log = []
        engine = TesterEngine(lib, log.append)
        made.clear()                                                # forget validation probes
        engines.append(engine)
        return engine, log, gate, made, camera

    yield make
    gate.set()
    for engine in engines:
        engine.stop()


def wait_for(condition, seconds=4.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def is_running(engine):
    return engine.running and engine.snapshot().status == "Running"


# ---------------------------------------------------------------- R2

def test_activate_before_start_cannot_start_the_real_mouse_active(running_engine):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera", real_mouse=True)
    engine.set_active(True)
    engine.start()
    assert wait_for(lambda: is_running(engine))
    time.sleep(0.05)
    assert not engine.snapshot().report.active


def test_activate_while_loading_is_ignored(running_engine):
    engine, log, gate, made, camera = running_engine()
    gate.clear()
    engine.configure(mode="camera", real_mouse=True)
    engine.start()
    assert wait_for(lambda: made)                                   # controller made, no frame yet
    engine.set_active(True)
    gate.set()
    assert wait_for(lambda: is_running(engine))
    time.sleep(0.05)
    assert not engine.snapshot().report.active


def test_activate_does_not_carry_into_real_mouse(running_engine):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera")
    engine.start()
    assert wait_for(lambda: is_running(engine))
    engine.set_active(True)
    engine.configure(real_mouse=True)
    assert wait_for(lambda: any(getattr(c, "mouse_requested", False) for c in made))
    time.sleep(0.1)
    assert not engine.snapshot().report.active


def test_activate_after_stop_does_not_survive_restart(running_engine):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera", real_mouse=True)
    engine.start()
    assert wait_for(lambda: is_running(engine))
    engine.stop()
    engine.set_active(True)
    engine.start()
    assert wait_for(lambda: is_running(engine))
    time.sleep(0.05)
    assert not engine.snapshot().report.active


def test_activate_works_once_running(running_engine):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera", real_mouse=True)
    engine.start()
    assert wait_for(lambda: is_running(engine))
    time.sleep(0.03)
    engine.set_active(True)
    assert wait_for(lambda: engine.snapshot().report.active)


# ---------------------------------------------------------------- R6, R7

def test_screen_starts_from_the_saved_setting(make_app, lib):
    doc = IniDoc(calibrated(lib))
    doc.set("mapping", "screen_target", "all_monitors")
    write_user_settings(doc.text())
    app = make_app()
    assert app.monitors_var.get()
    assert app.engine.config.all_monitors


def test_parity_label_reports_a_different_screen(make_app):
    app = make_app()
    app.mode_var.set("hand")
    app.monitors_var.set(True)
    app._options_changed()
    assert "screen" in app.parity_label.cget("text").lower()


def test_parity_label_reports_unsaved_tuning(make_app):
    app = make_app()
    app.mode_var.set("hand")
    app.sliders["wink"].set(0.4)
    app._push_tuning()
    assert "not saved" in app.parity_label.cget("text").lower()


def test_camera_asks_for_mjpg_before_the_size(running_engine):
    import cv2
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera")
    engine.start()
    assert wait_for(lambda: is_running(engine))
    assert camera.calls[0] == (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


# ---------------------------------------------------------------- R8

def pinch_until_held(engine, made):
    engine.configure(mode="hand")
    engine.start()
    assert wait_for(lambda: engine.snapshot().counters.left_down == 1), "setup: no pinch"
    return made[-1]


def test_blocked_camera_read_still_releases(running_engine):
    engine, log, gate, made, camera = running_engine()
    controller = pinch_until_held(engine, made)
    gate.clear()
    assert wait_for(lambda: controller.count(osi_capi.EVENT_UP) == 1, seconds=1.5)


def test_stop_releases_at_once_while_a_read_is_blocked(running_engine):
    engine, log, gate, made, camera = running_engine()
    controller = pinch_until_held(engine, made)
    gate.clear()
    engine.stop(timeout=0.2)
    assert wait_for(lambda: controller.count(osi_capi.EVENT_UP) == 1, seconds=0.2)


def test_escape_pauses_the_real_mouse_while_a_read_is_blocked(running_engine, monkeypatch):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="hand", real_mouse=True)                  # record-only controllers, see the fixture
    engine.start()
    assert wait_for(lambda: is_running(engine))
    time.sleep(0.03)
    engine.set_active(True)
    assert wait_for(lambda: engine.snapshot().report.active)
    gate.clear()
    time.sleep(0.05)
    monkeypatch.setattr(TesterEngine, "_escape_pressed", staticmethod(lambda: True))
    assert wait_for(lambda: any("Esc" in line for line in log), seconds=0.4)


# ---------------------------------------------------------------- smaller bugs found on the way

def test_paused_output_does_not_yield_when_the_mouse_moves(worker, monkeypatch):
    engine, state, log, cv2 = worker("hand")
    state.last_report = osi_capi.Report(active=False)
    state.expected_cursor = (10, 10)
    monkeypatch.setattr(engine.lib, "cursor_position", lambda: (500, 500))
    assert not engine._physical_mouse_check(state, time.perf_counter())


def test_recording_failure_does_not_stop_the_test(worker, tmp_path):
    engine, state, log, cv2 = worker("camera")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    engine._handle_recording(state, blocker / "cannot.csv", state.config)
    assert state.recorder is None and any("record" in line.lower() for line in log)


def test_stop_clears_the_recording_state(running_engine, tmp_path):
    engine, log, gate, made, camera = running_engine()
    engine.configure(mode="camera")
    engine.start()
    assert wait_for(lambda: is_running(engine))
    engine.start_recording(tmp_path / "run.csv")
    assert wait_for(lambda: engine.snapshot().recording)
    engine.stop()
    assert not engine.snapshot().recording


def test_middle_click_counts_in_the_accuracy_test(make_app, monkeypatch):
    app = make_app()
    app.test.start(dwell=False)
    seen = []
    monkeypatch.setattr(app.test, "update", lambda cursor, clicked: seen.append(clicked))
    snap = replace(app.engine.snapshot(), running=True, report=osi_capi.Report(cursor=(100, 100)),
                   counters=tester_engine.Counters(middle_click=1))
    app._update_screen(snap)
    assert seen == [True]


def test_gaze_source_matches_engine_vertical_sign_and_directions(monkeypatch):
    tester_engine.prepare_ai_imports()
    import gaze_intent as gi

    class FakeTracker:
        def __init__(self):
            self.outputs = iter([
                (-12.0, 10.0, 0.0, 0.0, 0, [object()]),   # UP-LEFT: negative yaw, positive pitch
                (12.0, -10.0, 0.0, 0.0, 0, [object()]),   # DOWN-RIGHT: positive yaw, negative pitch
            ])

        def process_frame(self, rgb, ts):
            return next(self.outputs)

        def shutdown(self):
            pass

    monkeypatch.setattr(gi, "GazeIntent", FakeTracker)
    src = tester_engine.GazeSource()
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        up_left = src.process(blank)
        down_right = src.process(blank)
    finally:
        src.close()

    assert up_left.point is not None and down_right.point is not None
    assert up_left.point[0] == pytest.approx(0.3, abs=1e-9)
    assert up_left.point[1] == pytest.approx(0.5 - 10.0 / 60.0, abs=1e-9)
    assert down_right.point[0] == pytest.approx(0.7, abs=1e-9)
    assert down_right.point[1] == pytest.approx(0.5 + 10.0 / 60.0, abs=1e-9)
    assert up_left.point[0] < down_right.point[0] and up_left.point[1] < down_right.point[1]


def test_uncalibrated_default_gaze_region_keeps_candidate_neutral_inside_screen(lib):
    defaults = IniDoc(lib.default_settings_ini())
    defaults.set("mapping", "cursor_source", "gaze_only")
    c = lib.controller(defaults.text(), all_monitors=False, real_mouse=False)
    try:
        target, _ = c.geometry()
        # Candidate HGB abs_pitch + ET rel_yaw golden neutral: abs_yaw=+0.837 deg -> 0.514, abs_pitch=-16.277 deg -> 0.771
        r = c.update(0.0, None, osi_capi.TrackerInput((0.514, 0.771), 0))
        assert r.cursor is not None
        cx, cy = r.cursor
        assert target[0] + 50 < cx < target[0] + target[2] - 50
        assert target[1] + 50 < cy < target[1] + target[3] - 50
    finally:
        c.close()


def test_clicks_disabled_keeps_gaze_blink_lock_and_blocks_clicks(lib, worker, monkeypatch):
    # Direct C API check
    doc = IniDoc(lib.default_settings_ini())
    doc.set("mapping", "cursor_source", "gaze_only")
    c = lib.controller(doc.text(), all_monitors=False, real_mouse=False)
    try:
        c.set_active(True)
        c.set_clicks_enabled(False)
        t = 0.0
        for _ in range(6):
            r = c.update(t, None, osi_capi.TrackerInput((0.50, 0.77), 0))
            t += 1.0 / 32.0
        base_cursor = r.cursor
        for _ in range(12):
            rw = c.update(t, None, osi_capi.TrackerInput((0.95, 0.05), 1))
            assert rw.cursor == base_cursor
            assert not any(
                e.kind in (osi_capi.EVENT_CLICK, osi_capi.EVENT_DOWN) for e in c.drain_events()
            )
            t += 1.0 / 32.0
    finally:
        c.close()

    # Worker _step check: clicks=False must still forward gaze.state (not overwrite with 0)
    class _WinkGaze(FakeGaze):
        def process(self, rgb):
            return tester_engine.GazeObs(point=(0.95, 0.05), state=1, ms=1.0, face=True)

    engine, state, _, cv2 = worker("gaze")
    state.gaze = _WinkGaze()
    seen_states = []
    orig_update = state.controller.update

    def _spy_update(t_val, hand_in, gaze_in):
        if gaze_in is not None:
            seen_states.append(gaze_in.state)
        return orig_update(t_val, hand_in, gaze_in)

    monkeypatch.setattr(state.controller, "update", _spy_update)
    cfg = replace(state.config, clicks=False)
    engine._apply_config(state, cfg, engine.settings_text(), cv2)
    engine._step(state, cfg, cv2)
    assert seen_states == [1]
    assert state.controller.count(osi_capi.EVENT_CLICK) == 0



# ---- [Engineer 5] P1 #4: Non-finite validation & recovery across Tester & osi_capi ----

@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_smoother_rejects_nonfinite_and_recovers(dsp_filter, bad):
    for filt in (dsp_filter, None):
        gate = Smoother(filt)
        pt0, trk0 = gate.apply((0.3, 0.4), TRACKED, 1.0)
        assert pt0 is not None and trk0 == TRACKED
        assert gate.apply((bad, 0.4), TRACKED, 1.1) == (None, LOST)
        assert gate.apply((0.3, bad), TRACKED, 1.2) == (None, LOST)
        # Held must not replay after invalid reset
        assert gate.apply((0.3, 0.4), HELD, 1.3) == (None, LOST)
        # Clean frame recovers immediately
        pt_rec, trk_rec = gate.apply((0.6, 0.7), TRACKED, 1.4)
        assert pt_rec is not None and trk_rec == TRACKED
        assert math.isfinite(pt_rec[0]) and math.isfinite(pt_rec[1])


def test_hand_source_rejects_nonfinite_index_and_hand_and_recovers():
    # Bypass __init__ to test HandSource.process normalization without loading MediaPipe weights
    blank = np.zeros((16, 16, 3), dtype=np.uint8)

    # Index mode
    src_idx = tester_engine.HandSource.__new__(tester_engine.HandSource)
    src_idx.kind = "index"

    class _FakeIdx:
        def __init__(self, seq):
            self._seq = iter(seq)

        def track(self, _):
            return next(self._seq)

    src_idx.tracker = _FakeIdx([
        type("R", (), {"valid": True, "x": float("inf"), "y": 0.5, "camera_x": 0.5, "camera_y": 0.5, "status": "TRACKING"})(),
        type("R", (), {"valid": True, "x": 0.4, "y": 0.6, "camera_x": 0.4, "camera_y": 0.6, "status": "TRACKING"})(),
    ])
    obs_bad = src_idx.process(blank)
    assert obs_bad.point is None and obs_bad.tracking == LOST
    obs_ok = src_idx.process(blank)
    assert obs_ok.point == (0.4, 0.6) and obs_ok.tracking == TRACKED

    # Hand gesture mode
    src_hand = tester_engine.HandSource.__new__(tester_engine.HandSource)
    src_hand.kind = "hand"

    class _FakeHand:
        def __init__(self):
            self.status = "DETECTED"
            self._last_bbox = None
            self.last_confidence = 0.9
            self._seq = iter([
                (float("nan"), 0.5, 1),
                (float("inf"), 0.5, 1),
                (0.25, 0.75, 1),
            ])

        def process_frame(self, _):
            return next(self._seq)

    src_hand.tracker = _FakeHand()
    for _ in range(2):
        bad_obs = src_hand.process(blank)
        assert bad_obs.point is None and bad_obs.tracking == LOST
    ok_obs = src_hand.process(blank)
    assert ok_obs.point == (0.25, 0.75) and ok_obs.tracking == TRACKED and ok_obs.state == 1


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_osi_capi_rejects_nonfinite_tracker_points_and_recovers(lib, bad):
    ini = lib.default_settings_ini()
    c = lib.controller(ini, all_monitors=False, real_mouse=False)
    try:
        c.set_active(True)
        # Before any valid sample, non-finite input must not produce a cursor
        r0 = c.update(0.05, osi_capi.TrackerInput((bad, 0.5), 0, TRACKED), None)
        assert r0.cursor is None and not r0.moved

        # Establish a valid cursor, then inject non-finite X and Y
        good = c.update(0.10, osi_capi.TrackerInput((0.5, 0.5), 0, TRACKED), None)
        assert good.cursor is not None and good.moved
        r1 = c.update(0.20, osi_capi.TrackerInput((bad, 0.5), 0, TRACKED), None)
        assert not r1.moved and r1.cursor == good.cursor
        r2 = c.update(0.30, osi_capi.TrackerInput((0.5, bad), 0, TRACKED), None)
        assert not r2.moved and r2.cursor == good.cursor

        # Next clean sample moves cleanly to the new position
        rec = c.update(0.40, osi_capi.TrackerInput((0.6, 0.4), 0, TRACKED), None)
        assert rec.moved and rec.cursor is not None and rec.cursor != good.cursor
    finally:
        c.close()


def test_old_placeholder_gaze_region_counts_as_uncalibrated(lib):
    old = IniDoc(lib.default_settings_ini())
    old.lines = [line for line in old.lines if not line.startswith("settings_version")]
    old.set("mapping", "gaze_region", "0.3 0.3 0.7 0.7")
    assert gaze_is_uncalibrated(old.text(), lib.default_settings_ini())
