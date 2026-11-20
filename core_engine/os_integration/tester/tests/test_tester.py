"""Tests for the HandEye tester logic. No camera, no real mouse.

Run from the repository root:
    .venv\\Scripts\\python.exe -m pytest core_engine\\os_integration\\tester\\tests -q
Needs a built os_integration_capi.dll (out/build/...); those tests skip without it.
"""

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osi_capi  # noqa: E402
from tester_app import AccuracyTest, _region  # noqa: E402
from tester_engine import (MODES, CsvRecorder, GazeObs, HandObs, IniDoc, Smoother,  # noqa: E402
                           format_value)


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
    def __init__(self):
        self.calls = []

    def apply(self, point, t):
        self.calls.append(point)
        return None if point is None else (point[0] + 1, point[1] + 1)

    def close(self):
        pass


def test_smoother_only_filters_fresh_points():
    f = FakeFilter()
    s = Smoother(f)
    assert s.apply((0.1, 0.2), osi_capi.TRACKED, 0.0) == (1.1, 1.2)
    assert s.apply((9, 9), osi_capi.HELD, 0.1) == (1.1, 1.2)          # held: cached point, no filter call
    assert s.apply(None, osi_capi.LOST, 0.2) is None                   # lost: reset
    assert f.calls == [(0.1, 0.2), None]
    assert Smoother(None).apply((0.5, 0.5), osi_capi.TRACKED, 0) == (0.5, 0.5)


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
