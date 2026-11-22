"""Offline unit tests for analyze_gaze_fixation_diag.py (no webcam, no OS mouse)."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import pytest
import tl2cgen

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER_PATH = REPO_ROOT / "ai_pipeline" / "gaze_intent" / "tests" / "analyze_gaze_fixation_diag.py"
MODELS_DIR = REPO_ROOT / "ai_pipeline" / "gaze_intent" / "models"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_gaze_fixation_diag", str(ANALYZER_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ANALYZER = _load_analyzer()


def _write_synthetic_run(
    run_dir: Path,
    frame_specs: Sequence[Dict[str, Any]],
    *,
    y_sign_mult: float = -1.0,
    diag_csv: Optional[Path] = None,
) -> Path:
    """Build a valid E4REC v2 trace.txt + COMPLETE + 32-column gaze_diag.csv using live DLLs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    yaw_pred = tl2cgen.Predictor(str(MODELS_DIR / "gaze_yaw.dll"))
    pitch_pred = tl2cgen.Predictor(str(MODELS_DIR / "gaze_pitch.dll"))

    trace_lines = [
        "E4REC BEGIN version=2 channel=gaze capacity=4096 min_cutoff=0.8 "
        "beta=6 derivative_cutoff=1.5 min_dt=1e-6 reset_gap=0.25"
    ]
    diag_csv = diag_csv or (run_dir / "gaze_diag.csv")
    diag_csv.parent.mkdir(parents=True, exist_ok=True)
    has_initialized_dsp = False

    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ANALYZER.EXPECTED_DIAG_COLUMNS)

        last_accepted_t: float | None = None
        cached_pub = False
        prev_ms = -1

        for idx, spec in enumerate(frame_specs):
            t_str = f"{float(spec['t_s']):.9f}"
            t_val = float(t_str)
            cap_ms = int(t_val * 1000.0)
            ms = max(cap_ms, prev_ms + 1)
            prev_ms = ms
            state = int(spec.get("state", 0))
            detected = int(spec.get("detected", 1))
            sx = float(spec.get("sx", 0.5))
            sy = float(spec.get("sy", 0.5))
            hp = float(spec.get("head_pitch_deg", 0.0))
            hy = float(spec.get("head_yaw_deg", 0.0))
            hr = float(spec.get("head_roll_deg", 0.0))

            if detected == 1 and state != -1:
                feat = np.array([[sx, sy, hp, hy, hr]], dtype=np.float32)
                dm = tl2cgen.DMatrix(feat)
                ry = float(np.asarray(yaw_pred.predict(dm)).reshape(-1)[0])
                rp = float(np.asarray(pitch_pred.predict(dm)).reshape(-1)[0])
                ay = ry + hy
                ap = rp + hp
                nx = float(np.float32(spec.get("override_nx", np.clip(0.5 + ay / 60.0, 0.0, 1.0))))
                ny = float(np.float32(spec.get("override_ny", np.clip(0.5 + y_sign_mult * (ap / 60.0), 0.0, 1.0))))
                ok = 1
                n_lm = 484
                x_val, y_val = repr(nx), repr(ny)
                adapter = 1
                called = 1
                input_t = t_str
                ix, iy = repr(nx), repr(ny)
                if last_accepted_t is None:
                    status = 0
                elif (t_val - last_accepted_t) > 0.25:
                    status = 2
                else:
                    status = 1
                last_accepted_t = t_val
                cached_pub = True
                resets = 0
                has, dx, dy = 1, repr(nx), repr(ny)
                outcome = 0
                pub, px, py = 1, repr(nx), repr(ny)
                out_trk = 0
            else:
                ry = rp = ay = ap = 0.0
                nx = ny = -1.0
                ok = 1 if state == -1 and detected == 1 else 0
                n_lm = 484 if state == -1 and detected == 1 else 0
                x_val = y_val = -1
                adapter = called = 0
                input_t = 0
                ix = iy = 0
                status = -1
                resets = 1 if cached_pub else 0
                cached_pub = False
                last_accepted_t = None
                has = dx = dy = 0
                outcome = -1
                pub = px = py = 0
                out_trk = 2

            fields = dict(
                frame=idx, ms=ms, cap=cap_ms, t=t_str, n=n_lm, ok=ok,
                x=x_val, y=y_val, state=state, trk=-1, missed=0,
                adapter=adapter, called=called, input_t=input_t, ix=ix, iy=iy,
                status=status, has=has, dx=dx, dy=dy, outcome=outcome,
                resets=resets, pub=pub, px=px, py=py, out_trk=out_trk, generation=1,
            )
            trace_lines.append("E4REC F " + " ".join(f"{k}={v}" for k, v in fields.items()))

            writer.writerow([
                idx, ms, t_str, detected, state,
                sx, sx, sx, sy, sy, sy,
                hp, hy, hr,
                ry, rp, ap, ay,
                nx, ny,
                0.3, 0.4, 0.4, 0.4, 0.35, 0.4,
                0.6, 0.4, 0.7, 0.4, 0.65, 0.4,
            ])

    n_total = len(frame_specs)
    trace_lines.append(
        f"E4REC END submitted={n_total} attempted={n_total} stored={n_total} overflow=0 config_changes=0 complete=1"
    )
    (run_dir / "trace.txt").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return diag_csv


def _make_target_segment(
    start_t: float, count: int, dt: float, *, sx: float, sy: float, hy: float = 0.0, hp: float = 0.0
) -> List[Dict[str, Any]]:
    return [
        {"t_s": start_t + i * dt, "sx": sx, "sy": sy, "head_yaw_deg": hy, "head_pitch_deg": hp}
        for i in range(count)
    ]


def test_fixed_vertical_y_sign_and_direction_checks(tmp_path: Path):
    run_dir = tmp_path / "run_fixed"
    frames: List[Dict[str, Any]] = []
    # 30 frames per direction: LEFT, RIGHT, UP, DOWN
    frames.extend(_make_target_segment(1.0, 30, 0.033, sx=0.38, sy=0.00, hy=-10.0, hp=0.0))
    frames.extend(_make_target_segment(3.0, 30, 0.033, sx=0.62, sy=0.00, hy=+10.0, hp=0.0))
    frames.extend(_make_target_segment(5.0, 30, 0.033, sx=0.50, sy=-0.25, hy=0.0, hp=0.0))
    frames.extend(_make_target_segment(7.0, 30, 0.033, sx=0.50, sy=+0.05, hy=0.0, hp=0.0))

    diag_csv = _write_synthetic_run(run_dir, frames, y_sign_mult=-1.0)
    out_json = tmp_path / "report.json"
    rc = ANALYZER.main([
        "--run-dir", str(run_dir),
        "--diag-csv", str(diag_csv),
        "--windows", "LEFT:1.0-2.5,RIGHT:3.0-4.5,UP:5.0-6.5,DOWN:7.0-8.5",
        "--candidate-models-dir", str(MODELS_DIR),
        "--out-json", str(out_json),
        "--out-md", str(tmp_path / "report.md"),
    ])
    assert rc == 0
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["integrity"]["alignment_ok"] is True
    assert rep["integrity"]["logged_y_sign"] == "fixed"
    for w in rep["windows"]:
        assert w["deployed_replay_verification"]["max_abs_err_norm_x"] < 1e-6
        assert w["deployed_replay_verification"]["max_abs_err_norm_y"] < 1e-6
        assert w["raw_x"]["pass_all"] is True
        assert w["raw_y"]["pass_all"] is True

    dirs = {d["comparison"]: d for d in rep["directions"]}
    assert dirs["HORIZONTAL_LEFT_TO_RIGHT"]["delta_raw_x_mean"] > 0.30
    assert dirs["VERTICAL_UP_TO_DOWN"]["delta_raw_y_mean"] > 0.10
    assert dirs["VERTICAL_UP_TO_DOWN"]["candidate_delta_raw_y_mean"] > 0.10


def test_legacy_vertical_y_sign_auto_detected_and_candidate_uses_fixed_y(tmp_path: Path):
    run_dir = tmp_path / "run_legacy"
    frames: List[Dict[str, Any]] = []
    frames.extend(_make_target_segment(1.0, 30, 0.033, sx=0.50, sy=-0.25, hy=0.0, hp=0.0))
    frames.extend(_make_target_segment(3.0, 30, 0.033, sx=0.50, sy=+0.05, hy=0.0, hp=0.0))

    diag_csv = _write_synthetic_run(run_dir, frames, y_sign_mult=+1.0)
    out_json = tmp_path / "legacy_report.json"
    rc = ANALYZER.main([
        "--run-dir", str(run_dir),
        "--diag-csv", str(diag_csv),
        "--windows", "UP:1.0-2.5,DOWN:3.0-4.5",
        "--candidate-models-dir", str(MODELS_DIR),
        "--out-json", str(out_json),
        "--out-md", str(tmp_path / "legacy_report.md"),
    ])
    assert rc == 0
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["integrity"]["logged_y_sign"] == "legacy"
    for w in rep["windows"]:
        assert w["deployed_replay_verification"]["max_abs_err_norm_y"] < 1e-6
    dirs = {d["comparison"]: d for d in rep["directions"]}
    # Legacy logged trace had inverted Y (<0), while fixed Y and candidate shadow have positive UP->DOWN (>0)
    assert dirs["VERTICAL_UP_TO_DOWN"]["delta_raw_y_mean"] < -0.10
    assert dirs["VERTICAL_UP_TO_DOWN"]["delta_raw_y_fixed_sign_mean"] > 0.10
    assert dirs["VERTICAL_UP_TO_DOWN"]["candidate_delta_raw_y_mean"] > 0.10


def test_wash_in_and_blink_plus_minus_2_frame_exclusion(tmp_path: Path):
    run_dir = tmp_path / "run_wash_blink"
    frames: List[Dict[str, Any]] = []
    # Window [1.0, 3.0): 60 frames at 30 Hz
    # First 0.5s (15 frames): transient jump at t=1.0..1.4s
    for i in range(60):
        t_s = 1.0 + i * (1.0 / 30.0)
        if i < 12:
            # Wash-in transient: huge jump that must be trimmed by wash_in=0.5s
            frames.append({"t_s": t_s, "sx": 0.35 if i % 2 == 0 else 0.65, "sy": 0.50, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})
        elif i in (33, 34, 36, 37):
            # +/-2 neighbor frames around blink at i=35 with corrupted edge coordinates
            frames.append({"t_s": t_s, "sx": 0.35, "sy": 0.20, "head_yaw_deg": -8.0, "head_pitch_deg": 8.0})
        elif i == 35:
            # Blink frame
            frames.append({"t_s": t_s, "state": 3, "sx": 0.50, "sy": 0.50, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})
        else:
            frames.append({"t_s": t_s, "sx": 0.50, "sy": 0.50, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})

    diag_csv = _write_synthetic_run(run_dir, frames, y_sign_mult=-1.0)
    sched_path = run_dir / "cued_schedule.json"
    sched_path.write_text(json.dumps({
        "schema_version": 1,
        "calib_ready_s": 0.5,
        "targets": [{"name": "CENTER", "target_start_s": 1.0, "target_end_s": 3.0}],
    }), encoding="utf-8")

    out_json = tmp_path / "wash_blink_report.json"
    rc = ANALYZER.main([
        "--run-dir", str(run_dir),
        "--diag-csv", str(diag_csv),
        "--cued-schedule", str(sched_path),
        "--out-json", str(out_json),
        "--out-md", str(tmp_path / "wash_blink_report.md"),
    ])
    assert rc == 0
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["analysis_params"]["wash_in_s"] == 0.5
    assert rep["analysis_params"]["blink_pad_frames"] == 2
    w = rep["windows"][0]
    assert w["exclusions"]["wash_in"] == 15
    assert w["exclusions"]["nonneutral"] == 1
    assert w["exclusions"]["blink_pad"] == 4
    # Remaining settled frames: 60 - 15 - 5 = 40 frames, all completely clean and steady
    assert w["eligible_frames"] == 40
    assert w["raw_x"]["pass_all"] is True
    assert w["raw_y"]["pass_all"] is True
    assert w["raw_x"]["range"] < 1e-6


def test_cued_readiness_gate_target_plan_and_capture_command():
    gate = ANALYZER.GazeReadinessGate()
    # Calibration lines (s=-1) must not trigger readiness
    assert not gate.feed("cap 30 | ai 30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.2 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(-1.00,-1.00 s=-1) hand(0.00,0.00 s=0) | os frames 10", 1.0)
    # First s=0 line records previous_valid=True, not yet ready
    assert not gate.feed("cap 30 | ai 30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.2 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.51,0.72 s=0) hand(0.00,0.00 s=0) | os frames 40", 2.0)
    # Second consecutive s=0 line with advancing os frames triggers readiness
    assert gate.feed("cap 30 | ai 30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.2 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.51,0.72 s=0) hand(0.00,0.00 s=0) | os frames 70", 3.0)
    assert gate.calib_ready_time == 3.0

    plan = ANALYZER.build_cued_target_plan()
    assert [p["name"] for p in plan] == ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    for p in plan:
        assert p["offset_from_ready_end_s"] - p["offset_from_ready_start_s"] == pytest.approx(10.0)

    cmd = ANALYZER.build_gaze_capture_command(Path("C:/exe/core_engine.exe"), Path("C:/run"), Path("C:/run/gaze_diag.csv"), 0, 80)
    assert "--os" in cmd and cmd[cmd.index("--os") + 1] == "log"
    assert "--no-hand" in cmd
    assert "--filter" in cmd and cmd[cmd.index("--filter") + 1] == "dsp"
    assert "--gaze-filter" in cmd and cmd[cmd.index("--gaze-filter") + 1] == "0.8,6,1.5"
    assert "win32" not in cmd


def test_mocked_cued_capture_writes_schedule_and_analyzes(tmp_path: Path):
    run_dir = tmp_path / "mock_cued_run"
    fake_exe = tmp_path / "core_engine.exe"
    fake_exe.write_text("stub", encoding="utf-8")

    # Build a synthetic capture covering t=0..15s
    frames: List[Dict[str, Any]] = []
    # t=0..1.0s: calibration (state=-1)
    for i in range(30):
        frames.append({"t_s": i * 0.033, "state": -1, "sx": 0.5, "sy": 0.5})
    # calib_ready_s=0.0, initial_settle_s=1.0, hold_s=2.0, gap_s=0.5 ->
    # CENTER(1.0-3.0), LEFT(3.5-5.5), RIGHT(6.0-8.0), UP(8.5-10.5), DOWN(11.0-13.0)
    for i in range(30, 450):
        t_s = i * 0.033333
        if 1.0 <= t_s < 3.0:
            frames.append({"t_s": t_s, "sx": 0.50, "sy": 0.00, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})
        elif 3.5 <= t_s < 5.5:
            frames.append({"t_s": t_s, "sx": 0.38, "sy": 0.00, "head_yaw_deg": -10.0, "head_pitch_deg": 0.0})
        elif 6.0 <= t_s < 8.0:
            frames.append({"t_s": t_s, "sx": 0.62, "sy": 0.00, "head_yaw_deg": 10.0, "head_pitch_deg": 0.0})
        elif 8.5 <= t_s < 10.5:
            frames.append({"t_s": t_s, "sx": 0.50, "sy": -0.25, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})
        elif 11.0 <= t_s < 13.0:
            frames.append({"t_s": t_s, "sx": 0.50, "sy": +0.05, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})
        else:
            frames.append({"t_s": t_s, "sx": 0.50, "sy": 0.00, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0})

    class FakeClock:
        def __init__(self) -> None:
            self.now = 100.0
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            return self.now

        def sleep(self, dt: float) -> None:
            self.now += dt

    fake_clock = FakeClock()

    class FakeProc:
        def __init__(self, cmd: List[str], **_kwargs: Any) -> None:
            self.cmd = cmd
            self.returncode = 0
            dsp_dir = Path(cmd[cmd.index("--dsp-record-dir") + 1])
            diag_csv_path = Path(cmd[cmd.index("--gaze-diag-log") + 1])
            # core_engine OutputSession strictly requires --dsp-record-dir NOT to pre-exist, while parent exists.
            assert not dsp_dir.exists(), f"--dsp-record-dir must not pre-exist: {dsp_dir}"
            assert dsp_dir.parent.is_dir()
            _write_synthetic_run(dsp_dir, frames, y_sign_mult=-1.0, diag_csv=diag_csv_path)
            self.stdout = io.StringIO(
                "OS: log (stdout). Starts PAUSED\n"
                "cap 30 | ai 30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.0 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.51,0.72 s=0) hand(0.00,0.00 s=0) | os frames 35\n"
                "cap 30 | ai 30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.0 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.51,0.72 s=0) hand(0.00,0.00 s=0) | os frames 65\n"
            )

        def poll(self):
            # Stay alive until simulated clock reaches 116.5s (16.5s elapsed)
            return None if fake_clock.now < 116.5 else 0

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    # Even if run_dir was already created as an empty folder beforehand, --cued-capture must still succeed.
    run_dir.mkdir(parents=True, exist_ok=True)

    with (
        mock.patch.object(ANALYZER.subprocess, "Popen", side_effect=FakeProc),
        mock.patch.object(ANALYZER.time, "monotonic", side_effect=fake_clock.monotonic),
        mock.patch.object(ANALYZER.time, "sleep", side_effect=fake_clock.sleep),
    ):
        out_json = tmp_path / "cued_report.json"
        rc = ANALYZER.main([
            "--cued-capture",
            "--exe", str(fake_exe),
            "--run-dir", str(run_dir),
            "--hold-s", "2.0",
            "--gap-s", "0.5",
            "--no-audio",
            "--yes",
            "--out-json", str(out_json),
            "--out-md", str(tmp_path / "cued_report.md"),
        ])

    assert rc == 0
    assert (run_dir / "dsp_record" / "trace.txt").is_file()
    assert (run_dir / "dsp_record" / "COMPLETE").is_file()
    assert (run_dir / "gaze_diag.csv").is_file()
    sched = json.loads((run_dir / "cued_schedule.json").read_text(encoding="utf-8"))
    assert "calib_ready_s" in sched
    assert [t["name"] for t in sched["targets"]] == ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    for t in sched["targets"]:
        assert t["target_end_s"] > t["target_start_s"]

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert Path(rep["integrity"]["dsp_record_dir"]) == (run_dir / "dsp_record").resolve()
    assert Path(rep["integrity"]["artifact_dir"]) == run_dir.resolve()
    assert rep["analysis_params"]["wash_in_s"] == 0.5
    assert rep["analysis_params"]["blink_pad_frames"] == 2
    assert len(rep["windows"]) == 5
    for w in rep["windows"]:
        assert w["eligible_frames"] >= 25
        assert w["raw_x"]["pass_all"] is True
        assert w["raw_y"]["pass_all"] is True
    dirs = {d["comparison"]: d for d in rep["directions"]}
    assert dirs["HORIZONTAL_LEFT_TO_RIGHT"]["delta_raw_x_mean"] > 0.30
    assert dirs["VERTICAL_UP_TO_DOWN"]["delta_raw_y_mean"] > 0.10

    # Offline re-analysis with ONLY --run-dir + wash-in + blink-pad auto-discovers dsp_record/, gaze_diag.csv, cued_schedule.json
    rc_re = ANALYZER.main([
        "--run-dir", str(run_dir),
        "--wash-in-s", "0.5",
        "--blink-pad-frames", "2",
    ])
    assert rc_re == 0
    re_rep = json.loads((run_dir / "gaze_attribution_report.json").read_text(encoding="utf-8"))
    assert len(re_rep["windows"]) == 5
    assert (run_dir / "gaze_attribution_report.md").is_file()


def test_cued_capture_refuses_existing_dsp_record_dir(tmp_path: Path):
    run_dir = tmp_path / "existing_run"
    (run_dir / "dsp_record").mkdir(parents=True, exist_ok=True)
    fake_exe = tmp_path / "core_engine.exe"
    fake_exe.write_text("stub", encoding="utf-8")

    with mock.patch.object(ANALYZER.subprocess, "Popen") as popen_mock:
        with pytest.raises(RuntimeError, match="Recorder output directory already exists"):
            ANALYZER.run_cued_capture(
                exe=fake_exe,
                run_dir=run_dir,
                diag_csv=run_dir / "gaze_diag.csv",
                camera=0,
                targets=["CENTER"],
                hold_s=1.0,
                gap_s=0.0,
                audio=False,
                yes=True,
            )
    popen_mock.assert_not_called()


def test_cued_capture_surfaces_fast_exit_stderr_tail_never_empty(tmp_path: Path):
    run_dir = tmp_path / "fast_fail_run"
    fake_exe = tmp_path / "fake_core_engine.exe"
    fake_exe.write_text("stub", encoding="utf-8")

    helper = tmp_path / "fast_fail_engine.py"
    helper.write_text(
        "import sys\n"
        "sys.stderr.write('[recording] startup failed: simulated fast exit error\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(4)\n",
        encoding="utf-8",
    )
    real_popen = ANALYZER.subprocess.Popen

    def _spawn_helper(cmd: List[str], **kwargs: Any):
        return real_popen([sys.executable, str(helper), *cmd[1:]], **kwargs)

    with mock.patch.object(ANALYZER.subprocess, "Popen", side_effect=_spawn_helper):
        with pytest.raises(RuntimeError) as exc_info:
            ANALYZER.run_cued_capture(
                exe=fake_exe,
                run_dir=run_dir,
                diag_csv=run_dir / "gaze_diag.csv",
                camera=0,
                targets=["CENTER"],
                hold_s=1.0,
                gap_s=0.0,
                audio=False,
                yes=True,
            )
    msg = str(exc_info.value)
    assert "exited with code 4" in msg
    assert "[recording] startup failed: simulated fast exit error" in msg


def test_cued_capture_real_subprocess_pipe_buffer_sync_and_ctrl_break(tmp_path: Path):
    run_dir = tmp_path / "pipe_sync_run"
    fake_exe = tmp_path / "fake_core_engine.exe"
    fake_exe.write_text("stub", encoding="utf-8")

    # Pre-build a synthetic recording payload template we can save from the helper subprocess.
    seed_dir = tmp_path / "_seed"
    frames: List[Dict[str, Any]] = []
    # t=0..0.5s: calibration (state=-1); t=0.5..5.5s: active steady CENTER frames (state=0) at 30 Hz
    for i in range(15):
        frames.append({"t_s": i * 0.033333, "state": -1, "sx": 0.5, "sy": 0.5})
    for i in range(15, 180):
        frames.append({"t_s": i * 0.033333, "state": 0, "sx": 0.5, "sy": 0.0})
    _write_synthetic_run(seed_dir, frames, y_sign_mult=-1.0)
    trace_text = (seed_dir / "trace.txt").read_text(encoding="utf-8")
    diag_text = (seed_dir / "gaze_diag.csv").read_text(encoding="utf-8")

    helper = tmp_path / "delayed_pipe_engine.py"
    helper.write_text(
        f"""
import os, signal, sys, time
from pathlib import Path

args = sys.argv[1:]
dsp_dir = Path(args[args.index('--dsp-record-dir') + 1])
diag_csv = Path(args[args.index('--gaze-diag-log') + 1])
if dsp_dir.exists():
    sys.stderr.write(f"[recording] startup failed: Recorder output directory already exists: {{dsp_dir}}\\n")
    sys.exit(4)
dsp_dir.mkdir(parents=False, exist_ok=False)

stop = False
def _on_sig(signum, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, _on_sig)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, _on_sig)

t0 = time.monotonic()
# Simulate Windows pipe buffering: print OS line WITHOUT flush, then wait 1.0s before flushing with first status line.
sys.stdout.write("OS: OS integration (dry run, mouse untouched). Starts PAUSED\\n")
time.sleep(1.0)
sys.stdout.write("cap  31 | ai  31 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.0 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.50,0.72 s=0) hand(0.00,0.00 s=0) | os frames 31\\n")
sys.stdout.flush()

for os_f in (61, 91, 121, 151, 180):
    target_dt = (os_f - 1) * 0.033333
    while not stop and (time.monotonic() - t0) < target_dt:
        time.sleep(0.02)
    if stop:
        break
    sys.stdout.write(f"cap  30 | ai  30 fps | gaze 1.0 ms hand 0.0 ms | latency avg 1.0 max 2.0 ms | evict 0 stale 0 pyerr 0 | slots 8/8 | gaze(0.50,0.72 s=0) hand(0.00,0.00 s=0) | os frames {{os_f}}\\n")
    sys.stdout.flush()

# Wait briefly for CTRL_BREAK_EVENT / SIGINT from run_cued_capture if targets just finished
deadline = time.monotonic() + 3.0
while not stop and time.monotonic() < deadline:
    time.sleep(0.02)

(dsp_dir / "trace.txt").write_text({trace_text!r}, encoding="utf-8")
(dsp_dir / "COMPLETE").write_text("ok\\n", encoding="utf-8")
diag_csv.write_text({diag_text!r}, encoding="utf-8")
sys.exit(0)
""",
        encoding="utf-8",
    )
    real_popen = ANALYZER.subprocess.Popen

    def _spawn_helper(cmd: List[str], **kwargs: Any):
        return real_popen([sys.executable, str(helper), *cmd[1:]], **kwargs)

    with mock.patch.object(ANALYZER.subprocess, "Popen", side_effect=_spawn_helper):
        rc = ANALYZER.main([
            "--cued-capture",
            "--exe", str(fake_exe),
            "--run-dir", str(run_dir),
            "--targets", "CENTER",
            "--hold-s", "1.5",
            "--gap-s", "0.0",
            "--no-audio",
            "--yes",
        ])

    assert rc == 0
    sched = json.loads((run_dir / "cued_schedule.json").read_text(encoding="utf-8"))
    assert sched["time_sync_method"] == "trace_status_sync"
    # Even with the 1.0s stdout buffer delay on "OS:", calib_ready_s must stay synced near trace capture_s (<= 2.6s)
    assert 1.5 <= sched["calib_ready_s"] <= 2.6
    tgt = sched["targets"][0]
    assert tgt["target_end_s"] - tgt["target_start_s"] == pytest.approx(1.5, abs=0.15)
    rep = json.loads((run_dir / "gaze_attribution_report.json").read_text(encoding="utf-8"))
    assert rep["integrity"]["alignment_ok"] is True
    assert rep["windows"][0]["eligible_frames"] >= 25
    assert rep["windows"][0]["raw_x"]["pass_all"] is True


def test_resolve_run_paths_flat_cued_subdir_and_explicit_layouts(tmp_path: Path):
    # 1. Flat legacy layout
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "trace.txt").write_text("stub", encoding="utf-8")
    (flat / "gaze_diag.csv").write_text("stub", encoding="utf-8")
    dsp, diag, sched = ANALYZER.resolve_run_paths(flat)
    assert dsp == flat.resolve()
    assert diag == (flat / "gaze_diag.csv").resolve()
    assert sched is None

    # 2. Cued subdir layout
    cued = tmp_path / "cued"
    (cued / "dsp_record").mkdir(parents=True)
    (cued / "dsp_record" / "trace.txt").write_text("stub", encoding="utf-8")
    (cued / "gaze_diag.csv").write_text("stub", encoding="utf-8")
    (cued / "cued_schedule.json").write_text("{}", encoding="utf-8")
    dsp2, diag2, sched2 = ANALYZER.resolve_run_paths(cued)
    assert dsp2 == (cued / "dsp_record").resolve()
    assert diag2 == (cued / "gaze_diag.csv").resolve()
    assert sched2 == (cued / "cued_schedule.json").resolve()

    # 3. Separate explicit dsp_record_dir referenced from cued_schedule.json
    sep_run = tmp_path / "sep_run"
    sep_rec = tmp_path / "sep_rec"
    sep_run.mkdir()
    sep_rec.mkdir()
    (sep_rec / "trace.txt").write_text("stub", encoding="utf-8")
    (sep_run / "gaze_diag.csv").write_text("stub", encoding="utf-8")
    (sep_run / "cued_schedule.json").write_text(json.dumps({"dsp_record_dir": str(sep_rec)}), encoding="utf-8")
    dsp3, diag3, sched3 = ANALYZER.resolve_run_paths(sep_run)
    assert dsp3 == sep_rec.resolve()
    assert diag3 == (sep_run / "gaze_diag.csv").resolve()
    assert sched3 == (sep_run / "cued_schedule.json").resolve()

