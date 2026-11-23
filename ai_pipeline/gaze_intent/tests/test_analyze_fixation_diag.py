"""Unit tests for ai_pipeline/gaze_intent/tests/analyze_gaze_fixation_diag.py."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import analyze_gaze_fixation_diag as agd  # noqa: E402


def _make_synthetic_run(
    base_dir: Path,
    *,
    num_frames: int = 35,
    dt_s: float = 0.033,
    gap_after_idx: Optional[int] = None,
    gap_dt_s: float = 0.140,
    x_val: float = 0.5,
    y_val: float = 0.5,
) -> tuple[Path, Path, Path]:
    """Create a valid synthetic E4REC v2 trace.txt + paired 32-col gaze_diag.csv + cued_schedule.json."""
    run_dir = base_dir
    dsp_dir = run_dir / "dsp_record"
    dsp_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = run_dir / "gaze_diag.csv"
    trace_path = dsp_dir / "trace.txt"

    times_s: List[float] = []
    t = 1.000
    for i in range(num_frames):
        times_s.append(t)
        step = gap_dt_s if (gap_after_idx is not None and i == gap_after_idx) else dt_s
        t += step

    trace_lines = [
        "E4REC BEGIN version=2 channel=gaze capacity=4096 min_cutoff=0.8 beta=6 derivative_cutoff=1.5 min_dt=0.0001 reset_gap=0.5"
    ]
    diag_rows: List[Dict[str, str]] = []
    prev_ms = -1

    for idx, cap_s in enumerate(times_s):
        cap_ms = int(math.floor(cap_s * 1000.0 + 1e-9))
        obs_ms = max(cap_ms, prev_ms + 1)
        prev_ms = obs_ms
        status = 0 if idx == 0 else 1

        trace_lines.append(
            f"E4REC F frame={idx} ms={obs_ms} cap={cap_ms} t={cap_s:.9f} n=484 ok=1 "
            f"x={x_val:.9f} y={y_val:.9f} state=0 trk=-1 missed=0 adapter=1 called=1 "
            f"input_t={cap_s:.9f} ix={x_val:.9f} iy={y_val:.9f} status={status} has=1 "
            f"dx={x_val:.9f} dy={y_val:.9f} outcome=0 resets=0 pub=1 "
            f"px={x_val:.9f} py={y_val:.9f} out_trk=0 generation=1"
        )
        diag_rows.append({
            "frame_id": str(idx),
            "ts_ms": str(obs_ms),
            "capture_s": f"{cap_s:.9f}",
            "detected": "1",
            "state": "0",
            "left_iris_x": "0.5",
            "right_iris_x": "0.5",
            "sx": "0.5",
            "left_iris_y": "0.0",
            "right_iris_y": "0.0",
            "sy": "0.0",
            "head_pitch_deg": "0.0",
            "head_yaw_deg": "0.0",
            "head_roll_deg": "0.0",
            "rel_pitch_deg": "0.0",
            "rel_yaw_deg": "0.0",
            "abs_pitch_deg": "0.0",
            "abs_yaw_deg": "0.0",
            "norm_x": f"{x_val:.9f}",
            "norm_y": f"{y_val:.9f}",
            "lm33_x": "0.40", "lm33_y": "0.40",
            "lm133_x": "0.46", "lm133_y": "0.40",
            "lm468_x": "0.43", "lm468_y": "0.40",
            "lm362_x": "0.54", "lm362_y": "0.40",
            "lm263_x": "0.60", "lm263_y": "0.40",
            "lm473_x": "0.57", "lm473_y": "0.40",
        })

    trace_lines.append(
        f"E4REC END submitted={num_frames} attempted={num_frames} stored={num_frames} "
        "overflow=0 config_changes=0 complete=1"
    )
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    (dsp_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")

    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agd.EXPECTED_DIAG_COLUMNS))
        writer.writeheader()
        writer.writerows(diag_rows)

    sched_path = run_dir / "cued_schedule.json"
    sched_path.write_text(json.dumps({
        "schema_version": 1,
        "targets": [
            {"name": "CENTER", "target_start_s": times_s[0], "target_end_s": times_s[-1] + 0.01}
        ],
    }), encoding="utf-8")
    return run_dir, dsp_dir, diag_csv


def _rewrite_diag_row(diag_csv: Path, row_idx: int, **updates: str) -> None:
    with diag_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for k, v in updates.items():
        rows[row_idx][k] = v
    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agd.EXPECTED_DIAG_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_valid_run_passes_alignment_and_gap_gates(tmp_path: Path):
    run_dir, _, _ = _make_synthetic_run(tmp_path / "valid_run", num_frames=35, dt_s=0.033)
    out_json = run_dir / "out.json"
    out_md = run_dir / "out.md"

    rc = agd.main([
        "--run-dir", str(run_dir),
        "--wash-in-s", "0.0",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0

    report = json.loads(out_json.read_text(encoding="utf-8"))
    integ = report["integrity"]
    assert integ["alignment_ok"] is True
    assert integ["ts_ms_mismatches"] == 0
    assert integ["nonfinite_errors"] == 0
    assert integ["norm_x_mismatches"] == 0
    assert integ["norm_y_mismatches"] == 0
    assert integ["frame_id_non_increasing"] == 0
    assert integ["ts_ms_non_increasing"] == 0
    assert integ["capture_s_non_increasing"] == 0

    win = report["windows"][0]
    assert win["raw_x"]["n"] == 35
    assert win["raw_x"]["max_gap_ms"] < agd.MAX_FRESH_GAP_MS
    assert win["raw_x"]["pass_gap"] is True
    assert win["raw_x"]["pass_all"] is True
    assert win["raw_y"]["pass_gap"] is True
    assert win["raw_y"]["pass_all"] is True

    md = out_md.read_text(encoding="utf-8")
    assert "INVALID RUN" not in md
    assert "Max Gap ms (<=100)" in md


def test_fresh_gap_over_100ms_fails_window_verdict(tmp_path: Path):
    run_dir, _, _ = _make_synthetic_run(
        tmp_path / "gap_run", num_frames=35, dt_s=0.033, gap_after_idx=15, gap_dt_s=0.145
    )
    out_json = run_dir / "out.json"
    out_md = run_dir / "out.md"

    rc = agd.main([
        "--run-dir", str(run_dir),
        "--wash-in-s", "0.0",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["integrity"]["alignment_ok"] is True

    win = report["windows"][0]
    for axis in ("raw_x", "raw_y"):
        assert win[axis]["pass_n"] is True
        assert win[axis]["pass_sd"] is True
        assert win[axis]["pass_range"] is True
        assert win[axis]["pass_slope"] is True
        assert win[axis]["max_gap_ms"] > agd.MAX_FRESH_GAP_MS
        assert win[axis]["pass_gap"] is False
        assert win[axis]["pass_all"] is False


@pytest.mark.parametrize(
    "mutation_name,row_idx,updates,expected_field",
    [
        ("ts_ms_mismatch", 5, {"ts_ms": "999999"}, "ts_ms_mismatches"),
        ("norm_delta_over_tol", 6, {"norm_x": "0.500010000"}, "norm_x_mismatches"),
        ("nan_in_detected_sx", 7, {"sx": "nan"}, "nonfinite_errors"),
        ("nan_in_detected_norm_x", 8, {"norm_x": "nan"}, "nonfinite_errors"),
        ("non_monotonic_capture_s", 10, {"capture_s": "1.000000000"}, "capture_s_non_increasing"),
        ("non_monotonic_ts_ms", 11, {"ts_ms": "1000"}, "ts_ms_non_increasing"),
    ],
)
def test_alignment_integrity_violations_fail_and_exit_nonzero(
    tmp_path: Path, capsys, mutation_name: str, row_idx: int, updates: Dict[str, str], expected_field: str
):
    run_dir, _, diag_csv = _make_synthetic_run(tmp_path / mutation_name, num_frames=35, dt_s=0.033)
    _rewrite_diag_row(diag_csv, row_idx, **updates)

    out_json = run_dir / "out.json"
    out_md = run_dir / "out.md"
    rc = agd.main([
        "--run-dir", str(run_dir),
        "--wash-in-s", "0.0",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 1
    captured = capsys.readouterr().out
    assert "INVALID RUN / ALIGNMENT FAILED" in captured

    report = json.loads(out_json.read_text(encoding="utf-8"))
    integ = report["integrity"]
    assert integ["alignment_ok"] is False
    assert integ[expected_field] > 0

    md = out_md.read_text(encoding="utf-8")
    assert "INVALID RUN / FAILED ALIGNMENT & INTEGRITY CHECKS" in md
