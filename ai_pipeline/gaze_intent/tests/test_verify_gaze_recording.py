"""Offline integrity regressions for native gaze CSV and DSP recorder pairs."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import verify_gaze_recording


def _make_synthetic_run(
    base_dir: Path,
    *,
    num_frames: int = 35,
    dt_s: float = 0.033,
    gap_after_idx: Optional[int] = None,
    gap_dt_s: float = 0.140,
    x_val: float = 0.5,
    y_val: float = 0.5,
    inactive_indices: Optional[set[int]] = None,
) -> tuple[Path, Path, Path]:
    """Create a valid synthetic E4REC v2 trace.txt + paired 32-col gaze_diag.csv + cued_schedule.json."""
    run_dir = base_dir
    dsp_dir = run_dir / "dsp_record"
    dsp_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = run_dir / "gaze_diag.csv"
    trace_path = dsp_dir / "trace.txt"
    inactive = inactive_indices or set()

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
    initialized = False

    for idx, cap_s in enumerate(times_s):
        cap_ms = int(math.floor(cap_s * 1000.0 + 1e-9))
        obs_ms = max(cap_ms, prev_ms + 1)
        prev_ms = obs_ms
        is_inactive = idx in inactive

        if is_inactive:
            resets_flag = 1 if initialized else 0
            initialized = False
            trace_lines.append(
                f"E4REC F frame={idx} ms={obs_ms} cap={cap_ms} t={cap_s:.9f} n=484 ok=1 "
                f"x=-1 y=-1 state=-1 trk=-1 missed=0 adapter=0 called=0 "
                f"input_t=0 ix=0 iy=0 status=-1 has=0 "
                f"dx=0 dy=0 outcome=-1 resets={resets_flag} pub=0 "
                f"px=0 py=0 out_trk=2 generation=1"
            )
            row = {col: "nan" for col in verify_gaze_recording.EXPECTED_DIAG_COLUMNS}
            row["frame_id"] = str(idx)
            row["ts_ms"] = str(obs_ms)
            row["capture_s"] = f"{cap_s:.9f}"
            row["detected"] = "0"
            row["state"] = "-1"
            row["norm_x"] = "-1.000000000"
            row["norm_y"] = "-1.000000000"
            diag_rows.append(row)
        else:
            status = 0 if not initialized else 1
            initialized = True
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
        writer = csv.DictWriter(f, fieldnames=list(verify_gaze_recording.EXPECTED_DIAG_COLUMNS))
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
        rows = list(csv.reader(f))
    header = rows[0]
    col_to_idx = {name: i for i, name in enumerate(header)}
    for k, v in updates.items():
        rows[row_idx + 1][col_to_idx[k]] = v
    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def test_clean_and_paired_damage(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    result_ok = verify_gaze_recording.verify(dsp, diag)
    assert result_ok["alignment_ok"]
    assert result_ok["schema_errors"] == 0
    _rewrite_diag_row(diag, 5, ts_ms="999999", norm_x="0.602128")
    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"]["ts_ms"] == result["mismatches"]["norm_x"] == 1
    assert result["max_abs_normalized_difference"]["norm_x"] == pytest.approx(0.102128)


def test_legitimate_inactive_nan_placeholders_pass(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={0, 1, 10})
    result = verify_gaze_recording.verify(dsp, diag)
    assert result["alignment_ok"]
    assert result["schema_errors"] == 0
    assert not any(result["mismatches"].values())


@pytest.mark.parametrize("key,value", [
    ("frame_id", "42"),
    ("capture_s", "nan"),
    ("ts_ms", "1000"),
    ("norm_y", "inf"),
    ("sx", "nan"),
    ("detected", "0"),
])
def test_bad_diagnostic(tmp_path, key, value):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    _rewrite_diag_row(diag, 5, **{key: value})
    assert not verify_gaze_recording.verify(dsp, diag)["alignment_ok"]


@pytest.mark.parametrize("damage", ["marker", "overflow", "missing row", "header"])
def test_incomplete_pair(tmp_path, damage):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    if damage == "marker":
        (dsp / "INCOMPLETE").touch()
    elif damage == "overflow":
        path = dsp / "trace.txt"
        path.write_text(path.read_text(encoding="utf-8").replace("overflow=0", "overflow=1"), encoding="utf-8")
    elif damage == "header":
        lines = diag.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace("lm473_y", "lm473_bad")
        diag.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        diag.write_text("\n".join(diag.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_gaze_recording.verify(dsp, diag)


def test_defect1_surplus_33rd_field_rejected_active_and_inactive(tmp_path):
    for label, inactive_set, target_row in (("active", set(), 5), ("inactive", {5}, 5)):
        sub = tmp_path / label
        _, dsp, diag = _make_synthetic_run(sub, inactive_indices=inactive_set)
        lines = diag.read_text(encoding="utf-8").splitlines()
        lines[target_row + 1] = lines[target_row + 1] + ",999"
        diag.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = verify_gaze_recording.verify(dsp, diag)
        assert not result["alignment_ok"], label
        assert result["mismatches"]["surplus_fields"] == 1
        assert result["schema_errors"] >= 1


def test_defect2_missing_inactive_lm473_y_rejected_without_crash(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={5})
    lines = diag.read_text(encoding="utf-8").splitlines()
    row_fields = lines[6].split(",")
    assert len(row_fields) == 32
    lines[6] = ",".join(row_fields[:-1])  # drop lm473_y -> 31 fields
    diag.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"]["missing_fields"] == 1
    assert result["schema_errors"] >= 1


def test_defect3_unreadable_text_in_inactive_lm473_y_rejected_without_crash(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={5})
    _rewrite_diag_row(diag, 5, lm473_y="bad_text")

    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"]["invalid_numeric"] == 1
    assert result["schema_errors"] >= 1


@pytest.mark.parametrize("bad_val", ["inf", "-inf", "Infinity", "-Infinity"])
def test_infinity_in_inactive_auxiliary_column_rejected(tmp_path, bad_val):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={5})
    _rewrite_diag_row(diag, 5, lm473_y=bad_val)

    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"]["infinite_values"] == 1
    assert result["mismatches"]["nonfinite"] == 1


@pytest.mark.parametrize("field,bad_val,expected_key", [
    ("detected", "2", "invalid_detected"),
    ("detected", "-1", "invalid_detected"),
    ("state", "4", "invalid_state"),
    ("frame_id", "5.5", "invalid_integer"),
    ("ts_ms", "-10", "invalid_integer"),
    ("sx", "", "invalid_numeric"),
    ("norm_x", "nan", "nonfinite"),
])
def test_schema_and_domain_corruptions_return_controlled_failures(tmp_path, field, bad_val, expected_key):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={5} if field == "norm_x" else set())
    _rewrite_diag_row(diag, 5, **{field: bad_val})

    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"][expected_key] >= 1


def test_cli_exits_zero_on_clean_and_one_on_corrupted(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path, inactive_indices={0})
    script = Path(verify_gaze_recording.__file__).resolve()
    ok_proc = subprocess.run([sys.executable, str(script), str(dsp), str(diag)], capture_output=True, text=True)
    assert ok_proc.returncode == 0
    assert json.loads(ok_proc.stdout)["alignment_ok"] is True

    _rewrite_diag_row(diag, 0, lm473_y="corrupt")
    bad_proc = subprocess.run([sys.executable, str(script), str(dsp), str(diag)], capture_output=True, text=True)
    assert bad_proc.returncode == 1
    bad_json = json.loads(bad_proc.stdout)
    assert bad_json["alignment_ok"] is False
    assert bad_json["mismatches"]["invalid_numeric"] == 1
