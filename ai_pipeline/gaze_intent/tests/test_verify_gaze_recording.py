"""Offline integrity regressions for native gaze CSV and DSP recorder pairs."""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        rows = list(csv.DictReader(f))
    for k, v in updates.items():
        rows[row_idx][k] = v
    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(verify_gaze_recording.EXPECTED_DIAG_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_clean_and_paired_damage(tmp_path):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    assert verify_gaze_recording.verify(dsp, diag)["alignment_ok"]
    _rewrite_diag_row(diag, 5, ts_ms="999999", norm_x="0.602128")
    result = verify_gaze_recording.verify(dsp, diag)
    assert not result["alignment_ok"]
    assert result["mismatches"]["ts_ms"] == result["mismatches"]["norm_x"] == 1
    assert result["max_abs_normalized_difference"]["norm_x"] == pytest.approx(0.102128)


@pytest.mark.parametrize("key,value", [("frame_id","42"), ("capture_s","nan"),
                         ("ts_ms","1000"), ("norm_y","inf"), ("sx","nan"), ("detected","0")])
def test_bad_diagnostic(tmp_path, key, value):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    _rewrite_diag_row(diag, 5, **{key: value})
    assert not verify_gaze_recording.verify(dsp, diag)["alignment_ok"]


@pytest.mark.parametrize("damage", ["marker", "overflow", "missing row"])
def test_incomplete_pair(tmp_path, damage):
    _, dsp, diag = _make_synthetic_run(tmp_path)
    if damage == "marker":
        (dsp / "INCOMPLETE").touch()
    elif damage == "overflow":
        path = dsp / "trace.txt"
        path.write_text(path.read_text().replace("overflow=0", "overflow=1"))
    else:
        diag.write_text("\n".join(diag.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        verify_gaze_recording.verify(dsp, diag)
