"""Check a saved native gaze diagnostic against its Engineer 4 recorder frame by frame.

Run with the path to the recorder directory and its matching gaze_diag.csv.
Prints JSON and exits nonzero on inconsistent or incomplete evidence. This is
an offline integrity check; qualification windows and model accuracy are separate.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

IMPORT_VS_TRACE_PATH = Path(__file__).resolve().parents[3] / "dsp_filter/evaluation/import_vs_trace.py"
COORD_TOL = 1e-6
EXPECTED_DIAG_COLUMNS = (
    "frame_id", "ts_ms", "capture_s", "detected", "state",
    "left_iris_x", "right_iris_x", "sx", "left_iris_y", "right_iris_y", "sy",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "rel_pitch_deg", "rel_yaw_deg", "abs_pitch_deg", "abs_yaw_deg",
    "norm_x", "norm_y",
    "lm33_x", "lm33_y", "lm133_x", "lm133_y", "lm468_x", "lm468_y",
    "lm362_x", "lm362_y", "lm263_x", "lm263_y", "lm473_x", "lm473_y",
)


def _load_import_vs_trace():
    spec = importlib.util.spec_from_file_location("e4_import_vs_trace", str(IMPORT_VS_TRACE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {IMPORT_VS_TRACE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(recorder_dir: Path, diag_csv: Path) -> dict:
    if not (recorder_dir / "COMPLETE").is_file() or (recorder_dir / "INCOMPLETE").exists():
        raise ValueError("Recorder completion marker missing or incomplete")
    importer = _load_import_vs_trace()
    lines = (recorder_dir / "trace.txt").read_text(encoding="utf-8").splitlines()
    records = importer.convert_recorder(lines, "gaze")
    with diag_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_DIAG_COLUMNS:
            raise ValueError("Unexpected gaze diagnostic schema")
        observations = list(reader)
    if not records or len(records) != len(observations):
        raise ValueError("Diagnostic and recorder row counts disagree or are empty")

    errors = dict(frame_id=0, ts_ms=0, capture_s=0, state=0, detected=0,
                  norm_x=0, norm_y=0, nonfinite=0, non_increasing=0)
    previous = None
    max_delta = {"norm_x": 0.0, "norm_y": 0.0}
    for fields, diag in zip(records, observations):
        rec = dict(zip(importer.RECORDER_COLUMNS, fields))
        for key, other in (("frame_id", "frame_id"), ("ts_ms", "observation_ms"), ("state", "state")):
            errors[key] += int(diag[key]) != int(rec[other])
        times = int(diag["frame_id"]), int(diag["ts_ms"]), float(diag["capture_s"])
        if previous is not None and any(now <= last for now, last in zip(times, previous)):
            errors["non_increasing"] += 1
        previous = times
        capture = float(rec["dsp_timestamp_seconds"])
        errors["capture_s"] += not math.isclose(times[2], capture, abs_tol=1e-12, rel_tol=0)
        valid = int(diag["detected"])
        has_position = all(0.0 <= float(rec["tracker_" + axis]) <= 1.0 for axis in ("x", "y"))
        errors["detected"] += valid != int(has_position)
        if valid == 1 and any(not math.isfinite(float(diag[key])) for key in EXPECTED_DIAG_COLUMNS[5:]):
            errors["nonfinite"] += 1
        for axis in ("x", "y"):
            key = "norm_" + axis
            delta = abs(float(diag[key]) - float(rec["tracker_" + axis]))
            if math.isfinite(delta):
                max_delta[key] = max(max_delta[key], delta)
            errors[key] += not math.isfinite(delta) or delta > COORD_TOL
    return {"alignment_ok": not any(errors.values()), "frames": len(records),
            "mismatches": errors, "max_abs_normalized_difference": max_delta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorder_dir", type=Path)
    parser.add_argument("diag_csv", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.recorder_dir, args.diag_csv)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"alignment_ok": False, "error": str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["alignment_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
