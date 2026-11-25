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
import re
from pathlib import Path
from typing import Dict, List, Optional

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
DIAG_ALWAYS_FINITE_FLOAT_COLUMNS = ("capture_s", "norm_x", "norm_y")
DIAG_INACTIVE_NAN_ALLOWED_COLUMNS = (
    "left_iris_x", "right_iris_x", "sx", "left_iris_y", "right_iris_y", "sy",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "rel_pitch_deg", "rel_yaw_deg", "abs_pitch_deg", "abs_yaw_deg",
    "lm33_x", "lm33_y", "lm133_x", "lm133_y", "lm468_x", "lm468_y",
    "lm362_x", "lm362_y", "lm263_x", "lm263_y", "lm473_x", "lm473_y",
)
DIAG_ALL_FLOAT_COLUMNS = DIAG_ALWAYS_FINITE_FLOAT_COLUMNS + DIAG_INACTIVE_NAN_ALLOWED_COLUMNS
ALLOWED_DETECTED_VALUES = (0, 1)
ALLOWED_STATE_VALUES = (-1, 0, 1, 2, 3)
_STRICT_INT_RE = re.compile(r"^[+-]?[0-9]+$")


def _load_import_vs_trace():
    spec = importlib.util.spec_from_file_location("e4_import_vs_trace", str(IMPORT_VS_TRACE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {IMPORT_VS_TRACE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_strict_int(raw: object) -> Optional[int]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or not _STRICT_INT_RE.match(text):
        return None
    try:
        return int(text, 10)
    except ValueError:
        return None


def _parse_strict_float(raw: object) -> Optional[float]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def verify(recorder_dir: Path, diag_csv: Path) -> dict:
    if not (recorder_dir / "COMPLETE").is_file() or (recorder_dir / "INCOMPLETE").exists():
        raise ValueError("Recorder completion marker missing or incomplete")
    importer = _load_import_vs_trace()
    lines = (recorder_dir / "trace.txt").read_text(encoding="utf-8").splitlines()
    records = importer.convert_recorder(lines, "gaze")

    with diag_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(header) != EXPECTED_DIAG_COLUMNS:
            raise ValueError("Unexpected gaze diagnostic schema")
        raw_observations = list(reader)

    if not records or len(records) != len(raw_observations):
        raise ValueError("Diagnostic and recorder row counts disagree or are empty")

    expected_width = len(EXPECTED_DIAG_COLUMNS)
    always_finite_set = set(DIAG_ALWAYS_FINITE_FLOAT_COLUMNS)
    inactive_nan_allowed_set = set(DIAG_INACTIVE_NAN_ALLOWED_COLUMNS)

    errors: Dict[str, int] = {
        "frame_id": 0,
        "ts_ms": 0,
        "capture_s": 0,
        "state": 0,
        "detected": 0,
        "norm_x": 0,
        "norm_y": 0,
        "nonfinite": 0,
        "non_increasing": 0,
        "surplus_fields": 0,
        "missing_fields": 0,
        "invalid_numeric": 0,
        "infinite_values": 0,
        "invalid_integer": 0,
        "invalid_detected": 0,
        "invalid_state": 0,
    }
    error_samples: List[str] = []

    def _record_sample(msg: str) -> None:
        if len(error_samples) < 25:
            error_samples.append(msg)

    previous: Optional[tuple[int, int, float]] = None
    max_delta = {"norm_x": 0.0, "norm_y": 0.0}

    for row_idx, (fields, raw_row) in enumerate(zip(records, raw_observations)):
        if len(raw_row) > expected_width:
            errors["surplus_fields"] += 1
            _record_sample(f"row {row_idx}: surplus fields ({len(raw_row)} > {expected_width})")
            continue
        if len(raw_row) < expected_width:
            errors["missing_fields"] += 1
            _record_sample(f"row {row_idx}: missing fields ({len(raw_row)} < {expected_width})")
            continue

        diag = dict(zip(EXPECTED_DIAG_COLUMNS, raw_row))
        rec = dict(zip(importer.RECORDER_COLUMNS, fields))

        # 1. Strict integer & enum validation
        d_fid = _parse_strict_int(diag["frame_id"])
        if d_fid is None or d_fid < 0:
            errors["invalid_integer"] += 1
            _record_sample(f"row {row_idx}: invalid frame_id={diag['frame_id']!r}")

        d_ts = _parse_strict_int(diag["ts_ms"])
        if d_ts is None or d_ts < 0:
            errors["invalid_integer"] += 1
            _record_sample(f"row {row_idx}: invalid ts_ms={diag['ts_ms']!r}")

        d_det = _parse_strict_int(diag["detected"])
        if d_det is None:
            errors["invalid_integer"] += 1
            _record_sample(f"row {row_idx}: invalid detected={diag['detected']!r}")
        elif d_det not in ALLOWED_DETECTED_VALUES:
            errors["invalid_detected"] += 1
            _record_sample(f"row {row_idx}: disallowed detected={d_det}")

        d_state = _parse_strict_int(diag["state"])
        if d_state is None:
            errors["invalid_integer"] += 1
            _record_sample(f"row {row_idx}: invalid state={diag['state']!r}")
        elif d_state not in ALLOWED_STATE_VALUES:
            errors["invalid_state"] += 1
            _record_sample(f"row {row_idx}: disallowed state={d_state}")

        # 2. Validate ALL 28 declared float columns on BOTH active and inactive rows
        parsed_floats: Dict[str, Optional[float]] = {}
        for col in DIAG_ALL_FLOAT_COLUMNS:
            val = _parse_strict_float(diag[col])
            parsed_floats[col] = val
            if val is None:
                errors["invalid_numeric"] += 1
                _record_sample(f"row {row_idx}: unreadable numeric field {col}={diag[col]!r}")
            elif math.isinf(val):
                errors["infinite_values"] += 1
                errors["nonfinite"] += 1
                _record_sample(f"row {row_idx}: infinite value in {col}={diag[col]!r}")
            elif math.isnan(val):
                if col in always_finite_set:
                    errors["nonfinite"] += 1
                    _record_sample(f"row {row_idx}: NaN in always-finite column {col}")
                elif d_det == 0 and col in inactive_nan_allowed_set:
                    # Legitimate inactive NaN placeholder per 32-column schema
                    pass
                else:
                    errors["nonfinite"] += 1
                    _record_sample(f"row {row_idx}: NaN in column {col} with detected={d_det}")

        # 3. Recorder field comparison & strict monotonicity
        r_fid = int(rec["frame_id"])
        r_ts = int(rec["observation_ms"])
        r_state = int(rec["state"])
        r_cap = float(rec["dsp_timestamp_seconds"])
        r_tx = float(rec["tracker_x"])
        r_ty = float(rec["tracker_y"])

        if d_fid is not None and r_fid is not None and d_fid != r_fid:
            errors["frame_id"] += 1
        if d_ts is not None and r_ts is not None and d_ts != r_ts:
            errors["ts_ms"] += 1
        if d_state is not None and r_state is not None and d_state != r_state:
            errors["state"] += 1

        d_cap = parsed_floats["capture_s"]
        if d_fid is not None and d_ts is not None and d_cap is not None and math.isfinite(d_cap):
            times = (d_fid, d_ts, d_cap)
            if previous is not None and any(now <= last for now, last in zip(times, previous)):
                errors["non_increasing"] += 1
            previous = times

        if d_cap is None or r_cap is None or not math.isfinite(d_cap) or not math.isfinite(r_cap):
            errors["capture_s"] += 1
        elif not math.isclose(d_cap, r_cap, abs_tol=1e-12, rel_tol=0.0):
            errors["capture_s"] += 1

        if r_tx is not None and r_ty is not None and math.isfinite(r_tx) and math.isfinite(r_ty):
            has_position = (0.0 <= r_tx <= 1.0) and (0.0 <= r_ty <= 1.0)
            if d_det in ALLOWED_DETECTED_VALUES and d_det != int(has_position):
                errors["detected"] += 1
        else:
            errors["detected"] += 1

        for axis, r_coord in (("x", r_tx), ("y", r_ty)):
            key = "norm_" + axis
            d_coord = parsed_floats[key]
            if d_coord is None or r_coord is None or not math.isfinite(d_coord) or not math.isfinite(r_coord):
                errors[key] += 1
            else:
                delta = abs(d_coord - r_coord)
                max_delta[key] = max(max_delta[key], delta)
                if delta > COORD_TOL:
                    errors[key] += 1

    schema_errors = (
        errors["surplus_fields"]
        + errors["missing_fields"]
        + errors["invalid_numeric"]
        + errors["infinite_values"]
        + errors["invalid_integer"]
        + errors["invalid_detected"]
        + errors["invalid_state"]
    )
    return {
        "alignment_ok": not any(errors.values()),
        "frames": len(records),
        "schema_errors": schema_errors,
        "mismatches": errors,
        "max_abs_normalized_difference": max_delta,
        "error_samples": error_samples,
    }


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
