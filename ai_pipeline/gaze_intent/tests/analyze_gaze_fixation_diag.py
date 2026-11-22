#!/usr/bin/env python
"""
Offline Gaze Fixation Stability, Attribution, A/B Shadow Replay & Cued Capture Tool.

Evaluates a completed Engineer 4 recorder run (`--dsp-record-dir`) paired with
C++ native gaze diagnostics (`--gaze-diag-log`) without touching DSP parameters
or moving the OS mouse.

Capabilities:
  1. E4REC v2 integrity & 1:1 `frame_id` / `capture_s` alignment verification.
  2. Raw fixation stability against Engineer 4 limits (SD <= 0.010, range <= 0.030,
     |OLS slope| <= 0.005/s, >= 25 samples) for both raw_x and raw_y, with
     configurable wash-in trimming (500 ms default for cued captures) and +-2 frame
     blink/dropout boundary exclusion.
  3. Horizontal (LEFT/RIGHT) & vertical (UP/DOWN) coordinate sign verification
     matching `core_engine/include/gaze_math.hpp` (`norm_y = clamp01(0.5 - pitch / 60)`).
  4. Quantitative variance & top-5 jump decomposition separating:
       (a) Iris landmark jitter (sx, sy)
       (b) Head-pose variation (direct additive angle + tree-input effect)
       (c) Treelite prediction cliffs / local +-0.005 sensitivity
  5. Optional read-only A/B shadow replay against candidate DLLs (`--candidate-models-dir`).
  6. Optional safe cued-capture launcher (`--cued-capture`, `--os log --no-hand --filter dsp`)
     guiding CENTER, LEFT, RIGHT, UP, DOWN fixations and logging `calib_ready_s` and
     per-target `target_start_s`/`target_end_s` to `<run-dir>/cued_schedule.json`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tl2cgen

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOYED_MODELS_DIR = REPO_ROOT / "ai_pipeline" / "gaze_intent" / "models"
DEFAULT_ENGINE_EXE = REPO_ROOT / "out" / "build" / "windows-release" / "core_engine.exe"
IMPORT_VS_TRACE_PATH = REPO_ROOT / "dsp_filter" / "evaluation" / "import_vs_trace.py"

SPAN_DEG = 60.0
FIXED_Y_SIGN = -1.0  # norm_y = clamp01(0.5 - abs_pitch_deg / 60.0)
LEGACY_Y_SIGN = 1.0  # pre-fix norm_y = clamp01(0.5 + abs_pitch_deg / 60.0)
MIN_SAMPLES = 25
LIMIT_SD = 0.010
LIMIT_RANGE = 0.030
LIMIT_SLOPE = 0.005  # units / second
DEFAULT_CUED_WASH_IN_S = 0.5
DEFAULT_BLINK_PAD_FRAMES = 2
DEFAULT_CUED_TARGETS = ("CENTER", "LEFT", "RIGHT", "UP", "DOWN")
GAZE_FILTER_SPEC = "0.8,6,1.5"

GAZE_STATUS_RE = re.compile(
    r"^cap\s+(\d+)\s*\|\s*ai\s+(\d+)\s+fps\s*\|.*?"
    r"pyerr\s+(\d+).*?gaze\(([^,]+),([^ ]+)\s+s=(-?\d+)\).*?os frames\s+(\d+)\s*$"
)

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


def _file_inventory(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "bytes": None, "sha256": None}
    data = path.read_bytes()
    return {
        "path": str(path),
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def _models_inventory(models_dir: Path, include_gesture: bool = True) -> Dict[str, Dict[str, Any]]:
    names = ["gaze_yaw.dll", "gaze_pitch.dll"]
    if include_gesture:
        names.append("gesture_model.dll")
    return {name: _file_inventory(models_dir / name) for name in names}


def _clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _norm_coord_x(abs_yaw_deg: np.ndarray) -> np.ndarray:
    return _clamp01(0.5 + abs_yaw_deg / SPAN_DEG)


def _norm_coord_y(abs_pitch_deg: np.ndarray, y_sign: float = FIXED_Y_SIGN) -> np.ndarray:
    return _clamp01(0.5 + y_sign * (abs_pitch_deg / SPAN_DEG))


def _predict_angles(
    yaw_pred: tl2cgen.Predictor,
    pitch_pred: tl2cgen.Predictor,
    features_5d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(features_5d) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    f32 = np.ascontiguousarray(features_5d, dtype=np.float32)
    dm = tl2cgen.DMatrix(f32)
    rel_yaw = np.asarray(yaw_pred.predict(dm), dtype=np.float64).reshape(-1)
    rel_pitch = np.asarray(pitch_pred.predict(dm), dtype=np.float64).reshape(-1)
    return rel_yaw, rel_pitch


def _compute_axis_metrics(values: np.ndarray, times_s: np.ndarray) -> Dict[str, Any]:
    n = int(len(values))
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
            "range": None,
            "pop_sd": None,
            "ols_slope_per_s": None,
            "max_abs_step": None,
            "clipped_zero_or_one": 0,
            "pass_n": False,
            "pass_sd": False,
            "pass_range": False,
            "pass_slope": False,
            "pass_all": False,
        }
    mean_v = float(np.mean(values))
    med_v = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr_v = float(q75 - q25)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    rng_v = float(max_v - min_v)
    sd_v = float(np.sqrt(np.mean((values - mean_v) ** 2)))  # population SD (pstdev)

    if n >= 2:
        t_c = times_s - float(np.mean(times_s))
        denom = float(np.sum(t_c * t_c))
        slope = float(np.sum(t_c * (values - mean_v)) / denom) if denom > 0.0 else 0.0
        max_step = float(np.max(np.abs(np.diff(values))))
    else:
        slope = 0.0
        max_step = 0.0

    clipped = int(np.sum((values <= 0.0) | (values >= 1.0)))
    pass_n = n >= MIN_SAMPLES
    pass_sd = sd_v <= LIMIT_SD
    pass_range = rng_v <= LIMIT_RANGE
    pass_slope = abs(slope) <= LIMIT_SLOPE
    return {
        "n": n,
        "mean": mean_v,
        "median": med_v,
        "iqr": iqr_v,
        "min": min_v,
        "max": max_v,
        "range": rng_v,
        "pop_sd": sd_v,
        "ols_slope_per_s": slope,
        "max_abs_step": max_step,
        "clipped_zero_or_one": clipped,
        "pass_n": pass_n,
        "pass_sd": pass_sd,
        "pass_range": pass_range,
        "pass_slope": pass_slope,
        "pass_all": bool(pass_n and pass_sd and pass_range and pass_slope),
    }


def parse_windows_spec(spec: str) -> List[Tuple[str, float, float]]:
    windows: List[Tuple[str, float, float]] = []
    seen = set()
    for token in spec.split(","):
        tok = token.strip()
        if not tok:
            continue
        if ":" not in tok or "-" not in tok.split(":", 1)[1]:
            raise ValueError(f"Invalid window spec {tok!r}; expected NAME:START-END (seconds)")
        name, span = tok.split(":", 1)
        name = name.strip()
        s_str, e_str = span.split("-", 1)
        start_s, end_s = float(s_str), float(e_str)
        if not name or name in seen or not (0.0 <= start_s < end_s):
            raise ValueError(f"Invalid window bounds/name in {tok!r}")
        seen.add(name)
        windows.append((name, start_s, end_s))
    if not windows:
        raise ValueError("No valid windows parsed from --windows")
    return windows


def load_cued_schedule(schedule_path: Path) -> Tuple[Dict[str, Any], List[Tuple[str, float, float]]]:
    data = json.loads(schedule_path.read_text(encoding="utf-8"))
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"Invalid cued_schedule.json (missing targets): {schedule_path}")
    windows: List[Tuple[str, float, float]] = []
    seen: Set[str] = set()
    for entry in targets:
        name = str(entry["name"]).strip()
        start_s = float(entry["target_start_s"])
        end_s = float(entry["target_end_s"])
        if not name or name in seen or not (0.0 <= start_s < end_s):
            raise ValueError(f"Invalid cued target entry {entry!r} in {schedule_path}")
        seen.add(name)
        windows.append((name, start_s, end_s))
    return data, windows


def auto_detect_windows(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    eligible = [r for r in records if r["eligible"]]
    if not eligible:
        t0 = float(records[0]["capture_s"])
        t1 = float(records[-1]["capture_s"]) + 1e-6
        return [("ALL_FRAMES", t0, t1)]
    t_start = float(eligible[0]["capture_s"])
    t_end = float(eligible[-1]["capture_s"]) + 1e-6
    dur = t_end - t_start
    windows = [("ACTIVE_ALL", t_start, t_end)]
    if dur >= 12.0:
        mid = t_start + 0.5 * dur
        margin = 1.0 if dur >= 16.0 else 0.0
        windows.append(("ACTIVE_FIRST_HALF", t_start, mid - margin))
        windows.append(("ACTIVE_SECOND_HALF", mid + margin, t_end))
    return windows


def compute_blink_pad_indices(rows: Sequence[Dict[str, Any]], pad_frames: int) -> Set[int]:
    if pad_frames <= 0 or not rows:
        return set()
    padded: Set[int] = set()
    n = len(rows)
    for idx, r in enumerate(rows):
        # Pad around blink/wink (state > 0) or face-loss/incomplete landmarks (state != -1)
        is_blink_or_dropout = (
            r["detected"] != 1
            or r["landmarks"] != 484
            or r["state"] > 0
        )
        if is_blink_or_dropout:
            lo = max(0, idx - pad_frames)
            hi = min(n - 1, idx + pad_frames)
            for j in range(lo, hi + 1):
                padded.add(j)
    return padded


class GazeReadinessGate:
    """Detect two consecutive calibrated, neutral live gaze status lines."""

    def __init__(self) -> None:
        self.first_status_time: Optional[float] = None
        self.calib_ready_time: Optional[float] = None
        self.last_os_frames: Optional[int] = None
        self.previous_valid = False

    def feed(self, line: str, now_monotonic: float) -> bool:
        if not math.isfinite(now_monotonic):
            raise ValueError("status timestamp must be finite")
        m = GAZE_STATUS_RE.fullmatch(line.strip())
        if not m:
            return False
        if self.first_status_time is None:
            self.first_status_time = now_monotonic
        cap_fps, ai_fps, pyerr, gx_s, gy_s, state_s, os_frames_s = m.groups()
        gx, gy = float(gx_s), float(gy_s)
        state = int(state_s)
        os_frames = int(os_frames_s)
        valid = (
            int(cap_fps) >= 0
            and int(ai_fps) > 0
            and int(pyerr) == 0
            and state == 0
            and math.isfinite(gx)
            and math.isfinite(gy)
            and 0.0 <= gx <= 1.0
            and 0.0 <= gy <= 1.0
            and os_frames > 0
        )
        advanced = self.last_os_frames is not None and os_frames > self.last_os_frames
        became_ready = self.calib_ready_time is None and valid and self.previous_valid and advanced
        self.previous_valid = valid
        self.last_os_frames = os_frames
        if became_ready:
            self.calib_ready_time = now_monotonic
        return became_ready


def build_cued_target_plan(
    targets: Sequence[str] = DEFAULT_CUED_TARGETS,
    hold_s: float = 10.0,
    gap_s: float = 2.0,
    initial_settle_s: float = 1.0,
) -> List[Dict[str, Any]]:
    if hold_s <= DEFAULT_CUED_WASH_IN_S:
        raise ValueError("hold_s must exceed 0.5s wash-in")
    if gap_s < 0.0 or initial_settle_s < 0.0:
        raise ValueError("gap_s and initial_settle_s must be non-negative")
    plan: List[Dict[str, Any]] = []
    cursor = initial_settle_s
    for idx, raw_name in enumerate(targets):
        name = str(raw_name).strip().upper()
        if not name:
            raise ValueError("Empty target name in cued plan")
        start_rel = cursor
        end_rel = start_rel + hold_s
        plan.append({
            "index": idx,
            "name": name,
            "offset_from_ready_start_s": start_rel,
            "offset_from_ready_end_s": end_rel,
        })
        cursor = end_rel + gap_s
    return plan


def build_gaze_capture_command(
    exe: Path,
    dsp_record_dir: Path,
    diag_csv: Path,
    camera: int,
    seconds: int,
) -> List[str]:
    return [
        str(exe),
        "--camera", str(camera),
        "--filter", "dsp",
        "--os", "log",
        "--no-hand",
        "--gaze-filter", GAZE_FILTER_SPEC,
        "--seconds", str(seconds),
        "--dsp-record-dir", str(dsp_record_dir),
        "--gaze-diag-log", str(diag_csv),
    ]


def _extract_trace_capture_times(trace_path: Path) -> List[float]:
    times: List[float] = []
    if not trace_path.is_file():
        return times
    for ln in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[0] == "E4REC" and parts[1] == "F":
            for token in parts[2:]:
                if token.startswith("t="):
                    times.append(float(token[2:]))
                    break
    return times


def _resolve_capture_epoch_mono(
    status_samples: Sequence[Tuple[int, float]],
    trace_times: Sequence[float],
    epoch_mono: Optional[float],
    launch_mono: float,
) -> Tuple[float, str]:
    cands = [
        rx_t - trace_times[os_frames - 1]
        for os_frames, rx_t in status_samples
        if 1 <= os_frames <= len(trace_times)
    ]
    tail = cands[1:] if len(cands) >= 3 else cands
    if len(tail) >= 2 and (max(tail) - min(tail)) <= 0.25:
        return float(np.median(tail)), "trace_status_sync"
    fallback = epoch_mono if epoch_mono is not None else launch_mono
    return fallback, "console_os_fallback"


def resolve_run_paths(
    run_dir: Path,
    dsp_record_dir: Optional[Path] = None,
    diag_csv: Optional[Path] = None,
    cued_schedule: Optional[Path] = None,
) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve (dsp_record_dir, diag_csv, cued_schedule_path) for flat or cued-subdir layouts."""
    run_dir = run_dir.resolve()
    sched_path = cued_schedule.resolve() if cued_schedule is not None else (
        (run_dir / "cued_schedule.json") if (run_dir / "cued_schedule.json").is_file() else None
    )
    sched_data: Dict[str, Any] = {}
    if sched_path is not None and sched_path.is_file():
        try:
            sched_data = json.loads(sched_path.read_text(encoding="utf-8"))
        except Exception:
            sched_data = {}

    if dsp_record_dir is not None:
        resolved_dsp = dsp_record_dir.resolve()
    elif any((run_dir / marker).is_file() for marker in ("trace.txt", "COMPLETE", "INCOMPLETE")):
        resolved_dsp = run_dir
    elif any((run_dir / "dsp_record" / marker).is_file() for marker in ("trace.txt", "COMPLETE", "INCOMPLETE")):
        resolved_dsp = run_dir / "dsp_record"
    elif isinstance(sched_data.get("dsp_record_dir"), str):
        cand = Path(sched_data["dsp_record_dir"])
        if not cand.is_absolute():
            cand = run_dir / cand
        resolved_dsp = cand.resolve() if cand.exists() else (
            (run_dir / "dsp_record") if (run_dir / "dsp_record").exists() else run_dir
        )
    else:
        resolved_dsp = run_dir

    if diag_csv is not None:
        resolved_diag = diag_csv.resolve()
    elif (run_dir / "gaze_diag.csv").is_file():
        resolved_diag = run_dir / "gaze_diag.csv"
    elif (resolved_dsp / "gaze_diag.csv").is_file():
        resolved_diag = resolved_dsp / "gaze_diag.csv"
    elif isinstance(sched_data.get("diag_csv"), str):
        cand_diag = Path(sched_data["diag_csv"])
        if not cand_diag.is_absolute():
            cand_diag = run_dir / cand_diag
        resolved_diag = cand_diag.resolve() if cand_diag.is_file() else (run_dir / "gaze_diag.csv")
    else:
        resolved_diag = run_dir / "gaze_diag.csv"

    return resolved_dsp, resolved_diag, sched_path



def load_and_align_run(run_dir: Path, diag_csv: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    complete_marker = run_dir / "COMPLETE"
    incomplete_marker = run_dir / "INCOMPLETE"
    trace_path = run_dir / "trace.txt"
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing recorder trace: {trace_path}")
    if not diag_csv.is_file():
        raise FileNotFoundError(f"Missing native gaze diag CSV: {diag_csv}")

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    header_meta: Dict[str, str] = {}
    footer_meta: Dict[str, str] = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2 and parts[0] == "E4REC" and parts[1] == "BEGIN":
            header_meta = dict(p.split("=", 1) for p in parts[2:])
        elif len(parts) >= 2 and parts[0] == "E4REC" and parts[1] == "END":
            footer_meta = dict(p.split("=", 1) for p in parts[2:])

    e4 = _load_import_vs_trace()
    converted = e4.convert_recorder(lines, "gaze")
    col_names = e4.RECORDER_COLUMNS
    trace_rows = [dict(zip(col_names, row)) for row in converted]

    with diag_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        diag_cols = tuple(reader.fieldnames or ())
        if diag_cols != EXPECTED_DIAG_COLUMNS:
            raise ValueError(
                f"gaze_diag.csv header mismatch: expected 32 canonical columns, got {len(diag_cols)}"
            )
        diag_rows = list(reader)

    if len(trace_rows) != len(diag_rows):
        raise ValueError(
            f"Row count mismatch: trace.txt has {len(trace_rows)} rows vs gaze_diag.csv {len(diag_rows)} rows"
        )

    aligned: List[Dict[str, Any]] = []
    fid_mismatches = 0
    ts_ms_mismatches = 0
    cap_s_mismatches = 0
    state_mismatches = 0
    max_cap_s_diff = 0.0
    max_dx = 0.0
    max_dy = 0.0

    for tr, dg in zip(trace_rows, diag_rows):
        t_fid = int(tr["frame_id"])
        d_fid = int(dg["frame_id"])
        t_obs_ms = int(tr["observation_ms"])
        d_ts_ms = int(dg["ts_ms"])
        t_cap_s = float(tr["dsp_timestamp_seconds"])
        d_cap_s = float(dg["capture_s"])
        t_state = int(tr["state"])
        d_state = int(dg["state"])
        t_tx = float(tr["tracker_x"])
        t_ty = float(tr["tracker_y"])
        d_nx = float(dg["norm_x"])
        d_ny = float(dg["norm_y"])

        if t_fid != d_fid:
            fid_mismatches += 1
        if t_obs_ms != d_ts_ms:
            ts_ms_mismatches += 1
        cap_diff = abs(t_cap_s - d_cap_s)
        max_cap_s_diff = max(max_cap_s_diff, cap_diff)
        if cap_diff > 1e-12:
            cap_s_mismatches += 1
        if t_state != d_state:
            state_mismatches += 1
        dx = abs(t_tx - d_nx)
        dy = abs(t_ty - d_ny)
        max_dx = max(max_dx, dx)
        max_dy = max(max_dy, dy)

        raw_x = float(tr["raw_x"]) if tr["raw_x"] != "" else None
        raw_y = float(tr["raw_y"]) if tr["raw_y"] != "" else None
        filt_x = float(tr["filtered_x"]) if tr["filtered_x"] != "" else None
        filt_y = float(tr["filtered_y"]) if tr["filtered_y"] != "" else None
        landmarks = int(tr["landmarks"])
        detected = int(dg["detected"])
        published = int(tr["published"])
        eligible = (
            detected == 1
            and d_state == 0
            and landmarks == 484
            and published == 1
            and raw_x is not None
            and raw_y is not None
            and math.isfinite(raw_x)
            and math.isfinite(raw_y)
        )

        aligned.append(
            {
                "frame_id": t_fid,
                "ts_ms": t_obs_ms,
                "capture_s": t_cap_s,
                "diag_capture_s": d_cap_s,
                "landmarks": landmarks,
                "detected": detected,
                "state": d_state,
                "published": published,
                "status": tr["status"],
                "raw_x": raw_x,
                "raw_y": raw_y,
                "filtered_x": filt_x,
                "filtered_y": filt_y,
                "sx": float(dg["sx"]),
                "sy": float(dg["sy"]),
                "head_pitch_deg": float(dg["head_pitch_deg"]),
                "head_yaw_deg": float(dg["head_yaw_deg"]),
                "head_roll_deg": float(dg["head_roll_deg"]),
                "rel_pitch_deg": float(dg["rel_pitch_deg"]),
                "rel_yaw_deg": float(dg["rel_yaw_deg"]),
                "abs_pitch_deg": float(dg["abs_pitch_deg"]),
                "abs_yaw_deg": float(dg["abs_yaw_deg"]),
                "norm_x": d_nx,
                "norm_y": d_ny,
                "eligible": eligible,
            }
        )

    detected_rows = [r for r in aligned if r["detected"] == 1]
    if detected_rows:
        ap_arr = np.array([r["abs_pitch_deg"] for r in detected_rows], dtype=np.float64)
        ny_arr = np.array([r["norm_y"] for r in detected_rows], dtype=np.float64)
        err_fixed = float(np.max(np.abs(_norm_coord_y(ap_arr, FIXED_Y_SIGN) - ny_arr)))
        err_legacy = float(np.max(np.abs(_norm_coord_y(ap_arr, LEGACY_Y_SIGN) - ny_arr)))
        if err_fixed <= err_legacy:
            logged_y_sign = "fixed"
            logged_y_sign_mult = FIXED_Y_SIGN
        else:
            logged_y_sign = "legacy"
            logged_y_sign_mult = LEGACY_Y_SIGN
    else:
        err_fixed = err_legacy = 0.0
        logged_y_sign = "fixed"
        logged_y_sign_mult = FIXED_Y_SIGN

    integrity = {
        "run_dir": str(run_dir),
        "diag_csv": str(diag_csv),
        "complete_marker_present": complete_marker.is_file(),
        "incomplete_marker_absent": not incomplete_marker.exists(),
        "trace_header": header_meta,
        "trace_footer": footer_meta,
        "overflow": int(footer_meta.get("overflow", -1)),
        "config_changes": int(footer_meta.get("config_changes", -1)),
        "complete_flag": int(footer_meta.get("complete", 0)),
        "trace_rows": len(trace_rows),
        "diag_rows": len(diag_rows),
        "diag_columns": len(diag_cols),
        "frame_id_mismatches": fid_mismatches,
        "ts_ms_mismatches": ts_ms_mismatches,
        "capture_s_mismatches": cap_s_mismatches,
        "state_mismatches": state_mismatches,
        "max_abs_capture_s_diff": max_cap_s_diff,
        "max_abs_delta_norm_x": max_dx,
        "max_abs_delta_norm_y": max_dy,
        "logged_y_sign": logged_y_sign,
        "logged_y_sign_mult": logged_y_sign_mult,
        "y_sign_err_fixed": err_fixed,
        "y_sign_err_legacy": err_legacy,
        "alignment_ok": bool(
            complete_marker.is_file()
            and not incomplete_marker.exists()
            and int(footer_meta.get("overflow", -1)) == 0
            and int(footer_meta.get("config_changes", -1)) == 0
            and int(footer_meta.get("complete", 0)) == 1
            and fid_mismatches == 0
            and cap_s_mismatches == 0
            and state_mismatches == 0
        ),
    }
    return integrity, aligned


def _local_sensitivity_stats(
    yaw_pred: tl2cgen.Predictor,
    pitch_pred: tl2cgen.Predictor,
    X: np.ndarray,
    base_ry: np.ndarray,
    base_rp: np.ndarray,
    delta: float = 0.005,
) -> Dict[str, Any]:
    if len(X) == 0:
        return {}
    out: Dict[str, Any] = {"delta": delta}
    for col_idx, feat_name, pred, base_deg in (
        (0, "sx", yaw_pred, base_ry),
        (1, "sy", pitch_pred, base_rp),
    ):
        Xp = X.copy()
        Xp[:, col_idx] += delta
        Xm = X.copy()
        Xm[:, col_idx] -= delta
        dm_p = tl2cgen.DMatrix(np.ascontiguousarray(Xp, dtype=np.float32))
        dm_m = tl2cgen.DMatrix(np.ascontiguousarray(Xm, dtype=np.float32))
        pred_p = np.asarray(pred.predict(dm_p), dtype=np.float64).reshape(-1)
        pred_m = np.asarray(pred.predict(dm_m), dtype=np.float64).reshape(-1)
        diffs_deg = np.maximum(np.abs(pred_p - base_deg), np.abs(pred_m - base_deg))
        out[feat_name] = {
            "max_abs_deg": float(np.max(diffs_deg)),
            "p95_abs_deg": float(np.percentile(diffs_deg, 95)),
            "mean_abs_deg": float(np.mean(diffs_deg)),
            "max_abs_norm": float(np.max(diffs_deg) / SPAN_DEG),
        }
    return out


def _top_jumps(
    win_rows: Sequence[Dict[str, Any]],
    axis: str,
    top_k: int = 5,
    y_sign_mult: float = FIXED_Y_SIGN,
) -> List[Dict[str, Any]]:
    if len(win_rows) < 2:
        return []
    pairs = []
    for i in range(1, len(win_rows)):
        p, c = win_rows[i - 1], win_rows[i]
        if axis == "x":
            val_p = p["norm_x"]
            val_c = c["norm_x"]
        else:
            val_p = float(_norm_coord_y(np.array([p["abs_pitch_deg"]]), y_sign_mult)[0])
            val_c = float(_norm_coord_y(np.array([c["abs_pitch_deg"]]), y_sign_mult)[0])
        dn = val_c - val_p
        dsx = c["sx"] - p["sx"]
        dsy = c["sy"] - p["sy"]
        dry = c["rel_yaw_deg"] - p["rel_yaw_deg"]
        drp = c["rel_pitch_deg"] - p["rel_pitch_deg"]
        gain_yaw = abs(dry / dsx) if abs(dsx) > 1e-7 else (float("inf") if abs(dry) > 1e-6 else 0.0)
        gain_pitch = abs(drp / dsy) if abs(dsy) > 1e-7 else (float("inf") if abs(drp) > 1e-6 else 0.0)
        pairs.append(
            (
                abs(dn),
                {
                    "prev_frame_id": p["frame_id"],
                    "curr_frame_id": c["frame_id"],
                    "dt_ms": (c["capture_s"] - p["capture_s"]) * 1000.0,
                    "norm_prev": val_p,
                    "norm_curr": val_c,
                    "delta_norm": dn,
                    "delta_sx": dsx,
                    "delta_sy": dsy,
                    "delta_head_yaw": c["head_yaw_deg"] - p["head_yaw_deg"],
                    "delta_head_pitch": c["head_pitch_deg"] - p["head_pitch_deg"],
                    "delta_rel_yaw": dry,
                    "delta_rel_pitch": drp,
                    "delta_abs_yaw": c["abs_yaw_deg"] - p["abs_yaw_deg"],
                    "delta_abs_pitch": c["abs_pitch_deg"] - p["abs_pitch_deg"],
                    "gain_rel_yaw_per_sx": None if not math.isfinite(gain_yaw) else gain_yaw,
                    "gain_rel_pitch_per_sy": None if not math.isfinite(gain_pitch) else gain_pitch,
                },
            )
        )
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in pairs[:top_k]]


def analyze_window(
    name: str,
    start_s: float,
    end_s: float,
    all_rows: Sequence[Dict[str, Any]],
    dep_yaw: tl2cgen.Predictor,
    dep_pitch: tl2cgen.Predictor,
    cand_yaw: Optional[tl2cgen.Predictor] = None,
    cand_pitch: Optional[tl2cgen.Predictor] = None,
    *,
    logged_y_sign_mult: float = FIXED_Y_SIGN,
    wash_in_s: float = 0.0,
    blink_pad_frames: int = DEFAULT_BLINK_PAD_FRAMES,
    padded_bad_indices: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    if padded_bad_indices is None:
        padded_bad_indices = compute_blink_pad_indices(all_rows, blink_pad_frames)

    eff_start_s = start_s + wash_in_s
    if eff_start_s >= end_s:
        raise ValueError(f"Window {name} has empty effective span after wash_in_s={wash_in_s}: [{eff_start_s}, {end_s})")

    in_raw_win = [r for r in all_rows if start_s <= r["capture_s"] < end_s]
    in_win_indexed = [(i, r) for i, r in enumerate(all_rows) if eff_start_s <= r["capture_s"] < end_s]
    wash_in_excluded = len(in_raw_win) - len(in_win_indexed)

    exclusions = {
        "wash_in": wash_in_excluded,
        "calibration": 0,
        "lost_or_unusable": 0,
        "nonneutral": 0,
        "blink_pad": 0,
        "unpublished": 0,
    }
    eligible: List[Dict[str, Any]] = []
    for idx, r in in_win_indexed:
        if r["state"] == -1:
            exclusions["calibration"] += 1
        elif r["detected"] != 1 or r["landmarks"] != 484:
            exclusions["lost_or_unusable"] += 1
        elif r["state"] != 0:
            exclusions["nonneutral"] += 1
        elif idx in padded_bad_indices:
            exclusions["blink_pad"] += 1
        elif r["published"] != 1 or r["raw_x"] is None or r["raw_y"] is None:
            exclusions["unpublished"] += 1
        else:
            eligible.append(r)

    if not eligible:
        return {
            "window": name,
            "raw_interval_s": [start_s, end_s],
            "interval_s": [eff_start_s, end_s],
            "wash_in_s": wash_in_s,
            "blink_pad_frames": blink_pad_frames,
            "total_frames_in_window": len(in_win_indexed),
            "eligible_frames": 0,
            "exclusions": exclusions,
            "raw_x": _compute_axis_metrics(np.empty(0), np.empty(0)),
            "raw_y": _compute_axis_metrics(np.empty(0), np.empty(0)),
        }

    times = np.array([r["capture_s"] for r in eligible], dtype=np.float64)
    raw_x = np.array([r["raw_x"] for r in eligible], dtype=np.float64)
    raw_y = np.array([r["raw_y"] for r in eligible], dtype=np.float64)
    filt_x = np.array([r["filtered_x"] for r in eligible], dtype=np.float64)
    filt_y = np.array([r["filtered_y"] for r in eligible], dtype=np.float64)
    logged_ap = np.array([r["abs_pitch_deg"] for r in eligible], dtype=np.float64)
    raw_y_fixed_sign = _norm_coord_y(logged_ap, FIXED_Y_SIGN)

    X = np.array(
        [
            [r["sx"], r["sy"], r["head_pitch_deg"], r["head_yaw_deg"], r["head_roll_deg"]]
            for r in eligible
        ],
        dtype=np.float32,
    )
    med = np.median(X.astype(np.float64), axis=0).astype(np.float32)

    # Replay deployed DLLs to confirm exact 5-D reconstruction against recorded trace
    rep_ry, rep_rp = _predict_angles(dep_yaw, dep_pitch, X)
    logged_ry = np.array([r["rel_yaw_deg"] for r in eligible], dtype=np.float64)
    logged_rp = np.array([r["rel_pitch_deg"] for r in eligible], dtype=np.float64)
    hy = X[:, 3].astype(np.float64)
    hp = X[:, 2].astype(np.float64)
    rep_ax = _norm_coord_x(rep_ry + hy)
    rep_ay = _norm_coord_y(rep_rp + hp, logged_y_sign_mult)

    # Counterfactuals use canonical fixed Y sign (0.5 - pitch / 60.0)
    # Counterfactual 1: Iris-only (vary sx, sy; hold head_pitch, head_yaw, head_roll at window median)
    X_iris = X.copy()
    X_iris[:, 2] = med[2]
    X_iris[:, 3] = med[3]
    X_iris[:, 4] = med[4]
    ry_iris, rp_iris = _predict_angles(dep_yaw, dep_pitch, X_iris)
    nx_iris = _norm_coord_x(ry_iris + float(med[3]))
    ny_iris = _norm_coord_y(rp_iris + float(med[2]), FIXED_Y_SIGN)

    # Counterfactual 2: Head-pose total (vary head_*; hold sx, sy at window median)
    X_head = X.copy()
    X_head[:, 0] = med[0]
    X_head[:, 1] = med[1]
    ry_head_in, rp_head_in = _predict_angles(dep_yaw, dep_pitch, X_head)
    nx_head_tot = _norm_coord_x(ry_head_in + hy)
    ny_head_tot = _norm_coord_y(rp_head_in + hp, FIXED_Y_SIGN)

    # Direct additive head angle only (hold rel_* at median prediction)
    rel_med_ry, rel_med_rp = _predict_angles(dep_yaw, dep_pitch, med.reshape(1, 5))
    nx_head_add = _norm_coord_x(float(rel_med_ry[0]) + hy)
    ny_head_add = _norm_coord_y(float(rel_med_rp[0]) + hp, FIXED_Y_SIGN)

    # Head pose entering regressor only (hold additive head angle at median)
    nx_head_tree = _norm_coord_x(ry_head_in + float(med[3]))
    ny_head_tree = _norm_coord_y(rp_head_in + float(med[2]), FIXED_Y_SIGN)

    raw_x_stats = _compute_axis_metrics(raw_x, times)
    raw_y_stats = _compute_axis_metrics(raw_y, times)
    raw_y_fixed_stats = _compute_axis_metrics(raw_y_fixed_sign, times)
    var_x = (raw_x_stats["pop_sd"] or 0.0) ** 2
    var_y = (raw_y_fixed_stats["pop_sd"] or 0.0) ** 2

    def _cf_summary(arr_x: np.ndarray, arr_y: np.ndarray) -> Dict[str, Any]:
        mx = _compute_axis_metrics(arr_x, times)
        my = _compute_axis_metrics(arr_y, times)
        return {
            "x_sd": mx["pop_sd"],
            "x_range": mx["range"],
            "x_var_share_pct": (100.0 * (mx["pop_sd"] ** 2) / var_x) if var_x > 0 else 0.0,
            "y_sd": my["pop_sd"],
            "y_range": my["range"],
            "y_var_share_pct": (100.0 * (my["pop_sd"] ** 2) / var_y) if var_y > 0 else 0.0,
        }

    feature_summary = {}
    for k in ("sx", "sy", "head_yaw_deg", "head_pitch_deg", "head_roll_deg", "rel_yaw_deg", "rel_pitch_deg", "abs_yaw_deg", "abs_pitch_deg"):
        vals = np.array([r[k] for r in eligible], dtype=np.float64)
        feature_summary[k] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "pop_sd": float(np.std(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        }

    res: Dict[str, Any] = {
        "window": name,
        "raw_interval_s": [start_s, end_s],
        "interval_s": [eff_start_s, end_s],
        "wash_in_s": wash_in_s,
        "blink_pad_frames": blink_pad_frames,
        "total_frames_in_window": len(in_win_indexed),
        "eligible_frames": len(eligible),
        "max_eligible_gap_ms": float(np.max(np.diff(times) * 1000.0)) if len(times) >= 2 else 0.0,
        "exclusions": exclusions,
        "raw_x": raw_x_stats,
        "raw_y": raw_y_stats,
        "raw_y_fixed_sign": raw_y_fixed_stats,
        "filtered_x": _compute_axis_metrics(filt_x, times),
        "filtered_y": _compute_axis_metrics(filt_y, times),
        "feature_summary": feature_summary,
        "deployed_replay_verification": {
            "logged_y_sign_mult": logged_y_sign_mult,
            "max_abs_err_rel_yaw_deg": float(np.max(np.abs(rep_ry - logged_ry))),
            "max_abs_err_rel_pitch_deg": float(np.max(np.abs(rep_rp - logged_rp))),
            "max_abs_err_norm_x": float(np.max(np.abs(rep_ax - raw_x))),
            "max_abs_err_norm_y": float(np.max(np.abs(rep_ay - raw_y))),
        },
        "variance_decomposition": {
            "iris_only_vs_median_head": _cf_summary(nx_iris, ny_iris),
            "head_total_vs_median_iris": _cf_summary(nx_head_tot, ny_head_tot),
            "head_direct_additive_only": _cf_summary(nx_head_add, ny_head_add),
            "head_regressor_input_only": _cf_summary(nx_head_tree, ny_head_tree),
        },
        "local_cliff_sensitivity_pm005": _local_sensitivity_stats(
            dep_yaw, dep_pitch, X, rep_ry, rep_rp, delta=0.005
        ),
        "top_5_jumps_x": _top_jumps(eligible, "x", top_k=5),
        "top_5_jumps_y": _top_jumps(eligible, "y", top_k=5, y_sign_mult=FIXED_Y_SIGN),
    }

    if cand_yaw is not None and cand_pitch is not None:
        c_ry, c_rp = _predict_angles(cand_yaw, cand_pitch, X)
        c_ayaw = c_ry + hy
        c_apitch = c_rp + hp
        c_nx = _norm_coord_x(c_ayaw)
        c_ny = _norm_coord_y(c_apitch, FIXED_Y_SIGN)
        res["candidate_shadow_replay"] = {
            "raw_x": _compute_axis_metrics(c_nx, times),
            "raw_y": _compute_axis_metrics(c_ny, times),
            "rel_yaw_mean_deg": float(np.mean(c_ry)),
            "rel_pitch_mean_deg": float(np.mean(c_rp)),
            "abs_yaw_mean_deg": float(np.mean(c_ayaw)),
            "abs_pitch_mean_deg": float(np.mean(c_apitch)),
            "local_cliff_sensitivity_pm005": _local_sensitivity_stats(
                cand_yaw, cand_pitch, X, c_ry, c_rp, delta=0.005
            ),
        }
    return res


def compute_direction_checks(windows_res: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {w["window"].upper(): w for w in windows_res if w.get("eligible_frames", 0) > 0}
    checks: List[Dict[str, Any]] = []

    def _find_match(substr: str) -> Optional[Dict[str, Any]]:
        for k, v in by_name.items():
            if substr in k:
                return v
        return None

    def _pair_summary(label: str, w_a: Dict[str, Any], w_b: Dict[str, Any]) -> Dict[str, Any]:
        dx_mean = w_b["raw_x"]["mean"] - w_a["raw_x"]["mean"]
        dx_med = w_b["raw_x"]["median"] - w_a["raw_x"]["median"]
        dy_mean = w_b["raw_y"]["mean"] - w_a["raw_y"]["mean"]
        dy_med = w_b["raw_y"]["median"] - w_a["raw_y"]["median"]
        ry_fix_a = w_a.get("raw_y_fixed_sign", w_a["raw_y"])
        ry_fix_b = w_b.get("raw_y_fixed_sign", w_b["raw_y"])
        dy_fix_mean = ry_fix_b["mean"] - ry_fix_a["mean"]
        fa, fb = w_a["feature_summary"], w_b["feature_summary"]
        entry: Dict[str, Any] = {
            "comparison": label,
            "window_a": w_a["window"],
            "window_b": w_b["window"],
            "delta_raw_x_mean": dx_mean,
            "delta_raw_x_median": dx_med,
            "delta_raw_y_mean": dy_mean,
            "delta_raw_y_median": dy_med,
            "delta_raw_y_fixed_sign_mean": dy_fix_mean,
            "delta_sx_mean": fb["sx"]["mean"] - fa["sx"]["mean"],
            "delta_sy_mean": fb["sy"]["mean"] - fa["sy"]["mean"],
            "delta_head_yaw_mean": fb["head_yaw_deg"]["mean"] - fa["head_yaw_deg"]["mean"],
            "delta_head_pitch_mean": fb["head_pitch_deg"]["mean"] - fa["head_pitch_deg"]["mean"],
            "delta_rel_yaw_mean": fb["rel_yaw_deg"]["mean"] - fa["rel_yaw_deg"]["mean"],
            "delta_rel_pitch_mean": fb["rel_pitch_deg"]["mean"] - fa["rel_pitch_deg"]["mean"],
            "delta_abs_yaw_mean": fb["abs_yaw_deg"]["mean"] - fa["abs_yaw_deg"]["mean"],
            "delta_abs_pitch_mean": fb["abs_pitch_deg"]["mean"] - fa["abs_pitch_deg"]["mean"],
        }
        if "candidate_shadow_replay" in w_a and "candidate_shadow_replay" in w_b:
            ca, cb = w_a["candidate_shadow_replay"], w_b["candidate_shadow_replay"]
            entry["candidate_delta_raw_x_mean"] = cb["raw_x"]["mean"] - ca["raw_x"]["mean"]
            entry["candidate_delta_raw_y_mean"] = cb["raw_y"]["mean"] - ca["raw_y"]["mean"]
            entry["candidate_delta_rel_yaw_mean"] = cb["rel_yaw_mean_deg"] - ca["rel_yaw_mean_deg"]
            entry["candidate_delta_rel_pitch_mean"] = cb["rel_pitch_mean_deg"] - ca["rel_pitch_mean_deg"]
        return entry

    left_w, right_w = _find_match("LEFT"), _find_match("RIGHT")
    up_w, down_w = _find_match("UP"), _find_match("DOWN")
    if left_w and right_w and left_w is not right_w:
        checks.append(_pair_summary("HORIZONTAL_LEFT_TO_RIGHT", left_w, right_w))
    if up_w and down_w and up_w is not down_w:
        checks.append(_pair_summary("VERTICAL_UP_TO_DOWN", up_w, down_w))
    valid_wins = [w for w in windows_res if w.get("eligible_frames", 0) > 0]
    if not checks and len(valid_wins) >= 2:
        checks.append(_pair_summary(f"{valid_wins[0]['window']}_TO_{valid_wins[-1]['window']}", valid_wins[0], valid_wins[-1]))
    return checks


def format_markdown_report(report: Dict[str, Any]) -> str:
    integ = report["integrity"]
    artifact_dir = integ.get("artifact_dir") or integ["run_dir"]
    dsp_dir = integ.get("dsp_record_dir") or integ["run_dir"]
    run_line = f"- **Run directory:** `{artifact_dir}`"
    if dsp_dir != artifact_dir:
        run_line += f" | **DSP recorder dir:** `{dsp_dir}`"
    lines = [
        "# Gaze Fixation Stability, Attribution & A/B Shadow Replay Report",
        "",
        "## 1. Alignment & Integrity",
        run_line,
        f"- **Diag CSV:** `{integ['diag_csv']}` (`{integ['diag_rows']}` rows, `{integ['diag_columns']}` columns)",
        f"- **Recorder markers:** `COMPLETE={integ['complete_marker_present']}`, `INCOMPLETE_absent={integ['incomplete_marker_absent']}`, `overflow={integ['overflow']}`, `config_changes={integ['config_changes']}`, `complete={integ['complete_flag']}`",
        f"- **DSP config (E4REC BEGIN):** `min_cutoff={integ['trace_header'].get('min_cutoff')}`, `beta={integ['trace_header'].get('beta')}`, `derivative_cutoff={integ['trace_header'].get('derivative_cutoff')}`",
        f"- **1:1 Alignment:** `frame_id_mismatches={integ['frame_id_mismatches']}`, `capture_s_mismatches={integ['capture_s_mismatches']}` (`max_diff={integ['max_abs_capture_s_diff']:.3e} s`), `state_mismatches={integ['state_mismatches']}`, `max_delta_norm=({integ['max_abs_delta_norm_x']:.3e}, {integ['max_abs_delta_norm_y']:.3e})`, `logged_y_sign={integ['logged_y_sign']}`",
        "",
        "### Model Inventory",
        "| Role | File | Bytes | SHA-256 |",
        "| --- | --- | ---: | --- |",
    ]
    for role, inv_map in (("Deployed", report["deployed_models"]), ("Candidate", report.get("candidate_models") or {})):
        for name, info in inv_map.items():
            lines.append(f"| {role} | `{name}` | {info.get('bytes')} | `{info.get('sha256')}` |")

    lines += ["", "## 2. Raw Fixation Stability & Attribution by Window"]
    for w in report["windows"]:
        lines.append(
            f"### Window `{w['window']}` ([{w['interval_s'][0]:.3f}s, {w['interval_s'][1]:.3f}s), "
            f"wash_in={w.get('wash_in_s', 0.0):.2f}s, blink_pad=+-{w.get('blink_pad_frames', 0)}f)"
        )
        lines.append(f"- Frames in window: `{w['total_frames_in_window']}`, Eligible (`state=0, detected=1`): `{w['eligible_frames']}`, Exclusions: `{w['exclusions']}`")
        if w["eligible_frames"] == 0:
            lines.append("- **No eligible frames in window.**")
            continue
        rx, ry = w["raw_x"], w["raw_y"]
        lines.extend([
            "",
            "| Axis | N (>=25) | Mean | SD (<=0.010) | Range (<=0.030) | |Slope|/s (<=0.005) | Max Step | Clipped 0/1 | Verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| `raw_x` | {rx['n']} | {rx['mean']:.5f} | {rx['pop_sd']:.6f} | {rx['range']:.6f} | {abs(rx['ols_slope_per_s']):.6f} | {rx['max_abs_step']:.6f} | {rx['clipped_zero_or_one']} | **{'PASS' if rx['pass_all'] else 'FAIL'}** |",
            f"| `raw_y` | {ry['n']} | {ry['mean']:.5f} | {ry['pop_sd']:.6f} | {ry['range']:.6f} | {abs(ry['ols_slope_per_s']):.6f} | {ry['max_abs_step']:.6f} | {ry['clipped_zero_or_one']} | **{'PASS' if ry['pass_all'] else 'FAIL'}** |",
        ])
        vd = w["variance_decomposition"]
        lines.extend([
            "",
            "**Counterfactual Variance Decomposition (holding opposite group at median):**",
            f"- **Iris-only (`sx/sy` vary):** `X SD={vd['iris_only_vs_median_head']['x_sd']:.6f}` (`{vd['iris_only_vs_median_head']['x_var_share_pct']:.1f}%` var), `Y SD={vd['iris_only_vs_median_head']['y_sd']:.6f}` (`{vd['iris_only_vs_median_head']['y_var_share_pct']:.1f}%` var)",
            f"- **Head-pose total (`head_*` vary):** `X SD={vd['head_total_vs_median_iris']['x_sd']:.6f}` (`{vd['head_total_vs_median_iris']['x_var_share_pct']:.1f}%` var), `Y SD={vd['head_total_vs_median_iris']['y_sd']:.6f}` (`{vd['head_total_vs_median_iris']['y_var_share_pct']:.1f}%` var)",
            f"- **Direct additive head angle only:** `X SD={vd['head_direct_additive_only']['x_sd']:.6f}`, `Y SD={vd['head_direct_additive_only']['y_sd']:.6f}`",
            f"- **Deployed local `±0.005` sensitivity:** `sx` max=`{w['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}°` (`p95={w['local_cliff_sensitivity_pm005']['sx']['p95_abs_deg']:.3f}°`), `sy` max=`{w['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}°`",
        ])
        if "candidate_shadow_replay" in w:
            cs = w["candidate_shadow_replay"]
            crx, cry = cs["raw_x"], cs["raw_y"]
            lines.extend([
                "",
                "**A/B Shadow Replay (Deployed vs. Candidate on identical 5-D frames; fixed Y sign `0.5 - pitch/60`):**",
                "| Model | Axis | Mean | SD | Range | |Slope|/s | Max Step | `±0.005` Max | Verdict |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| Deployed | `X` | {rx['mean']:.5f} | {rx['pop_sd']:.6f} | {rx['range']:.6f} | {abs(rx['ols_slope_per_s']):.6f} | {rx['max_abs_step']:.6f} | {w['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}° | **{'PASS' if rx['pass_all'] else 'FAIL'}** |",
                f"| Candidate | `X` | {crx['mean']:.5f} | {crx['pop_sd']:.6f} | {crx['range']:.6f} | {abs(crx['ols_slope_per_s']):.6f} | {crx['max_abs_step']:.6f} | {cs['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}° | **{'PASS' if crx['pass_all'] else 'FAIL'}** |",
                f"| Deployed (logged Y) | `Y` | {ry['mean']:.5f} | {ry['pop_sd']:.6f} | {ry['range']:.6f} | {abs(ry['ols_slope_per_s']):.6f} | {ry['max_abs_step']:.6f} | {w['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}° | **{'PASS' if ry['pass_all'] else 'FAIL'}** |",
                f"| Candidate (fixed Y) | `Y` | {cry['mean']:.5f} | {cry['pop_sd']:.6f} | {cry['range']:.6f} | {abs(cry['ols_slope_per_s']):.6f} | {cry['max_abs_step']:.6f} | {cs['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}° | **{'PASS' if cry['pass_all'] else 'FAIL'}** |",
            ])
        for axis_label, key in (("X (`norm_x`)", "top_5_jumps_x"), ("Y (`norm_y` fixed sign)", "top_5_jumps_y")):
            jumps = w.get(key) or []
            if jumps:
                lines.extend([
                    "",
                    f"**Top {len(jumps)} Consecutive Jumps — {axis_label}:**",
                    "| Frames | `dt (ms)` | `norm_prev -> norm_curr` | `Δnorm` | `Δsx` | `Δsy` | `Δhead_yaw` | `Δhead_pitch` | `Δrel_yaw` | `Δrel_pitch` | `|Δrel_yaw/Δsx|` |",
                    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ])
                for j in jumps:
                    gain = f"{j['gain_rel_yaw_per_sx']:.1f}" if j["gain_rel_yaw_per_sx"] is not None else "inf"
                    lines.append(
                        f"| {j['prev_frame_id']}->{j['curr_frame_id']} | {j['dt_ms']:.1f} | "
                        f"{j['norm_prev']:.5f}->{j['norm_curr']:.5f} | {j['delta_norm']:+.5f} | "
                        f"{j['delta_sx']:+.5f} | {j['delta_sy']:+.5f} | "
                        f"{j['delta_head_yaw']:+.3f}° | {j['delta_head_pitch']:+.3f}° | "
                        f"{j['delta_rel_yaw']:+.3f}° | {j['delta_rel_pitch']:+.3f}° | {gain} |"
                    )

    if report["directions"]:
        lines += ["", "## 3. Direction Verification"]
        for d in report["directions"]:
            lines.append(
                f"- **{d['comparison']}** (`{d['window_a']}` -> `{d['window_b']}`): "
                f"`Δraw_x={d['delta_raw_x_mean']:+.5f}`, `Δraw_y(logged)={d['delta_raw_y_mean']:+.5f}`, "
                f"`Δraw_y(fixed)={d['delta_raw_y_fixed_sign_mean']:+.5f}`, "
                f"`Δsx={d['delta_sx_mean']:+.5f}`, `Δsy={d['delta_sy_mean']:+.5f}`, "
                f"`Δrel_yaw={d['delta_rel_yaw_mean']:+.3f}°`, `Δrel_pitch={d['delta_rel_pitch_mean']:+.3f}°`"
            )
            if "candidate_delta_raw_x_mean" in d:
                lines.append(
                    f"  - Candidate shadow: `Δraw_x={d['candidate_delta_raw_x_mean']:+.5f}`, "
                    f"`Δraw_y(fixed)={d['candidate_delta_raw_y_mean']:+.5f}`, "
                    f"`Δrel_yaw={d['candidate_delta_rel_yaw_mean']:+.3f}°`, "
                    f"`Δrel_pitch={d['candidate_delta_rel_pitch_mean']:+.3f}°`"
                )
    return "\n".join(lines) + "\n"


def _speak_cue_async(text: str, enabled: bool) -> None:
    if not enabled or os.name != "nt":
        return
    def _worker() -> None:
        try:
            ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            safe_txt = text.replace("'", "''")
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{safe_txt}'); $s.Dispose()"
            )
            subprocess.run(
                [str(ps), "-NoProfile", "-NonInteractive", "-Command", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def run_cued_capture(
    *,
    exe: Path,
    run_dir: Path,
    diag_csv: Path,
    camera: int,
    targets: Sequence[str],
    hold_s: float,
    gap_s: float,
    audio: bool,
    yes: bool,
    readiness_timeout_s: float = 30.0,
    dsp_record_dir: Optional[Path] = None,
) -> Path:
    run_dir = run_dir.resolve()
    diag_csv = diag_csv.resolve()
    dsp_dir = (dsp_record_dir or (run_dir / "dsp_record")).resolve()

    if not exe.is_file():
        raise FileNotFoundError(f"Missing core_engine executable: {exe}")
    if (run_dir / "trace.txt").exists() or (run_dir / "COMPLETE").exists():
        raise RuntimeError(f"Refusing to overwrite existing recording in {run_dir}")
    if dsp_dir.exists():
        raise RuntimeError(
            f"Recorder output directory already exists (use a fresh --run-dir or --dsp-record-dir): {dsp_dir}"
        )

    plan = build_cued_target_plan(targets=targets, hold_s=hold_s, gap_s=gap_s, initial_settle_s=1.0)
    total_target_s = plan[-1]["offset_from_ready_end_s"] + 2.0
    seconds = int(math.ceil(total_target_s + min(readiness_timeout_s, 15.0)))
    if total_target_s > 65.0:
        print(
            f"[WARNING] Planned cued duration (~{total_target_s:.1f}s) may approach the 4096-frame "
            "DSP recorder capacity at 60 FPS.",
            flush=True,
        )
    cmd = build_gaze_capture_command(exe, dsp_dir, diag_csv, camera, seconds)

    print(f"This command WILL open camera {camera} in safe --os log --no-hand --filter dsp mode (NO mouse control).")
    print("Command:", " ".join(cmd))
    if not yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Cued capture requires interactive confirmation or --yes")
        if input("Type RECORD to start cued gaze capture: ").strip() != "RECORD":
            raise RuntimeError("Cued capture cancelled by user.")

    # core_engine's OutputSession requires dsp_dir NOT to exist yet, while its parent must exist.
    run_dir.mkdir(parents=True, exist_ok=True)
    dsp_dir.parent.mkdir(parents=True, exist_ok=True)
    diag_parent = diag_csv.parent
    if diag_parent != dsp_dir and dsp_dir not in diag_parent.parents:
        diag_parent.mkdir(parents=True, exist_ok=True)

    clock = time.monotonic
    launch_mono = clock()
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(cmd, **popen_kwargs)

    inbox: "queue.Queue[Tuple[str, float]]" = queue.Queue()

    def _drain() -> None:
        if proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                inbox.put((raw_line.rstrip("\r\n"), clock()))
        except Exception:
            pass

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    gate = GazeReadinessGate()
    epoch_mono: Optional[float] = None
    next_idx = 0
    active_idx: Optional[int] = None
    raw_targets: List[Dict[str, Any]] = []
    status_samples: List[Tuple[int, float]] = []
    console_tail: "collections.deque[str]" = collections.deque(maxlen=200)
    all_targets_done_at: Optional[float] = None
    stop_sent = False

    def _handle_line(line: str, rx_time: float) -> None:
        nonlocal epoch_mono
        console_tail.append(line)
        if epoch_mono is None and line.startswith("OS: "):
            epoch_mono = rx_time
        m = GAZE_STATUS_RE.fullmatch(line.strip())
        if m:
            status_samples.append((int(m.group(7)), rx_time))
        if gate.feed(line, rx_time):
            if epoch_mono is None:
                epoch_mono = (gate.first_status_time - 1.0) if gate.first_status_time is not None else launch_mono
            print(f"[Cued Capture] Calibration ready at capture_s~={gate.calib_ready_time - epoch_mono:.2f}s", flush=True)
            _speak_cue_async("Calibration ready.", audio)

    def _drain_inbox() -> None:
        while True:
            try:
                line, rx_time = inbox.get_nowait()
            except queue.Empty:
                break
            _handle_line(line, rx_time)

    print("[Cued Capture] Hold steady neutral gaze for initial calibration...", flush=True)
    _speak_cue_async("Hold still facing center for calibration.", audio)

    try:
        while proc.poll() is None:
            now = clock()
            _drain_inbox()

            t_base_live = epoch_mono if epoch_mono is not None else launch_mono
            if gate.calib_ready_time is None:
                if now - launch_mono > readiness_timeout_s:
                    raise RuntimeError("Timed out waiting for live calibrated gaze readiness (state=0)")
            else:
                rel_ready = now - gate.calib_ready_time
                if active_idx is None and next_idx < len(plan):
                    item = plan[next_idx]
                    if rel_ready >= item["offset_from_ready_start_s"]:
                        active_idx = next_idx
                        t_start_s = now - t_base_live
                        msg = f"Look at {item['name']} and hold still."
                        print(f"[CUE START t={t_start_s:.2f}s] {msg}", flush=True)
                        _speak_cue_async(f"Look {item['name'].lower()}.", audio)
                        raw_targets.append({
                            "name": item["name"],
                            "start_mono": now,
                            "end_mono": None,
                        })
                elif active_idx is not None:
                    item = plan[active_idx]
                    if rel_ready >= item["offset_from_ready_end_s"]:
                        t_end_s = now - t_base_live
                        raw_targets[active_idx]["end_mono"] = now
                        print(f"[CUE END   t={t_end_s:.2f}s] Done {item['name']}.", flush=True)
                        active_idx = None
                        next_idx += 1
                        if next_idx >= len(plan):
                            all_targets_done_at = now
                            print("[Cued Capture] All targets complete; finishing capture cleanly...", flush=True)
                            _speak_cue_async("All targets complete.", audio)
                elif all_targets_done_at is not None and not stop_sent and (now - all_targets_done_at) >= 1.5:
                    stop_sent = True
                    sig = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT) if os.name == "nt" else signal.SIGINT
                    try:
                        if hasattr(proc, "send_signal"):
                            proc.send_signal(sig)
                    except Exception:
                        pass
            time.sleep(0.01)
    finally:
        if proc.poll() is None:
            if stop_sent:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=5)
            else:
                proc.terminate()
                proc.wait(timeout=5)
        else:
            proc.wait(timeout=5)
        drain_thread.join(timeout=5)
        _drain_inbox()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

    tail_list = list(console_tail)
    if proc.returncode != 0:
        raise RuntimeError(f"core_engine exited with code {proc.returncode}; tail={tail_list[-15:]}")
    if gate.calib_ready_time is None or len(raw_targets) != len(plan) or any(t["end_mono"] is None for t in raw_targets):
        raise RuntimeError(f"Cued capture ended before all targets completed; tail={tail_list[-15:]}")

    trace_times = _extract_trace_capture_times(dsp_dir / "trace.txt")
    t_base, sync_method = _resolve_capture_epoch_mono(status_samples, trace_times, epoch_mono, launch_mono)
    scheduled_targets = [
        {
            "name": t["name"],
            "target_start_s": round(float(t["start_mono"]) - t_base, 4),
            "target_end_s": round(float(t["end_mono"]) - t_base, 4),
        }
        for t in raw_targets
    ]
    schedule_payload = {
        "schema_version": 1,
        "command": cmd,
        "dsp_record_dir": str(dsp_dir),
        "diag_csv": str(diag_csv),
        "time_sync_method": sync_method,
        "gaze_filter": GAZE_FILTER_SPEC,
        "wash_in_s": DEFAULT_CUED_WASH_IN_S,
        "blink_pad_frames": DEFAULT_BLINK_PAD_FRAMES,
        "calib_ready_s": round(gate.calib_ready_time - t_base, 4),
        "targets": scheduled_targets,
    }
    sched_path = run_dir / "cued_schedule.json"
    sched_path.write_text(json.dumps(schedule_payload, indent=2), encoding="utf-8")
    return sched_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory (artifacts/reports, or legacy flat recorder dir containing trace.txt)")
    ap.add_argument("--dsp-record-dir", type=Path, default=None, help="Recorder directory for trace.txt/COMPLETE (default for --cued-capture: <run-dir>/dsp_record; offline auto-detects <run-dir> or <run-dir>/dsp_record)")
    ap.add_argument("--diag-csv", type=Path, default=None, help="Path to gaze_diag.csv (default: <run-dir>/gaze_diag.csv)")
    ap.add_argument("--windows", type=str, default=None, help="Comma-separated NAME:START_S-END_S intervals, e.g. 'LEFT:5-15,RIGHT:17-27'")
    ap.add_argument("--auto-windows", action="store_true", help="Automatically segment active tracking windows")
    ap.add_argument("--cued-schedule", type=Path, default=None, help="Path to cued_schedule.json (auto-detected in <run-dir> if present and --windows/--auto-windows omitted)")
    ap.add_argument("--wash-in-s", type=float, default=None, help="Seconds to trim from the start of each window (default: 0.5 with cued schedule, else 0.0)")
    ap.add_argument("--blink-pad-frames", type=int, default=DEFAULT_BLINK_PAD_FRAMES, help="Frames to exclude before/after blink or face loss (default: 2)")
    ap.add_argument("--deployed-models-dir", type=Path, default=DEFAULT_DEPLOYED_MODELS_DIR)
    ap.add_argument("--candidate-models-dir", type=Path, default=None, help="Optional directory containing candidate gaze_yaw.dll / gaze_pitch.dll")
    ap.add_argument("--out-json", type=Path, default=None, help="Output JSON report path (default: <run-dir>/gaze_attribution_report.json)")
    ap.add_argument("--out-md", type=Path, default=None, help="Output Markdown report path (default: <run-dir>/gaze_attribution_report.md)")
    # Cued live capture options
    ap.add_argument("--cued-capture", action="store_true", help="Launch safe cued core_engine.exe capture (--os log --no-hand --filter dsp) then analyze")
    ap.add_argument("--exe", type=Path, default=DEFAULT_ENGINE_EXE, help="Path to core_engine.exe for --cued-capture")
    ap.add_argument("--camera", type=int, default=0, help="Camera index for --cued-capture")
    ap.add_argument("--targets", type=str, default=",".join(DEFAULT_CUED_TARGETS), help="Comma-separated cued targets")
    ap.add_argument("--hold-s", type=float, default=10.0, help="Seconds per cued target (default: 10.0)")
    ap.add_argument("--gap-s", type=float, default=2.0, help="Transition gap seconds between targets (default: 2.0)")
    ap.add_argument("--no-audio", action="store_true", help="Disable spoken TTS cues during --cued-capture")
    ap.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompt for --cued-capture")
    args = ap.parse_args(argv)

    if args.windows is not None and args.auto_windows:
        ap.error("Specify either --windows or --auto-windows, not both.")
    if args.blink_pad_frames < 0:
        ap.error("--blink-pad-frames must be >= 0")

    run_dir = args.run_dir.resolve()
    out_json = (args.out_json or (run_dir / "gaze_attribution_report.json")).resolve()
    out_md = (args.out_md or (run_dir / "gaze_attribution_report.md")).resolve()

    cued_sched_path = args.cued_schedule.resolve() if args.cued_schedule else None
    capture_dsp_dir: Optional[Path] = args.dsp_record_dir.resolve() if args.dsp_record_dir else None
    capture_diag_csv = (args.diag_csv or (run_dir / "gaze_diag.csv")).resolve()
    if args.cued_capture:
        if capture_dsp_dir is None:
            capture_dsp_dir = run_dir / "dsp_record"
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        cued_sched_path = run_cued_capture(
            exe=args.exe.resolve(),
            run_dir=run_dir,
            dsp_record_dir=capture_dsp_dir,
            diag_csv=capture_diag_csv,
            camera=args.camera,
            targets=targets,
            hold_s=args.hold_s,
            gap_s=args.gap_s,
            audio=not args.no_audio,
            yes=args.yes,
        )

    resolved_dsp_dir, resolved_diag_csv, auto_sched_path = resolve_run_paths(
        run_dir,
        dsp_record_dir=capture_dsp_dir if args.cued_capture else args.dsp_record_dir,
        diag_csv=capture_diag_csv if args.cued_capture else args.diag_csv,
        cued_schedule=cued_sched_path,
    )
    if cued_sched_path is None and args.windows is None and not args.auto_windows:
        cued_sched_path = auto_sched_path

    integrity, rows = load_and_align_run(resolved_dsp_dir, resolved_diag_csv)
    integrity["artifact_dir"] = str(run_dir)
    integrity["dsp_record_dir"] = str(resolved_dsp_dir)
    dep_inv = _models_inventory(args.deployed_models_dir.resolve(), include_gesture=True)
    dep_yaw = tl2cgen.Predictor(str(args.deployed_models_dir.resolve() / "gaze_yaw.dll"))
    dep_pitch = tl2cgen.Predictor(str(args.deployed_models_dir.resolve() / "gaze_pitch.dll"))

    cand_inv = None
    cand_yaw = cand_pitch = None
    if args.candidate_models_dir is not None:
        cdir = args.candidate_models_dir.resolve()
        cand_inv = _models_inventory(cdir, include_gesture=False)
        for req in ("gaze_yaw.dll", "gaze_pitch.dll"):
            if not (cdir / req).is_file():
                raise FileNotFoundError(f"Missing candidate DLL in {cdir}: {req}")
        cand_yaw = tl2cgen.Predictor(str(cdir / "gaze_yaw.dll"))
        cand_pitch = tl2cgen.Predictor(str(cdir / "gaze_pitch.dll"))

    cued_meta = None
    if args.windows:
        windows_spec = parse_windows_spec(args.windows)
    elif cued_sched_path is not None and not args.auto_windows:
        cued_meta, windows_spec = load_cued_schedule(cued_sched_path)
    else:
        windows_spec = auto_detect_windows(rows)

    wash_in_s = args.wash_in_s if args.wash_in_s is not None else (
        DEFAULT_CUED_WASH_IN_S if cued_meta is not None else 0.0
    )
    padded_bad = compute_blink_pad_indices(rows, args.blink_pad_frames)

    windows_res = [
        analyze_window(
            name, s, e, rows, dep_yaw, dep_pitch, cand_yaw, cand_pitch,
            logged_y_sign_mult=integrity["logged_y_sign_mult"],
            wash_in_s=wash_in_s,
            blink_pad_frames=args.blink_pad_frames,
            padded_bad_indices=padded_bad,
        )
        for name, s, e in windows_spec
    ]
    directions = compute_direction_checks(windows_res)

    report = {
        "integrity": integrity,
        "deployed_models": dep_inv,
        "candidate_models": cand_inv,
        "cued_schedule": cued_meta,
        "analysis_params": {
            "wash_in_s": wash_in_s,
            "blink_pad_frames": args.blink_pad_frames,
            "fixed_y_sign": FIXED_Y_SIGN,
        },
        "limits": {
            "min_samples": MIN_SAMPLES,
            "max_pop_sd": LIMIT_SD,
            "max_range": LIMIT_RANGE,
            "max_abs_ols_slope_per_s": LIMIT_SLOPE,
        },
        "windows": windows_res,
        "directions": directions,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_text = format_markdown_report(report)
    out_md.write_text(md_text, encoding="utf-8")

    print(f"=== ALIGNMENT & INTEGRITY ===")
    print(
        f"rows={integrity['trace_rows']} COMPLETE={integrity['complete_marker_present']} "
        f"overflow={integrity['overflow']} config_changes={integrity['config_changes']} "
        f"fid_mis={integrity['frame_id_mismatches']} cap_s_mis={integrity['capture_s_mismatches']} "
        f"y_sign={integrity['logged_y_sign']} "
        f"max_dnorm=({integrity['max_abs_delta_norm_x']:.3e}, {integrity['max_abs_delta_norm_y']:.3e})"
    )
    for w in windows_res:
        if w["eligible_frames"] == 0:
            print(f"[{w['window']}] NO ELIGIBLE FRAMES (total={w['total_frames_in_window']})")
            continue
        rx, ry = w["raw_x"], w["raw_y"]
        vd = w["variance_decomposition"]
        print(
            f"[{w['window']}] n={rx['n']} | "
            f"raw_x: mean={rx['mean']:.5f} SD={rx['pop_sd']:.6f} rng={rx['range']:.6f} |slope|={abs(rx['ols_slope_per_s']):.6f} [{'PASS' if rx['pass_all'] else 'FAIL'}] | "
            f"raw_y: mean={ry['mean']:.5f} SD={ry['pop_sd']:.6f} rng={ry['range']:.6f} |slope|={abs(ry['ols_slope_per_s']):.6f} [{'PASS' if ry['pass_all'] else 'FAIL'}]"
        )
        print(
            f"    Attribution SD: iris_only=({vd['iris_only_vs_median_head']['x_sd']:.6f}, {vd['iris_only_vs_median_head']['y_sd']:.6f}) "
            f"head_total=({vd['head_total_vs_median_iris']['x_sd']:.6f}, {vd['head_total_vs_median_iris']['y_sd']:.6f}) "
            f"cliff_max(+-0.005)=({w['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}deg, {w['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}deg)"
        )
        if "candidate_shadow_replay" in w:
            cs = w["candidate_shadow_replay"]
            crx, cry = cs["raw_x"], cs["raw_y"]
            print(
                f"    Candidate A/B : "
                f"X: mean={crx['mean']:.5f} SD={crx['pop_sd']:.6f} rng={crx['range']:.6f} [{'PASS' if crx['pass_all'] else 'FAIL'}] | "
                f"Y: mean={cry['mean']:.5f} SD={cry['pop_sd']:.6f} rng={cry['range']:.6f} [{'PASS' if cry['pass_all'] else 'FAIL'}] | "
                f"cliff_max=({cs['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}deg, {cs['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}deg)"
            )
    for d in directions:
        line = (
            f"Direction {d['comparison']}: d_raw_x={d['delta_raw_x_mean']:+.5f}, "
            f"d_raw_y(logged)={d['delta_raw_y_mean']:+.5f}, d_raw_y(fixed)={d['delta_raw_y_fixed_sign_mean']:+.5f}, "
            f"d_sx={d['delta_sx_mean']:+.5f}, d_sy={d['delta_sy_mean']:+.5f}"
        )
        if "candidate_delta_raw_x_mean" in d:
            line += f" | cand_d_raw_x={d['candidate_delta_raw_x_mean']:+.5f}, cand_d_raw_y(fixed)={d['candidate_delta_raw_y_mean']:+.5f}"
        print(line)
    print(f"Reports written: {out_json} | {out_md}")
    return 0 if integrity["alignment_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
