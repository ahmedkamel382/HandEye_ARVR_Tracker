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
     guiding CENTER, LEFT, RIGHT, UP, DOWN fixations and logging `calib_ready_s`,
     per-target `target_start_s`/`target_end_s`, clock-sync diagnostics, and software SAPI
     dispatch telemetry to `<run-dir>/cued_schedule.json`.

Exit codes:
  0: Archive integrity and alignment OK (and qualification PASS when `--require-qualification-pass` is set).
  1: Archive integrity, CSV schema, or recorder alignment FAILED (`alignment_ok=False`).
  2: Archive integrity and alignment OK, but fixation qualification FAILED under `--require-qualification-pass`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
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
MAX_FRESH_GAP_MS = 100.0
COORD_TOL = 1e-6
DEFAULT_CUED_WASH_IN_S = 0.5
DEFAULT_BLINK_PAD_FRAMES = 2
DEFAULT_PRE_CUE_S = 2.5
LOW_DELIVERED_FPS_WARN_THRESHOLD = 25.0
DEFAULT_CUED_TARGETS = ("CENTER", "LEFT", "RIGHT", "UP", "DOWN")
MANDATORY_HORIZONTAL_TARGETS: Tuple[str, ...] = ("CENTER", "LEFT", "RIGHT")
MANDATORY_VERTICAL_TARGETS: Tuple[str, ...] = ("CENTER", "UP", "DOWN")
GAZE_FILTER_SPEC = "0.8,6,1.5"

EXIT_CODE_OK = 0
EXIT_CODE_INTEGRITY_FAILED = 1
EXIT_CODE_QUALIFICATION_FAILED = 2

AUDIO_TIMING_SCOPE = "software_sapi_dispatch_only_not_physical_acoustic_onset"
ALLOWED_DETECTED_VALUES: Tuple[int, ...] = (0, 1)
ALLOWED_STATE_VALUES: Tuple[int, ...] = (-1, 0, 1, 2, 3)

STRICT_INT_RE = re.compile(r"^[+-]?\d+$")
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

DIAG_INT_COLUMNS = ("frame_id", "ts_ms", "detected", "state")
DIAG_ALWAYS_FINITE_FLOAT_COLUMNS = ("capture_s", "norm_x", "norm_y")
DIAG_INACTIVE_NAN_ALLOWED_COLUMNS = (
    "left_iris_x", "right_iris_x", "sx", "left_iris_y", "right_iris_y", "sy",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "rel_pitch_deg", "rel_yaw_deg", "abs_pitch_deg", "abs_yaw_deg",
    "lm33_x", "lm33_y", "lm133_x", "lm133_y", "lm468_x", "lm468_y",
    "lm362_x", "lm362_y", "lm263_x", "lm263_y", "lm473_x", "lm473_y",
)
DIAG_FLOAT_COLUMNS = (
    "left_iris_x", "right_iris_x", "sx", "left_iris_y", "right_iris_y", "sy",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "rel_pitch_deg", "rel_yaw_deg", "abs_pitch_deg", "abs_yaw_deg",
    "norm_x", "norm_y",
    "lm33_x", "lm33_y", "lm133_x", "lm133_y", "lm468_x", "lm468_y",
    "lm362_x", "lm362_y", "lm263_x", "lm263_y", "lm473_x", "lm473_y",
)


def _parse_strict_int(raw: Any) -> Optional[int]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or not STRICT_INT_RE.fullmatch(s):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_strict_float(raw: Any) -> Optional[float]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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


def _detect_pitch_mode(pitch_pred: tl2cgen.Predictor) -> str:
    """Detect whether gaze_pitch.dll predicts rel_pitch (legacy) or direct abs_pitch (HGB).

    For rel_pitch models trained on `true_pitch - head_pitch`, increasing head_pitch from
    -20 deg to +20 deg while holding iris features fixed drives the prediction strongly downward
    (delta < -20 deg). Direct abs_pitch models output nearly constant world pitch (< 20 deg delta).
    """
    probe = np.array(
        [[0.5, 0.0, -20.0, 0.0, 0.0], [0.5, 0.0, 20.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    preds = np.asarray(pitch_pred.predict(tl2cgen.DMatrix(probe)), dtype=np.float64).reshape(-1)
    return "rel_pitch" if (float(preds[1] - preds[0]) < -20.0) else "abs_pitch"


def _predict_angles(
    yaw_pred: tl2cgen.Predictor,
    pitch_pred: tl2cgen.Predictor,
    features_5d: np.ndarray,
    pitch_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(features_5d) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if pitch_mode is None:
        pitch_mode = _detect_pitch_mode(pitch_pred)
    f32 = np.ascontiguousarray(features_5d, dtype=np.float32)
    dm = tl2cgen.DMatrix(f32)
    rel_yaw = np.asarray(yaw_pred.predict(dm), dtype=np.float64).reshape(-1)
    raw_pitch = np.asarray(pitch_pred.predict(dm), dtype=np.float64).reshape(-1)
    if pitch_mode == "abs_pitch":
        head_pitch = f32[:, 2].astype(np.float64)
        rel_pitch = raw_pitch - head_pitch
    else:
        rel_pitch = raw_pitch
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
            "max_gap_ms": None,
            "clipped_zero_or_one": 0,
            "pass_n": False,
            "pass_sd": False,
            "pass_range": False,
            "pass_slope": False,
            "pass_gap": False,
            "pass_all": False,
        }
    finite_ok = bool(np.all(np.isfinite(values)) and np.all(np.isfinite(times_s)))
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
        max_gap_ms = float(np.max(np.diff(times_s) * 1000.0))
    else:
        slope = 0.0
        max_step = 0.0
        max_gap_ms = 0.0

    clipped = int(np.sum((values <= 0.0) | (values >= 1.0)))
    pass_n = n >= MIN_SAMPLES
    pass_sd = finite_ok and (sd_v <= LIMIT_SD)
    pass_range = finite_ok and (rng_v <= LIMIT_RANGE)
    pass_slope = finite_ok and (abs(slope) <= LIMIT_SLOPE)
    pass_gap = finite_ok and (max_gap_ms <= MAX_FRESH_GAP_MS)
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
        "max_gap_ms": max_gap_ms,
        "clipped_zero_or_one": clipped,
        "pass_n": pass_n,
        "pass_sd": pass_sd,
        "pass_range": pass_range,
        "pass_slope": pass_slope,
        "pass_gap": pass_gap,
        "pass_all": bool(finite_ok and pass_n and pass_sd and pass_range and pass_slope and pass_gap),
    }


def parse_windows_spec(spec: str) -> List[Tuple[str, float, float]]:
    windows: List[Tuple[str, float, float]] = []
    seen: Set[str] = set()
    prev_end: Optional[float] = None
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
        if (
            not name
            or name in seen
            or not (math.isfinite(start_s) and math.isfinite(end_s))
            or not (0.0 <= start_s < end_s)
        ):
            raise ValueError(f"Invalid window bounds/name in {tok!r}")
        if prev_end is not None and start_s < prev_end:
            raise ValueError(
                f"Window {name!r} ([{start_s}, {end_s})) overlaps or precedes previous end {prev_end}"
            )
        seen.add(name)
        prev_end = end_s
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
    prev_end: Optional[float] = None
    for entry in targets:
        if not isinstance(entry, dict) or "name" not in entry or "target_start_s" not in entry or "target_end_s" not in entry:
            raise ValueError(f"Invalid cued target entry {entry!r} in {schedule_path}")
        name = str(entry["name"]).strip()
        start_s = float(entry["target_start_s"])
        end_s = float(entry["target_end_s"])
        if (
            not name
            or name in seen
            or not (math.isfinite(start_s) and math.isfinite(end_s))
            or not (0.0 <= start_s < end_s)
        ):
            raise ValueError(f"Invalid cued target entry {entry!r} in {schedule_path}")
        if prev_end is not None and start_s < prev_end:
            raise ValueError(
                f"Cued target {name!r} ([{start_s}, {end_s})) overlaps or precedes previous end {prev_end} in {schedule_path}"
            )
        seen.add(name)
        prev_end = end_s
        windows.append((name, start_s, end_s))
    return data, windows


def auto_detect_windows(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    if not records:
        return []
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


def _compute_cadence_fps(timestamps_s: Sequence[float]) -> Optional[float]:
    if len(timestamps_s) < 2:
        return None
    span_s = float(timestamps_s[-1]) - float(timestamps_s[0])
    if not math.isfinite(span_s) or span_s <= 0.0:
        return None
    return float(len(timestamps_s) - 1) / span_s


def build_cued_target_plan(
    targets: Sequence[str] = DEFAULT_CUED_TARGETS,
    hold_s: float = 10.0,
    gap_s: float = 2.0,
    initial_settle_s: float = 1.0,
    pre_cue_s: float = DEFAULT_PRE_CUE_S,
) -> List[Dict[str, Any]]:
    if hold_s <= DEFAULT_CUED_WASH_IN_S:
        raise ValueError("hold_s must exceed 0.5s wash-in")
    if gap_s < 0.0 or initial_settle_s < 0.0:
        raise ValueError("gap_s and initial_settle_s must be non-negative")
    if not math.isfinite(pre_cue_s) or pre_cue_s < 0.0:
        raise ValueError("pre_cue_s must be finite and non-negative")
    effective_gap_pre_cue = min(float(pre_cue_s), max(0.0, float(gap_s) - 0.5)) if pre_cue_s > 0.0 else 0.0
    plan: List[Dict[str, Any]] = []
    cursor = initial_settle_s
    for idx, raw_name in enumerate(targets):
        name = str(raw_name).strip().upper()
        if not name:
            raise ValueError("Empty target name in cued plan")
        start_rel = cursor
        end_rel = start_rel + hold_s
        if idx == 0:
            lead_s = (
                min(effective_gap_pre_cue, max(0.0, float(initial_settle_s) - 0.25))
                if (effective_gap_pre_cue > 0.0 and gap_s > 0.0 and initial_settle_s > 0.0)
                else 0.0
            )
        else:
            lead_s = effective_gap_pre_cue
        pre_cue_rel = (start_rel - lead_s) if lead_s > 0.0 else None
        plan.append({
            "index": idx,
            "name": name,
            "pre_cue_lead_s": lead_s,
            "offset_from_ready_pre_cue_s": pre_cue_rel,
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
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
) -> List[str]:
    cmd = [
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
    if width is not None:
        w = int(width)
        if w <= 0:
            raise ValueError(f"--width must be > 0, got {width}")
        cmd.extend(["--width", str(w)])
    if height is not None:
        h = int(height)
        if h <= 0:
            raise ValueError(f"--height must be > 0, got {height}")
        cmd.extend(["--height", str(h)])
    if fps is not None:
        fps_f = float(fps)
        if not math.isfinite(fps_f) or fps_f <= 0.0:
            raise ValueError(f"--fps must be finite and > 0, got {fps}")
        fps_str = str(int(fps_f)) if fps_f.is_integer() else f"{fps_f:g}"
        cmd.extend(["--fps", fps_str])
    return cmd


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


def _resolve_capture_epoch_with_diagnostics(
    status_samples: Sequence[Tuple[int, float]],
    trace_times: Sequence[float],
    epoch_mono: Optional[float],
    launch_mono: float,
    wall_origin_unix_s: Optional[float] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    raw_entries: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    cands: List[float] = []
    for idx, (os_frames, rx_t) in enumerate(status_samples):
        if 1 <= os_frames <= len(trace_times):
            tr_t = float(trace_times[os_frames - 1])
            cand_offset = float(rx_t) - tr_t
            valid_indices.append(idx)
            cands.append(cand_offset)
            raw_entries.append({
                "sample_index": idx,
                "os_frames": int(os_frames),
                "rx_mono_s": round(float(rx_t), 6),
                "rx_since_launch_s": round(float(rx_t) - float(launch_mono), 6),
                "trace_capture_s": round(tr_t, 6),
                "candidate_offset_mono_s": round(cand_offset, 6),
                "inlier": False,
            })
        else:
            raw_entries.append({
                "sample_index": idx,
                "os_frames": int(os_frames),
                "rx_mono_s": round(float(rx_t), 6),
                "rx_since_launch_s": round(float(rx_t) - float(launch_mono), 6),
                "trace_capture_s": None,
                "candidate_offset_mono_s": None,
                "inlier": False,
            })

    tail_start = 1 if len(cands) >= 3 else 0
    tail = cands[tail_start:]
    inliers: List[float] = []
    inlier_cand_positions: List[int] = []
    t_base: float
    sync_method: str

    if len(tail) >= 2:
        med = float(np.median(tail))
        for pos in range(tail_start, len(cands)):
            if abs(cands[pos] - med) <= 0.25:
                inliers.append(cands[pos])
                inlier_cand_positions.append(pos)
        if len(inliers) >= 2 and len(inliers) * 2 >= len(tail):
            t_base = float(np.median(inliers))
            sync_method = "trace_status_sync"
        else:
            inliers = []
            inlier_cand_positions = []
            t_base = float(epoch_mono) if epoch_mono is not None else float(launch_mono)
            sync_method = "console_os_fallback"
    elif len(cands) == 1:
        inliers = [cands[0]]
        inlier_cand_positions = [0]
        t_base = float(cands[0])
        sync_method = "trace_status_sync_single"
    else:
        t_base = float(epoch_mono) if epoch_mono is not None else float(launch_mono)
        sync_method = "console_os_fallback"

    for pos in inlier_cand_positions:
        raw_entries[valid_indices[pos]]["inlier"] = True

    inlier_spread_ms: Optional[float] = None
    mad_ms: Optional[float] = None
    uncertainty_ms: Optional[float] = None
    if len(inliers) >= 2:
        arr = np.array(inliers, dtype=np.float64)
        inlier_spread_ms = round(float(np.max(arr) - np.min(arr)) * 1000.0, 3)
        mad_ms = round(float(np.median(np.abs(arr - float(np.median(arr))))) * 1000.0, 3)
        uncertainty_ms = round(max(inlier_spread_ms / 2.0, 1.4826 * mad_ms), 3)
    elif len(inliers) == 1:
        inlier_spread_ms = 0.0
        mad_ms = 0.0
        uncertainty_ms = None

    wall_iso: Optional[str] = None
    if wall_origin_unix_s is not None and math.isfinite(wall_origin_unix_s):
        wall_iso = datetime.datetime.fromtimestamp(
            wall_origin_unix_s, tz=datetime.timezone.utc
        ).isoformat()

    diagnostics = {
        "capture_clock_origin": "core_engine Pipeline::t0 (std::chrono::steady_clock at pipeline start; trace.txt E4REC F t= seconds)",
        "wall_clock_origin_monotonic_s": round(float(launch_mono), 6),
        "wall_clock_origin_unix_s": round(float(wall_origin_unix_s), 6) if wall_origin_unix_s is not None else None,
        "wall_clock_origin_utc_iso": wall_iso,
        "conversion_formula": "capture_s = host_monotonic_s - clock_offset_mono_s",
        "sync_method": sync_method,
        "clock_offset_mono_s": round(float(t_base), 6),
        "console_epoch_mono_s": round(float(epoch_mono), 6) if epoch_mono is not None else None,
        "inlier_threshold_s": 0.25,
        "num_status_samples": len(status_samples),
        "num_valid_candidates": len(cands),
        "num_inliers": len(inliers),
        "inlier_spread_ms": inlier_spread_ms,
        "mad_ms": mad_ms,
        "uncertainty_estimate_ms": uncertainty_ms,
        "uncertainty_note": (
            "Estimated from 1 Hz stdout status reception timestamps vs trace.txt E4REC F capture_s; "
            "reflects host/engine clock alignment uncertainty, NOT physical acoustic latency."
        ),
        "raw_status_samples": raw_entries,
    }
    return t_base, sync_method, diagnostics


def _resolve_capture_epoch_mono(
    status_samples: Sequence[Tuple[int, float]],
    trace_times: Sequence[float],
    epoch_mono: Optional[float],
    launch_mono: float,
) -> Tuple[float, str]:
    t_base, sync_method, _ = _resolve_capture_epoch_with_diagnostics(
        status_samples, trace_times, epoch_mono, launch_mono
    )
    return t_base, sync_method


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

    integrity_errors: List[str] = []

    def _record_err(msg: str) -> None:
        if len(integrity_errors) < 25:
            integrity_errors.append(msg)

    lines = trace_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_meta: Dict[str, str] = {}
    footer_meta: Dict[str, str] = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2 and parts[0] == "E4REC" and parts[1] == "BEGIN":
            header_meta = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
        elif len(parts) >= 2 and parts[0] == "E4REC" and parts[1] == "END":
            footer_meta = dict(p.split("=", 1) for p in parts[2:] if "=" in p)

    trace_parse_errors = 0
    trace_rows: List[Dict[str, Any]] = []
    try:
        e4 = _load_import_vs_trace()
        converted = e4.convert_recorder(lines, "gaze")
        col_names = e4.RECORDER_COLUMNS
        trace_rows = [dict(zip(col_names, row)) for row in converted]
    except Exception as exc:
        trace_parse_errors += 1
        _record_err(f"trace.txt parse error: {type(exc).__name__}: {exc}")

    with diag_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header_row = next(reader, None)
        diag_cols = tuple(header_row) if header_row is not None else ()
        raw_diag_rows = list(reader)

    expected_col_count = len(EXPECTED_DIAG_COLUMNS)
    header_valid = diag_cols == EXPECTED_DIAG_COLUMNS
    if not header_valid:
        _record_err(
            f"gaze_diag.csv header mismatch: expected {expected_col_count} canonical columns, "
            f"got {len(diag_cols)}"
        )

    row_count_mismatch = len(trace_rows) != len(raw_diag_rows)
    if row_count_mismatch:
        _record_err(
            f"Row count mismatch: trace.txt has {len(trace_rows)} rows vs gaze_diag.csv {len(raw_diag_rows)} rows"
        )

    aligned: List[Dict[str, Any]] = []
    fid_mismatches = 0
    ts_ms_mismatches = 0
    cap_s_mismatches = 0
    state_mismatches = 0
    norm_x_mismatches = 0
    norm_y_mismatches = 0
    frame_id_non_increasing = 0
    ts_ms_non_increasing = 0
    capture_s_non_increasing = 0
    nonfinite_errors = 0
    surplus_field_rows = 0
    missing_field_rows = 0
    invalid_integer_fields = 0
    invalid_detected_values = 0
    invalid_state_values = 0
    invalid_numeric_fields = 0
    infinite_value_errors = 0
    recorder_correspondence_errors = 0
    max_cap_s_diff = 0.0
    max_dx = 0.0
    max_dy = 0.0

    # Validate row widths across ALL CSV rows even if row counts differ.
    if header_valid:
        for row_idx, raw_row in enumerate(raw_diag_rows):
            if len(raw_row) > expected_col_count:
                surplus_field_rows += 1
                _record_err(f"diag row {row_idx}: surplus fields ({len(raw_row)} > {expected_col_count})")
            elif len(raw_row) < expected_col_count:
                missing_field_rows += 1
                _record_err(f"diag row {row_idx}: missing fields ({len(raw_row)} < {expected_col_count})")

    prev_fid: Optional[int] = None
    prev_ts_ms: Optional[int] = None
    prev_cap_s: Optional[float] = None
    always_finite_set = set(DIAG_ALWAYS_FINITE_FLOAT_COLUMNS)
    inactive_nan_allowed_set = set(DIAG_INACTIVE_NAN_ALLOWED_COLUMNS)

    if header_valid and trace_parse_errors == 0:
        for row_idx, (tr, raw_row) in enumerate(zip(trace_rows, raw_diag_rows)):
            if len(raw_row) != expected_col_count:
                continue

            dg = dict(zip(EXPECTED_DIAG_COLUMNS, raw_row))
            row_parse_ok = True

            # 1. Strict integer parsing & domain validation on diag row
            d_fid = _parse_strict_int(dg["frame_id"])
            if d_fid is None or d_fid < 0:
                invalid_integer_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: invalid frame_id={dg['frame_id']!r}")

            d_ts_ms = _parse_strict_int(dg["ts_ms"])
            if d_ts_ms is None or d_ts_ms < 0:
                invalid_integer_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: invalid ts_ms={dg['ts_ms']!r}")

            detected = _parse_strict_int(dg["detected"])
            if detected is None:
                invalid_integer_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: invalid detected={dg['detected']!r}")
            elif detected not in ALLOWED_DETECTED_VALUES:
                invalid_detected_values += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: disallowed detected={detected} (allowed: {ALLOWED_DETECTED_VALUES})")

            d_state = _parse_strict_int(dg["state"])
            if d_state is None:
                invalid_integer_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: invalid state={dg['state']!r}")
            elif d_state not in ALLOWED_STATE_VALUES:
                invalid_state_values += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: disallowed state={d_state} (allowed: {ALLOWED_STATE_VALUES})")

            # 2. Strict float parsing across ALL 28 float columns (capture_s + 27 numeric cols)
            d_cap_s = _parse_strict_float(dg["capture_s"])
            if d_cap_s is None:
                invalid_numeric_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: unreadable capture_s={dg['capture_s']!r}")
            elif math.isinf(d_cap_s):
                infinite_value_errors += 1
                nonfinite_errors += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: infinite capture_s={dg['capture_s']!r}")
            elif math.isnan(d_cap_s):
                nonfinite_errors += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: NaN capture_s")
            elif d_cap_s < 0.0:
                invalid_numeric_fields += 1
                row_parse_ok = False
                _record_err(f"diag row {row_idx}: negative capture_s={d_cap_s}")

            parsed_diag: Dict[str, float] = {}
            for col in DIAG_FLOAT_COLUMNS:
                val = _parse_strict_float(dg[col])
                if val is None:
                    invalid_numeric_fields += 1
                    row_parse_ok = False
                    _record_err(f"diag row {row_idx}: unreadable numeric {col}={dg[col]!r}")
                    continue
                parsed_diag[col] = val
                if math.isinf(val):
                    infinite_value_errors += 1
                    nonfinite_errors += 1
                    _record_err(f"diag row {row_idx}: infinite value in {col}={dg[col]!r}")
                elif math.isnan(val):
                    if col in always_finite_set:
                        nonfinite_errors += 1
                        _record_err(f"diag row {row_idx}: NaN in always-finite column {col}")
                    elif detected == 1:
                        nonfinite_errors += 1
                        _record_err(f"diag row {row_idx}: NaN in active (detected=1) column {col}")
                    elif detected == 0 and col in inactive_nan_allowed_set:
                        # Legitimate inactive placeholder per 32-column schema.
                        pass
                    else:
                        nonfinite_errors += 1
                        _record_err(f"diag row {row_idx}: NaN in column {col} with detected={detected}")

            # 3. Parse trace row fields safely
            try:
                t_fid = int(tr["frame_id"])
                t_obs_ms = int(tr["observation_ms"])
                t_cap_s = float(tr["dsp_timestamp_seconds"])
                t_state = int(tr["state"])
                t_tx = float(tr["tracker_x"])
                t_ty = float(tr["tracker_y"])
                raw_x = float(tr["raw_x"]) if tr["raw_x"] != "" else None
                raw_y = float(tr["raw_y"]) if tr["raw_y"] != "" else None
                filt_x = float(tr["filtered_x"]) if tr["filtered_x"] != "" else None
                filt_y = float(tr["filtered_y"]) if tr["filtered_y"] != "" else None
                landmarks = int(tr["landmarks"])
                published = int(tr["published"])
            except (ValueError, TypeError, KeyError) as exc:
                recorder_correspondence_errors += 1
                _record_err(f"trace row {row_idx}: malformed field ({type(exc).__name__}: {exc})")
                continue

            if t_state not in ALLOWED_STATE_VALUES:
                invalid_state_values += 1
                _record_err(f"trace row {row_idx}: disallowed state={t_state}")

            if not row_parse_ok or "norm_x" not in parsed_diag or "norm_y" not in parsed_diag:
                continue

            assert d_fid is not None and d_ts_ms is not None and detected is not None and d_state is not None
            assert d_cap_s is not None
            d_nx = parsed_diag["norm_x"]
            d_ny = parsed_diag["norm_y"]

            # 4. 1:1 alignment & monotonicity
            if t_fid != d_fid:
                fid_mismatches += 1
            if t_obs_ms != d_ts_ms:
                ts_ms_mismatches += 1
            if t_state != d_state:
                state_mismatches += 1

            if prev_fid is not None and (t_fid <= prev_fid or d_fid <= prev_fid):
                frame_id_non_increasing += 1
            if prev_ts_ms is not None and (t_obs_ms <= prev_ts_ms or d_ts_ms <= prev_ts_ms):
                ts_ms_non_increasing += 1
            if prev_cap_s is not None:
                if not (math.isfinite(t_cap_s) and math.isfinite(d_cap_s)) or t_cap_s <= prev_cap_s or d_cap_s <= prev_cap_s:
                    capture_s_non_increasing += 1
            prev_fid = t_fid
            prev_ts_ms = t_obs_ms
            if math.isfinite(t_cap_s):
                prev_cap_s = t_cap_s

            if math.isfinite(t_cap_s) and math.isfinite(d_cap_s):
                cap_diff = abs(t_cap_s - d_cap_s)
                max_cap_s_diff = max(max_cap_s_diff, cap_diff)
                if cap_diff > 1e-12:
                    cap_s_mismatches += 1
            else:
                nonfinite_errors += 1
                cap_s_mismatches += 1

            if math.isfinite(t_tx) and math.isfinite(d_nx):
                dx = abs(t_tx - d_nx)
                max_dx = max(max_dx, dx)
                if dx > COORD_TOL:
                    norm_x_mismatches += 1
            else:
                if not math.isfinite(t_tx):
                    nonfinite_errors += 1
                norm_x_mismatches += 1

            if math.isfinite(t_ty) and math.isfinite(d_ny):
                dy = abs(t_ty - d_ny)
                max_dy = max(max_dy, dy)
                if dy > COORD_TOL:
                    norm_y_mismatches += 1
            else:
                if not math.isfinite(t_ty):
                    nonfinite_errors += 1
                norm_y_mismatches += 1

            # 5. Recorder correspondence & trace finiteness
            if detected == 0 and published == 1:
                recorder_correspondence_errors += 1
                _record_err(f"row {row_idx}: detected=0 but trace published=1")
            if detected == 0 and math.isfinite(d_nx) and math.isfinite(d_ny) and (d_nx != -1.0 or d_ny != -1.0):
                recorder_correspondence_errors += 1
                _record_err(f"row {row_idx}: detected=0 expects sentinel norm=(-1,-1), got ({d_nx}, {d_ny})")
            if published == 1 and (detected != 1 or landmarks != 484):
                recorder_correspondence_errors += 1
                _record_err(f"row {row_idx}: published=1 inconsistent with detected={detected}, landmarks={landmarks}")
            if detected == 1 and d_state != -1 and landmarks != 484:
                recorder_correspondence_errors += 1
                _record_err(f"row {row_idx}: active detected=1 expects landmarks=484, got {landmarks}")

            for opt_val in (raw_x, raw_y, filt_x, filt_y):
                if opt_val is not None and not math.isfinite(opt_val):
                    nonfinite_errors += 1
            if published == 1 and (
                raw_x is None or raw_y is None or filt_x is None or filt_y is None
                or not (math.isfinite(raw_x) and math.isfinite(raw_y) and math.isfinite(filt_x) and math.isfinite(filt_y))
            ):
                nonfinite_errors += 1

            diag_core_finite = all(
                math.isfinite(parsed_diag[k])
                for k in (
                    "sx", "sy", "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
                    "rel_pitch_deg", "rel_yaw_deg", "abs_pitch_deg", "abs_yaw_deg", "norm_x", "norm_y",
                )
            )
            eligible = (
                detected == 1
                and d_state == 0
                and landmarks == 484
                and published == 1
                and math.isfinite(t_cap_s)
                and raw_x is not None
                and raw_y is not None
                and filt_x is not None
                and filt_y is not None
                and math.isfinite(raw_x)
                and math.isfinite(raw_y)
                and math.isfinite(filt_x)
                and math.isfinite(filt_y)
                and diag_core_finite
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
                    "sx": parsed_diag["sx"],
                    "sy": parsed_diag["sy"],
                    "head_pitch_deg": parsed_diag["head_pitch_deg"],
                    "head_yaw_deg": parsed_diag["head_yaw_deg"],
                    "head_roll_deg": parsed_diag["head_roll_deg"],
                    "rel_pitch_deg": parsed_diag["rel_pitch_deg"],
                    "rel_yaw_deg": parsed_diag["rel_yaw_deg"],
                    "abs_pitch_deg": parsed_diag["abs_pitch_deg"],
                    "abs_yaw_deg": parsed_diag["abs_yaw_deg"],
                    "norm_x": d_nx,
                    "norm_y": d_ny,
                    "eligible": eligible,
                }
            )

    detected_rows = [
        r for r in aligned
        if r["detected"] == 1 and math.isfinite(r["abs_pitch_deg"]) and math.isfinite(r["norm_y"])
    ]
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

    overflow_val = _parse_strict_int(footer_meta.get("overflow", "-1"))
    config_changes_val = _parse_strict_int(footer_meta.get("config_changes", "-1"))
    complete_flag_val = _parse_strict_int(footer_meta.get("complete", "0"))
    overflow_i = overflow_val if overflow_val is not None else -1
    config_changes_i = config_changes_val if config_changes_val is not None else -1
    complete_flag_i = complete_flag_val if complete_flag_val is not None else 0

    schema_errors = (
        (0 if header_valid else 1)
        + (1 if row_count_mismatch else 0)
        + trace_parse_errors
        + surplus_field_rows
        + missing_field_rows
        + invalid_integer_fields
        + invalid_detected_values
        + invalid_state_values
        + invalid_numeric_fields
        + infinite_value_errors
        + recorder_correspondence_errors
    )

    alignment_ok = bool(
        complete_marker.is_file()
        and not incomplete_marker.exists()
        and overflow_i == 0
        and config_changes_i == 0
        and complete_flag_i == 1
        and header_valid
        and not row_count_mismatch
        and trace_parse_errors == 0
        and len(trace_rows) > 0
        and len(raw_diag_rows) == len(trace_rows)
        and len(aligned) == len(trace_rows)
        and len(diag_cols) == expected_col_count
        and schema_errors == 0
        and fid_mismatches == 0
        and ts_ms_mismatches == 0
        and cap_s_mismatches == 0
        and state_mismatches == 0
        and norm_x_mismatches == 0
        and norm_y_mismatches == 0
        and frame_id_non_increasing == 0
        and ts_ms_non_increasing == 0
        and capture_s_non_increasing == 0
        and nonfinite_errors == 0
        and math.isfinite(max_cap_s_diff)
        and math.isfinite(max_dx)
        and math.isfinite(max_dy)
        and max_dx <= COORD_TOL
        and max_dy <= COORD_TOL
    )

    integrity = {
        "run_dir": str(run_dir),
        "diag_csv": str(diag_csv),
        "complete_marker_present": complete_marker.is_file(),
        "incomplete_marker_absent": not incomplete_marker.exists(),
        "trace_header": header_meta,
        "trace_footer": footer_meta,
        "overflow": overflow_i,
        "config_changes": config_changes_i,
        "complete_flag": complete_flag_i,
        "header_valid": header_valid,
        "row_count_mismatch": row_count_mismatch,
        "trace_parse_errors": trace_parse_errors,
        "trace_rows": len(trace_rows),
        "diag_rows": len(raw_diag_rows),
        "aligned_rows": len(aligned),
        "diag_columns": len(diag_cols),
        "surplus_field_rows": surplus_field_rows,
        "missing_field_rows": missing_field_rows,
        "invalid_integer_fields": invalid_integer_fields,
        "invalid_detected_values": invalid_detected_values,
        "invalid_state_values": invalid_state_values,
        "invalid_numeric_fields": invalid_numeric_fields,
        "infinite_value_errors": infinite_value_errors,
        "recorder_correspondence_errors": recorder_correspondence_errors,
        "schema_errors": schema_errors,
        "frame_id_mismatches": fid_mismatches,
        "ts_ms_mismatches": ts_ms_mismatches,
        "capture_s_mismatches": cap_s_mismatches,
        "state_mismatches": state_mismatches,
        "norm_x_mismatches": norm_x_mismatches,
        "norm_y_mismatches": norm_y_mismatches,
        "frame_id_non_increasing": frame_id_non_increasing,
        "ts_ms_non_increasing": ts_ms_non_increasing,
        "capture_s_non_increasing": capture_s_non_increasing,
        "nonfinite_errors": nonfinite_errors,
        "integrity_errors": integrity_errors,
        "coord_tol": COORD_TOL,
        "max_abs_capture_s_diff": max_cap_s_diff,
        "max_abs_delta_norm_x": max_dx,
        "max_abs_delta_norm_y": max_dy,
        "logged_y_sign": logged_y_sign,
        "logged_y_sign_mult": logged_y_sign_mult,
        "y_sign_err_fixed": err_fixed,
        "y_sign_err_legacy": err_legacy,
        "alignment_ok": alignment_ok,
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
    strict_protocol_wash_in: bool = False,
) -> Dict[str, Any]:
    if padded_bad_indices is None:
        padded_bad_indices = compute_blink_pad_indices(all_rows, blink_pad_frames)

    eff_start_s = start_s + wash_in_s
    if eff_start_s >= end_s:
        raise ValueError(f"Window {name} has empty effective span after wash_in_s={wash_in_s}: [{eff_start_s}, {end_s})")

    in_raw_win = [(i, r) for i, r in enumerate(all_rows) if start_s <= r["capture_s"] < end_s]

    exclusions = {
        "wash_in": 0,
        "calibration": 0,
        "lost_or_unusable": 0,
        "nonneutral": 0,
        "blink_pad": 0,
        "unpublished": 0,
    }
    eligible: List[Dict[str, Any]] = []

    def _exclusion_reason(idx: int, r: Dict[str, Any]) -> Optional[str]:
        if r["state"] == -1:
            return "calibration"
        if r["detected"] != 1 or r["landmarks"] != 484:
            return "lost_or_unusable"
        if r["state"] != 0:
            return "nonneutral"
        if idx in padded_bad_indices:
            return "blink_pad"
        if r["published"] != 1 or r["raw_x"] is None or r["raw_y"] is None:
            return "unpublished"
        return None

    raw_win_times = [float(r["capture_s"]) for _, r in in_raw_win]
    win_delivered_fps = _compute_cadence_fps(raw_win_times)
    win_duration_s = float(end_s - start_s)
    win_delivered_fps_by_duration = (float(len(in_raw_win)) / win_duration_s) if win_duration_s > 0.0 else None

    if strict_protocol_wash_in:
        total_frames_in_window = len(in_raw_win)
        # Anchor the initial wash-in countdown to target_start_s so frame-arrival latency
        # inside [start_s, end_s) never shifts the initial boundary by an extra frame;
        # interior exclusions or >100 ms gaps still restart wash_in_s from recovery.
        wash_until_s: Optional[float] = eff_start_s if wash_in_s > 0.0 else None
        prev_cap_s: Optional[float] = start_s
        saw_clean = False
        for idx, r in in_raw_win:
            cap_s = float(r["capture_s"])
            reason = _exclusion_reason(idx, r)
            if reason is not None:
                exclusions[reason] += 1
                wash_until_s = None
                saw_clean = False
            else:
                gap_exceeded = (
                    prev_cap_s is not None and (cap_s - prev_cap_s) * 1000.0 > MAX_FRESH_GAP_MS
                )
                if not saw_clean and wash_until_s is not None and not gap_exceeded:
                    pass
                elif not saw_clean or gap_exceeded:
                    wash_until_s = (cap_s + wash_in_s) if wash_in_s > 0.0 else None
                saw_clean = True
                if wash_until_s is not None and (cap_s + 1e-9) < wash_until_s:
                    exclusions["wash_in"] += 1
                else:
                    eligible.append(r)
            prev_cap_s = cap_s
    else:
        in_win_indexed = [
            (i, r)
            for i, r in in_raw_win
            if (wash_in_s <= 0.0 or (r["capture_s"] + 1e-9) >= eff_start_s)
        ]
        exclusions["wash_in"] = len(in_raw_win) - len(in_win_indexed)
        total_frames_in_window = len(in_win_indexed)
        for idx, r in in_win_indexed:
            reason = _exclusion_reason(idx, r)
            if reason is not None:
                exclusions[reason] += 1
            else:
                eligible.append(r)

    if not eligible:
        return {
            "window": name,
            "raw_interval_s": [start_s, end_s],
            "interval_s": [start_s if strict_protocol_wash_in else eff_start_s, end_s],
            "wash_in_s": wash_in_s,
            "strict_protocol_wash_in": strict_protocol_wash_in,
            "blink_pad_frames": blink_pad_frames,
            "total_frames_in_raw_window": len(in_raw_win),
            "total_frames_in_window": total_frames_in_window,
            "eligible_frames": 0,
            "delivered_fps": win_delivered_fps,
            "delivered_fps_by_duration": win_delivered_fps_by_duration,
            "eligible_fps": None,
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

    dep_pitch_mode = _detect_pitch_mode(dep_pitch)

    # Replay deployed DLLs to confirm exact 5-D reconstruction against recorded trace
    rep_ry, rep_rp = _predict_angles(dep_yaw, dep_pitch, X, pitch_mode=dep_pitch_mode)
    logged_ry = np.array([r["rel_yaw_deg"] for r in eligible], dtype=np.float64)
    logged_rp = np.array([r["rel_pitch_deg"] for r in eligible], dtype=np.float64)
    hy = X[:, 3].astype(np.float64)
    hp = X[:, 2].astype(np.float64)
    rep_ax = _norm_coord_x(rep_ry + hy)
    rep_ay = _norm_coord_y(rep_rp + hp, logged_y_sign_mult)
    dep_base_raw_pitch = (rep_rp + hp) if dep_pitch_mode == "abs_pitch" else rep_rp

    # Counterfactuals use canonical fixed Y sign (0.5 - pitch / 60.0)
    # Counterfactual 1: Iris-only (vary sx, sy; hold head_pitch, head_yaw, head_roll at window median)
    X_iris = X.copy()
    X_iris[:, 2] = med[2]
    X_iris[:, 3] = med[3]
    X_iris[:, 4] = med[4]
    ry_iris, rp_iris = _predict_angles(dep_yaw, dep_pitch, X_iris, pitch_mode=dep_pitch_mode)
    nx_iris = _norm_coord_x(ry_iris + float(med[3]))
    ny_iris = _norm_coord_y(rp_iris + float(med[2]), FIXED_Y_SIGN)

    # Counterfactual 2: Head-pose total (vary head_*; hold sx, sy at window median)
    X_head = X.copy()
    X_head[:, 0] = med[0]
    X_head[:, 1] = med[1]
    ry_head_in, rp_head_in = _predict_angles(dep_yaw, dep_pitch, X_head, pitch_mode=dep_pitch_mode)
    nx_head_tot = _norm_coord_x(ry_head_in + hy)
    ny_head_tot = _norm_coord_y(rp_head_in + hp, FIXED_Y_SIGN)

    # Direct additive head angle only (hold rel_* at median prediction)
    rel_med_ry, rel_med_rp = _predict_angles(dep_yaw, dep_pitch, med.reshape(1, 5), pitch_mode=dep_pitch_mode)
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

    win_eligible_fps = _compute_cadence_fps(times.tolist())
    res: Dict[str, Any] = {
        "window": name,
        "raw_interval_s": [start_s, end_s],
        "interval_s": [start_s if strict_protocol_wash_in else eff_start_s, end_s],
        "wash_in_s": wash_in_s,
        "strict_protocol_wash_in": strict_protocol_wash_in,
        "blink_pad_frames": blink_pad_frames,
        "total_frames_in_raw_window": len(in_raw_win),
        "total_frames_in_window": total_frames_in_window,
        "eligible_frames": len(eligible),
        "delivered_fps": win_delivered_fps,
        "delivered_fps_by_duration": win_delivered_fps_by_duration,
        "eligible_fps": win_eligible_fps,
        "max_eligible_gap_ms": float(np.max(np.diff(times) * 1000.0)) if len(times) >= 2 else 0.0,
        "exclusions": exclusions,
        "raw_x": raw_x_stats,
        "raw_y": raw_y_stats,
        "raw_y_fixed_sign": raw_y_fixed_stats,
        "filtered_x": _compute_axis_metrics(filt_x, times),
        "filtered_y": _compute_axis_metrics(filt_y, times),
        "feature_summary": feature_summary,
        "deployed_replay_verification": {
            "deployed_pitch_mode": dep_pitch_mode,
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
            dep_yaw, dep_pitch, X, rep_ry, dep_base_raw_pitch, delta=0.005
        ),
        "top_5_jumps_x": _top_jumps(eligible, "x", top_k=5),
        "top_5_jumps_y": _top_jumps(eligible, "y", top_k=5, y_sign_mult=FIXED_Y_SIGN),
    }

    if cand_yaw is not None and cand_pitch is not None:
        cand_pitch_mode = _detect_pitch_mode(cand_pitch)
        c_ry, c_rp = _predict_angles(cand_yaw, cand_pitch, X, pitch_mode=cand_pitch_mode)
        c_base_raw_pitch = (c_rp + hp) if cand_pitch_mode == "abs_pitch" else c_rp
        c_ayaw = c_ry + hy
        c_apitch = c_rp + hp
        c_nx = _norm_coord_x(c_ayaw)
        c_ny = _norm_coord_y(c_apitch, FIXED_Y_SIGN)
        res["candidate_shadow_replay"] = {
            "pitch_mode": cand_pitch_mode,
            "raw_x": _compute_axis_metrics(c_nx, times),
            "raw_y": _compute_axis_metrics(c_ny, times),
            "rel_yaw_mean_deg": float(np.mean(c_ry)),
            "rel_pitch_mean_deg": float(np.mean(c_rp)),
            "abs_yaw_mean_deg": float(np.mean(c_ayaw)),
            "abs_pitch_mean_deg": float(np.mean(c_apitch)),
            "local_cliff_sensitivity_pm005": _local_sensitivity_stats(
                cand_yaw, cand_pitch, X, c_ry, c_base_raw_pitch, delta=0.005
            ),
        }
    return res


def compute_direction_checks(windows_res: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {w["window"].upper(): w for w in windows_res if w.get("eligible_frames", 0) > 0}
    checks: List[Dict[str, Any]] = []

    def _find_match(substr: str) -> Optional[Dict[str, Any]]:
        if substr in by_name:
            return by_name[substr]
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


def compute_qualification_summary(
    integrity: Dict[str, Any],
    windows_res: Sequence[Dict[str, Any]],
    directions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    archive_integrity_ok = bool(integrity.get("alignment_ok", False))
    alignment_ok = bool(integrity.get("alignment_ok", False))

    by_window: Dict[str, Dict[str, Any]] = {}
    by_window_upper: Dict[str, Dict[str, Any]] = {}
    passed_axes = 0
    total_axes = 0
    for w in windows_res:
        rx_pass = bool(w.get("raw_x", {}).get("pass_all", False))
        ry_pass = bool(w.get("raw_y", {}).get("pass_all", False))
        w_passed = int(rx_pass) + int(ry_pass)
        passed_axes += w_passed
        total_axes += 2
        entry = {
            "raw_x": rx_pass,
            "raw_y": ry_pass,
            "passed": w_passed,
            "total": 2,
            "pass_both": rx_pass and ry_pass,
        }
        by_window[w["window"]] = entry
        by_window_upper[w["window"].upper()] = entry

    dirs_by_comp = {d["comparison"]: d for d in directions}
    horiz_dir = dirs_by_comp.get("HORIZONTAL_LEFT_TO_RIGHT")
    vert_dir = dirs_by_comp.get("VERTICAL_UP_TO_DOWN")

    horiz_dir_pass: Optional[bool] = None
    if horiz_dir is not None:
        dx = horiz_dir.get("delta_raw_x_mean")
        horiz_dir_pass = bool(dx is not None and math.isfinite(dx) and dx > 0.0)

    vert_dir_pass: Optional[bool] = None
    if vert_dir is not None:
        dy = vert_dir.get("delta_raw_y_mean")
        vert_dir_pass = bool(dy is not None and math.isfinite(dy) and dy > 0.0)

    cardinal_checks = [c for c in (horiz_dir_pass, vert_dir_pass) if c is not None]
    direction_verification_pass: Optional[bool] = (
        all(cardinal_checks) if cardinal_checks else None
    )

    window_names_upper = set(by_window_upper.keys())
    has_duplicate_windows = len(windows_res) != len(window_names_upper)
    horiz_present = all(k in window_names_upper for k in MANDATORY_HORIZONTAL_TARGETS)
    vert_present = all(k in window_names_upper for k in MANDATORY_VERTICAL_TARGETS)
    horiz_attempted = bool({"LEFT", "RIGHT"} & window_names_upper)
    vert_attempted = bool({"UP", "DOWN"} & window_names_upper)

    missing_horiz_targets = (
        [k for k in MANDATORY_HORIZONTAL_TARGETS if k not in window_names_upper]
        if horiz_attempted and not horiz_present
        else []
    )
    missing_vert_targets = (
        [k for k in MANDATORY_VERTICAL_TARGETS if k not in window_names_upper]
        if vert_attempted and not vert_present
        else []
    )
    known_mandatory_targets = set(MANDATORY_HORIZONTAL_TARGETS) | set(MANDATORY_VERTICAL_TARGETS)
    unexpected_windows = sorted(window_names_upper - known_mandatory_targets)

    mandatory_target_set_complete = bool(
        (horiz_present or vert_present)
        and (not horiz_attempted or horiz_present)
        and (not vert_attempted or vert_present)
        and not unexpected_windows
        and not has_duplicate_windows
    )

    horiz_qual_pass: Optional[bool] = None
    if horiz_present:
        horiz_qual_pass = bool(
            archive_integrity_ok
            and alignment_ok
            and not has_duplicate_windows
            and all(by_window_upper[k]["pass_both"] for k in MANDATORY_HORIZONTAL_TARGETS)
            and horiz_dir_pass is True
        )
    elif horiz_attempted:
        horiz_qual_pass = False

    vert_qual_pass: Optional[bool] = None
    if vert_present:
        vert_qual_pass = bool(
            archive_integrity_ok
            and alignment_ok
            and not has_duplicate_windows
            and all(by_window_upper[k]["pass_both"] for k in MANDATORY_VERTICAL_TARGETS)
            and vert_dir_pass is True
        )
    elif vert_attempted:
        vert_qual_pass = False

    all_evaluated_axes_pass = bool(total_axes > 0 and passed_axes == total_axes)
    overall_qualification_pass = bool(
        archive_integrity_ok
        and alignment_ok
        and mandatory_target_set_complete
        and all_evaluated_axes_pass
        and direction_verification_pass is True
        and (horiz_qual_pass is not False)
        and (vert_qual_pass is not False)
    )

    return {
        "archive_integrity_ok": archive_integrity_ok,
        "alignment_ok": alignment_ok,
        "per_axis_pass_counts": {
            "passed": passed_axes,
            "total": total_axes,
            "by_window": by_window,
        },
        "all_evaluated_axes_pass": all_evaluated_axes_pass,
        "mandatory_target_set_complete": mandatory_target_set_complete,
        "horizontal_axes_present": horiz_present,
        "vertical_axes_present": vert_present,
        "missing_horizontal_targets": missing_horiz_targets,
        "missing_vertical_targets": missing_vert_targets,
        "unexpected_windows": unexpected_windows,
        "duplicate_window_names": has_duplicate_windows,
        "horizontal_direction_pass": horiz_dir_pass,
        "vertical_direction_pass": vert_dir_pass,
        "direction_verification_pass": direction_verification_pass,
        "horizontal_qualification_pass": horiz_qual_pass,
        "vertical_qualification_pass": vert_qual_pass,
        "overall_qualification_pass": overall_qualification_pass,
        "scored_responsiveness_authorized": False,
        "scored_responsiveness_note": (
            "Scored gaze responsiveness remains ON HOLD until Engineer 4 independently reviews "
            "and authorizes the complete qualification evidence and exact executable configuration."
        ),
    }


def format_markdown_report(report: Dict[str, Any]) -> str:
    integ = report["integrity"]
    cadence = report.get("cadence") or {}
    qual = report.get("qualification") or {}
    artifact_dir = integ.get("artifact_dir") or integ["run_dir"]
    dsp_dir = integ.get("dsp_record_dir") or integ["run_dir"]
    run_line = f"- **Run directory:** `{artifact_dir}`"
    if dsp_dir != artifact_dir:
        run_line += f" | **DSP recorder dir:** `{dsp_dir}`"
    req_fps = cadence.get("requested_fps")
    deliv_fps = cadence.get("delivered_fps_overall")
    req_fps_str = f"{req_fps:.2f}" if req_fps is not None else "unspecified"
    deliv_fps_str = f"{deliv_fps:.2f}" if deliv_fps is not None else "n/a"
    lines = [
        "# Gaze Fixation Stability, Attribution & A/B Shadow Replay Report",
        "",
    ]
    if not integ.get("alignment_ok", False):
        lines.extend([
            "> [!CAUTION]",
            "> **INVALID RUN / FAILED ALIGNMENT & INTEGRITY CHECKS.** Do not use metrics below as valid qualification evidence.",
            "",
        ])
    if cadence.get("low_fps_warning"):
        lines.extend([
            "> [!WARNING]",
            f"> **LOW DELIVERED FPS WARNING:** delivered=`{deliv_fps_str}` FPS (< `{cadence.get('low_fps_threshold', LOW_DELIVERED_FPS_WARN_THRESHOLD):.1f}` FPS, low windows=`{cadence.get('low_fps_windows')}`); check room lighting / webcam auto-exposure.",
            "",
        ])
    lines.extend([
        "## 1. Alignment & Integrity",
        f"- **Alignment & integrity verdict:** **{'VALID (`alignment_ok=True`)' if integ.get('alignment_ok') else 'INVALID (`alignment_ok=False`)'}**",
        run_line,
        f"- **Diag CSV:** `{integ['diag_csv']}` (`{integ['diag_rows']}` rows, `{integ['diag_columns']}` columns, `header_valid={integ.get('header_valid', True)}`)",
        f"- **Cadence (`{cadence.get('delivered_fps_formula', '(N - 1) / (last_capture_s - first_capture_s)')}`):** `requested_fps={req_fps_str}`, `delivered_fps_overall={deliv_fps_str}`, `capture_span_s={cadence.get('capture_span_s')}`, `max_frame_gap_ms={cadence.get('max_frame_gap_ms')}`, `low_fps_warning={cadence.get('low_fps_warning', False)}`",
        f"- **Recorder markers:** `COMPLETE={integ['complete_marker_present']}`, `INCOMPLETE_absent={integ['incomplete_marker_absent']}`, `overflow={integ['overflow']}`, `config_changes={integ['config_changes']}`, `complete={integ['complete_flag']}`",
        f"- **DSP config (E4REC BEGIN):** `min_cutoff={integ['trace_header'].get('min_cutoff')}`, `beta={integ['trace_header'].get('beta')}`, `derivative_cutoff={integ['trace_header'].get('derivative_cutoff')}`",
        f"- **1:1 Alignment (`COORD_TOL={integ.get('coord_tol', COORD_TOL):.1e}`):** `frame_id_mismatches={integ['frame_id_mismatches']}`, `ts_ms_mismatches={integ.get('ts_ms_mismatches', 0)}`, `capture_s_mismatches={integ['capture_s_mismatches']}` (`max_diff={integ['max_abs_capture_s_diff']:.3e} s`), `state_mismatches={integ['state_mismatches']}`, `norm_mismatches=({integ.get('norm_x_mismatches', 0)}, {integ.get('norm_y_mismatches', 0)})`, `max_delta_norm=({integ['max_abs_delta_norm_x']:.3e}, {integ['max_abs_delta_norm_y']:.3e})`, `logged_y_sign={integ['logged_y_sign']}`",
        f"- **Schema & correspondence validation:** `schema_errors={integ.get('schema_errors', 0)}`, `surplus_field_rows={integ.get('surplus_field_rows', 0)}`, `missing_field_rows={integ.get('missing_field_rows', 0)}`, `invalid_integer_fields={integ.get('invalid_integer_fields', 0)}`, `invalid_detected_values={integ.get('invalid_detected_values', 0)}`, `invalid_state_values={integ.get('invalid_state_values', 0)}`, `invalid_numeric_fields={integ.get('invalid_numeric_fields', 0)}`, `infinite_value_errors={integ.get('infinite_value_errors', 0)}`, `recorder_correspondence_errors={integ.get('recorder_correspondence_errors', 0)}`",
        f"- **Monotonicity & finiteness:** `frame_id_non_increasing={integ.get('frame_id_non_increasing', 0)}`, `ts_ms_non_increasing={integ.get('ts_ms_non_increasing', 0)}`, `capture_s_non_increasing={integ.get('capture_s_non_increasing', 0)}`, `nonfinite_errors={integ.get('nonfinite_errors', 0)}`",
    ])
    if integ.get("integrity_errors"):
        lines.append("- **Integrity error samples:**")
        for err in integ["integrity_errors"][:10]:
            lines.append(f"  - `{err}`")
    lines.extend([
        "",
        "### Model Inventory",
        "| Role | File | Bytes | SHA-256 |",
        "| --- | --- | ---: | --- |",
    ])
    for role, inv_map in (("Deployed", report["deployed_models"]), ("Candidate", report.get("candidate_models") or {})):
        for name, info in inv_map.items():
            lines.append(f"| {role} | `{name}` | {info.get('bytes')} | `{info.get('sha256')}` |")

    if qual:
        counts = qual.get("per_axis_pass_counts") or {}
        lines.extend([
            "",
            "## 2. Qualification Summary (Separate from Archive Integrity)",
            f"- **Archive integrity (`archive_integrity_ok`):** `{qual.get('archive_integrity_ok')}` | **Alignment (`alignment_ok`):** `{qual.get('alignment_ok')}`",
            f"- **Mandatory target set complete (`mandatory_target_set_complete`):** `{qual.get('mandatory_target_set_complete')}` (missing_horiz=`{qual.get('missing_horizontal_targets', [])}`, missing_vert=`{qual.get('missing_vertical_targets', [])}`, unexpected=`{qual.get('unexpected_windows', [])}`)",
            f"- **Per-axis fixation pass count:** `{counts.get('passed', 0)}/{counts.get('total', 0)}` (`all_evaluated_axes_pass={qual.get('all_evaluated_axes_pass')}`)",
            f"- **Horizontal qualification (`CENTER/LEFT/RIGHT`):** `{qual.get('horizontal_qualification_pass')}` (direction: `{qual.get('horizontal_direction_pass')}`)",
            f"- **Vertical qualification (`CENTER/UP/DOWN`):** `{qual.get('vertical_qualification_pass')}` (direction: `{qual.get('vertical_direction_pass')}`)",
            f"- **Direction verification (`direction_verification_pass`):** `{qual.get('direction_verification_pass')}`",
            f"- **Overall fixation qualification (`overall_qualification_pass`):** **{'PASS' if qual.get('overall_qualification_pass') else 'FAIL'}**",
            f"- **Scored responsiveness authorized (`scored_responsiveness_authorized`):** `{qual.get('scored_responsiveness_authorized', False)}` ({qual.get('scored_responsiveness_note', '')})",
            "- **Exit code policy:** `0` = integrity OK (and qualification PASS if `--require-qualification-pass`), `1` = integrity/schema/alignment FAILED, `2` = integrity OK but qualification FAILED under `--require-qualification-pass`.",
        ])

    lines += ["", "## 3. Raw Fixation Stability & Attribution by Window"]
    for w in report["windows"]:
        wash_mode = "strict-protocol" if w.get("strict_protocol_wash_in") else "window-start"
        w_fps = w.get("delivered_fps")
        w_fps_str = f"{w_fps:.2f} FPS" if w_fps is not None else "n/a"
        lines.append(
            f"### Window `{w['window']}` ([{w['interval_s'][0]:.3f}s, {w['interval_s'][1]:.3f}s), "
            f"wash_in={w.get('wash_in_s', 0.0):.2f}s ({wash_mode}), blink_pad=+-{w.get('blink_pad_frames', 0)}f)"
        )
        lines.append(
            f"- Delivered FPS: `{w_fps_str}` (raw window frames: `{w.get('total_frames_in_raw_window', w['total_frames_in_window'])}`), "
            f"Frames in window: `{w['total_frames_in_window']}`, Eligible (`state=0, detected=1`): `{w['eligible_frames']}`, "
            f"Exclusions: `{w['exclusions']}`"
        )
        if w["eligible_frames"] == 0:
            lines.append("- **No eligible frames in window.**")
            continue
        rx, ry = w["raw_x"], w["raw_y"]
        lines.extend([
            "",
            f"| Axis | N (>=25) | Mean | SD (<=0.010) | Range (<=0.030) | |Slope|/s (<=0.005) | Max Gap ms (<={MAX_FRESH_GAP_MS:.0f}) | Max Step | Clipped 0/1 | Verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| `raw_x` | {rx['n']} | {rx['mean']:.5f} | {rx['pop_sd']:.6f} | {rx['range']:.6f} | {abs(rx['ols_slope_per_s']):.6f} | {rx['max_gap_ms']:.1f} | {rx['max_abs_step']:.6f} | {rx['clipped_zero_or_one']} | **{'PASS' if rx['pass_all'] else 'FAIL'}** |",
            f"| `raw_y` | {ry['n']} | {ry['mean']:.5f} | {ry['pop_sd']:.6f} | {ry['range']:.6f} | {abs(ry['ols_slope_per_s']):.6f} | {ry['max_gap_ms']:.1f} | {ry['max_abs_step']:.6f} | {ry['clipped_zero_or_one']} | **{'PASS' if ry['pass_all'] else 'FAIL'}** |",
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
                f"| Model | Axis | Mean | SD | Range | |Slope|/s | Max Gap ms (<={MAX_FRESH_GAP_MS:.0f}) | Max Step | `±0.005` Max | Verdict |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| Deployed | `X` | {rx['mean']:.5f} | {rx['pop_sd']:.6f} | {rx['range']:.6f} | {abs(rx['ols_slope_per_s']):.6f} | {rx['max_gap_ms']:.1f} | {rx['max_abs_step']:.6f} | {w['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}° | **{'PASS' if rx['pass_all'] else 'FAIL'}** |",
                f"| Candidate | `X` | {crx['mean']:.5f} | {crx['pop_sd']:.6f} | {crx['range']:.6f} | {abs(crx['ols_slope_per_s']):.6f} | {crx['max_gap_ms']:.1f} | {crx['max_abs_step']:.6f} | {cs['local_cliff_sensitivity_pm005']['sx']['max_abs_deg']:.3f}° | **{'PASS' if crx['pass_all'] else 'FAIL'}** |",
                f"| Deployed (logged Y) | `Y` | {ry['mean']:.5f} | {ry['pop_sd']:.6f} | {ry['range']:.6f} | {abs(ry['ols_slope_per_s']):.6f} | {ry['max_gap_ms']:.1f} | {ry['max_abs_step']:.6f} | {w['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}° | **{'PASS' if ry['pass_all'] else 'FAIL'}** |",
                f"| Candidate (fixed Y) | `Y` | {cry['mean']:.5f} | {cry['pop_sd']:.6f} | {cry['range']:.6f} | {abs(cry['ols_slope_per_s']):.6f} | {cry['max_gap_ms']:.1f} | {cry['max_abs_step']:.6f} | {cs['local_cliff_sensitivity_pm005']['sy']['max_abs_deg']:.3f}° | **{'PASS' if cry['pass_all'] else 'FAIL'}** |",
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
        lines += ["", "## 4. Direction Verification"]
        for d in report["directions"]:
            lines.append(
                f"- **{d['comparison']}** (`{d['window_a']}` -> `{d['window_b']}`): "
                f"`Δraw_x={d['delta_raw_x_mean']:+.5f}`, `Δraw_y(logged)={d['delta_raw_y_mean']:+.5f}`, "
                f"`Δraw_y_median={d['delta_raw_y_median']:+.5f}`, "
                f"`Δraw_y(fixed)={d['delta_raw_y_fixed_sign_mean']:+.5f}`, "
                f"`Δsx={d['delta_sx_mean']:+.5f}`, `Δsy={d['delta_sy_mean']:+.5f}`, "
                f"`Δrel_yaw={d['delta_rel_yaw_mean']:+.3f}°`, `Δrel_pitch={d['delta_rel_pitch_mean']:+.3f}°`, "
                f"`Δabs_pitch={d['delta_abs_pitch_mean']:+.3f}°`"
            )
            if "candidate_delta_raw_x_mean" in d:
                lines.append(
                    f"  - Candidate shadow: `Δraw_x={d['candidate_delta_raw_x_mean']:+.5f}`, "
                    f"`Δraw_y(fixed)={d['candidate_delta_raw_y_mean']:+.5f}`, "
                    f"`Δrel_yaw={d['candidate_delta_rel_yaw_mean']:+.3f}°`, "
                    f"`Δrel_pitch={d['candidate_delta_rel_pitch_mean']:+.3f}°`"
                )
    return "\n".join(lines) + "\n"


class _CueSpeaker:
    """Persistent pre-warmed Windows SAPI speech worker with software dispatch telemetry.

    NOTE: Timings recorded here measure software pipe-write and SAPI `SpeakAsync()`
    acknowledgement only (`AUDIO_TIMING_SCOPE`), NOT physical acoustic speaker onset.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled and os.name == "nt")
        self._proc: Optional[subprocess.Popen[str]] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._pending: Optional[Tuple[str, float]] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._start_perf: Optional[float] = None
        self._warm_perf: Optional[float] = None
        self._seq: int = 0
        self._events: List[Dict[str, Any]] = []
        self._events_by_seq: Dict[int, Dict[str, Any]] = {}

    def start(self) -> None:
        if not self.enabled or self._proc is not None:
            return
        self._start_perf = time.perf_counter()
        try:
            ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 3; "
                "$s.Speak(''); "
                "Write-Output 'WARM'; "
                "[Console]::Out.Flush(); "
                "$seq = 0; "
                "while ($null -ne ($line = [Console]::In.ReadLine())) { "
                "  if ($line -eq '__EXIT__') { break } "
                "  $seq++; "
                "  $sw = [System.Diagnostics.Stopwatch]::StartNew(); "
                "  $s.SpeakAsyncCancelAll(); "
                "  $null = $s.SpeakAsync($line); "
                "  $sw.Stop(); "
                "  Write-Output ('ACK|' + $seq + '|' + [math]::Round($sw.Elapsed.TotalMilliseconds, 3)); "
                "  [Console]::Out.Flush(); "
                "} "
                "$s.SpeakAsyncCancelAll(); "
                "$s.Dispose()"
            )
            self._proc = subprocess.Popen(
                [str(ps), "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._thread = threading.Thread(target=self._read_worker_stdout, daemon=True)
            self._thread.start()
        except Exception:
            self._proc = None

    def _read_worker_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                rx_perf = time.perf_counter()
                line = raw_line.strip()
                if line == "WARM":
                    self._warm_perf = rx_perf
                    self._ready.set()
                    with self._lock:
                        pending = self._pending
                        self._pending = None
                        if pending and not self._closed:
                            self._write_line_locked(pending[0], enqueue_mono=pending[1], buffered_until_warm=True)
                elif line.startswith("ACK|"):
                    parts = line.split("|")
                    if len(parts) == 3:
                        ack_seq = _parse_strict_int(parts[1])
                        ps_ms = _parse_strict_float(parts[2])
                        if ack_seq is not None:
                            with self._lock:
                                ev = self._events_by_seq.get(ack_seq)
                                if ev is not None:
                                    if ps_ms is not None and math.isfinite(ps_ms):
                                        ev["ps_speak_async_ms"] = round(ps_ms, 3)
                                    w_perf = ev.get("_write_start_perf")
                                    if isinstance(w_perf, float):
                                        ev["ps_ack_delay_ms"] = round((rx_perf - w_perf) * 1000.0, 3)
        except Exception:
            pass

    def _write_line_locked(
        self,
        text: str,
        *,
        enqueue_mono: float,
        buffered_until_warm: bool = False,
    ) -> Optional[Dict[str, Any]]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return None
        clean = text.replace("\r", " ").replace("\n", " ").strip()
        if not clean:
            return None
        self._seq += 1
        seq = self._seq
        w_t0 = time.perf_counter()
        ev: Dict[str, Any] = {
            "seq": seq,
            "text": clean,
            "enqueue_mono_s": float(enqueue_mono),
            "buffered_until_warm": buffered_until_warm,
            "write_ok": False,
            "worker_write_call_ms": None,
            "ps_speak_async_ms": None,
            "ps_ack_delay_ms": None,
            "_write_start_perf": w_t0,
        }
        self._events.append(ev)
        self._events_by_seq[seq] = ev
        ok = False
        try:
            proc.stdin.write(clean + "\n")
            proc.stdin.flush()
            ok = True
        except Exception:
            ok = False
        w_t1 = time.perf_counter()
        ev["write_ok"] = ok
        ev["worker_write_call_ms"] = round((w_t1 - w_t0) * 1000.0, 4)
        return ev

    def speak(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._closed:
            return None
        enqueue_mono = time.monotonic()
        if self._ready.is_set():
            with self._lock:
                return self._write_line_locked(text, enqueue_mono=enqueue_mono, buffered_until_warm=False)
        with self._lock:
            self._pending = (text, enqueue_mono)
        return None

    def get_telemetry(
        self,
        *,
        launch_mono: float,
        t_base: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            alive = bool(proc is not None and proc.poll() is None and not self._closed)
            warm_delay_ms = (
                round((self._warm_perf - self._start_perf) * 1000.0, 3)
                if (self._start_perf is not None and self._warm_perf is not None)
                else None
            )
            public_events: List[Dict[str, Any]] = []
            for ev in self._events:
                enq_m = float(ev["enqueue_mono_s"])
                public_events.append({
                    "seq": ev["seq"],
                    "text": ev["text"],
                    "enqueue_wall_s": round(enq_m - launch_mono, 4),
                    "enqueue_capture_s": round(enq_m - t_base, 4) if t_base is not None else None,
                    "buffered_until_warm": ev["buffered_until_warm"],
                    "write_ok": ev["write_ok"],
                    "worker_write_call_ms": ev["worker_write_call_ms"],
                    "ps_speak_async_ms": ev["ps_speak_async_ms"],
                    "ps_ack_delay_ms": ev["ps_ack_delay_ms"],
                })
        return {
            "audio_timing_scope": AUDIO_TIMING_SCOPE,
            "acoustic_onset_measured": False,
            "zero_dispatch_explanation": (
                "Earlier 0.0 ms dispatch readings occurred because time.monotonic() on Windows has "
                "~15.6 ms tick resolution while pipe writes complete in <0.2 ms. Current telemetry "
                "uses time.perf_counter() for worker_write_call_ms and PowerShell Stopwatch/ACK for "
                "ps_speak_async_ms and ps_ack_delay_ms; it explicitly does NOT measure physical acoustic onset."
            ),
            "audio_cues_enabled": self.enabled,
            "speaker_warmed": self._ready.is_set(),
            "speaker_warm_delay_ms": warm_delay_ms,
            "speaker_alive_at_telemetry": alive,
            "events": public_events,
        }

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                proc.stdin.write("__EXIT__\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


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
                "$s.Rate = 3; "
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
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    pre_cue_s: float = DEFAULT_PRE_CUE_S,
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

    plan = build_cued_target_plan(
        targets=targets, hold_s=hold_s, gap_s=gap_s, initial_settle_s=1.0, pre_cue_s=pre_cue_s
    )
    total_target_s = plan[-1]["offset_from_ready_end_s"] + 2.0
    seconds = int(math.ceil(total_target_s + min(readiness_timeout_s, 15.0)))
    if total_target_s > 65.0:
        print(
            f"[WARNING] Planned cued duration (~{total_target_s:.1f}s) may approach the 4096-frame "
            "DSP recorder capacity at 60 FPS.",
            flush=True,
        )
    cmd = build_gaze_capture_command(
        exe, dsp_dir, diag_csv, camera, seconds, width=width, height=height, fps=fps
    )

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

    speaker = _CueSpeaker(enabled=audio)
    speaker.start()

    clock = time.monotonic
    launch_unix_s = time.time()
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
    pre_cue_state: Optional[Dict[str, Any]] = None
    raw_targets: List[Dict[str, Any]] = []
    status_samples: List[Tuple[int, float]] = []
    console_tail: "collections.deque[str]" = collections.deque(maxlen=200)
    all_targets_done_at: Optional[float] = None
    stop_sent = False
    speaker_telemetry: Optional[Dict[str, Any]] = None

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
            print(
                f"[Cued Capture] Calibration ready (+{gate.calib_ready_time - launch_mono:.2f}s since launch); "
                "starting target cues...",
                flush=True,
            )
            speaker.speak("Ready.")

    def _drain_inbox() -> None:
        while True:
            try:
                line, rx_time = inbox.get_nowait()
            except queue.Empty:
                break
            _handle_line(line, rx_time)

    print("[Cued Capture] Hold steady neutral gaze for initial calibration...", flush=True)
    speaker.speak("Hold center for calibration.")

    try:
        while proc.poll() is None:
            now = clock()
            _drain_inbox()

            if gate.calib_ready_time is None:
                if now - launch_mono > readiness_timeout_s:
                    raise RuntimeError("Timed out waiting for live calibrated gaze readiness (state=0)")
            else:
                rel_ready = now - gate.calib_ready_time
                if active_idx is None and next_idx < len(plan):
                    item = plan[next_idx]
                    pre_cue_off = item["offset_from_ready_pre_cue_s"]
                    if (
                        pre_cue_state is None
                        and pre_cue_off is not None
                        and rel_ready >= pre_cue_off
                        and rel_ready < item["offset_from_ready_start_s"]
                    ):
                        pc_t0 = clock()
                        pc_perf0 = time.perf_counter()
                        prep_msg = f"Prepare: move eyes to {item['name']} now."
                        print(
                            f"[PRE-CUE   +{rel_ready:.2f}s from ready | wall=+{pc_t0 - launch_mono:.2f}s] {prep_msg}",
                            flush=True,
                        )
                        pc_ev = speaker.speak(prep_msg)
                        pc_perf1 = time.perf_counter()
                        pre_cue_state = {
                            "pre_cue_mono": pc_t0,
                            "pre_cue_dispatch_ms": round((pc_perf1 - pc_perf0) * 1000.0, 4),
                            "pre_cue_speaker_seq": pc_ev["seq"] if isinstance(pc_ev, dict) else None,
                        }
                    if rel_ready >= item["offset_from_ready_start_s"]:
                        active_idx = next_idx
                        st_t0 = clock()
                        st_perf0 = time.perf_counter()
                        msg = f"Hold {item['name']} still."
                        print(
                            f"[CUE START +{rel_ready:.2f}s from ready | wall=+{st_t0 - launch_mono:.2f}s] {msg}",
                            flush=True,
                        )
                        st_ev = speaker.speak(f"Hold {item['name'].lower()}.")
                        st_perf1 = time.perf_counter()
                        raw_targets.append({
                            "name": item["name"],
                            "pre_cue_mono": pre_cue_state["pre_cue_mono"] if pre_cue_state else None,
                            "pre_cue_dispatch_ms": pre_cue_state["pre_cue_dispatch_ms"] if pre_cue_state else None,
                            "pre_cue_speaker_seq": pre_cue_state["pre_cue_speaker_seq"] if pre_cue_state else None,
                            "start_mono": st_t0,
                            "cue_start_dispatch_ms": round((st_perf1 - st_perf0) * 1000.0, 4),
                            "cue_start_speaker_seq": st_ev["seq"] if isinstance(st_ev, dict) else None,
                            "end_mono": None,
                        })
                        pre_cue_state = None
                elif active_idx is not None:
                    item = plan[active_idx]
                    if rel_ready >= item["offset_from_ready_end_s"]:
                        raw_targets[active_idx]["end_mono"] = now
                        print(
                            f"[CUE END   +{rel_ready:.2f}s from ready | wall=+{now - launch_mono:.2f}s] "
                            f"Done {item['name']}.",
                            flush=True,
                        )
                        active_idx = None
                        next_idx += 1
                        if next_idx >= len(plan):
                            all_targets_done_at = now
                            print("[Cued Capture] All targets complete; finishing capture cleanly...", flush=True)
                            speaker.speak("Complete.")
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
        # CRITICAL: sample speaker telemetry BEFORE speaker.close() so speaker_alive_at_telemetry
        # reflects live worker health rather than post-shutdown state.
        if hasattr(speaker, "get_telemetry"):
            speaker_telemetry = speaker.get_telemetry(launch_mono=launch_mono)
        speaker.close()

    tail_list = list(console_tail)
    if proc.returncode != 0:
        raise RuntimeError(f"core_engine exited with code {proc.returncode}; tail={tail_list[-15:]}")
    if gate.calib_ready_time is None or len(raw_targets) != len(plan) or any(t["end_mono"] is None for t in raw_targets):
        raise RuntimeError(f"Cued capture ended before all targets completed; tail={tail_list[-15:]}")

    trace_times = _extract_trace_capture_times(dsp_dir / "trace.txt")
    t_base, sync_method, clock_sync_diag = _resolve_capture_epoch_with_diagnostics(
        status_samples, trace_times, epoch_mono, launch_mono, wall_origin_unix_s=launch_unix_s
    )
    calib_ready_s = round(gate.calib_ready_time - t_base, 4)
    effective_gap_pre_cue = min(float(pre_cue_s), max(0.0, float(gap_s) - 0.5)) if pre_cue_s > 0.0 else 0.0

    # Enrich speaker telemetry with capture_s timestamps now that t_base is resolved
    events_by_seq: Dict[int, Dict[str, Any]] = {}
    if speaker_telemetry is not None:
        for ev in speaker_telemetry.get("events") or []:
            if ev.get("enqueue_wall_s") is not None:
                ev["enqueue_capture_s"] = round(launch_mono + float(ev["enqueue_wall_s"]) - t_base, 4)
            if isinstance(ev.get("seq"), int):
                events_by_seq[ev["seq"]] = ev
    else:
        speaker_telemetry = {
            "audio_timing_scope": AUDIO_TIMING_SCOPE,
            "acoustic_onset_measured": False,
            "audio_cues_enabled": bool(audio),
            "speaker_warmed": False,
            "speaker_warm_delay_ms": None,
            "speaker_alive_at_telemetry": False,
            "events": [],
        }

    scheduled_targets: List[Dict[str, Any]] = []
    for t in raw_targets:
        pc_ev = events_by_seq.get(t["pre_cue_speaker_seq"]) if t.get("pre_cue_speaker_seq") is not None else None
        st_ev = events_by_seq.get(t["cue_start_speaker_seq"]) if t.get("cue_start_speaker_seq") is not None else None
        scheduled_targets.append({
            "name": t["name"],
            "pre_cue_s": round(float(t["pre_cue_mono"]) - t_base, 4) if t["pre_cue_mono"] is not None else None,
            "pre_cue_wall_s": round(float(t["pre_cue_mono"]) - launch_mono, 4) if t["pre_cue_mono"] is not None else None,
            "pre_cue_dispatch_ms": t["pre_cue_dispatch_ms"],
            "pre_cue_worker_write_call_ms": pc_ev["worker_write_call_ms"] if pc_ev else None,
            "pre_cue_ps_speak_async_ms": pc_ev["ps_speak_async_ms"] if pc_ev else None,
            "pre_cue_ps_ack_delay_ms": pc_ev["ps_ack_delay_ms"] if pc_ev else None,
            "target_start_s": round(float(t["start_mono"]) - t_base, 4),
            "cue_start_wall_s": round(float(t["start_mono"]) - launch_mono, 4),
            "cue_start_dispatch_ms": t["cue_start_dispatch_ms"],
            "cue_start_worker_write_call_ms": st_ev["worker_write_call_ms"] if st_ev else None,
            "cue_start_ps_speak_async_ms": st_ev["ps_speak_async_ms"] if st_ev else None,
            "cue_start_ps_ack_delay_ms": st_ev["ps_ack_delay_ms"] if st_ev else None,
            "target_end_s": round(float(t["end_mono"]) - t_base, 4),
            "cue_end_wall_s": round(float(t["end_mono"]) - launch_mono, 4),
        })

    schedule_payload = {
        "schema_version": 2,
        "command": cmd,
        "dsp_record_dir": str(dsp_dir),
        "diag_csv": str(diag_csv),
        "camera": camera,
        "width": width,
        "height": height,
        "fps": fps,
        "requested_pre_cue_s": pre_cue_s,
        "pre_cue_s": effective_gap_pre_cue,
        "audio_cues_enabled": bool(audio),
        "audio_timing_scope": AUDIO_TIMING_SCOPE,
        "time_sync_method": sync_method,
        "first_frame_capture_s": round(float(trace_times[0]), 4) if trace_times else None,
        "gaze_filter": GAZE_FILTER_SPEC,
        "wash_in_s": DEFAULT_CUED_WASH_IN_S,
        "blink_pad_frames": DEFAULT_BLINK_PAD_FRAMES,
        "calib_ready_s": calib_ready_s,
        "clock_sync_diagnostics": clock_sync_diag,
        "audio_dispatch_telemetry": speaker_telemetry,
        "targets": scheduled_targets,
    }
    sched_path = run_dir / "cued_schedule.json"
    sched_path.write_text(json.dumps(schedule_payload, indent=2), encoding="utf-8")
    wins_summary = ", ".join(
        f"{t['name']}=[{t['target_start_s']:.3f}, {t['target_end_s']:.3f})"
        for t in scheduled_targets
    )
    print(
        f"[Cued Capture] Synced schedule ({sync_method}): calib_ready_s={calib_ready_s:.4f}s | {wins_summary}",
        flush=True,
    )
    return sched_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory (artifacts/reports, or legacy flat recorder dir containing trace.txt)")
    ap.add_argument("--dsp-record-dir", type=Path, default=None, help="Recorder directory for trace.txt/COMPLETE (default for --cued-capture: <run-dir>/dsp_record; offline auto-detects <run-dir> or <run-dir>/dsp_record)")
    ap.add_argument("--diag-csv", type=Path, default=None, help="Path to gaze_diag.csv (default: <run-dir>/gaze_diag.csv)")
    ap.add_argument("--windows", type=str, default=None, help="Comma-separated NAME:START_S-END_S intervals, e.g. 'LEFT:5-15,RIGHT:17-27'")
    ap.add_argument("--auto-windows", action="store_true", help="Automatically segment active tracking windows")
    ap.add_argument("--cued-schedule", type=Path, default=None, help="Path to cued_schedule.json (auto-detected in <run-dir> if present and --windows/--auto-windows omitted)")
    ap.add_argument("--wash-in-s", type=float, default=None, help="Seconds to trim from the start of each window (finite, >= 0; default: 0.5 with cued schedule, else 0.0)")
    ap.add_argument(
        "--strict-protocol-wash-in",
        action="store_true",
        help="Restart the wash-in countdown inside a window whenever a frame is excluded or gap > 100 ms (matches evaluate_holds.py)",
    )
    ap.add_argument(
        "--require-qualification-pass",
        action="store_true",
        help="Return exit code 2 when archive integrity is OK but physical fixation qualification fails (default descriptive mode returns 0 if alignment_ok else 1)",
    )
    ap.add_argument("--blink-pad-frames", type=int, default=DEFAULT_BLINK_PAD_FRAMES, help="Frames to exclude before/after blink or face loss (default: 2)")
    ap.add_argument("--deployed-models-dir", type=Path, default=DEFAULT_DEPLOYED_MODELS_DIR)
    ap.add_argument("--candidate-models-dir", type=Path, default=None, help="Optional directory containing candidate gaze_yaw.dll / gaze_pitch.dll")
    ap.add_argument("--out-json", type=Path, default=None, help="Output JSON report path (default: <run-dir>/gaze_attribution_report.json)")
    ap.add_argument("--out-md", type=Path, default=None, help="Output Markdown report path (default: <run-dir>/gaze_attribution_report.md)")
    # Cued live capture options
    ap.add_argument("--cued-capture", action="store_true", help="Launch safe cued core_engine.exe capture (--os log --no-hand --filter dsp) then analyze")
    ap.add_argument("--exe", type=Path, default=DEFAULT_ENGINE_EXE, help="Path to core_engine.exe for --cued-capture")
    ap.add_argument("--camera", type=int, default=0, help="Camera index for --cued-capture")
    ap.add_argument("--width", type=int, default=None, help="Optional camera capture width for --cued-capture (forwarded to core_engine.exe)")
    ap.add_argument("--height", type=int, default=None, help="Optional camera capture height for --cued-capture (forwarded to core_engine.exe)")
    ap.add_argument("--fps", type=float, default=None, help="Optional camera capture FPS for --cued-capture (forwarded to core_engine.exe)")
    ap.add_argument("--targets", type=str, default=",".join(DEFAULT_CUED_TARGETS), help="Comma-separated cued targets")
    ap.add_argument("--hold-s", type=float, default=10.0, help="Seconds per cued target (default: 10.0)")
    ap.add_argument("--gap-s", type=float, default=2.0, help="Transition gap seconds between targets (default: 2.0)")
    ap.add_argument(
        "--pre-cue-s",
        type=float,
        default=DEFAULT_PRE_CUE_S,
        help="Seconds before each target start to announce eye-movement prep prompt (capped to gap_s - 0.5; 0 to disable; default: 2.5)",
    )
    ap.add_argument("--no-audio", action="store_true", help="Disable spoken TTS cues during --cued-capture")
    ap.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompt for --cued-capture")
    args = ap.parse_args(argv)

    if args.windows is not None and args.auto_windows:
        ap.error("Specify either --windows or --auto-windows, not both.")
    if args.wash_in_s is not None and (not math.isfinite(args.wash_in_s) or args.wash_in_s < 0.0):
        ap.error("--wash-in-s must be finite and >= 0")
    if args.blink_pad_frames < 0:
        ap.error("--blink-pad-frames must be >= 0")
    if args.width is not None and args.width <= 0:
        ap.error("--width must be > 0")
    if args.height is not None and args.height <= 0:
        ap.error("--height must be > 0")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0.0):
        ap.error("--fps must be > 0")
    if not math.isfinite(args.hold_s) or args.hold_s <= DEFAULT_CUED_WASH_IN_S:
        ap.error(f"--hold-s must be finite and > {DEFAULT_CUED_WASH_IN_S}")
    if not math.isfinite(args.gap_s) or args.gap_s < 0.0:
        ap.error("--gap-s must be finite and >= 0")
    if not math.isfinite(args.pre_cue_s) or args.pre_cue_s < 0.0:
        ap.error("--pre-cue-s must be finite and >= 0")

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
            width=args.width,
            height=args.height,
            fps=args.fps,
            pre_cue_s=args.pre_cue_s,
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

    if args.wash_in_s is not None:
        wash_in_s = args.wash_in_s
        wash_in_source = "cli_override"
    elif cued_meta is not None:
        wash_in_s = DEFAULT_CUED_WASH_IN_S
        wash_in_source = "cued_schedule_default"
    else:
        wash_in_s = 0.0
        wash_in_source = "analyzer_default"
    padded_bad = compute_blink_pad_indices(rows, args.blink_pad_frames)

    if integrity["schema_errors"] > 0 or integrity["nonfinite_errors"] > 0:
        windows_res = []
    else:
        windows_res = [
            analyze_window(
                name, s, e, rows, dep_yaw, dep_pitch, cand_yaw, cand_pitch,
                logged_y_sign_mult=integrity["logged_y_sign_mult"],
                wash_in_s=wash_in_s,
                blink_pad_frames=args.blink_pad_frames,
                padded_bad_indices=padded_bad,
                strict_protocol_wash_in=args.strict_protocol_wash_in,
            )
            for name, s, e in windows_spec
        ]
    directions = compute_direction_checks(windows_res)
    qualification = compute_qualification_summary(integrity, windows_res, directions)

    all_cap_times = [float(r["capture_s"]) for r in rows if math.isfinite(r["capture_s"])]
    delivered_fps_overall = _compute_cadence_fps(all_cap_times)
    capture_span_s = (
        float(all_cap_times[-1] - all_cap_times[0])
        if len(all_cap_times) >= 2
        else 0.0
    )
    max_frame_gap_ms = (
        float(np.max(np.diff(np.array(all_cap_times, dtype=np.float64)) * 1000.0))
        if len(all_cap_times) >= 2
        else 0.0
    )
    raw_req_fps = cued_meta.get("fps") if isinstance(cued_meta, dict) else args.fps
    requested_fps = float(raw_req_fps) if raw_req_fps is not None else None
    low_fps_windows = [
        w["window"]
        for w in windows_res
        if w.get("delivered_fps") is not None and w["delivered_fps"] < LOW_DELIVERED_FPS_WARN_THRESHOLD
    ]
    low_fps_overall = (
        delivered_fps_overall is not None
        and delivered_fps_overall < LOW_DELIVERED_FPS_WARN_THRESHOLD
    )
    low_fps_warning = bool(low_fps_overall or low_fps_windows)
    cadence = {
        "requested_fps": requested_fps,
        "delivered_fps_overall": delivered_fps_overall,
        "delivered_fps_formula": "(N - 1) / (last_capture_s - first_capture_s)",
        "low_fps_threshold": LOW_DELIVERED_FPS_WARN_THRESHOLD,
        "low_fps_warning": low_fps_warning,
        "low_fps_windows": low_fps_windows,
        "total_rows": len(rows),
        "capture_span_s": capture_span_s,
        "max_frame_gap_ms": max_frame_gap_ms,
    }

    if not integrity["alignment_ok"]:
        exit_code = EXIT_CODE_INTEGRITY_FAILED
    elif args.require_qualification_pass and not qualification["overall_qualification_pass"]:
        exit_code = EXIT_CODE_QUALIFICATION_FAILED
    else:
        exit_code = EXIT_CODE_OK

    report = {
        "integrity": integrity,
        "qualification": qualification,
        "exit_code_policy": {
            "require_qualification_pass": bool(args.require_qualification_pass),
            "computed_exit_code": exit_code,
            "codes": {
                "0": "Archive integrity and alignment OK (and qualification PASS when --require-qualification-pass is set).",
                "1": "Archive integrity, CSV schema, or recorder alignment FAILED (alignment_ok=False).",
                "2": "Archive integrity and alignment OK, but fixation qualification FAILED under --require-qualification-pass.",
            },
        },
        "cadence": cadence,
        "deployed_models": dep_inv,
        "candidate_models": cand_inv,
        "cued_schedule": cued_meta,
        "analysis_params": {
            "wash_in_s": wash_in_s,
            "wash_in_source": wash_in_source,
            "strict_protocol_wash_in": args.strict_protocol_wash_in,
            "blink_pad_frames": args.blink_pad_frames,
            "fixed_y_sign": FIXED_Y_SIGN,
        },
        "limits": {
            "min_samples": MIN_SAMPLES,
            "max_pop_sd": LIMIT_SD,
            "max_range": LIMIT_RANGE,
            "max_abs_ols_slope_per_s": LIMIT_SLOPE,
            "max_fresh_gap_ms": MAX_FRESH_GAP_MS,
            "coord_tol": COORD_TOL,
        },
        "windows": windows_res,
        "directions": directions,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_text = format_markdown_report(report)
    out_md.write_text(md_text, encoding="utf-8")

    if not integrity["alignment_ok"]:
        print("=== INVALID RUN / ALIGNMENT FAILED ===")
    print("=== ALIGNMENT & INTEGRITY ===")
    print(
        f"alignment_ok={integrity['alignment_ok']} rows={integrity['trace_rows']} "
        f"COMPLETE={integrity['complete_marker_present']} "
        f"overflow={integrity['overflow']} config_changes={integrity['config_changes']} "
        f"schema_err={integrity['schema_errors']} "
        f"fid_mis={integrity['frame_id_mismatches']} ts_mis={integrity['ts_ms_mismatches']} "
        f"cap_s_mis={integrity['capture_s_mismatches']} state_mis={integrity['state_mismatches']} "
        f"norm_mis=({integrity['norm_x_mismatches']}, {integrity['norm_y_mismatches']}) "
        f"non_inc=({integrity['frame_id_non_increasing']}, {integrity['ts_ms_non_increasing']}, {integrity['capture_s_non_increasing']}) "
        f"nonfinite={integrity['nonfinite_errors']} y_sign={integrity['logged_y_sign']} "
        f"max_dnorm=({integrity['max_abs_delta_norm_x']:.3e}, {integrity['max_abs_delta_norm_y']:.3e})"
    )
    q_counts = qualification["per_axis_pass_counts"]
    print(
        f"Qualification: overall={'PASS' if qualification['overall_qualification_pass'] else 'FAIL'} "
        f"axes={q_counts['passed']}/{q_counts['total']} "
        f"horiz={qualification['horizontal_qualification_pass']} "
        f"vert={qualification['vertical_qualification_pass']} "
        f"dir={qualification['direction_verification_pass']} "
        f"scored_authorized={qualification['scored_responsiveness_authorized']} "
        f"exit_code={exit_code}"
    )
    req_fps_disp = f"{requested_fps:.2f}" if requested_fps is not None else "unspecified"
    deliv_fps_disp = f"{delivered_fps_overall:.2f}" if delivered_fps_overall is not None else "n/a"
    print(
        f"Cadence: requested_fps={req_fps_disp} | delivered_fps={deliv_fps_disp} "
        f"(span={capture_span_s:.2f}s, max_gap={max_frame_gap_ms:.1f}ms)"
    )
    if low_fps_warning:
        fps_note = deliv_fps_disp if low_fps_overall else f"windows={low_fps_windows}"
        print(
            f"LOW DELIVERED FPS WARNING: {fps_note} < {LOW_DELIVERED_FPS_WARN_THRESHOLD:.0f} FPS; "
            "check room lighting / webcam auto-exposure"
        )
    for w in windows_res:
        if w["eligible_frames"] == 0:
            print(f"[{w['window']}] NO ELIGIBLE FRAMES (total={w['total_frames_in_window']})")
            continue
        rx, ry = w["raw_x"], w["raw_y"]
        vd = w["variance_decomposition"]
        w_fps_disp = f"{w['delivered_fps']:.2f}" if w.get("delivered_fps") is not None else "n/a"
        print(
            f"[{w['window']}] n={rx['n']} fps={w_fps_disp} | "
            f"raw_x: mean={rx['mean']:.5f} SD={rx['pop_sd']:.6f} rng={rx['range']:.6f} |slope|={abs(rx['ols_slope_per_s']):.6f} gap={rx['max_gap_ms']:.1f}ms [{'PASS' if rx['pass_all'] else 'FAIL'}] | "
            f"raw_y: mean={ry['mean']:.5f} SD={ry['pop_sd']:.6f} rng={ry['range']:.6f} |slope|={abs(ry['ols_slope_per_s']):.6f} gap={ry['max_gap_ms']:.1f}ms [{'PASS' if ry['pass_all'] else 'FAIL'}]"
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
                f"X: mean={crx['mean']:.5f} SD={crx['pop_sd']:.6f} rng={crx['range']:.6f} gap={crx['max_gap_ms']:.1f}ms [{'PASS' if crx['pass_all'] else 'FAIL'}] | "
                f"Y: mean={cry['mean']:.5f} SD={cry['pop_sd']:.6f} rng={cry['range']:.6f} gap={cry['max_gap_ms']:.1f}ms [{'PASS' if cry['pass_all'] else 'FAIL'}] | "
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
