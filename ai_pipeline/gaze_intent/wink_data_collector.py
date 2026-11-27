"""Cued Blink / Wink Dataset Collector v2 (schema-enforced, runtime-parity).

Fixes the root causes of legacy ``blink_dataset.csv`` corruption:
  1. Explicit ``schema_version`` + fixed 29-column header (refuses to append on mismatch).
  2. Non-destructive per-session output path (never silently appends onto legacy CSVs).
  3. Timed ``CueEventStateMachine`` (prepare -> active -> release -> auto-return neutral)
     instead of latched key toggles, preventing ~1000 ms reaction-lag label contamination.
  4. Enables ``output_face_blendshapes=True`` and records eye-aligned MediaPipe blendshapes,
     raw + baseline + normalized EAR, gaze context (sx, sy, head pose), and live HybridBlinkDetector
     state for full Engineer 4-style auditability.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

# Ensure sibling gaze_intent module is importable whether invoked directly or via pytest.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gaze_intent import (  # noqa: E402
    HybridBlinkDetector,
    LEFT_EYE,
    LEFT_IRIS,
    RIGHT_EYE,
    RIGHT_IRIS,
    calculate_ear,
    extract_eye_blendshapes,
    extract_gaze_features,
)

BLINK_COLLECTOR_SCHEMA_VERSION = "blink_collector_v2"

BLINK_COLLECTOR_HEADER: Tuple[str, ...] = (
    "schema_version",
    "subject_id",
    "session_id",
    "frame_id",
    "ts_ms",
    "capture_s",
    "detected",
    "cue_label",
    "cue_phase",
    "raw_ear_l",
    "raw_ear_r",
    "base_ear_l",
    "base_ear_r",
    "norm_ear_l",
    "norm_ear_r",
    "bs_blink_l",
    "bs_blink_r",
    "bs_lookdown_l",
    "bs_lookdown_r",
    "bs_squint_l",
    "bs_squint_r",
    "bs_wide_l",
    "bs_wide_r",
    "sx",
    "sy",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "hybrid_state",
)

VALID_CUE_LABELS = frozenset((-1, 0, 1, 2, 3))
VALID_HYBRID_STATES = frozenset((-1, 0, 1, 2, 3))
VALID_CUE_PHASES = frozenset((
    "neutral",
    "prepare",
    "active_left_wink",
    "active_right_wink",
    "active_closure",
    "active_downward_gaze",
    "release",
))


@dataclass(frozen=True)
class CueSpec:
    name: str
    target_label: int
    active_phase: str
    prepare_s: float
    active_s: float
    release_s: float


DEFAULT_CUE_SPECS: Dict[str, CueSpec] = {
    "left_wink": CueSpec("left_wink", 1, "active_left_wink", 0.25, 0.65, 0.25),
    "right_wink": CueSpec("right_wink", 2, "active_right_wink", 0.25, 0.65, 0.25),
    "closure": CueSpec("closure", 3, "active_closure", 0.25, 1.20, 0.25),
    "downward_gaze": CueSpec("downward_gaze", 0, "active_downward_gaze", 0.25, 1.20, 0.25),
}


class CueEventStateMachine:
    """Timed cue window state machine with transition guards and auto-return to neutral.

    During ``prepare`` and ``release`` guard windows, ``cue_label`` is ``-1`` so boundary
    reaction-time frames are cleanly separated from steady active/neutral frames.
    """

    def __init__(self, cue_specs: Optional[Dict[str, CueSpec]] = None) -> None:
        self._specs: Dict[str, CueSpec] = dict(cue_specs or DEFAULT_CUE_SPECS)
        self._active_spec: Optional[CueSpec] = None
        self._start_s: Optional[float] = None

    def trigger(self, cue_name: str, now_s: float) -> None:
        if cue_name not in self._specs:
            raise ValueError(f"Unknown cue_name: {cue_name!r}")
        if not math.isfinite(now_s):
            raise ValueError(f"Non-finite timestamp: {now_s!r}")
        self._active_spec = self._specs[cue_name]
        self._start_s = float(now_s)

    def reset_neutral(self) -> None:
        self._active_spec = None
        self._start_s = None

    def current(self, now_s: float) -> Tuple[int, str]:
        """Return ``(cue_label, cue_phase)`` at monotonic time ``now_s``."""
        if not math.isfinite(now_s):
            raise ValueError(f"Non-finite timestamp: {now_s!r}")
        if self._active_spec is None or self._start_s is None:
            return 0, "neutral"

        elapsed = float(now_s) - self._start_s
        if elapsed < 0.0:
            self.reset_neutral()
            return 0, "neutral"

        spec = self._active_spec
        if elapsed < spec.prepare_s:
            return -1, "prepare"
        if elapsed < spec.prepare_s + spec.active_s:
            return spec.target_label, spec.active_phase
        if elapsed < spec.prepare_s + spec.active_s + spec.release_s:
            return -1, "release"

        self.reset_neutral()
        return 0, "neutral"


def _require_finite(name: str, value: float) -> float:
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Field '{name}' must be finite, got {value!r}")
    return fval


def _require_unit_blendshape(name: str, value: float) -> float:
    fval = _require_finite(name, value)
    if fval < 0.0 or fval > 1.0:
        raise ValueError(f"Blendshape '{name}' must be in [0.0, 1.0], got {fval!r}")
    return fval


def build_collector_row(
    *,
    subject_id: str,
    session_id: str,
    frame_id: int,
    ts_ms: int,
    capture_s: float,
    detected: int,
    cue_label: int,
    cue_phase: str,
    raw_ear_l: float,
    raw_ear_r: float,
    base_ear_l: float,
    base_ear_r: float,
    norm_ear_l: float,
    norm_ear_r: float,
    blendshapes: Sequence[float],
    sx: float,
    sy: float,
    head_pitch_deg: float,
    head_yaw_deg: float,
    head_roll_deg: float,
    hybrid_state: int,
) -> Tuple[object, ...]:
    """Validate and assemble a single 29-column v2 blink collector CSV row."""
    if not subject_id or not str(subject_id).strip():
        raise ValueError("subject_id must be a non-empty string")
    if not session_id or not str(session_id).strip():
        raise ValueError("session_id must be a non-empty string")
    if int(frame_id) < 0:
        raise ValueError(f"frame_id must be >= 0, got {frame_id!r}")
    if int(ts_ms) < 0:
        raise ValueError(f"ts_ms must be >= 0, got {ts_ms!r}")
    if int(detected) not in (0, 1):
        raise ValueError(f"detected must be 0 or 1, got {detected!r}")
    if int(cue_label) not in VALID_CUE_LABELS:
        raise ValueError(f"Invalid cue_label: {cue_label!r}")
    if cue_phase not in VALID_CUE_PHASES:
        raise ValueError(f"Invalid cue_phase: {cue_phase!r}")
    if int(hybrid_state) not in VALID_HYBRID_STATES:
        raise ValueError(f"Invalid hybrid_state: {hybrid_state!r}")
    if len(blendshapes) != 8:
        raise ValueError(f"Expected 8 eye-aligned blendshapes, got {len(blendshapes)}")

    bs_names = (
        "bs_blink_l", "bs_blink_r",
        "bs_lookdown_l", "bs_lookdown_r",
        "bs_squint_l", "bs_squint_r",
        "bs_wide_l", "bs_wide_r",
    )
    bs_vals = tuple(_require_unit_blendshape(n, v) for n, v in zip(bs_names, blendshapes))

    row = (
        BLINK_COLLECTOR_SCHEMA_VERSION,
        str(subject_id).strip(),
        str(session_id).strip(),
        int(frame_id),
        int(ts_ms),
        f"{_require_finite('capture_s', capture_s):.6f}",
        int(detected),
        int(cue_label),
        cue_phase,
        f"{_require_finite('raw_ear_l', raw_ear_l):.6f}",
        f"{_require_finite('raw_ear_r', raw_ear_r):.6f}",
        f"{_require_finite('base_ear_l', base_ear_l):.6f}",
        f"{_require_finite('base_ear_r', base_ear_r):.6f}",
        f"{_require_finite('norm_ear_l', norm_ear_l):.6f}",
        f"{_require_finite('norm_ear_r', norm_ear_r):.6f}",
        *(f"{v:.6f}" for v in bs_vals),
        f"{_require_finite('sx', sx):.6f}",
        f"{_require_finite('sy', sy):.6f}",
        f"{_require_finite('head_pitch_deg', head_pitch_deg):.6f}",
        f"{_require_finite('head_yaw_deg', head_yaw_deg):.6f}",
        f"{_require_finite('head_roll_deg', head_roll_deg):.6f}",
        int(hybrid_state),
    )
    assert len(row) == len(BLINK_COLLECTOR_HEADER)
    return row


class BlinkCollectorCsvWriter:
    """Schema-safe CSV writer that refuses to append onto mismatched headers."""

    def __init__(self, output_path: str) -> None:
        self.output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._fp = None
        self._writer = None

    def open(self) -> None:
        file_exists = os.path.isfile(self.output_path) and os.path.getsize(self.output_path) > 0
        if file_exists:
            with open(self.output_path, mode="r", encoding="utf-8", newline="") as rf:
                reader = csv.reader(rf)
                existing_header = tuple(next(reader, ()))
            if existing_header != BLINK_COLLECTOR_HEADER:
                raise ValueError(
                    f"Schema mismatch in existing CSV '{self.output_path}': "
                    f"found {len(existing_header)} columns {existing_header!r}, "
                    f"expected {len(BLINK_COLLECTOR_HEADER)} columns for {BLINK_COLLECTOR_SCHEMA_VERSION}."
                )

        self._fp = open(self.output_path, mode="a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)
        if not file_exists:
            self._writer.writerow(BLINK_COLLECTOR_HEADER)
            self._fp.flush()

    def write_row(self, row: Sequence[object]) -> None:
        if self._writer is None:
            raise RuntimeError("BlinkCollectorCsvWriter is not open")
        if len(row) != len(BLINK_COLLECTOR_HEADER):
            raise ValueError(
                f"Row width mismatch: got {len(row)}, expected {len(BLINK_COLLECTOR_HEADER)}"
            )
        self._writer.writerow(row)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._writer = None

    def __enter__(self) -> "BlinkCollectorCsvWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def default_session_output_path(script_dir: str, session_id: Optional[str] = None) -> str:
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(script_dir, "datasets", f"blink_v2_session_{sid}.csv")


class DataCollector:
    def __init__(self, output_file: Optional[str] = None, subject_id: str = "subject_01") -> None:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.script_dir = _THIS_DIR
        self.subject_id = str(subject_id).strip() or "subject_01"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = output_file or default_session_output_path(self.script_dir, self.session_id)

        model_path = os.path.abspath(os.path.join(self.script_dir, "..", "..", "assets", "face_landmarker.task"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at:\n{model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.hybrid_blink = HybridBlinkDetector()
        self.cue_sm = CueEventStateMachine()

    def run(self) -> None:
        import cv2
        import mediapipe as mp

        with BlinkCollectorCsvWriter(self.output_file) as csv_writer:
            cap = cv2.VideoCapture(0)
            print("--- BLINK V2 CUED DATA COLLECTOR STARTED ---")
            print(f"Session CSV: {self.output_file}")
            print("Look neutrally at the camera for robust open-eye calibration...")

            frame_id = 0
            t0 = time.monotonic()
            last_ts_ms = 0

            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    now_s = time.monotonic() - t0
                    ts_ms = max(last_ts_ms + 1, int(round(now_s * 1000.0)))
                    last_ts_ms = ts_ms
                    frame_id += 1

                    frame = cv2.flip(frame, 1)
                    frame_h, frame_w, _ = frame.shape

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    results = self.detector.detect_for_video(mp_image, ts_ms)

                    cue_label, cue_phase = self.cue_sm.current(now_s)

                    if results.face_landmarks and results.facial_transformation_matrixes:
                        first_face_landmarks = results.face_landmarks[0]
                        pose_matrix = results.facial_transformation_matrixes[0]
                        bs_list = getattr(results, "face_blendshapes", None)
                        blendshapes = extract_eye_blendshapes(bs_list[0]) if bs_list else None

                        raw_left_ear = calculate_ear(first_face_landmarks, LEFT_EYE, frame_w, frame_h)
                        raw_right_ear = calculate_ear(first_face_landmarks, RIGHT_EYE, frame_w, frame_h)
                        _, _, sx, _, _, sy, head_pitch, head_yaw, head_roll = extract_gaze_features(
                            first_face_landmarks, pose_matrix
                        )

                        ear_valid = (
                            math.isfinite(raw_left_ear) and math.isfinite(raw_right_ear)
                            and all(math.isfinite(v) for v in (sx, sy, head_pitch, head_yaw, head_roll))
                            and blendshapes is not None
                        )
                        if not ear_valid:
                            self.hybrid_blink.reset_tracking_state()
                            cv2.putText(frame, "INVALID LANDMARKS - skipped", (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        elif not self.hybrid_blink.is_calibrated:
                            was_calibrated = self.hybrid_blink.is_calibrated
                            self.hybrid_blink.update_calibration(
                                raw_left_ear, raw_right_ear, head_pitch, sy, blendshapes
                            )
                            cv2.putText(
                                frame,
                                f"CALIBRATING: {self.hybrid_blink.current_calibration_frames}"
                                f"/{self.hybrid_blink.calibration_frame_count}",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2,
                            )
                            if not was_calibrated and self.hybrid_blink.is_calibrated:
                                print("\nCalibration Complete!")
                                print("Press '1' for timed LEFT WINK cue")
                                print("Press '2' for timed RIGHT WINK cue")
                                print("Press '3' for timed SUSTAINED CLOSURE cue")
                                print("Press '4' for timed DOWNWARD GAZE negative cue")
                                print("Press 'SPACE' to abort cue to NEUTRAL | 'Q' to quit")
                        else:
                            state, score_l, score_r, norm_l, norm_r, _ = self.hybrid_blink.step(
                                raw_left_ear, raw_right_ear, head_pitch, sy, ts_ms, blendshapes
                            )
                            row = build_collector_row(
                                subject_id=self.subject_id,
                                session_id=self.session_id,
                                frame_id=frame_id,
                                ts_ms=ts_ms,
                                capture_s=now_s,
                                detected=1,
                                cue_label=cue_label,
                                cue_phase=cue_phase,
                                raw_ear_l=raw_left_ear,
                                raw_ear_r=raw_right_ear,
                                base_ear_l=self.hybrid_blink.baseline_left_ear,
                                base_ear_r=self.hybrid_blink.baseline_right_ear,
                                norm_ear_l=norm_l,
                                norm_ear_r=norm_r,
                                blendshapes=blendshapes,
                                sx=sx,
                                sy=sy,
                                head_pitch_deg=head_pitch,
                                head_yaw_deg=head_yaw,
                                head_roll_deg=head_roll,
                                hybrid_state=state,
                            )
                            csv_writer.write_row(row)

                            cv2.putText(
                                frame,
                                f"CUE: {cue_phase} (label={cue_label}) | HYBRID={state}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            )
                            cv2.putText(
                                frame,
                                f"nEAR L={norm_l:.2f} R={norm_r:.2f} | Score L={score_l:.2f} R={score_r:.2f}",
                                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                            )

                        for idx in LEFT_EYE + RIGHT_EYE:
                            lm = first_face_landmarks[idx]
                            cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 2, (0, 255, 0), -1)
                        for idx in (LEFT_IRIS, RIGHT_IRIS):
                            lm = first_face_landmarks[idx]
                            cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 3, (0, 0, 255), -1)
                    else:
                        self.hybrid_blink.reset_tracking_state()
                        cv2.putText(frame, "NO FACE - window reset", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow("Blink v2 Cued Data Collector (Q to Quit)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        break
                    elif key == ord(" "):
                        self.cue_sm.reset_neutral()
                    elif key == ord("1"):
                        self.cue_sm.trigger("left_wink", now_s)
                    elif key == ord("2"):
                        self.cue_sm.trigger("right_wink", now_s)
                    elif key == ord("3"):
                        self.cue_sm.trigger("closure", now_s)
                    elif key == ord("4"):
                        self.cue_sm.trigger("downward_gaze", now_s)
            finally:
                cap.release()
                cv2.destroyAllWindows()
                self.detector.close()
        print(f"Data successfully saved to:\n{self.output_file}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
