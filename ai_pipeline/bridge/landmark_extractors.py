"""Landmark extractors for the C++ core engine (Engineer 1's bridge).

Python's ONLY job in the final pipeline is to run MediaPipe and hand back raw landmarks.
Everything else (iris/gaze maths, EAR, 30-frame calibration, the Random Forest, the hand pinch/fist
classifier + vote) runs natively in C++ -- see core_engine/include/gaze_math.hpp and hand_math.hpp, which are
line-for-line ports of gaze_intent.py, hand_tracker.py and hand_features.py (kept as the reference implementations).

Contract (called from core_engine/include/py_bridge.hpp, always on ONE thread):
    extractor = FaceLandmarkExtractor()          # or HandLandmarkExtractor(); no arguments
    pts = extractor.process_frame(rgb, ts_ms)
        rgb   : uint8 ndarray (H, W, 3), C-contiguous, ALREADY mirrored and in RGB order.
                It is a view of memory owned by the C++ core: use it during the call, never keep it.
        ts_ms : int, strictly increasing milliseconds
        pts   : float32 ndarray (N, 3) of the first face / hand, or an array of shape (0, 3) when nothing is
                detected.   FaceLandmarkExtractor: N = 484 = the 478 normalised (x, y, z) face points followed by
                6 rows that carry MediaPipe's 4x4 facial transformation matrix (16 floats, row-major, then 2 zeros):
                gaze_math.hpp turns it into the head pose (pitch / yaw / roll), like gaze_intent.py does.
                No matrix from MediaPipe counts as no face.
                HandLandmarkExtractor: N = 42 = the 21 normalised IMAGE landmarks (rows 0..20, used for the
                cursor) followed by the 21 WORLD landmarks (rows 21..41, metres, used for the 24 gesture features).
                Both groups must contain exactly 21 landmarks: a 20+22 split that also sums to 42 is malformed
                and is reported as a bridge error (exception), while the diagnostic still records the counts.
                This keeps malformed input distinguishable from an ordinary no-hand frame at the C++ boundary.
    extractor.shutdown()                         # releases the MediaPipe resources (called once at exit)

Rules: no OS access, no flipping/colour conversion. Exceptions are allowed: the C++ side counts them
and reports a bridge error, which is distinguishable from a no-hand frame. Never raise for "no face / no hand".

Environment:
    AI_PIPELINE_ROOT   same meaning as in gaze_intent.py / hand_tracker.py (default: the folder above this one)
    HAND_RUNNING_MODE  VIDEO (default, what hand_tracker.py uses: MediaPipe tracks the hand between frames)
                       or IMAGE (palm detection on every frame; use it together with --hand-roi)
    HAND_LANDMARK_DIAG 1/true/yes/on: write one scalar CSV line per processed hand frame (default: off).
                       This is observation only: it does not change the returned array or the tracking path.
    HAND_LANDMARK_DIAG_FILE  optional CSV path (default: hand_landmark_diag_<pid>.csv in the current folder)
"""
import csv
import os
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_EMPTY = np.zeros((0, 3), dtype=np.float32)
_POSE_ROWS = 6          # 4x4 matrix = 16 floats = 5.33 rows of 3 -> 6 rows, zero padded


def _asset_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("AI_PIPELINE_ROOT", os.path.abspath(os.path.join(here, "..")))
    path = os.path.abspath(os.path.join(project_root, "..", "assets", name))
    if not os.path.exists(path):
        raise FileNotFoundError(f"[CRITICAL ERROR] MediaPipe model missing at:\n{path}")
    return path


def _to_array(landmarks) -> np.ndarray:
    out = np.empty((len(landmarks), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


# One row per processed hand frame. ts_ms is the same integer the engine passes to
# process_frame (TrackingResult::ts_ms / "observation_ms" in the E4DSP trace), so the
# diagnostic joins the trial log exactly. "seq" is a per-extractor counter, not frame_id.
_DIAG_FIELDS = (
    "seq", "ts_ms", "mode", "img_n", "world_n", "n", "reason",
    "min_x", "max_x", "min_y", "max_y", "edge_l", "edge_r", "edge_t", "edge_b",
)


def _diag_enabled() -> bool:
    return os.environ.get("HAND_LANDMARK_DIAG", "").strip().lower() in ("1", "true", "yes", "on")


def _describe_hand(result, seq: int, ts_ms: int, mode: str) -> dict:
    """Classify one MediaPipe hand result for the diagnostic log.

    Mirrors the exact branch used by process_frame: the outer lists decide whether a
    result is empty; the inner counts are recorded so a short list is visible as
    MALFORMED instead of being silently packed. No value is changed here.
    """
    groups_img = getattr(result, "hand_landmarks", None) or []
    groups_world = getattr(result, "hand_world_landmarks", None) or []
    img_present = bool(groups_img)
    world_present = bool(groups_world)
    img_n = len(groups_img[0]) if img_present else 0
    world_n = len(groups_world[0]) if world_present else 0

    if not img_present and not world_present:
        reason = "BOTH_EMPTY"
    elif img_present and not world_present:
        reason = "IMAGE_ONLY"
    elif not img_present and world_present:
        reason = "WORLD_ONLY"
    elif img_n != 21 or world_n != 21:
        reason = "MALFORMED"
    else:
        reason = "OK"

    # The native bridge stores landmarks as float32. A value that is finite in
    # float64 can overflow to Inf when cast, so checking only the Python floats
    # would report "OK / n=42" for a payload the engine actually rejects. Check
    # the exact representation used downstream and report the rejection.
    if reason == "OK":
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                coords = np.asarray(
                    [[float(p.x), float(p.y), float(p.z)] for p in groups_img[0]] +
                    [[float(p.x), float(p.y), float(p.z)] for p in groups_world[0]],
                    dtype=np.float64,
                )
                finite32 = bool(np.isfinite(coords.astype(np.float32)).all())
        except Exception:
            finite32 = False
        if not finite32:
            reason = "NONFINITE"

    info = {
        "seq": seq,
        "ts_ms": ts_ms,
        "mode": mode,
        "img_n": img_n,
        "world_n": world_n,
        "n": 42 if reason == "OK" else 0,   # rows actually published to the native engine
        "reason": reason,
        "min_x": "", "max_x": "", "min_y": "", "max_y": "",
        "edge_l": "", "edge_r": "", "edge_t": "", "edge_b": "",
    }
    if img_n and reason != "NONFINITE":
        try:
            xs = [float(p.x) for p in groups_img[0]]
            ys = [float(p.y) for p in groups_img[0]]
        except Exception:
            xs, ys = [], []
        if xs and ys:
            info.update(
                min_x=round(min(xs), 6),
                max_x=round(max(xs), 6),
                min_y=round(min(ys), 6),
                max_y=round(max(ys), 6),
                edge_l=round(min(xs), 6),
                edge_r=round(1.0 - max(xs), 6),
                edge_t=round(min(ys), 6),
                edge_b=round(1.0 - max(ys), 6),
            )
    return info


class _HandDiagWriter:
    """One CSV line per processed hand frame. Enabled only by HAND_LANDMARK_DIAG."""

    def __init__(self, path=None):
        self.path = path or os.environ.get("HAND_LANDMARK_DIAG_FILE") or f"hand_landmark_diag_{os.getpid()}.csv"
        self._file = None
        self._writer = None
        self._failed = False
        self._rows = 0

    def write(self, info: dict) -> None:
        if self._failed:
            return
        try:
            if self._file is None:
                self._file = open(self.path, "w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=_DIAG_FIELDS)
                self._writer.writeheader()
                try:
                    print(f"[hand-diag] writing {self.path}", file=sys.stderr)
                except Exception:
                    pass
            self._writer.writerow(info)
            self._rows += 1
            # Flush in batches, not on every frame: an fsync per row is visible
            # in the inference timing. shutdown() always flushes the remainder.
            if self._rows % 64 == 0:
                self._file.flush()
        except Exception as exc:
            self._failed = True
            try:
                print(f"[hand-diag] disabled after write error: {exc}", file=sys.stderr)
            except Exception:
                pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None


def _pack_hand(result, diag=None, seq: int = 0, ts_ms: int = 0, mode: str = "") -> np.ndarray:
    """Pack image + world landmarks exactly as before; optionally log the decision.

    The diagnostic is called before the early return so the image/world split is
    visible even when the packed result is empty. A logging failure must never
    change the tracking path.
    """
    if diag is not None:
        try:
            diag.write(_describe_hand(result, seq, ts_ms, mode))
        except Exception:
            pass
    # "no hand" = a missing group; an incorrect group size is a bridge error, not a
    # missed detection. Raise so PyLandmarker::call returns -1 and main.cpp publishes
    # the existing bridge-error result, distinguishable from n == 0 / HELD grace.
    img = result.hand_landmarks
    world = result.hand_world_landmarks
    if not img or not world:
        return _EMPTY
    img_n, world_n = len(img[0]), len(world[0])
    if img_n != 21 or world_n != 21:
        raise RuntimeError(f"malformed hand payload: image={img_n}, world={world_n}; expected 21+21")
    return np.concatenate([_to_array(img[0]), _to_array(world[0])])


class FaceLandmarkExtractor:
    """478 face-mesh points (468 mesh + 10 iris) + the head-pose matrix. Same MediaPipe settings as gaze_intent.py."""

    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_asset_path("face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,      # head pose, needed by the gaze models
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return _EMPTY
        pts = _to_array(result.face_landmarks[0])
        pose = np.zeros((_POSE_ROWS, 3), dtype=np.float32)
        pose.reshape(-1)[:16] = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(-1)
        return np.concatenate([pts, pose])

    def shutdown(self):
        if getattr(self, "detector", None) is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


class HandLandmarkExtractor:
    """First hand: 21 image landmarks + 21 world landmarks, packed into one (42, 3) array.
    Same MediaPipe settings as hand_tracker.py (VIDEO mode, all three confidences 0.3)."""

    def __init__(self):
        mode_name = os.environ.get("HAND_RUNNING_MODE", "VIDEO").upper()
        if mode_name not in ("IMAGE", "VIDEO"):
            raise ValueError("HAND_RUNNING_MODE must be IMAGE or VIDEO")
        self._video = mode_name == "VIDEO"
        self._frame = 0
        self._diag = None
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_asset_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO if self._video else vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        if _diag_enabled():
            self._diag = _HandDiagWriter()

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            result = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.detector.detect(mp_image)
        self._frame += 1
        return _pack_hand(result, self._diag, self._frame, int(timestamp_ms),
                          "VIDEO" if self._video else "IMAGE")

    def shutdown(self):
        if getattr(self, "_diag", None) is not None:
            self._diag.close()
            self._diag = None
        if getattr(self, "detector", None) is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
