"""Index-finger pointer tracking without gesture recognition.

The tracker uses MediaPipe HandLandmarker in VIDEO mode and exposes landmark 8
(the index fingertip) as a normalized pointer position. It intentionally has no
gesture classifier, click logic, ROI recovery, or cursor prediction.
"""

from dataclasses import dataclass
import math
import os
import time
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from hand_features import INDEX_TIP


STATUS_TRACKED = "TRACKED"
STATUS_LOST = "LOST"
NO_ACTION_STATE = 0


@dataclass(frozen=True)
class FingerTrackingResult:
    """One tracking update. Pointer coordinates remain normalized to [0, 1]."""

    x: Optional[float]
    y: Optional[float]
    camera_x: Optional[float]
    camera_y: Optional[float]
    valid: bool
    status: str
    landmark_index: int = INDEX_TIP


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _validate_region(region: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(region) != 4:
        raise ValueError("tracking_region must contain (left, top, right, bottom)")
    left, top, right, bottom = (float(value) for value in region)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("tracking_region values must be finite")
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("tracking_region must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return left, top, right, bottom


def map_pointer(
    x: float,
    y: float,
    tracking_region: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
) -> Tuple[float, float]:
    """Map a camera-normalized point through a region into pointer coordinates."""
    left, top, right, bottom = _validate_region(tracking_region)
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("x and y must be finite")
    x, y = _clamp01(x), _clamp01(y)
    mapped_x = (x - left) / (right - left)
    mapped_y = (y - top) / (bottom - top)
    return _clamp01(mapped_x), _clamp01(mapped_y)


class IndexFingerTracker:
    """Track landmark 8 and expose a stable normalized pointer position.

    ``track()`` returns the rich result. ``process_frame()`` is a compatibility
    adapter for the existing hand contract; its gesture state is always 0 because
    this class does not recognize gestures. Callers must use ``tracking_valid``
    to distinguish a valid pointer from the lost fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tracking_region: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
        smoothing_alpha: float = 0.35,
        min_hand_detection_confidence: float = 0.3,
        min_hand_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get(
            "AI_PIPELINE_ROOT", os.path.abspath(os.path.join(self.script_dir, ".."))
        )
        assets_dir = os.path.abspath(os.path.join(project_root, "..", "assets"))

        if model_path is None:
            model_path = os.path.join(assets_dir, "hand_landmarker.task")
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Hand model missing at:\n{self.model_path}")

        self.tracking_region = _validate_region(tracking_region)
        smoothing_alpha = float(smoothing_alpha)
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must satisfy 0 < alpha <= 1")
        self.smoothing_alpha = smoothing_alpha

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

        self._start_time = time.perf_counter()
        self._last_timestamp_ms = -1
        self._smoothed: Optional[Tuple[float, float]] = None

        self.status = STATUS_LOST
        self.tracking_valid = False
        self.last_result = FingerTrackingResult(
            x=None,
            y=None,
            camera_x=None,
            camera_y=None,
            valid=False,
            status=STATUS_LOST,
        )

    def _lost(self) -> FingerTrackingResult:
        self._smoothed = None
        self.status = STATUS_LOST
        self.tracking_valid = False
        self.last_result = FingerTrackingResult(
            x=None,
            y=None,
            camera_x=None,
            camera_y=None,
            valid=False,
            status=STATUS_LOST,
        )
        return self.last_result

    @staticmethod
    def _parse(result) -> Optional[Tuple[float, float]]:
        try:
            hand_landmarks = getattr(result, "hand_landmarks", None)
            if not hand_landmarks:
                return None
            hand = hand_landmarks[0]
            if hand is None or len(hand) <= INDEX_TIP:
                return None
            point = hand[INDEX_TIP]
            x, y = float(point.x), float(point.y)
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            return x, y
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def _smooth(self, x: float, y: float) -> Tuple[float, float]:
        if self._smoothed is None or self.smoothing_alpha >= 1.0:
            self._smoothed = (x, y)
            return x, y
        previous_x, previous_y = self._smoothed
        alpha = self.smoothing_alpha
        smoothed_x = alpha * x + (1.0 - alpha) * previous_x
        smoothed_y = alpha * y + (1.0 - alpha) * previous_y
        self._smoothed = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def track(self, bgr_frame: np.ndarray) -> FingerTrackingResult:
        if self._detector is None:
            raise RuntimeError("IndexFingerTracker is closed")
        if (
            not isinstance(bgr_frame, np.ndarray)
            or bgr_frame.ndim != 3
            or bgr_frame.shape[2] != 3
        ):
            return self._lost()

        frame = cv2.flip(bgr_frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        detection = self._parse(self._detector.detect_for_video(mp_image, timestamp_ms))
        if detection is None:
            return self._lost()

        camera_x = _clamp01(detection[0])
        camera_y = _clamp01(detection[1])
        pointer_x, pointer_y = map_pointer(camera_x, camera_y, self.tracking_region)
        pointer_x, pointer_y = self._smooth(pointer_x, pointer_y)

        self.status = STATUS_TRACKED
        self.tracking_valid = True
        self.last_result = FingerTrackingResult(
            x=pointer_x,
            y=pointer_y,
            camera_x=camera_x,
            camera_y=camera_y,
            valid=True,
            status=STATUS_TRACKED,
        )
        return self.last_result

    def process_frame(self, bgr_frame: np.ndarray) -> Tuple[float, float, int]:
        """Compatibility adapter: valid -> (x, y, 0), lost -> (-1, -1, 0)."""
        result = self.track(bgr_frame)
        if not result.valid:
            return (-1.0, -1.0, NO_ACTION_STATE)
        return (float(result.x), float(result.y), NO_ACTION_STATE)

    def get_status(self) -> Tuple[str, bool]:
        return self.status, self.tracking_valid

    def close(self) -> None:
        detector = getattr(self, "_detector", None)
        if detector is not None:
            self._detector = None
            detector.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
