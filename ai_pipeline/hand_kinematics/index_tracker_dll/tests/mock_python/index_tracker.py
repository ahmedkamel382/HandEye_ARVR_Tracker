"""Deterministic stand-in used only by the hand_index_tracker DLL smoke test."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class FingerTrackingResult:
    x: Optional[float]
    y: Optional[float]
    camera_x: Optional[float]
    camera_y: Optional[float]
    valid: bool
    status: str
    landmark_index: int = 8


class IndexFingerTracker:
    def __init__(
        self,
        model_path=None,
        tracking_region: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
        smoothing_alpha: float = 0.35,
        min_hand_detection_confidence: float = 0.3,
        min_hand_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        **kwargs,
    ):
        del kwargs
        self.model_path = model_path
        self.tracking_region = tuple(float(value) for value in tracking_region)
        self.smoothing_alpha = float(smoothing_alpha)
        self.confidences = (
            float(min_hand_detection_confidence),
            float(min_hand_presence_confidence),
            float(min_tracking_confidence),
        )
        self.calls = 0

    def _verify_frame(self, frame):
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("expected a BGR frame")
        red = frame[:, :, 2]
        if not bool((red == 42).all()):
            raise ValueError("frame data or stride was copied incorrectly")

    def track(self, frame):
        self.calls += 1
        self._verify_frame(frame)

        if self.calls == 2:
            return FingerTrackingResult(
                x=None,
                y=None,
                camera_x=None,
                camera_y=None,
                valid=False,
                status="LOST",
                landmark_index=8,
            )

        left, _top, _right, bottom = self.tracking_region
        return FingerTrackingResult(
            x=left,
            y=bottom,
            camera_x=self.smoothing_alpha,
            camera_y=0.5,
            valid=True,
            status="TRACKED",
            landmark_index=8,
        )

    def close(self):
        self.calls = -1
