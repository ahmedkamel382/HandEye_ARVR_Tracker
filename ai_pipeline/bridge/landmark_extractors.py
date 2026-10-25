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
    extractor.shutdown()                         # releases the MediaPipe resources (called once at exit)

Rules: no OS access, no flipping/colour conversion, exceptions are allowed (the C++ side counts them
and treats the frame as "nothing detected"), never raise for "no face / no hand".

Environment:
    AI_PIPELINE_ROOT   same meaning as in gaze_intent.py / hand_tracker.py (default: the folder above this one)
    HAND_RUNNING_MODE  VIDEO (default, what hand_tracker.py uses: MediaPipe tracks the hand between frames)
                       or IMAGE (palm detection on every frame; use it together with --hand-roi)
"""
import os

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
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_asset_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO if self._video else vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            result = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.detector.detect(mp_image)
        # hand_tracker.py: "no hand" = no world landmarks OR no image landmarks
        if not result.hand_landmarks or not result.hand_world_landmarks:
            return _EMPTY
        return np.concatenate([_to_array(result.hand_landmarks[0]), _to_array(result.hand_world_landmarks[0])])

    def shutdown(self):
        if getattr(self, "detector", None) is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
