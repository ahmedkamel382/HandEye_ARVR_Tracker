"""TEST-ONLY stand-in for ai_pipeline/bridge/landmark_extractors.py (no MediaPipe needed).

The end-to-end test (tests/e2e/run_e2e.py) draws a synthetic video whose pixels CARRY the scene:
  * left-hand column (in the frame the engine delivers, i.e. AFTER mirroring + RGB conversion):
        x 0..39, y   0..39  white = a face is present
        x 0..39, y  40..79  white = left eye closed
        x 0..39, y  80..119 white = right eye closed
        x 0..39, y 120..159 white = hand pinched
        x 0..39, y 160..199 pure RED in the original BGR video -> must arrive as R high, B low (proves BGR->RGB)
  * a white disc (right of x=40) = the index fingertip position (present only when a hand is present)
These extractors decode that and return landmarks (the hand one returns the 42-row image+world array). They also behave like MediaPipe's VIDEO mode:
timestamps must strictly increase, otherwise they raise (the engine must count it, never crash).
Never use this folder as --ai-root for a real run.
"""
import numpy as np

import synthetic_landmarks as syn

_EMPTY = np.zeros((0, 3), dtype=np.float32)


def _check_frame(rgb):
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3):
        raise RuntimeError("frame must be a uint8 (H, W, 3) array")
    if not rgb.flags["C_CONTIGUOUS"]:
        raise RuntimeError("frame must be C-contiguous")


def _block_on(rgb, row):
    return float(rgb[row * 40 + 20, 20].mean()) > 128.0


class FaceLandmarkExtractor:
    def __init__(self):
        self._last_ts = -1

    def process_frame(self, rgb, timestamp_ms):
        _check_frame(rgb)
        if timestamp_ms <= self._last_ts:
            raise RuntimeError("timestamps must strictly increase")
        self._last_ts = timestamp_ms
        px = rgb[180, 20]                                     # block 4: red in the source video
        if not (px[0] > 200 and px[2] < 60):
            raise RuntimeError(f"frame is not RGB (R,G,B = {px.tolist()})")
        if not _block_on(rgb, 0):
            return _EMPTY
        k_left = 0.25 if _block_on(rgb, 1) else 1.0
        k_right = 0.25 if _block_on(rgb, 2) else 1.0
        return syn.make_face(k_left=k_left, k_right=k_right, iris=(0.5, 0.5))

    def shutdown(self):
        pass


class HandLandmarkExtractor:
    def __init__(self):
        self._last_ts = -1

    def process_frame(self, rgb, timestamp_ms):
        _check_frame(rgb)
        self._last_ts = timestamp_ms
        mask = rgb[:, 40:, :].min(axis=2) > 200               # the white disc (and nothing from the code column)
        ys, xs = np.nonzero(mask)
        if xs.size < 50:
            return _EMPTY
        h, w = rgb.shape[:2]
        tip = ((xs.mean() + 40) / w, ys.mean() / h)
        img = syn.make_hand(tip=tip, pinch=_block_on(rgb, 3))
        return syn.pack_hand(img, syn.make_hand_world(img))

    def shutdown(self):
        pass
