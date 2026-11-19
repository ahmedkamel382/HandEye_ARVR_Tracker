"""
Low-light preprocessing shared by hand_tracker.py and hand_data_collector.py.

Idea: only touch the frame when it is actually dark. In normal lighting the
frame passes through UNCHANGED (same array, zero cost beyond a tiny luma
measurement), so behaviour in good light is identical to the validated
baseline. In the dark we apply a gamma lift + CLAHE on the L channel
(LAB) to restore contrast before MediaPipe sees the frame.

Hysteresis (separate ON/OFF thresholds + smoothed luma) stops the enhancer
from flickering on/off around the threshold, which would otherwise make the
landmark statistics jump between frames.

IMPORTANT: this is a hypothesis, not a proven win. Enhancement can amplify
sensor noise. Measure it with test_hand_tracker.py (--no-enhance vs default)
on recordings made in the dark before claiming any improvement.
"""
import cv2
import numpy as np


class LowLightEnhancer:
    def __init__(self,
                 dark_on: float = 80.0,     # mean luma (0-255) below this -> enhancer turns ON
                 dark_off: float = 95.0,    # ...and must rise above this to turn OFF again
                 target_luma: float = 115.0,
                 clip_limit: float = 2.5,
                 tile_grid: int = 8,
                 smoothing: float = 0.2):   # EMA factor for the luma estimate
        if dark_off <= dark_on:
            raise ValueError("dark_off must be greater than dark_on (hysteresis)")
        self.dark_on = dark_on
        self.dark_off = dark_off
        self.target_luma = target_luma
        self.smoothing = smoothing
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        self._luma = None
        self._lut_cache = {}
        self.active = False

    @property
    def luma(self) -> float:
        """Smoothed mean luma of the recent frames (0-255). 0.0 before the first frame."""
        return 0.0 if self._luma is None else self._luma

    @staticmethod
    def measure_luma(bgr: np.ndarray) -> float:
        # Tiny thumbnail: ~0.1 ms and plenty accurate for a global brightness estimate.
        small = cv2.resize(bgr, (80, 60), interpolation=cv2.INTER_AREA)
        return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        key = round(gamma / 0.05) * 0.05          # quantize so the LUT cache stays tiny
        lut = self._lut_cache.get(key)
        if lut is None:
            lut = (np.power(np.arange(256) / 255.0, key) * 255.0).astype(np.uint8)
            self._lut_cache[key] = lut
        return lut

    def apply(self, bgr: np.ndarray):
        """Returns (frame, enhanced_flag). `frame` is the input array itself when not enhanced."""
        current = self.measure_luma(bgr)
        self._luma = current if self._luma is None else (
            (1.0 - self.smoothing) * self._luma + self.smoothing * current)

        if self.active:
            if self._luma > self.dark_off:
                self.active = False
        elif self._luma < self.dark_on:
            self.active = True

        if not self.active:
            return bgr, False

        # Gamma < 1 brightens. Chosen so the mean luma lands near target_luma.
        mean = max(self._luma, 1.0)
        gamma = float(np.clip(np.log(self.target_luma / 255.0) / np.log(mean / 255.0), 0.35, 1.0))
        lifted = cv2.LUT(bgr, self._gamma_lut(gamma))

        lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge((self._clahe.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True
