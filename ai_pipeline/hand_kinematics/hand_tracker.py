import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque, Counter
from typing import Optional, Tuple

from hand_features import (extract_features, result_to_points, load_gesture_model,
                           FEATURE_NAMES)
from hand_preprocess import LowLightEnhancer

# Tracking status of the most recent process_frame() call (see HandTracker.status).
STATUS_DETECTED = "DETECTED"          # primary VIDEO detector found the hand
STATUS_RECOVERED = "RECOVERED_ROI"    # primary missed; found by ROI re-detection around the last position
STATUS_HELD = "HELD"                  # nothing found; last output held (short dropout, NOT a detection)
STATUS_LOST = "LOST"                  # nothing found beyond the grace period -> (-1, -1, 0)
_OBSERVED = (STATUS_DETECTED, STATUS_RECOVERED)


class TrackerStats:
    """
    Objective measurements for comparing tracker configurations on the SAME
    recording protocol (research plan: "measure dropouts directly").

    Keep the hand in view for the whole recording when benchmarking: a frame
    with no hand in view is indistinguishable from a detection failure.
    """

    def __init__(self):
        self.frames = 0
        self.counts = {STATUS_DETECTED: 0, STATUS_RECOVERED: 0, STATUS_HELD: 0, STATUS_LOST: 0}
        self.primary_misses = 0          # frames where the main VIDEO detector found nothing
        self.enhanced_frames = 0
        self.latencies_ms = []
        self.dropout_runs = []           # lengths (frames) of consecutive HELD/LOST runs
        self.state_counts = {0: 0, 1: 0, 2: 0}
        self.status_changes = 0
        self.state_changes = 0
        self.lost_events = 0
        self.recovery_events = 0
        self.reacquired_events = 0
        self._last_status = None
        self._last_state = None
        self._stable_run = 0
        self._longest_stable_run = 0
        self._run = 0
        self._t_start = None
        self._t_end = None

    def record(self, status: str, primary_hit: bool, enhanced: bool, latency_ms: float,
               state: Optional[int] = None):
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now - latency_ms / 1000.0
        self._t_end = now
        self.frames += 1
        self.counts[status] += 1
        self.primary_misses += 0 if primary_hit else 1
        self.enhanced_frames += 1 if enhanced else 0
        self.latencies_ms.append(latency_ms)
        if self._last_status is not None and status != self._last_status:
            self.status_changes += 1
        if status == STATUS_LOST and self._last_status != STATUS_LOST:
            self.lost_events += 1
        if status == STATUS_RECOVERED and self._last_status != STATUS_RECOVERED:
            self.recovery_events += 1
        if status in _OBSERVED and self._last_status in (STATUS_HELD, STATUS_LOST):
            self.reacquired_events += 1
        self._last_status = status
        if state is not None:
            state = int(state)
            self.state_counts[state] = self.state_counts.get(state, 0) + 1
            if self._last_state is None or state == self._last_state:
                self._stable_run += 1
            else:
                self.state_changes += 1
                self._stable_run = 1
            self._last_state = state
            self._longest_stable_run = max(self._longest_stable_run, self._stable_run)
        if status in _OBSERVED:
            if self._run:
                self.dropout_runs.append(self._run)
                self._run = 0
        else:
            self._run += 1

    def summary(self) -> dict:
        runs = self.dropout_runs + ([self._run] if self._run else [])
        n = max(self.frames, 1)
        duration = (self._t_end - self._t_start) if self._t_start is not None else 0.0
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.zeros(1)
        pct = lambda v: round(100.0 * v / n, 2)
        return {
            "frames": self.frames,
            "duration_s": round(duration, 2),
            "fps": round(self.frames / duration, 1) if duration > 0 else 0.0,
            "detected_pct": pct(self.counts[STATUS_DETECTED]),
            "recovered_roi_pct": pct(self.counts[STATUS_RECOVERED]),
            "held_pct": pct(self.counts[STATUS_HELD]),
            "lost_pct": pct(self.counts[STATUS_LOST]),
            "primary_miss_pct": pct(self.primary_misses),
            "effective_dropout_pct": pct(self.counts[STATUS_HELD] + self.counts[STATUS_LOST]),
            "dropout_events": len(runs),
            "dropouts_per_min": round(len(runs) / (duration / 60.0), 2) if duration > 0 else 0.0,
            "longest_dropout_frames": max(runs) if runs else 0,
            "mean_dropout_frames": round(float(np.mean(runs)), 2) if runs else 0.0,
            "enhanced_pct": pct(self.enhanced_frames),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "status_changes": self.status_changes,
            "state_changes": self.state_changes,
            "longest_stable_frames": self._longest_stable_run,
            "lost_events": self.lost_events,
            "recovery_events": self.recovery_events,
            "reacquired_events": self.reacquired_events,
            "state_counts": dict(self.state_counts),
        }


class HandTracker:
    """
    Reference Python hand tracker and intended C++ bridge class.
    process_frame(BGR frame) -> (Cursor X, Cursor Y, Gesture State)

    The return contract is unchanged. Extra information is exposed as plain
    attributes (bindable from Pybind11 if the C++ side wants it):
        status          one of DETECTED / RECOVERED_ROI / HELD / LOST
        tracking_valid  True only when the hand was actually observed this frame
                        (DETECTED or RECOVERED_ROI). HELD/LOST frames are NOT detections.
        missed_frames   consecutive frames without a real observation (read-only):
                        0 for DETECTED/RECOVERED_ROI, 1..GRACE_FRAMES for HELD,
                        >GRACE_FRAMES for LOST. No observation yet also reports LOST.

    The active C++ core currently uses a native port (HandNative) rather than this
    Python class; see PROJECT_HANDOFF.md and CORE_ENGINE_INTEGRATION_REPORT.md.

    Classification: Random Forest over 3D joint angles / normalized 3D
    distances (hand_features.py, trained by model_trainer.py), followed by a
    majority vote over SMOOTHING_FRAMES and a MIN_CONFIDENCE gate.

    Robustness layers (each can be switched off to A/B-test it):
      1. VIDEO running mode  - MediaPipe tracks from the previous frame instead of
         re-detecting the palm from scratch every frame.
      2. Lowered detection thresholds (0.3) for hard poses (tight fist near the face).
      3. Low-light enhancement (enable_lowlight) - gamma + CLAHE, applied ONLY when the
         frame is dark (hysteresis-gated); normal-light frames pass through untouched.
      4. ROI recovery (enable_roi_recovery) - when the primary detector misses, retry on
         a crop around the last known hand position with a separate IMAGE-mode
         landmarker. Landmarks are mapped back to full-frame coordinates. If the ROI also
         fails we fall through to the grace period; the next frame always searches the
         full frame again, so the tracker cannot get stuck in a wrong region.
      5. Grace period (GRACE_FRAMES) - a short dropout holds the last output instead of
         snapping to Neutral; longer loss resets to (-1, -1, 0). With predict_cursor=True
         the held cursor is extrapolated with a damped velocity (state is never changed).

    Click safety: a held/predicted frame can never START a new gesture - the held state is
    the last state that was backed by a real observation, and a recovered detection still
    has to win the majority vote before the state can change.

    If hand_gesture_rf.pkl does not exist, a rough 3D geometric fallback is used
    (bootstrap only, UNTESTED thresholds) and a warning is printed.
    """

    SMOOTHING_FRAMES = 5
    MIN_CONFIDENCE = 0.55      # below this, the frame votes Neutral
    GRACE_FRAMES = 6           # ~0.2 s at 30 fps

    ROI_MARGIN = 0.5           # crop = last hand bbox expanded by this fraction on every side
    ROI_MIN_PX = 96            # never crop smaller than this (px)
    PREDICT_DECAY = 0.6        # per-held-frame damping of the extrapolated velocity
    VELOCITY_SMOOTHING = 0.5   # EMA factor for per-frame cursor velocity

    # Fallback-only (no trained model):
    FALLBACK_PINCH_RATIO = 0.30
    FALLBACK_CURL_RATIO = 1.30

    def __init__(self, enable_lowlight: bool = True, enable_roi_recovery: bool = True,
                 predict_cursor: bool = False):
        self.enable_lowlight = enable_lowlight
        self.enable_roi_recovery = enable_roi_recovery
        self.predict_cursor = predict_cursor

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get('AI_PIPELINE_ROOT', os.path.abspath(os.path.join(self.script_dir, '..')))
        assets_dir = os.path.abspath(os.path.join(project_root, '..', 'assets'))

        model_path = os.path.join(assets_dir, 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Hand model missing at:\n{model_path}")

        def make_options(mode):
            return vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=mode,
                num_hands=1,
                # 0.3 (down from 0.5): live tests showed correct Fist classification at
                # 0.83-0.92 confidence whenever detection succeeded - the gap was detection
                # recall, not classification quality.
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )

        # Primary: VIDEO mode (needs a strictly increasing timestamp per call).
        self.detector = vision.HandLandmarker.create_from_options(make_options(vision.RunningMode.VIDEO))
        self._start_time = time.time()
        self._last_timestamp_ms = -1

        # Secondary: IMAGE mode, used ONLY for ROI crops after a primary miss.
        self.roi_detector = None
        if self.enable_roi_recovery:
            self.roi_detector = vision.HandLandmarker.create_from_options(make_options(vision.RunningMode.IMAGE))

        self.enhancer = LowLightEnhancer() if self.enable_lowlight else None

        self.rf = load_gesture_model(os.path.join(assets_dir, 'hand_gesture_rf.pkl'))
        if self.rf is None:
            print("[WARNING] hand_gesture_rf.pkl not found -> using rough geometric fallback. "
                  "Run hand_data_collector.py then model_trainer.py.")

        self._votes = deque(maxlen=self.SMOOTHING_FRAMES)
        self.last_confidence = 0.0   # debug only, not part of the C++ bridge payload
        self.status = STATUS_LOST
        self.tracking_valid = False
        self.stats = TrackerStats()

        self._missed_frames = 0
        self._last_output = (-1.0, -1.0, 0)
        self._last_bbox = None       # (x0, y0, x1, y1) normalized, from the last observed hand
        self._last_cursor = None     # np.array([x, y]) normalized
        self._vel = np.zeros(2)

    # ------------------------------------------------------------------ helpers
    def reset_stats(self):
        self.stats = TrackerStats()

    def get_status(self) -> Tuple[str, bool]:
        return self.status, self.tracking_valid

    @property
    def missed_frames(self) -> int:
        """Consecutive frames without a real observation (read-only)."""
        return self._missed_frames

    def _classify(self, feats: np.ndarray) -> Tuple[int, float]:
        if self.rf is not None:
            proba = self.rf.predict_proba(feats.reshape(1, -1))[0]
            k = int(np.argmax(proba))
            cls, conf = int(self.rf.classes_[k]), float(proba[k])
            return (cls if conf >= self.MIN_CONFIDENCE else 0), conf

        # Fallback: 3D normalized distances (indices follow FEATURE_NAMES).
        pinch = feats[FEATURE_NAMES.index("thumb_to_index_tip")]
        tips = [feats[FEATURE_NAMES.index(f"wrist_to_{f}_tip")] for f in ("index", "middle", "ring", "pinky")]
        if pinch < self.FALLBACK_PINCH_RATIO:
            return 1, 1.0
        if sum(t < self.FALLBACK_CURL_RATIO for t in tips) >= 3:
            return 2, 1.0
        return 0, 1.0

    @staticmethod
    def _parse(result) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """MediaPipe result -> (world points (21,3), image landmarks (21,2) normalized) or None.

        Invalid payloads are rejected here, at the single boundary used by both the
        primary VIDEO detector and ROI recovery. This is the Tester-side equivalent of
        the native Core Engine 21+21 / finite-input safeguard: a malformed 20+22 split
        or a non-finite coordinate must never become a fresh cursor coordinate (for
        example an Infinity clipped to X = 1.0 by the cursor clamp below).
        """
        img = result.hand_landmarks
        wld = result.hand_world_landmarks
        if not img or not wld or len(img[0]) != 21 or len(wld[0]) != 21:
            return None
        points = result_to_points(result)
        lm_xy = np.array([[p.x, p.y] for p in img[0]], dtype=np.float64)
        if points is None or points.shape != (21, 3) or lm_xy.shape != (21, 2):
            return None
        if not np.isfinite(points).all() or not np.isfinite(lm_xy).all():
            return None
        return points, lm_xy

    def _roi_recover(self, rgb: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Re-detect on a crop around the last known hand; returns full-frame-mapped landmarks."""
        if self.roi_detector is None or self._last_bbox is None:
            return None
        h, w = rgb.shape[:2]
        bx0, by0, bx1, by1 = self._last_bbox
        cx, cy = (bx0 + bx1) / 2.0 * w, (by0 + by1) / 2.0 * h
        side = max((bx1 - bx0) * w, (by1 - by0) * h) * (1.0 + 2.0 * self.ROI_MARGIN)
        side = max(side, float(self.ROI_MIN_PX))
        # Square window (clipped to the frame size). Near a border the window is SHIFTED
        # inward rather than cut, so the crop keeps its size.
        cw, ch = int(min(side, w)), int(min(side, h))
        x0 = int(np.clip(cx - cw / 2.0, 0, w - cw))
        y0 = int(np.clip(cy - ch / 2.0, 0, h - ch))
        x1, y1 = x0 + cw, y0 + ch
        if cw < self.ROI_MIN_PX or ch < self.ROI_MIN_PX:
            return None

        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        res = self.roi_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        parsed = self._parse(res)
        if parsed is None:
            return None
        points, lm_crop = parsed
        # Crop-normalized -> full-frame-normalized. World points are metric around the
        # palm, so they need no mapping.
        lm_full = np.empty_like(lm_crop)
        lm_full[:, 0] = (x0 + lm_crop[:, 0] * cw) / w
        lm_full[:, 1] = (y0 + lm_crop[:, 1] * ch) / h
        return points, lm_full

    # ---------------------------------------------------------------- main API
    def process_frame(self, frame_array: np.ndarray) -> Tuple[float, float, int]:
        """
        State values: 0 = Neutral/Hover, 1 = Pinch (click/drag), 2 = Fist.
        No observation for more than GRACE_FRAMES consecutive frames -> (-1.0, -1.0, 0).
        Never raises on a missing hand; always returns (float, float, int).

        The first call with no prior observation reports LOST, not HELD. Misses
        1..GRACE_FRAMES after a valid observation report HELD and preserve the
        last valid coordinates; later misses report LOST. Both non-observed states
        set tracking_valid=False and are observable through missed_frames.
        """
        t0 = time.perf_counter()
        frame = cv2.flip(frame_array, 1)

        enhanced = False
        if self.enhancer is not None:
            frame, enhanced = self.enhancer.apply(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic timestamp required by VIDEO mode (two calls can share a millisecond).
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        det = self._parse(self.detector.detect_for_video(mp_image, timestamp_ms))
        primary_hit = det is not None
        status = STATUS_DETECTED

        if det is None and self.enable_roi_recovery:
            det = self._roi_recover(rgb)
            status = STATUS_RECOVERED

        if det is not None:
            points, lm_xy = det
            had_gap = self._missed_frames > 0
            self._missed_frames = 0

            cursor = np.clip(lm_xy[8], 0.0, 1.0)
            if had_gap or self._last_cursor is None:
                self._vel = np.zeros(2)          # a gap makes the frame-to-frame delta meaningless
            else:
                self._vel = (self.VELOCITY_SMOOTHING * self._vel
                             + (1.0 - self.VELOCITY_SMOOTHING) * (cursor - self._last_cursor))
            self._last_cursor = cursor
            self._last_bbox = (float(lm_xy[:, 0].min()), float(lm_xy[:, 1].min()),
                               float(lm_xy[:, 0].max()), float(lm_xy[:, 1].max()))

            state, self.last_confidence = self._classify(extract_features(points))
            self._votes.append(state)
            smoothed = Counter(self._votes).most_common(1)[0][0]
            self._last_output = (float(cursor[0]), float(cursor[1]), int(smoothed))
        else:
            self._missed_frames += 1
            if self._last_cursor is None or self._missed_frames > self.GRACE_FRAMES:
                # No prior valid observation, or genuine hand loss: clear everything;
                # the next frame searches the full frame.
                status = STATUS_LOST
                self._votes.clear()
                self.last_confidence = 0.0
                self._last_output = (-1.0, -1.0, 0)
                self._last_bbox = None
                self._last_cursor = None
                self._vel = np.zeros(2)
            else:
                status = STATUS_HELD
                if self.predict_cursor and self._last_cursor is not None:
                    step = self._vel * (self.PREDICT_DECAY ** self._missed_frames)
                    self._last_cursor = np.clip(self._last_cursor + step, 0.0, 1.0)
                    self._last_output = (float(self._last_cursor[0]), float(self._last_cursor[1]),
                                         self._last_output[2])
                # else: hold the previous output exactly as it was

        self.status = status
        self.tracking_valid = status in _OBSERVED
        self.stats.record(status, primary_hit, enhanced, (time.perf_counter() - t0) * 1000.0,
                          state=int(self._last_output[2]))
        return self._last_output
