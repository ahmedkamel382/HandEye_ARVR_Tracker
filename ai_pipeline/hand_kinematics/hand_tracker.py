import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque, Counter
from typing import Tuple

from hand_features import (extract_features, result_to_points, load_gesture_model,
                           FEATURE_NAMES)


class HandTracker:
    """
    THE C++ BRIDGE CLASS.  process_frame(BGR frame) -> (Cursor X, Cursor Y, Gesture State)

    Gesture classification = Random Forest over 3D joint angles / normalized
    3D distances (see hand_features.py), trained with model_trainer.py.
    This replaces the old 2D pixel-distance thresholds, whose Pinch/Fist
    distributions overlapped completely (no threshold could separate them).

    Stability layer 1 (classification): a majority vote over the last
    SMOOTHING_FRAMES predictions plus a minimum-confidence gate, so one
    noisy frame can't fire a click.

    Stability layer 2 (detection dropouts): MediaPipe's hand detector can
    intermittently fail to find a hand for a few consecutive frames on hard
    poses -- a tightly closed Fist held close to the face is the worst case,
    since it has few distinguishing edges and low skin/background contrast.
    Without this layer, a single missed frame used to clear the vote buffer
    and snap straight to (-1.0, -1.0, 0), bypassing the smoothing above
    entirely -- that produced the "flickers between all three states" bug,
    which was actually a *detection* dropout, not a *classification* error
    (confirmed live: real detections during the same dropouts classified
    Fist correctly at 0.83-0.92 confidence). Now a short run of missed
    frames (up to GRACE_FRAMES) just holds the last known output instead of
    resetting; only a longer, genuine hand-loss clears state to Neutral.

    Stability layer 3 (tracking mode): running in VIDEO mode (not IMAGE)
    lets MediaPipe reuse the previous frame's hand location as a tracking
    prior instead of re-running full palm detection from scratch on every
    single frame. Per Google's own docs, this is specifically what keeps
    the skeleton alive through hard, momentary poses -- live testing still
    showed occasional dropouts longer than GRACE_FRAMES under IMAGE mode
    (a held Fist briefly reporting Neutral/-1,-1 despite being clearly
    visible), which VIDEO mode's tracking continuity is meant to reduce at
    the source, with the grace period above as a remaining safety net for
    whatever dropouts still get through.

    If hand_gesture_rf.pkl does not exist yet, a rough 3D geometric fallback
    is used (bootstrap only, UNTESTED thresholds) and a warning is printed.
    Train the model to get reliable behavior.
    """

    SMOOTHING_FRAMES = 5
    MIN_CONFIDENCE = 0.55      # below this, the frame votes Neutral

    # Detection-dropout tolerance: consecutive frames with no hand found
    # before we actually clear the vote buffer and report (-1, -1, 0).
    # 6 frames (~0.2s at 30fps) comfortably covers the short dropouts seen
    # on a tight Fist near the face without masking a genuine hand-away.
    GRACE_FRAMES = 6

    # Fallback-only (no trained model):
    FALLBACK_PINCH_RATIO = 0.30
    FALLBACK_CURL_RATIO = 1.30

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.environ.get('AI_PIPELINE_ROOT', os.path.abspath(os.path.join(self.script_dir, '..')))
        assets_dir = os.path.abspath(os.path.join(project_root, '..', 'assets'))

        model_path = os.path.join(assets_dir, 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Hand model missing at:\n{model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            # VIDEO (not IMAGE): each process_frame() call is one frame of a
            # continuous webcam stream, so MediaPipe can track the hand
            # forward from the previous frame instead of re-detecting the
            # palm from scratch every time -- see Stability layer 3 above.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            # Lowered from 0.5 -> 0.3. A tight Fist near the face has few
            # clear edges and low skin/background contrast, which was
            # pushing MediaPipe's own detection confidence below the old
            # threshold on otherwise-valid frames (live tests showed
            # correct Fist classification at 0.83-0.92 model confidence
            # whenever detection succeeded -- the gap was detection
            # recall, not classification quality).
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # VIDEO mode requires a strictly increasing timestamp (ms) per call.
        self._start_time = time.time()
        self._last_timestamp_ms = -1

        self.rf = load_gesture_model(os.path.join(assets_dir, 'hand_gesture_rf.pkl'))
        if self.rf is None:
            print("[WARNING] hand_gesture_rf.pkl not found -> using rough geometric fallback. "
                  "Run hand_data_collector.py then model_trainer.py.")

        self._votes = deque(maxlen=self.SMOOTHING_FRAMES)
        self.last_confidence = 0.0   # debug only, not part of the C++ bridge payload

        # Grace-period state: last known good output + consecutive-miss counter.
        self._missed_frames = 0
        self._last_output = (-1.0, -1.0, 0)

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

    def process_frame(self, frame_array: np.ndarray) -> Tuple[float, float, int]:
        """
        State values: 0 = Neutral/Hover, 1 = Pinch (click/drag), 2 = Fist.
        No hand for more than GRACE_FRAMES consecutive frames -> (-1.0, -1.0, 0).
        A short dropout (<= GRACE_FRAMES) holds the last known output instead
        of resetting, since most such dropouts are detection misses on a
        hand that is still actually there (see class docstring).
        """
        frame = cv2.flip(frame_array, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic timestamp required by VIDEO mode. Wall-clock time is
        # normally plenty, but two calls can land in the same millisecond
        # on a fast machine -- force strictly increasing just in case.
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        points = result_to_points(result)
        if points is None or not result.hand_landmarks:
            self._missed_frames += 1
            if self._missed_frames > self.GRACE_FRAMES:
                # Genuine hand-loss: clear everything and report Neutral.
                self._votes.clear()
                self.last_confidence = 0.0
                self._last_output = (-1.0, -1.0, 0)
            # Either way, return the held output rather than snapping to
            # Neutral on every single missed frame.
            return self._last_output

        # Hand found again -- reset the miss counter.
        self._missed_frames = 0

        index_tip = result.hand_landmarks[0][8]
        cursor_x = max(0.0, min(1.0, index_tip.x))
        cursor_y = max(0.0, min(1.0, index_tip.y))

        state, self.last_confidence = self._classify(extract_features(points))
        self._votes.append(state)
        smoothed = Counter(self._votes).most_common(1)[0][0]

        self._last_output = (float(cursor_x), float(cursor_y), int(smoothed))
        return self._last_output
