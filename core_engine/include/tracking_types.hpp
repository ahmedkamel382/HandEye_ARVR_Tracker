// Data that flows between the stages of the pipeline:
//   AI thread -> TrackingResult (raw, per frame) -> main thread -> filter -> input policy -> OS output
#pragma once
#include <chrono>
#include <cstdint>
#include <optional>

namespace core {

using Clock = std::chrono::steady_clock;

// One tracker's answer for one frame.
//   x, y  : normalised [0,1] in the FULL mirrored frame. x < 0  => nothing detected.
//   state : module-specific.   gaze: -1 calibrating, 0 neutral, 1 L-wink, 2 R-wink, 3 closed
//                              hand:  0 neutral, 1 pinch, 2 fist
//   ok    : false if the Python call raised or the module is disabled
//           (x, y, state then hold the safe fallback -1, -1, 0)
//   gaze only: yaw_deg / pitch_deg = the absolute world gaze in degrees (eye-in-socket + head pose, smoothed:
//           exactly what gaze_intent.py returns), head_*_deg = the head pose alone. x, y are those angles
//           mapped linearly onto 0..1 (see GazeNative::kSpanDeg) so the OS stage can treat gaze like a pointer.
//           While calibrating (state -1) and with no face (state 0) all four angles are NaN and x = y = -1, so
//           detected() is false and no invalid value reaches the DSP stage.
// How fresh a HAND result is (the gaze module leaves it unset). Set on every hand result, including startup,
// loss and bridge errors, so no consumer has to guess from repeated coordinates (a still hand repeats them too).
//   Tracked : the hand was observed in THIS frame.
//   Held    : short dropout (up to HandParams::grace_frames misses after a real observation): x, y and the gesture
//             state are the last observed ones, nothing was measured. detected() stays true (ROI crop and preview need
//             the position), but the DSP must NOT treat it as a new measurement.
//   Lost    : no usable position: startup before the first observation, grace expired, or a bridge error.
// Mirrors the Python reference vocabulary (hand_tracker.py: detected / held / lost); detected() is NOT a freshness test.
enum class TrackingState : std::uint8_t { Tracked, Held, Lost };

inline const char* toString(TrackingState s) {
    switch (s) {
        case TrackingState::Tracked: return "TRACKED";
        case TrackingState::Held:    return "HELD";
        case TrackingState::Lost:    return "LOST";
    }
    return "?";
}

struct ModuleOut {
    float x = -1.0f;
    float y = -1.0f;
    int   state = 0;
    bool  ok = false;
    double yaw_deg = 0.0, pitch_deg = 0.0, head_yaw_deg = 0.0, head_pitch_deg = 0.0;
    // hand only (appended with defaults: existing aggregate initialisation keeps compiling; gaze leaves them as is):
    std::optional<TrackingState> tracking_state;   // unset = module without freshness info (gaze, disabled, Python error)
    int missed_frames = 0;                         // consecutive misses since the last real observation
    bool detected() const { return ok && x >= 0.0f && y >= 0.0f; }
};

struct TrackingResult {
    uint64_t          frame_id = 0;
    int64_t           ts_ms = 0;          // strictly increasing, ms since pipeline start (MediaPipe's clock)
    double            capture_s = -1.0;   // exact capture time, seconds since pipeline start; < 0 = not provided (frame_clock.hpp)
    Clock::time_point captured_at{};
    ModuleOut         gaze;
    ModuleOut         hand;
    double            gaze_ms = 0.0;      // time in the gaze stage (MediaPipe call + native math)
    double            hand_ms = 0.0;      // same for the hand stage
    // Optional Engineer 4 recording metadata: actual detector return counts, never inferred from retained points.
    // Populated only when numerical recording is requested; -1 otherwise (or a detector-call error).
    int               hand_landmarks = -1;
    int               gaze_landmarks = -1;
};

} // namespace core
