// Data that flows between the stages of the pipeline:
//   AI thread -> TrackingResult (raw, per frame) -> main thread -> filter -> input policy -> OS output
#pragma once
#include <chrono>
#include <cstdint>

namespace core {

using Clock = std::chrono::steady_clock;

// One tracker's answer for one frame.
//   x, y  : normalised [0,1] in the FULL mirrored frame. x < 0  => nothing detected.
//   state : module-specific.   gaze: -1 calibrating, 0 neutral, 1 L-wink, 2 R-wink, 3 closed
//                              hand:  0 neutral, 1 pinch, 2 fist
//   ok    : false if the Python call raised or the module is disabled
//           (x, y, state then hold the safe fallback -1, -1, 0)
struct ModuleOut {
    float x = -1.0f;
    float y = -1.0f;
    int   state = 0;
    bool  ok = false;
    bool detected() const { return ok && x >= 0.0f && y >= 0.0f; }
};

struct TrackingResult {
    uint64_t          frame_id = 0;
    int64_t           ts_ms = 0;          // strictly increasing, ms since pipeline start
    Clock::time_point captured_at{};
    ModuleOut         gaze;
    ModuleOut         hand;
    double            gaze_ms = 0.0;      // time in the gaze stage (MediaPipe call + native math)
    double            hand_ms = 0.0;      // same for the hand stage
};

} // namespace core
