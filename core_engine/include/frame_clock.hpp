// Frame timestamps: two clocks, on purpose.
//
//   ts_ms      integer milliseconds, STRICTLY increasing. Only MediaPipe VIDEO mode needs this (it rejects equal or
//              decreasing timestamps), so frames captured within the same millisecond are bumped by +1 ms.
//   capture_s  seconds since pipeline start, taken from the ORIGINAL capture time_point of the frame (steady_clock,
//              nanosecond resolution): no millisecond truncation and no MediaPipe bump. This is the time base of the DSP
//              filter and of the OS stage (debounce times, gesture timing).
//
// Neither is the sensor exposure time: the capture thread stamps the frame when cv::VideoCapture::read() returns. After a
// camera stall the buffered frames are read back to back and get nearly identical stamps. That limit is in the driver
// interface, not in this code.
//
// Header-only and free of OpenCV / Python so the unit tests can drive it.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>

#include "tracking_types.hpp"      // core::Clock

namespace core {

struct FrameTimes {
    int64_t ts_ms;       // strictly increasing, for MediaPipe
    double  capture_s;   // exact capture time, for the DSP / OS stage
};

// last_ts_ms: the ts_ms of the previous frame (-1 before the first). The caller stores the returned ts_ms.
inline FrameTimes frameTimes(Clock::time_point pipeline_start, Clock::time_point captured_at, int64_t last_ts_ms) {
    const auto since = captured_at - pipeline_start;
    FrameTimes t;
    t.ts_ms = std::max<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(since).count(), last_ts_ms + 1);
    t.capture_s = std::chrono::duration<double>(since).count();
    return t;
}

} // namespace core
