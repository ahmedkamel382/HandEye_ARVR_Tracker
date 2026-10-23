// Pre-allocated pool of full-size RGB frames + the tiny packet that travels
// through the lock-free queue. Pixels are NEVER copied between threads: the
// capture thread writes into a slot, the AI thread reads it in place, then
// returns the slot. Steady state = zero heap allocation, bounded memory.
#pragma once
#include <opencv2/core.hpp>
#include <concurrentqueue.h>
#include <cstdint>
#include <vector>
#include "tracking_types.hpp"

namespace core {

struct Packet {
    uint32_t          slot;
    uint64_t          frame_id;
    int64_t           ts_ms;
    Clock::time_point captured_at;
};

class FramePool {
public:
    FramePool(size_t n, cv::Size size) : size_(size), mats_(n), free_(n) {
        for (size_t i = 0; i < n; ++i) {
            mats_[i].create(size.height, size.width, CV_8UC3);   // one-time allocation
            free_.enqueue(static_cast<uint32_t>(i));
        }
    }
    bool      acquire(uint32_t& slot) { return free_.try_dequeue(slot); }
    void      release(uint32_t slot)  { free_.enqueue(slot); }
    cv::Mat&  mat(uint32_t slot)      { return mats_[slot]; }
    cv::Size  size() const            { return size_; }
    size_t    free_count() const      { return free_.size_approx(); }
    size_t    capacity() const        { return mats_.size(); }

private:
    cv::Size size_;
    std::vector<cv::Mat> mats_;
    moodycamel::ConcurrentQueue<uint32_t> free_;
};

} // namespace core
