#include "os_integration/cursor_stabilizer.hpp"

#include <cmath>

namespace osi {
namespace {

double distance(PixelPoint a, PixelPoint b) noexcept {
    return std::hypot(static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y);
}

} // namespace

void PositionHistory::push(double timestamp_seconds, PixelPoint position) noexcept {
    entries_[head_] = {timestamp_seconds, position};
    head_ = (head_ + 1) % kCapacity;
    if (size_ < kCapacity) {
        ++size_;
    }
}

std::optional<PixelPoint> PositionHistory::at_or_before(double timestamp_seconds) const noexcept {
    if (size_ == 0) {
        return std::nullopt;
    }
    // Walk from newest to oldest.
    for (std::size_t i = 0; i < size_; ++i) {
        const std::size_t index = (head_ + kCapacity - 1 - i) % kCapacity;
        if (entries_[index].timestamp <= timestamp_seconds) {
            return entries_[index].position;
        }
    }
    const std::size_t oldest = (head_ + kCapacity - size_) % kCapacity;
    return entries_[oldest].position;
}

std::optional<PixelPoint> CursorLock::update(std::optional<PixelPoint> live, bool gesture_active,
                                             bool button_held, double timestamp_seconds) noexcept {
    const double t = timestamp_seconds;
    gesture_active = gesture_active || button_held;
    if (live) {
        history_.push(t, *live);
    }
    if (!gesture_active) {
        suppressed_ = false;
    }

    if (locked_) {
        if (gesture_active) {
            last_active_ = t;
        }
        const bool moved_far = !dragging_ && live && params_.escape_px > 0.0 &&
                               distance(*live, live_at_lock_) > params_.escape_px;
        if (moved_far && button_held) {
            dragging_ = true;
            drag_offset_ = {anchor_.x - live_at_lock_.x, anchor_.y - live_at_lock_.y};
        } else if (moved_far) {
            // Not a click after all; release the cursor to the live position.
            locked_ = false;
            suppressed_ = gesture_active;
        } else if (!gesture_active && t - last_active_ >= params_.post_hold_seconds) {
            locked_ = false;
            dragging_ = false;
        }

        if (locked_) {
            if (dragging_ && gesture_active && live) {
                last_output_ = PixelPoint{live->x + drag_offset_.x, live->y + drag_offset_.y};
            }
            return last_output_;
        }
        dragging_ = false;
    }

    if (gesture_active && !suppressed_) {
        const auto anchor = history_.at_or_before(t - params_.lookback_seconds);
        if (anchor) {
            locked_ = true;
            dragging_ = false;
            anchor_ = *anchor;
            live_at_lock_ = live ? *live : *anchor;
            last_active_ = t;
            last_output_ = anchor_;
            return last_output_;
        }
    }
    if (live) {
        last_output_ = live;
    }
    return live;
}

void CursorLock::reset() noexcept {
    history_.clear();
    locked_ = false;
    dragging_ = false;
    suppressed_ = false;
    last_output_.reset();
    last_active_ = 0.0;
}

CursorSource SourceSelector::update(bool hand_available, bool gaze_available,
                                    double timestamp_seconds) noexcept {
    CursorSource desired = current_;
    switch (mode_) {
    case CursorSourceMode::HandOnly:
        desired = CursorSource::Hand;
        break;
    case CursorSourceMode::GazeOnly:
        desired = CursorSource::Gaze;
        break;
    case CursorSourceMode::HandPreferred:
        if (hand_available) {
            desired = CursorSource::Hand;
        } else if (gaze_available) {
            desired = CursorSource::Gaze;
        }
        break;
    }

    if (desired == current_) {
        pending_ = current_;
        return current_;
    }
    if (current_ == CursorSource::None) {
        current_ = pending_ = desired;
        return current_;
    }
    if (pending_ != desired) {
        pending_ = desired;
        pending_since_ = timestamp_seconds;
    }
    if (timestamp_seconds - pending_since_ >= switch_seconds_) {
        current_ = desired;
    }
    return current_;
}

void SourceSelector::reset() noexcept {
    current_ = CursorSource::None;
    pending_ = CursorSource::None;
    pending_since_ = 0.0;
}

} // namespace osi
