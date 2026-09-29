#pragma once

#include "os_integration/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace osi {

// Recent cursor positions for one input channel, newest last.
class PositionHistory {
public:
    static constexpr std::size_t kCapacity = 64; // ~2 s at 30 FPS

    void push(double timestamp_seconds, PixelPoint position) noexcept;
    // Newest position recorded at or before `timestamp_seconds`; the oldest
    // entry when everything is newer; empty when nothing was recorded.
    [[nodiscard]] std::optional<PixelPoint> at_or_before(double timestamp_seconds) const noexcept;
    void clear() noexcept { size_ = 0; }

private:
    struct Entry {
        double timestamp;
        PixelPoint position;
    };
    std::array<Entry, kCapacity> entries_{};
    std::size_t head_ = 0; // next write index
    std::size_t size_ = 0;
};

// Freezes one channel's cursor while a click gesture is forming.
//
// Pinching drags the index fingertip (the hand cursor) toward the thumb, and
// a wink corrupts the iris ratio (the gaze cursor). Without a lock, every
// click lands a few dozen pixels away from what the user was pointing at.
// When a gesture starts, the cursor snaps back to where it was `lookback`
// seconds earlier and stays there until the gesture ends plus `post_hold`.
//
// Moving further than `escape_px` from the lock point either starts a drag
// (the button is held: the cursor follows with a constant offset) or, when
// no button is held, cancels the lock because the user is just moving.
class CursorLock {
public:
    struct Params {
        double lookback_seconds = 0.10;
        double post_hold_seconds = 0.15;
        double escape_px = 30.0; // <= 0 disables escaping (used for gaze)
    };

    explicit CursorLock(const Params& params) noexcept : params_(params) {}

    // live: this frame's mapped position (empty if the tracker lost its target).
    // gesture_active: a click gesture is forming or held on this channel.
    // button_held: the gesture currently holds a mouse button (drag allowed).
    // Returns the position to show, or empty to leave the cursor where it is.
    std::optional<PixelPoint> update(std::optional<PixelPoint> live, bool gesture_active,
                                     bool button_held, double timestamp_seconds) noexcept;

    [[nodiscard]] bool locked() const noexcept { return locked_; }
    [[nodiscard]] bool dragging() const noexcept { return dragging_; }
    void reset() noexcept;

private:
    Params params_;
    PositionHistory history_;
    bool locked_ = false;
    bool dragging_ = false;
    bool suppressed_ = false; // escaped without a button; wait for the gesture to end
    PixelPoint anchor_{0, 0};
    PixelPoint live_at_lock_{0, 0};
    PixelPoint drag_offset_{0, 0};
    std::optional<PixelPoint> last_output_;
    double last_active_ = 0.0;
};

enum class CursorSource : std::uint8_t { None, Hand, Gaze };
enum class CursorSourceMode : std::uint8_t { HandPreferred, HandOnly, GazeOnly };

// Chooses which tracker drives the cursor. In HandPreferred mode the hand
// wins whenever it is tracked and gaze takes over when it is lost; a switch
// only happens after the new choice has been stable for `switch_seconds`,
// so a flickering hand detection cannot make the cursor jump back and forth.
class SourceSelector {
public:
    SourceSelector(CursorSourceMode mode, double switch_seconds) noexcept
        : mode_(mode), switch_seconds_(switch_seconds) {}

    CursorSource update(bool hand_available, bool gaze_available, double timestamp_seconds) noexcept;
    [[nodiscard]] CursorSource current() const noexcept { return current_; }
    void reset() noexcept;

private:
    CursorSourceMode mode_;
    double switch_seconds_;
    CursorSource current_ = CursorSource::None;
    CursorSource pending_ = CursorSource::None;
    double pending_since_ = 0.0;
};

} // namespace osi
