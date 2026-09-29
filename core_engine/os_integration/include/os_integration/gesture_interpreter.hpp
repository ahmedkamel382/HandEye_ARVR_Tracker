#pragma once

#include "os_integration/types.hpp"

#include <array>
#include <cstdint>

namespace osi {

// Confirms a per-frame classification only after it has persisted, so a
// single misclassified frame (common with the per-frame Random Forest) never
// clicks. Time is measured from the first to the latest frame showing the
// state; different states for up to `glitch_tolerance_seconds` are ignored.
class StateDebouncer {
public:
    static constexpr int kMaxStates = 4; // states are 0..3, 0 = neutral

    struct Timing {
        // onset_seconds[s]: how long state s must persist to be confirmed.
        std::array<double, kMaxStates> onset_seconds{};
        // release_seconds[s]: how long neutral must persist to leave state s.
        std::array<double, kMaxStates> release_seconds{};
        double glitch_tolerance_seconds = 0.05;
        // A gap without frames longer than this discards partial evidence.
        double max_gap_seconds = 0.25;
    };

    explicit StateDebouncer(const Timing& timing) noexcept : timing_(timing) {}

    // `raw` outside [0, kMaxStates) is treated as neutral. Returns the
    // confirmed state after this frame.
    int update(int raw, double timestamp_seconds) noexcept;
    [[nodiscard]] int confirmed() const noexcept { return confirmed_; }
    void reset() noexcept;

private:
    Timing timing_;
    int confirmed_ = 0;
    int candidate_ = 0;
    double candidate_since_ = 0.0;
    double candidate_last_seen_ = 0.0;
    bool has_history_ = false;
};

enum class FistAction : std::uint8_t { None, RightClick, MiddleClick, TogglePause };

struct GestureSettings {
    double glitch_tolerance_seconds = 0.05;
    double max_gap_seconds = 0.25;

    // Hand: pinch holds the left button (click, or drag while held).
    double pinch_onset_seconds = 0.06;
    double pinch_release_seconds = 0.08;
    double fist_onset_seconds = 0.30;
    FistAction fist_action = FistAction::RightClick;

    // Eyes: a deliberate wink must outlast the reflexive blink the gaze
    // model sometimes labels as a wink. Natural blinks close both eyes and
    // last 0.1-0.4 s, so a sustained closure needs to be much longer.
    double wink_onset_seconds = 0.25;
    double closure_onset_seconds = 1.0;
    bool closure_toggles_pause = true;
};

// Turns raw per-frame gesture states into edge-triggered mouse actions:
//   pinch start -> left down, pinch end -> left up (click or drag)
//   fist        -> FistAction (right click by default)
//   left wink   -> left click, right wink -> right click
//   sustained eye closure -> toggle pause (hands-free on/off)
// Pure logic: no clock, no OS calls, no allocation.
class GestureInterpreter {
public:
    explicit GestureInterpreter(const GestureSettings& settings) noexcept;

    // Missing trackers should be reported as Neutral. Calibrating counts as Neutral.
    ActionList update(HandGesture hand_raw, EyeState eye_raw, double timestamp_seconds) noexcept;

    [[nodiscard]] HandGesture confirmed_hand() const noexcept;
    [[nodiscard]] EyeState confirmed_eye() const noexcept;
    void reset() noexcept;

private:
    GestureSettings settings_;
    StateDebouncer hand_;
    StateDebouncer eye_;
};

} // namespace osi
