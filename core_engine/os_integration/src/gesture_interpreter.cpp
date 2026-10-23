#include "os_integration/gesture_interpreter.hpp"

namespace osi {
namespace {

StateDebouncer::Timing hand_timing(const GestureSettings& s) noexcept {
    StateDebouncer::Timing t;
    t.onset_seconds = {0.0, s.pinch_onset_seconds, s.fist_onset_seconds, 0.0};
    t.release_seconds = {0.0, s.pinch_release_seconds, s.pinch_release_seconds, 0.0};
    t.glitch_tolerance_seconds = s.glitch_tolerance_seconds;
    t.max_gap_seconds = s.max_gap_seconds;
    return t;
}

StateDebouncer::Timing eye_timing(const GestureSettings& s) noexcept {
    StateDebouncer::Timing t;
    t.onset_seconds = {0.0, s.wink_onset_seconds, s.wink_onset_seconds, s.closure_onset_seconds};
    // Leaving an eye action needs only the glitch window; the action already fired.
    t.release_seconds = {0.0, 0.0, 0.0, 0.0};
    t.glitch_tolerance_seconds = s.glitch_tolerance_seconds;
    t.max_gap_seconds = s.max_gap_seconds;
    return t;
}

void push_fist_action(ActionList& actions, FistAction action) noexcept {
    switch (action) {
    case FistAction::RightClick:
        actions.push({ActionType::Click, MouseButton::Right});
        break;
    case FistAction::MiddleClick:
        actions.push({ActionType::Click, MouseButton::Middle});
        break;
    case FistAction::TogglePause:
        actions.push({ActionType::TogglePause, MouseButton::Left});
        break;
    case FistAction::None:
        break;
    }
}

} // namespace

int StateDebouncer::update(int raw, double timestamp_seconds) noexcept {
    if (raw < 0 || raw >= kMaxStates) {
        raw = 0;
    }
    const double t = timestamp_seconds;
    if (!has_history_ || t - candidate_last_seen_ > timing_.max_gap_seconds) {
        // First frame, or the stream stalled: start collecting evidence afresh.
        has_history_ = true;
        candidate_ = raw;
        candidate_since_ = t;
        candidate_last_seen_ = t;
    } else if (raw == candidate_) {
        candidate_last_seen_ = t;
    } else if (t - candidate_last_seen_ > timing_.glitch_tolerance_seconds) {
        candidate_ = raw;
        candidate_since_ = t;
        candidate_last_seen_ = t;
    }
    // Otherwise: a short glitch inside the current candidate; ignore it.

    if (candidate_ != confirmed_) {
        const double required = candidate_ == 0
                                    ? timing_.release_seconds[static_cast<std::size_t>(confirmed_)]
                                    : timing_.onset_seconds[static_cast<std::size_t>(candidate_)];
        if (candidate_last_seen_ - candidate_since_ >= required) {
            confirmed_ = candidate_;
        }
    }
    return confirmed_;
}

void StateDebouncer::reset() noexcept {
    confirmed_ = 0;
    candidate_ = 0;
    candidate_since_ = 0.0;
    candidate_last_seen_ = 0.0;
    has_history_ = false;
}

GestureInterpreter::GestureInterpreter(const GestureSettings& settings) noexcept
    : settings_(settings), hand_(hand_timing(settings)), eye_(eye_timing(settings)) {}

ActionList GestureInterpreter::update(HandGesture hand_raw, EyeState eye_raw,
                                      double timestamp_seconds) noexcept {
    ActionList actions;

    const auto previous_hand = static_cast<HandGesture>(hand_.confirmed());
    const auto hand = static_cast<HandGesture>(
        hand_.update(static_cast<int>(hand_raw), timestamp_seconds));
    if (hand != previous_hand) {
        if (previous_hand == HandGesture::Pinch) {
            actions.push({ActionType::ButtonUp, MouseButton::Left});
        }
        if (hand == HandGesture::Pinch) {
            actions.push({ActionType::ButtonDown, MouseButton::Left});
        } else if (hand == HandGesture::Fist) {
            push_fist_action(actions, settings_.fist_action);
        }
    }

    const int eye_input = eye_raw == EyeState::Calibrating ? 0 : static_cast<int>(eye_raw);
    const auto previous_eye = static_cast<EyeState>(eye_.confirmed());
    const auto eye = static_cast<EyeState>(eye_.update(eye_input, timestamp_seconds));
    if (eye != previous_eye) {
        if (eye == EyeState::LeftWink) {
            actions.push({ActionType::Click, MouseButton::Left});
        } else if (eye == EyeState::RightWink) {
            actions.push({ActionType::Click, MouseButton::Right});
        } else if (eye == EyeState::SustainedClosure && settings_.closure_toggles_pause) {
            actions.push({ActionType::TogglePause, MouseButton::Left});
        }
    }
    return actions;
}

HandGesture GestureInterpreter::confirmed_hand() const noexcept {
    return static_cast<HandGesture>(hand_.confirmed());
}

EyeState GestureInterpreter::confirmed_eye() const noexcept {
    return static_cast<EyeState>(eye_.confirmed());
}

void GestureInterpreter::reset() noexcept {
    hand_.reset();
    eye_.reset();
}

} // namespace osi
