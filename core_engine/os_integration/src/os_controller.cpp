#include "os_integration/os_controller.hpp"

#include <cmath>
#include <stdexcept>

namespace osi {
namespace {

const Settings& checked(const Settings& settings) {
    const auto problems = validate(settings);
    if (!problems.empty()) {
        throw std::invalid_argument("Invalid OS integration settings: " + problems.front());
    }
    return settings;
}

CursorLock::Params hand_lock_params(const Settings& s) noexcept {
    return {s.stabilization.anchor_lookback_seconds, s.stabilization.post_click_hold_seconds,
            s.stabilization.drag_threshold_px};
}

// Gaze never escapes a lock by moving: during a wink the iris ratio is
// meaningless, so any apparent movement is noise.
CursorLock::Params gaze_lock_params(const Settings& s) noexcept {
    return {s.stabilization.anchor_lookback_seconds, s.stabilization.post_click_hold_seconds, 0.0};
}

std::optional<NormalizedPoint> usable(const std::optional<NormalizedPoint>& point) noexcept {
    if (point && is_normalized(*point)) {
        return point;
    }
    return std::nullopt;
}

std::size_t index(MouseButton button) noexcept {
    return static_cast<std::size_t>(button);
}

} // namespace

OsController::OsController(const Settings& settings, const ScreenGeometry& geometry,
                           IInputSink& sink)
    : settings_(checked(settings)),
      mapper_(geometry),
      sink_(sink),
      gestures_(settings.gestures),
      selector_(settings.mapping.cursor_source, settings.mapping.source_switch_seconds),
      hand_lock_(hand_lock_params(settings)),
      gaze_lock_(gaze_lock_params(settings)) {}

OsController::~OsController() {
    release_all_buttons();
}

FrameReport OsController::update(const FramePayload& frame) noexcept {
    FrameReport report;
    const double t = frame.timestamp_seconds;
    if (!std::isfinite(t) || (has_timestamp_ && t <= last_timestamp_)) {
        report.status = FrameStatus::RejectedTimestamp;
        report.source = selector_.current();
        report.cursor = last_cursor_;
        report.active = active_;
        report.inhibited = inhibited_;
        return report;
    }
    has_timestamp_ = true;
    last_timestamp_ = t;

    // States are trusted even when DSP dropped the point this frame: the
    // Python trackers already report Neutral whenever they lose the target.
    HandGesture hand_raw = HandGesture::Neutral;
    if (const auto gesture = hand_gesture_from_int(frame.hand.state)) {
        hand_raw = *gesture;
    } else {
        ++report.unknown_states;
    }
    EyeState eye_raw = EyeState::Neutral;
    if (const auto eye = eye_state_from_int(frame.gaze.state)) {
        eye_raw = *eye;
    } else {
        ++report.unknown_states;
    }
    const bool calibrating = eye_raw == EyeState::Calibrating;
    report.gaze_calibrating = calibrating;

    const ActionList actions = gestures_.update(hand_raw, eye_raw, t);

    // Map both channels every frame so each keeps its own history.
    const auto hand_point = usable(frame.hand.point);
    const auto gaze_point = usable(frame.gaze.point);
    std::optional<PixelPoint> hand_live;
    if (hand_point) {
        hand_live = mapper_.map(settings_.mapping.hand_region, *hand_point);
    }
    std::optional<PixelPoint> gaze_live;
    if (gaze_point && !calibrating) { // gaze_intent.py: -1 means "hold the cursor"
        gaze_live = mapper_.map(settings_.mapping.gaze_region, *gaze_point);
    }

    const CursorSource source =
        selector_.update(hand_live.has_value(), gaze_live.has_value(), t);
    const bool pinch_held = gestures_.confirmed_hand() == HandGesture::Pinch;
    const auto hand_out =
        hand_lock_.update(hand_live, hand_raw != HandGesture::Neutral, pinch_held, t);
    const bool eye_gesture = eye_raw != EyeState::Neutral && !calibrating;
    const auto gaze_out = gaze_lock_.update(gaze_live, eye_gesture, false, t);

    std::optional<PixelPoint> cursor;
    if (source == CursorSource::Hand) {
        cursor = hand_out;
    } else if (source == CursorSource::Gaze) {
        cursor = gaze_out;
    }
    if (cursor) {
        cursor = mapper_.clamp(*cursor);
        last_cursor_ = cursor;
    }
    report.source = source;
    report.cursor = last_cursor_;

    // Pause toggles work while paused (hands-free resume) but not while inhibited.
    for (const Action& action : actions) {
        if (action.type == ActionType::TogglePause && !inhibited_) {
            set_active(!active_);
            report.pause_toggled = true;
        }
    }

    if (can_output()) {
        if (cursor && cursor != last_sent_) {
            report.moved = true;
            last_sent_ = cursor;
            if (!sink_.move_to(*cursor, mapper_.to_absolute(*cursor))) {
                report.injection_failed = true;
            }
        }
        for (const Action& action : actions) {
            if (action.type == ActionType::TogglePause) {
                continue;
            }
            switch (execute(action)) {
            case Outcome::Sent:
                report.executed.push(action);
                break;
            case Outcome::Failed:
                report.executed.push(action);
                report.injection_failed = true;
                break;
            case Outcome::Skipped:
                break;
            }
        }
    }
    report.active = active_;
    report.inhibited = inhibited_;
    return report;
}

// Redundant requests (pressing a held button, releasing a free one, clicking
// a button a pinch is holding) are skipped.
OsController::Outcome OsController::execute(const Action& action) noexcept {
    bool& held = held_[index(action.button)];
    switch (action.type) {
    case ActionType::ButtonDown:
        if (held) {
            return Outcome::Skipped;
        }
        // A refused press is not held, so no matching release will be sent.
        held = sink_.button(action.button, ButtonAction::Down);
        return held ? Outcome::Sent : Outcome::Failed;
    case ActionType::ButtonUp:
        if (!held) {
            return Outcome::Skipped;
        }
        held = false;
        return sink_.button(action.button, ButtonAction::Up) ? Outcome::Sent : Outcome::Failed;
    case ActionType::Click:
        if (held) {
            return Outcome::Skipped;
        }
        return sink_.click(action.button) ? Outcome::Sent : Outcome::Failed;
    case ActionType::TogglePause:
        break;
    }
    return Outcome::Skipped;
}

void OsController::set_active(bool active) noexcept {
    if (active_ == active) {
        return;
    }
    active_ = active;
    if (!can_output()) {
        release_all_buttons();
    }
    last_sent_.reset(); // resend the position on resume
}

void OsController::set_inhibited(bool inhibited) noexcept {
    if (inhibited_ == inhibited) {
        return;
    }
    inhibited_ = inhibited;
    if (!can_output()) {
        release_all_buttons();
    }
    last_sent_.reset();
}

void OsController::release_all_buttons() noexcept {
    for (std::size_t i = 0; i < held_.size(); ++i) {
        if (held_[i]) {
            held_[i] = false;
            sink_.button(static_cast<MouseButton>(i), ButtonAction::Up);
        }
    }
}

bool OsController::button_held(MouseButton button) const noexcept {
    return held_[index(button)];
}

void OsController::apply_settings(const Settings& settings) {
    checked(settings);
    release_all_buttons();
    settings_ = settings;
    gestures_ = GestureInterpreter(settings.gestures);
    selector_ = SourceSelector(settings.mapping.cursor_source, settings.mapping.source_switch_seconds);
    hand_lock_ = CursorLock(hand_lock_params(settings));
    gaze_lock_ = CursorLock(gaze_lock_params(settings));
    reset_tracking_state();
}

void OsController::set_screen_geometry(const ScreenGeometry& geometry) {
    ScreenMapper mapper(geometry);
    release_all_buttons();
    mapper_ = mapper;
    reset_tracking_state();
}

void OsController::reset_tracking_state() noexcept {
    gestures_.reset();
    selector_.reset();
    hand_lock_.reset();
    gaze_lock_.reset();
    last_sent_.reset();
    last_cursor_.reset();
}

} // namespace osi
