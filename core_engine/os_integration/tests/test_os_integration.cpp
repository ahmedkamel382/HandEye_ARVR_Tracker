#include "os_integration/calibration.hpp"
#include "os_integration/cursor_stabilizer.hpp"
#include "os_integration/gesture_interpreter.hpp"
#include "os_integration/input_sink.hpp"
#include "os_integration/os_controller.hpp"
#include "os_integration/physical_input_guard.hpp"
#include "os_integration/screen_mapper.hpp"
#include "os_integration/settings.hpp"

#if defined(OSI_HAVE_WIN32)
#include "os_integration/win32_input.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace osi;

int passes = 0;
int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void close_to(double actual, double expected, double tolerance, const std::string& what) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            what + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template <typename Test>
void run_test(const char* name, Test&& test) {
    try {
        test();
        ++passes;
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cout << "FAIL " << name << ": " << error.what() << '\n';
    }
}

std::string str(PixelPoint p) {
    return "(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
}

void require_pixel(std::optional<PixelPoint> actual, PixelPoint expected, const std::string& what) {
    require(actual.has_value(), what + ": expected " + str(expected) + ", got nothing");
    require(*actual == expected, what + ": expected " + str(expected) + ", got " + str(*actual));
}

// Frame period that is exact in binary floating point (32 FPS).
constexpr double kDt = 1.0 / 32.0;
constexpr NormalizedPoint kCentre{0.5, 0.5};
const ScreenGeometry kFullHd{{0, 0, 1920, 1080}, {0, 0, 1920, 1080}};

// ------------------------------------------------------------ gesture helpers

std::vector<Action> feed(GestureInterpreter& interpreter, HandGesture hand, EyeState eye,
                         double& t, int frames) {
    std::vector<Action> actions;
    for (int i = 0; i < frames; ++i) {
        for (const Action& action : interpreter.update(hand, eye, t)) {
            actions.push_back(action);
        }
        t += kDt;
    }
    return actions;
}

bool same_actions(const std::vector<Action>& actual, const std::vector<Action>& expected) {
    return actual.size() == expected.size() && std::equal(actual.begin(), actual.end(), expected.begin());
}

// ------------------------------------------------------------ controller helpers

Settings unit_regions() {
    Settings settings;
    settings.mapping.hand_region = {0.0, 0.0, 1.0, 1.0};
    settings.mapping.gaze_region = {0.0, 0.0, 1.0, 1.0};
    return settings;
}

FramePayload hand_frame(double t, std::optional<NormalizedPoint> point,
                        HandGesture gesture = HandGesture::Neutral) {
    FramePayload frame;
    frame.timestamp_seconds = t;
    frame.hand = {point, static_cast<std::int32_t>(gesture)};
    return frame;
}

FramePayload gaze_frame(double t, std::optional<NormalizedPoint> point,
                        EyeState state = EyeState::Neutral) {
    FramePayload frame;
    frame.timestamp_seconds = t;
    frame.gaze = {point, static_cast<std::int32_t>(state)};
    return frame;
}

std::size_t count(const RecordingInputSink& sink, SinkEvent::Kind kind) {
    return static_cast<std::size_t>(std::count_if(
        sink.events.begin(), sink.events.end(), [kind](const SinkEvent& e) { return e.kind == kind; }));
}

std::optional<std::size_t> find_event(const RecordingInputSink& sink, SinkEvent::Kind kind,
                                      std::size_t from = 0) {
    for (std::size_t i = from; i < sink.events.size(); ++i) {
        if (sink.events[i].kind == kind) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<PixelPoint> last_move_before(const RecordingInputSink& sink, std::size_t index) {
    for (std::size_t i = index; i-- > 0;) {
        if (sink.events[i].kind == SinkEvent::Kind::Move) {
            return sink.events[i].pixel;
        }
    }
    return std::nullopt;
}

// ------------------------------------------------------------ calibration helpers

std::vector<TargetObservation> synthetic_observations(const InputRegion& truth, unsigned seed,
                                                      double noise, int outliers_per_target = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> jitter(-noise, noise);
    std::uniform_real_distribution<double> wild(0.0, 1.0);
    std::vector<TargetObservation> observations;
    for (const CalibrationTarget& target : default_calibration_targets()) {
        TargetObservation observation{target, {}};
        const double x = truth.x_min + target.u * (truth.x_max - truth.x_min);
        const double y = truth.y_min + target.v * (truth.y_max - truth.y_min);
        for (int i = 0; i < 20; ++i) {
            observation.samples.push_back({x + jitter(rng), y + jitter(rng)});
        }
        for (int i = 0; i < outliers_per_target; ++i) {
            observation.samples.push_back({wild(rng), wild(rng)});
        }
        observations.push_back(observation);
    }
    return observations;
}

void require_region(const InputRegion& actual, const InputRegion& expected, double tolerance) {
    close_to(actual.x_min, expected.x_min, tolerance, "x_min");
    close_to(actual.y_min, expected.y_min, tolerance, "y_min");
    close_to(actual.x_max, expected.x_max, tolerance, "x_max");
    close_to(actual.y_max, expected.y_max, tolerance, "y_max");
}

} // namespace

int main() {
    // ================================================================ types
    run_test("state_integer_contract", [] {
        require(hand_gesture_from_int(0) == HandGesture::Neutral, "hand 0");
        require(hand_gesture_from_int(1) == HandGesture::Pinch, "hand 1");
        require(hand_gesture_from_int(2) == HandGesture::Fist, "hand 2");
        require(!hand_gesture_from_int(3) && !hand_gesture_from_int(-1), "hand out of range");
        require(eye_state_from_int(-1) == EyeState::Calibrating, "eye -1");
        require(eye_state_from_int(3) == EyeState::SustainedClosure, "eye 3");
        require(!eye_state_from_int(4) && !eye_state_from_int(-2), "eye out of range");
    });
    run_test("normalized_point_validation", [] {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        require(is_normalized({0.0, 1.0}) && is_normalized({1.0, 0.0}), "bounds are inclusive");
        require(!is_normalized({-1.0, -1.0}), "python sentinel accepted");
        require(!is_normalized({-1e-12, 0.5}) && !is_normalized({0.5, 1.0 + 1e-12}), "outside accepted");
        require(!is_normalized({nan, 0.5}), "NaN accepted");
    });
    run_test("action_list_capacity", [] {
        ActionList list;
        for (std::size_t i = 0; i < ActionList::kCapacity; ++i) {
            require(list.push({ActionType::Click, MouseButton::Left}), "push within capacity");
        }
        require(!list.push({ActionType::Click, MouseButton::Left}), "push beyond capacity");
        require(list.size() == ActionList::kCapacity, "size");
    });

    // ================================================================ mapping
    run_test("region_linear_interpolation", [] {
        const InputRegion region{0.2, 0.3, 0.8, 0.7};
        auto u = region_to_unit(region, {0.5, 0.5});
        close_to(u.x, 0.5, 1e-12, "centre x");
        close_to(u.y, 0.5, 1e-12, "centre y");
        u = region_to_unit(region, {0.35, 0.4});
        close_to(u.x, 0.25, 1e-12, "quarter x");
        close_to(u.y, 0.25, 1e-12, "quarter y");
        u = region_to_unit(region, {0.2, 0.7});
        close_to(u.x, 0.0, 1e-12, "min edge");
        close_to(u.y, 1.0, 1e-12, "max edge");
    });
    run_test("region_clamps_to_screen_edges", [] {
        const InputRegion region{0.2, 0.3, 0.8, 0.7};
        const auto u = region_to_unit(region, {0.0, 1.0});
        require(u.x == 0.0 && u.y == 1.0, "outside input must pin to the edge");
    });
    run_test("inverted_region_flips_axis", [] {
        const InputRegion region{0.8, 0.7, 0.2, 0.3};
        auto u = region_to_unit(region, {0.8, 0.7});
        require(u.x == 0.0 && u.y == 0.0, "inverted min");
        u = region_to_unit(region, {0.2, 0.3});
        close_to(u.x, 1.0, 1e-12, "inverted max x");
        close_to(u.y, 1.0, 1e-12, "inverted max y");
    });
    run_test("region_validation", [] {
        require(is_valid_region({0.0, 0.0, 1.0, 1.0}), "unit region");
        require(is_valid_region({0.9, 0.9, 0.1, 0.1}), "inverted region");
        require(is_valid_region({-0.1, -0.2, 1.1, 1.3}), "extrapolated region");
        require(!is_valid_region({0.5, 0.0, 0.51, 1.0}), "degenerate x span");
        require(!is_valid_region({0.0, 0.0, 2.5, 1.0}), "corner out of range");
        require(!is_valid_region({std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0, 1.0}), "NaN");
    });
    run_test("unit_to_pixel_corners_and_centre", [] {
        const ScreenRect screen{0, 0, 1920, 1080};
        require(unit_to_pixel(screen, {0.0, 0.0}) == PixelPoint{0, 0}, "top-left");
        require(unit_to_pixel(screen, {1.0, 1.0}) == PixelPoint{1919, 1079}, "bottom-right");
        require(unit_to_pixel(screen, {0.5, 0.5}) == PixelPoint{960, 540}, "centre");
    });
    run_test("unit_to_pixel_negative_origin_monitor", [] {
        const ScreenRect left_monitor{-1920, -200, 1920, 1080};
        require(unit_to_pixel(left_monitor, {0.0, 0.0}) == PixelPoint{-1920, -200}, "top-left");
        require(unit_to_pixel(left_monitor, {1.0, 1.0}) == PixelPoint{-1, 879}, "bottom-right");
    });
    run_test("absolute_known_values", [] {
        const ScreenRect desk{0, 0, 1920, 1080};
        require(pixel_to_absolute(desk, {0, 0}) == AbsolutePoint{0, 0}, "origin");
        require(pixel_to_absolute(desk, {960, 540}) == AbsolutePoint{32768, 32768}, "centre");
        require(pixel_to_absolute(desk, {1919, 1079}).x == 65502, "last column");
    });
    run_test("absolute_round_trip_every_pixel", [] {
        std::size_t checked = 0;
        for (std::int32_t width : {1, 2, 3, 640, 800, 1024, 1280, 1366, 1920, 2560, 3440, 3840,
                                   5760, 7680, 65535, 65536}) {
            for (std::int32_t left : {0, -1920, 1234}) {
                const ScreenRect desk{left, left, width, width};
                std::int32_t previous = -1;
                for (std::int32_t offset = 0; offset < width; ++offset) {
                    const PixelPoint pixel{left + offset, left + offset};
                    const AbsolutePoint absolute = pixel_to_absolute(desk, pixel);
                    require(absolute.x >= 0 && absolute.x <= 65535, "absolute outside 0..65535");
                    require(absolute.x > previous || width > 65536, "absolute not increasing");
                    require(absolute_to_pixel(desk, absolute) == pixel,
                            "round trip failed at width " + std::to_string(width) + " offset " +
                                std::to_string(offset));
                    previous = absolute.x;
                    ++checked;
                }
            }
        }
        std::cout << "ROUND_TRIP pixels_checked=" << checked << '\n';
    });
    run_test("absolute_clamps_outside_desktop", [] {
        const ScreenRect desk{-1920, 0, 3840, 1080};
        require(pixel_to_absolute(desk, {-5000, -10}) == AbsolutePoint{0, 0}, "before origin");
        require(pixel_to_absolute(desk, {99999, 99999}) == pixel_to_absolute(desk, {1919, 1079}),
                "past the end");
    });
    run_test("screen_mapper_clips_and_rejects", [] {
        const ScreenMapper clipped({{0, 0, 1920, 1080}, {1000, 500, 2000, 2000}});
        const ScreenRect target = clipped.geometry().target;
        require(target.left == 1000 && target.top == 500 && target.width == 920 && target.height == 580,
                "target not clipped to the desktop");
        bool threw = false;
        try {
            ScreenMapper({{0, 0, 1920, 1080}, {5000, 0, 100, 100}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "target outside the desktop accepted");
        threw = false;
        try {
            ScreenMapper({{0, 0, 0, 1080}, {0, 0, 10, 10}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "empty desktop accepted");
    });

    // ================================================================ debouncer
    StateDebouncer::Timing timing;
    timing.onset_seconds = {0.0, 0.1, 0.1, 0.1};
    timing.release_seconds = {0.0, 0.08, 0.08, 0.08};

    run_test("debounce_requires_persistence", [timing] {
        StateDebouncer debouncer(timing);
        double t = 0.0;
        for (int i = 0; i < 4; ++i, t += kDt) {
            require(debouncer.update(1, t) == 0, "confirmed too early at frame " + std::to_string(i));
        }
        require(debouncer.update(1, t) == 1, "not confirmed after 0.125 s");
    });
    run_test("debounce_ignores_single_glitch", [timing] {
        StateDebouncer debouncer(timing);
        const int stream[] = {1, 1, 0, 1, 1};
        int confirmed = 0;
        double t = 0.0;
        for (int raw : stream) {
            confirmed = debouncer.update(raw, t);
            t += kDt;
        }
        require(confirmed == 1, "one-frame glitch restarted the timer");
    });
    run_test("debounce_two_frame_gap_restarts", [timing] {
        StateDebouncer debouncer(timing);
        const int stream[] = {1, 1, 0, 0, 1, 1, 1, 1, 1, 1};
        std::vector<int> confirmed;
        double t = 0.0;
        for (int raw : stream) {
            confirmed.push_back(debouncer.update(raw, t));
            t += kDt;
        }
        // Candidate restarts at frame 5 and needs 0.1 s: confirmed at frame 9.
        for (std::size_t i = 0; i < 9; ++i) {
            require(confirmed[i] == 0, "confirmed early at frame " + std::to_string(i));
        }
        require(confirmed[9] == 1, "not confirmed at frame 9");
    });
    run_test("debounce_release_needs_neutral_time", [timing] {
        StateDebouncer debouncer(timing);
        double t = 0.0;
        for (int i = 0; i < 6; ++i, t += kDt) {
            debouncer.update(1, t);
        }
        require(debouncer.confirmed() == 1, "setup");
        require(debouncer.update(0, t) == 1, "released on the first neutral frame");
        t += kDt;
        int frames = 1;
        while (debouncer.update(0, t) == 1) {
            t += kDt;
            require(++frames < 20, "never released");
        }
        require(frames >= 4, "released after only " + std::to_string(frames) + " frames");
    });
    run_test("debounce_gap_discards_partial_evidence", [timing] {
        StateDebouncer debouncer(timing);
        debouncer.update(1, 0.0);
        require(debouncer.update(1, 1.0) == 0, "stale evidence confirmed across a 1 s gap");
    });
    run_test("debounce_out_of_range_is_neutral", [timing] {
        StateDebouncer debouncer(timing);
        for (int i = 0; i < 10; ++i) {
            require(debouncer.update(9, i * kDt) == 0, "out-of-range state confirmed");
        }
    });

    // ================================================================ gestures
    run_test("pinch_is_press_then_release", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        std::vector<Action> all;
        for (auto [gesture, frames] : {std::pair{HandGesture::Neutral, 5}, std::pair{HandGesture::Pinch, 10},
                                       std::pair{HandGesture::Neutral, 10}}) {
            const auto actions = feed(g, gesture, EyeState::Neutral, t, frames);
            all.insert(all.end(), actions.begin(), actions.end());
        }
        require(same_actions(all, {{ActionType::ButtonDown, MouseButton::Left},
                                   {ActionType::ButtonUp, MouseButton::Left}}),
                "expected left down then left up");
    });
    run_test("brief_pinch_flicker_does_not_click", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        std::vector<Action> all;
        for (int flicker = 1; flicker <= 2; ++flicker) {
            auto a = feed(g, HandGesture::Neutral, EyeState::Neutral, t, 5);
            all.insert(all.end(), a.begin(), a.end());
            a = feed(g, HandGesture::Pinch, EyeState::Neutral, t, flicker);
            all.insert(all.end(), a.begin(), a.end());
        }
        auto a = feed(g, HandGesture::Neutral, EyeState::Neutral, t, 5);
        all.insert(all.end(), a.begin(), a.end());
        require(all.empty(), "a 1-2 frame pinch produced a click");
    });
    run_test("fist_right_clicks_once_per_fist", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        auto first = feed(g, HandGesture::Fist, EyeState::Neutral, t, 30);
        feed(g, HandGesture::Neutral, EyeState::Neutral, t, 10);
        auto second = feed(g, HandGesture::Fist, EyeState::Neutral, t, 30);
        const std::vector<Action> right_click{{ActionType::Click, MouseButton::Right}};
        require(same_actions(first, right_click) && same_actions(second, right_click),
                "expected exactly one right click per fist");
    });
    run_test("fist_action_is_configurable", [] {
        GestureSettings settings;
        settings.fist_action = FistAction::TogglePause;
        GestureInterpreter pause(settings);
        double t = 0.0;
        auto actions = feed(pause, HandGesture::Fist, EyeState::Neutral, t, 30);
        require(actions.size() == 1 && actions[0].type == ActionType::TogglePause, "toggle pause");
        settings.fist_action = FistAction::None;
        GestureInterpreter none(settings);
        t = 0.0;
        require(feed(none, HandGesture::Fist, EyeState::Neutral, t, 30).empty(), "none");
    });
    run_test("pinch_into_fist_releases_first", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        feed(g, HandGesture::Pinch, EyeState::Neutral, t, 10);
        const auto actions = feed(g, HandGesture::Fist, EyeState::Neutral, t, 30);
        require(same_actions(actions, {{ActionType::ButtonUp, MouseButton::Left},
                                       {ActionType::Click, MouseButton::Right}}),
                "expected left up before the fist action");
    });
    run_test("deliberate_wink_clicks_reflex_does_not", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        auto reflex = feed(g, HandGesture::Neutral, EyeState::LeftWink, t, 5); // 0.125 s
        feed(g, HandGesture::Neutral, EyeState::Neutral, t, 10);
        auto deliberate = feed(g, HandGesture::Neutral, EyeState::LeftWink, t, 12); // 0.34 s
        feed(g, HandGesture::Neutral, EyeState::Neutral, t, 10);
        auto right = feed(g, HandGesture::Neutral, EyeState::RightWink, t, 12);
        require(reflex.empty(), "reflex wink clicked");
        require(same_actions(deliberate, {{ActionType::Click, MouseButton::Left}}), "left wink");
        require(same_actions(right, {{ActionType::Click, MouseButton::Right}}), "right wink");
    });
    run_test("sustained_closure_toggles_pause_once", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        auto blink = feed(g, HandGesture::Neutral, EyeState::SustainedClosure, t, 10); // 0.28 s
        feed(g, HandGesture::Neutral, EyeState::Neutral, t, 10);
        auto closure = feed(g, HandGesture::Neutral, EyeState::SustainedClosure, t, 45);
        require(blink.empty(), "a natural blink toggled pause");
        require(closure.size() == 1 && closure[0].type == ActionType::TogglePause,
                "expected exactly one toggle");
        GestureSettings disabled;
        disabled.closure_toggles_pause = false;
        GestureInterpreter off(disabled);
        t = 0.0;
        require(feed(off, HandGesture::Neutral, EyeState::SustainedClosure, t, 45).empty(),
                "toggle fired while disabled");
    });
    run_test("gaze_calibration_state_is_neutral", [] {
        GestureInterpreter g(GestureSettings{});
        double t = 0.0;
        require(feed(g, HandGesture::Neutral, EyeState::Calibrating, t, 60).empty(),
                "calibrating produced actions");
        require(g.confirmed_eye() == EyeState::Neutral, "calibrating confirmed as a state");
    });

    // ================================================================ cursor lock
    run_test("history_lookup", [] {
        PositionHistory history;
        require(!history.at_or_before(1.0), "empty history returned a value");
        history.push(0.0, {1, 1});
        history.push(0.1, {2, 2});
        history.push(0.2, {3, 3});
        require_pixel(history.at_or_before(0.15), {2, 2}, "between entries");
        require_pixel(history.at_or_before(0.2), {3, 3}, "exact");
        require_pixel(history.at_or_before(-1.0), {1, 1}, "before everything returns oldest");
        for (int i = 0; i < 100; ++i) {
            history.push(1.0 + i, {i, i});
        }
        require_pixel(history.at_or_before(1000.0), {99, 99}, "newest after wrap");
        require_pixel(history.at_or_before(0.0), {36, 36}, "oldest retained after wrap");
    });
    run_test("lock_snaps_back_and_holds", [] {
        CursorLock lock({0.1, 0.15, 30.0});
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            lock.update(PixelPoint{100 + 10 * i, 100}, false, false, t);
        }
        // Frame 8 at t = 0.25; lookback 0.1 -> newest entry at or before 0.15 is frame 4 (x = 140).
        require_pixel(lock.update(PixelPoint{180, 100}, true, false, t), {140, 100}, "anchor");
        require(lock.locked(), "not locked");
        t += kDt;
        require_pixel(lock.update(PixelPoint{190, 105}, true, false, t), {140, 100}, "held while active");
        t += kDt;
        const double released_at = t - kDt; // last active frame
        std::optional<PixelPoint> output;
        while (t - released_at < 0.15) {
            output = lock.update(PixelPoint{200, 100}, false, false, t);
            require_pixel(output, {140, 100}, "post-click hold");
            t += kDt;
        }
        require_pixel(lock.update(PixelPoint{200, 100}, false, false, t), {200, 100}, "released");
        require(!lock.locked(), "still locked");
    });
    run_test("lock_escapes_when_moving_without_button", [] {
        CursorLock lock({0.1, 0.15, 30.0});
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            lock.update(PixelPoint{500, 500}, false, false, t);
        }
        lock.update(PixelPoint{505, 500}, true, false, t);
        t += kDt;
        require_pixel(lock.update(PixelPoint{560, 500}, true, false, t), {560, 500}, "escape");
        t += kDt;
        require_pixel(lock.update(PixelPoint{565, 500}, true, false, t), {565, 500}, "stays released");
        require(!lock.locked(), "relocked while the same gesture continued");
    });
    run_test("lock_drags_with_constant_offset", [] {
        CursorLock lock({0.1, 0.15, 30.0});
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            lock.update(PixelPoint{500, 500}, false, false, t);
        }
        require_pixel(lock.update(PixelPoint{509, 510}, true, false, t), {500, 500}, "anchor");
        t += kDt;
        require_pixel(lock.update(PixelPoint{509, 510}, true, true, t), {500, 500}, "held");
        t += kDt;
        require_pixel(lock.update(PixelPoint{700, 510}, true, true, t), {691, 500}, "drag offset");
        require(lock.dragging(), "not dragging");
        t += kDt;
        require_pixel(lock.update(PixelPoint{720, 530}, true, true, t), {711, 520}, "drag follows");
        t += kDt;
        require_pixel(lock.update(PixelPoint{900, 900}, false, false, t), {711, 520},
                      "holds the drop point after release");
    });
    run_test("gaze_lock_never_escapes", [] {
        CursorLock lock({0.1, 0.15, 0.0});
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            lock.update(PixelPoint{300, 300}, false, false, t);
        }
        for (int i = 0; i < 10; ++i, t += kDt) {
            require_pixel(lock.update(PixelPoint{1500, 50}, true, false, t), {300, 300},
                          "wink noise moved the gaze cursor");
        }
    });
    run_test("lock_without_history_passes_through", [] {
        CursorLock lock({0.1, 0.15, 30.0});
        require(!lock.update(std::nullopt, true, false, 0.0), "invented a position");
        require(!lock.locked(), "locked with nothing to anchor to");
    });

    // ================================================================ source selection
    run_test("hand_preferred_switching", [] {
        SourceSelector s(CursorSourceMode::HandPreferred, 0.2);
        double t = 0.0;
        require(s.update(false, true, t) == CursorSource::Gaze, "first source is immediate");
        for (int i = 0; i < 7; ++i) { // pending since frame 1; 0.1875 s < 0.2 at frame 7
            t += kDt;
            require(s.update(true, true, t) == CursorSource::Gaze, "switched to hand too early");
        }
        t += kDt;
        require(s.update(true, true, t) == CursorSource::Hand, "hand did not take over");
        for (int i = 0; i < 10; ++i) {
            t += kDt;
            s.update(false, false, t);
        }
        require(s.current() == CursorSource::Hand, "switched away with no alternative");
        for (int i = 0; i < 8; ++i) {
            t += kDt;
            s.update(false, true, t);
        }
        require(s.current() == CursorSource::Gaze, "gaze did not take over after hand loss");
    });
    run_test("hand_flicker_does_not_steal_cursor", [] {
        SourceSelector s(CursorSourceMode::HandPreferred, 0.2);
        double t = 0.0;
        s.update(false, true, t);
        for (int i = 0; i < 40; ++i) {
            t += kDt;
            require(s.update(i % 3 == 0, true, t) == CursorSource::Gaze, "flicker switched source");
        }
    });
    run_test("fixed_source_modes", [] {
        SourceSelector hand(CursorSourceMode::HandOnly, 0.2);
        SourceSelector gaze(CursorSourceMode::GazeOnly, 0.2);
        require(hand.update(false, true, 0.0) == CursorSource::Hand, "hand only");
        require(gaze.update(true, false, 0.0) == CursorSource::Gaze, "gaze only");
    });

    // ================================================================ controller
    run_test("controller_starts_paused", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        const auto report = c.update(hand_frame(0.0, kCentre));
        require(sink.events.empty(), "paused controller sent input");
        require(!report.active && report.cursor.has_value(), "report while paused");
        c.set_active(true);
        c.update(hand_frame(kDt, kCentre));
        require(count(sink, SinkEvent::Kind::Move) == 1, "no move after enabling");
    });
    run_test("controller_maps_to_pixels_and_absolute", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        c.update(hand_frame(0.0, kCentre));
        require(sink.events.size() == 1, "expected one move");
        require(sink.events[0].pixel == PixelPoint{960, 540}, "pixel");
        require(sink.events[0].absolute == AbsolutePoint{32768, 32768}, "absolute");
    });
    run_test("controller_skips_redundant_moves", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        for (int i = 0; i < 5; ++i) {
            c.update(hand_frame(i * kDt, kCentre));
        }
        require(count(sink, SinkEvent::Kind::Move) == 1, "identical positions resent");
    });
    run_test("controller_pinch_clicks_at_pre_pinch_position", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            c.update(hand_frame(t, kCentre));
        }
        // The pinch pulls the fingertip ~10 px away from the target.
        for (int i = 0; i < 6; ++i, t += kDt) {
            c.update(hand_frame(t, NormalizedPoint{0.505, 0.51}, HandGesture::Pinch));
        }
        const auto down = find_event(sink, SinkEvent::Kind::Down);
        require(down.has_value(), "no button down");
        require_pixel(last_move_before(sink, *down), {960, 540}, "click position");
        require(c.button_held(MouseButton::Left), "button not tracked as held");
        for (int i = 0; i < 12; ++i, t += kDt) {
            c.update(hand_frame(t, NormalizedPoint{0.505, 0.51}));
        }
        const auto up = find_event(sink, SinkEvent::Kind::Up, *down);
        require(up.has_value(), "no button up");
        require(!find_event(sink, SinkEvent::Kind::Move, 1).has_value() ||
                    *find_event(sink, SinkEvent::Kind::Move, 1) > *up,
                "cursor wandered between press and release");
    });
    run_test("controller_pinch_drag", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            c.update(hand_frame(t, kCentre));
        }
        for (int i = 0; i < 6; ++i, t += kDt) {
            c.update(hand_frame(t, NormalizedPoint{0.505, 0.51}, HandGesture::Pinch));
        }
        c.update(hand_frame(t, NormalizedPoint{0.6, 0.51}, HandGesture::Pinch));
        t += kDt;
        // live (1151, 550) + offset (960 - 969, 540 - 550)
        require(sink.events.back().kind == SinkEvent::Kind::Move &&
                    sink.events.back().pixel == PixelPoint{1142, 540},
                "drag did not follow with the lock offset");
        for (int i = 0; i < 12; ++i, t += kDt) {
            c.update(hand_frame(t, NormalizedPoint{0.6, 0.51}));
        }
        const auto up = find_event(sink, SinkEvent::Kind::Up);
        require(up.has_value(), "drag never released");
        require_pixel(last_move_before(sink, *up), {1142, 540}, "drop position");
    });
    run_test("controller_pause_releases_held_button", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 6; ++i, t += kDt) {
            c.update(hand_frame(t, kCentre, HandGesture::Pinch));
        }
        require(c.button_held(MouseButton::Left), "setup: pinch not held");
        c.set_active(false);
        require(count(sink, SinkEvent::Kind::Up) == 1 && !c.button_held(MouseButton::Left),
                "pausing did not release the button");
        for (int i = 0; i < 12; ++i, t += kDt) {
            c.update(hand_frame(t, kCentre));
        }
        require(count(sink, SinkEvent::Kind::Up) == 1, "released twice");
    });
    run_test("controller_inhibit_blocks_output_and_gestures", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_inhibited(true);
        double t = 0.0;
        bool toggled = false;
        for (int i = 0; i < 45; ++i, t += kDt) {
            toggled |= c.update(gaze_frame(t, kCentre, EyeState::SustainedClosure)).pause_toggled;
        }
        require(!toggled && !c.active(), "eye closure resumed control during inhibition");
        c.set_active(true);
        for (int i = 0; i < 10; ++i, t += kDt) { // long enough for the hand to take over
            c.update(hand_frame(t, kCentre));
        }
        require(sink.events.empty(), "inhibited controller sent input");
        c.set_inhibited(false);
        c.update(hand_frame(t, kCentre));
        require(count(sink, SinkEvent::Kind::Move) == 1, "no move after inhibition ended");
    });
    run_test("controller_eye_closure_resumes_hands_free", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            c.update(gaze_frame(t, kCentre));
        }
        bool toggled = false;
        for (int i = 0; i < 40 && !toggled; ++i, t += kDt) {
            require(sink.events.empty(), "input sent while paused");
            toggled = c.update(gaze_frame(t, kCentre, EyeState::SustainedClosure)).pause_toggled;
        }
        require(toggled && c.active(), "closure did not resume control");
        require(count(sink, SinkEvent::Kind::Click) == 0, "closure clicked");
    });
    run_test("controller_rejects_bad_timestamps", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        c.update(hand_frame(1.0, kCentre));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (double bad : {1.0, 0.5, nan, std::numeric_limits<double>::infinity()}) {
            const auto report = c.update(hand_frame(bad, NormalizedPoint{0.1, 0.1}));
            require(report.status == FrameStatus::RejectedTimestamp, "bad timestamp accepted");
        }
        require(sink.events.size() == 1, "rejected frame moved the cursor");
        require(c.update(hand_frame(1.1, NormalizedPoint{0.1, 0.1})).status == FrameStatus::Processed,
                "valid frame after rejection");
    });
    run_test("controller_ignores_unusable_points", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        c.update(hand_frame(0.0, NormalizedPoint{-1.0, -1.0}));
        const auto report = c.update(hand_frame(kDt, NormalizedPoint{nan, 0.5}));
        require(sink.events.empty() && report.source == CursorSource::None, "unusable point moved");
    });
    run_test("controller_counts_unknown_states", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        FramePayload frame = hand_frame(0.0, kCentre);
        frame.hand.state = 7;
        frame.gaze.state = -5;
        require(c.update(frame).unknown_states == 2, "unknown states not reported");
    });
    run_test("controller_reports_refused_injection", [] {
        RecordingInputSink sink;
        sink.fail = true;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        bool failed = false;
        for (int i = 0; i < 6; ++i, t += kDt) {
            failed |= c.update(hand_frame(t, kCentre, HandGesture::Pinch)).injection_failed;
        }
        require(failed, "failure not reported");
        require(!c.button_held(MouseButton::Left), "refused press recorded as held");
    });
    run_test("controller_gaze_holds_while_calibrating", [] {
        Settings settings = unit_regions();
        settings.mapping.cursor_source = CursorSourceMode::GazeOnly;
        RecordingInputSink sink;
        OsController c(settings, kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 10; ++i, t += kDt) {
            require(c.update(gaze_frame(t, kCentre, EyeState::Calibrating)).gaze_calibrating,
                    "calibration not reported");
        }
        require(sink.events.empty(), "cursor moved during gaze calibration");
        c.update(gaze_frame(t, kCentre));
        require(count(sink, SinkEvent::Kind::Move) == 1, "no move after calibration");
    });
    run_test("controller_wink_clicks_where_user_was_looking", [] {
        Settings settings = unit_regions();
        settings.mapping.cursor_source = CursorSourceMode::GazeOnly;
        RecordingInputSink sink;
        OsController c(settings, kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 8; ++i, t += kDt) {
            c.update(gaze_frame(t, kCentre));
        }
        // A closed eye makes the iris ratio jump; the cursor must not follow it.
        for (int i = 0; i < 12; ++i, t += kDt) {
            c.update(gaze_frame(t, NormalizedPoint{0.95, 0.05}, EyeState::LeftWink));
        }
        for (int i = 0; i < 3; ++i, t += kDt) {
            c.update(gaze_frame(t, kCentre));
        }
        require(count(sink, SinkEvent::Kind::Click) == 1, "wink did not click");
        require(count(sink, SinkEvent::Kind::Move) == 1, "wink noise moved the cursor");
    });
    run_test("controller_no_click_on_held_button", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        double t = 0.0;
        for (int i = 0; i < 20; ++i, t += kDt) {
            FramePayload frame = hand_frame(t, kCentre, HandGesture::Pinch);
            frame.gaze = {kCentre, static_cast<std::int32_t>(EyeState::LeftWink)};
            c.update(frame);
        }
        require(count(sink, SinkEvent::Kind::Down) == 1, "pinch not pressed");
        require(count(sink, SinkEvent::Kind::Click) == 0, "left click sent while left held");
    });
    run_test("controller_destructor_releases_buttons", [] {
        RecordingInputSink sink;
        {
            OsController c(unit_regions(), kFullHd, sink);
            c.set_active(true);
            for (int i = 0; i < 6; ++i) {
                c.update(hand_frame(i * kDt, kCentre, HandGesture::Pinch));
            }
        }
        require(count(sink, SinkEvent::Kind::Down) == 1 && count(sink, SinkEvent::Kind::Up) == 1,
                "button left pressed after destruction");
    });
    run_test("controller_settings_and_geometry_updates", [] {
        RecordingInputSink sink;
        OsController c(unit_regions(), kFullHd, sink);
        c.set_active(true);
        Settings bad = unit_regions();
        bad.mapping.hand_region = {0.5, 0.5, 0.5, 0.5};
        bool threw = false;
        try {
            c.apply_settings(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "invalid settings accepted");
        require(c.settings().mapping.hand_region.x_max == 1.0, "invalid settings partially applied");

        const ScreenGeometry dual{{-1920, 0, 3840, 1080}, {-1920, 0, 1920, 1080}};
        c.set_screen_geometry(dual);
        c.update(hand_frame(0.0, NormalizedPoint{0.0, 0.0}));
        c.update(hand_frame(kDt, NormalizedPoint{1.0, 1.0}));
        require(sink.events.size() == 2, "expected two moves");
        require(sink.events[0].pixel == PixelPoint{-1920, 0} &&
                    sink.events[0].absolute == AbsolutePoint{0, 0},
                "left monitor origin");
        require(sink.events[1].pixel == PixelPoint{-1, 1079} &&
                    sink.events[1].absolute == AbsolutePoint{32751, 65476},
                "left monitor far corner");
    });
    run_test("controller_rejects_invalid_construction", [] {
        RecordingInputSink sink;
        Settings bad;
        bad.gestures.wink_onset_seconds = -1.0;
        bool threw = false;
        try {
            OsController c(bad, kFullHd, sink);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "invalid settings accepted by constructor");
    });

    // ================================================================ settings
    run_test("default_settings_are_valid", [] {
        const auto problems = validate(Settings{});
        require(problems.empty(), problems.empty() ? "" : problems.front());
    });
    run_test("settings_round_trip", [] {
        Settings s;
        s.mapping.hand_region = {0.9, 0.1, 0.2, 0.8};
        s.mapping.gaze_region = {0.41, 0.37, 0.59, 0.61};
        s.mapping.cursor_source = CursorSourceMode::GazeOnly;
        s.mapping.screen_target = ScreenTarget::VirtualDesktop;
        s.stabilization.drag_threshold_px = 42.5;
        s.gestures.fist_action = FistAction::MiddleClick;
        s.gestures.wink_onset_seconds = 0.1;
        s.gestures.closure_toggles_pause = false;
        s.gaze_filter = {0.4, 7.25, 2.0};
        s.app.start_active = true;
        s.app.stale_input_seconds = 0.75;
        const auto loaded = parse_settings(serialize_settings(s));
        require(loaded.warnings.empty(), loaded.warnings.empty() ? "" : loaded.warnings.front());
        const Settings& r = loaded.settings;
        require_region(r.mapping.hand_region, s.mapping.hand_region, 0.0);
        require_region(r.mapping.gaze_region, s.mapping.gaze_region, 0.0);
        require(r.mapping.cursor_source == CursorSourceMode::GazeOnly, "cursor_source");
        require(r.mapping.screen_target == ScreenTarget::VirtualDesktop, "screen_target");
        require(r.stabilization.drag_threshold_px == 42.5, "drag threshold");
        require(r.gestures.fist_action == FistAction::MiddleClick, "fist action");
        require(r.gestures.wink_onset_seconds == 0.1, "wink onset (exact round trip)");
        require(!r.gestures.closure_toggles_pause, "closure toggle");
        require(r.gaze_filter.min_cutoff_hz == 0.4 && r.gaze_filter.beta == 7.25 &&
                    r.gaze_filter.derivative_cutoff_hz == 2.0,
                "gaze filter");
        require(r.app.start_active && r.app.stale_input_seconds == 0.75, "app");
    });
    run_test("settings_bad_values_keep_defaults", [] {
        const auto loaded = parse_settings(
            "[gestures]\n"
            "pinch_onset_seconds = -1\n"
            "fist_action = kick\n"
            "wink_onset_seconds = soon\n"
            "unknown_key = 3\n"
            "[mapping]\n"
            "hand_region = 0.1 0.2 0.3\n"
            "gaze_region = 0.5 0.1 0.505 0.9\n"
            "no equals sign\n"
            "[broken\n"
            "[app]\n"
            "stale_input_seconds = 0\n");
        require(loaded.warnings.size() == 9, "expected 9 warnings, got " +
                                                 std::to_string(loaded.warnings.size()));
        const Settings defaults;
        require(loaded.settings.gestures.pinch_onset_seconds == defaults.gestures.pinch_onset_seconds,
                "negative time accepted");
        require(loaded.settings.gestures.fist_action == defaults.gestures.fist_action, "bad enum");
        require_region(loaded.settings.mapping.hand_region, defaults.mapping.hand_region, 0.0);
        require_region(loaded.settings.mapping.gaze_region, defaults.mapping.gaze_region, 0.0);
        require(loaded.settings.app.stale_input_seconds == defaults.app.stale_input_seconds, "stale range");
        require(validate(loaded.settings).empty(), "parsed settings invalid");
    });
    run_test("settings_comments_and_whitespace", [] {
        const auto loaded = parse_settings(
            "# comment\r\n  [ gestures ]  ; trailing\r\n  pinch_onset_seconds=0.1 # note\r\n"
            "closure_toggles_pause = off\r\n");
        require(loaded.warnings.empty(), loaded.warnings.empty() ? "" : loaded.warnings.front());
        require(loaded.settings.gestures.pinch_onset_seconds == 0.1, "value");
        require(!loaded.settings.gestures.closure_toggles_pause, "bool alias");
    });
    run_test("settings_file_round_trip", [] {
        const auto dir = std::filesystem::temp_directory_path() / "osi_settings_test";
        std::filesystem::remove_all(dir);
        const auto path = dir / "nested" / "os_integration.ini";
        const auto missing = load_settings_file(path);
        require(missing.warnings.empty(), "missing file warned");
        Settings s;
        s.mapping.gaze_region = {0.62, 0.3, 0.38, 0.7};
        std::string error;
        require(save_settings_file(path, s, &error), "save failed: " + error);
        require(save_settings_file(path, s, &error), "overwrite failed: " + error);
        const auto loaded = load_settings_file(path);
        require(loaded.warnings.empty(), "reload warned");
        require_region(loaded.settings.mapping.gaze_region, s.mapping.gaze_region, 0.0);
        Settings bad;
        bad.app.stale_input_seconds = -1.0;
        require(!save_settings_file(path, bad, &error) && !error.empty(), "saved invalid settings");
        std::filesystem::remove_all(dir);
    });
    run_test("settings_validate_programmatic_values", [] {
        Settings s;
        s.app.stale_input_seconds = 0.0;
        s.hand_filter.min_cutoff_hz = 0.0;
        s.stabilization.anchor_lookback_seconds = std::numeric_limits<double>::quiet_NaN();
        require(validate(s).size() == 3, "expected three problems");
    });
    run_test("smoothing_presets", [] {
        Settings s;
        require(detect_smoothing_preset(s) == SmoothingPreset::Balanced, "defaults are Balanced");
        apply_smoothing_preset(s, SmoothingPreset::Smooth);
        close_to(s.hand_filter.min_cutoff_hz, 0.6, 1e-12, "smooth hand");
        close_to(s.gaze_filter.min_cutoff_hz, 0.4, 1e-12, "smooth gaze");
        require(detect_smoothing_preset(s) == SmoothingPreset::Smooth, "detect smooth");
        s.hand_filter.min_cutoff_hz = 3.3;
        require(!detect_smoothing_preset(s), "custom values matched a preset");
    });

    // ================================================================ calibration
    run_test("calibration_recovers_region", [] {
        const InputRegion truth{0.3, 0.25, 0.7, 0.65};
        const auto result = fit_input_region(synthetic_observations(truth, 7, 0.005));
        require(result.fit.has_value(), result.error);
        require_region(result.fit->region, truth, 0.01);
        require(result.fit->rms_error < 0.03, "rms too high");
    });
    run_test("calibration_detects_inverted_axis", [] {
        const InputRegion truth{0.62, 0.3, 0.38, 0.7};
        const auto result = fit_input_region(synthetic_observations(truth, 11, 0.003));
        require(result.fit.has_value(), result.error);
        require_region(result.fit->region, truth, 0.01);
    });
    run_test("calibration_ignores_outliers", [] {
        const InputRegion truth{0.2, 0.2, 0.8, 0.8};
        const auto result = fit_input_region(synthetic_observations(truth, 3, 0.002, 4));
        require(result.fit.has_value(), result.error);
        require_region(result.fit->region, truth, 0.01);
    });
    run_test("calibration_reports_problems", [] {
        auto few = synthetic_observations({0.2, 0.2, 0.8, 0.8}, 1, 0.0);
        few[2].samples.resize(2);
        require(!fit_input_region(few).fit && few.size() == 5, "too few samples accepted");

        auto still = synthetic_observations({0.5, 0.2, 0.505, 0.8}, 1, 0.0);
        const auto flat = fit_input_region(still);
        require(!flat.fit && flat.error.find("left-right") != std::string::npos, "no movement accepted");

        auto shuffled = synthetic_observations({0.2, 0.2, 0.8, 0.8}, 1, 0.0);
        std::swap(shuffled[1].samples, shuffled[3].samples);
        std::swap(shuffled[0].samples, shuffled[2].samples);
        require(!fit_input_region(shuffled).fit, "inconsistent targets accepted");

        shuffled.resize(2);
        require(!fit_input_region(shuffled).fit, "two targets accepted");
    });
    run_test("eye_recorder_runs", [] {
        EyeStateRecorder recorder(0.05);
        const EyeState stream[] = {EyeState::Neutral, EyeState::LeftWink, EyeState::LeftWink,
                                   EyeState::Neutral, EyeState::LeftWink, EyeState::LeftWink,
                                   EyeState::Neutral, EyeState::Neutral, EyeState::RightWink,
                                   EyeState::Calibrating};
        double t = 0.0;
        for (EyeState state : stream) {
            recorder.add(state, t);
            t += kDt;
        }
        const auto runs = recorder.runs();
        require(runs.size() == 2, "expected 2 runs, got " + std::to_string(runs.size()));
        require(runs[0].state == EyeState::LeftWink, "first run state");
        close_to(runs[0].duration(), 4 * kDt, 1e-12, "glitch merged into one run");
        require(runs[1].state == EyeState::RightWink && runs[1].duration() == 0.0, "second run");
    });
    run_test("eye_timing_recommendation", [] {
        const std::vector<StateRun> natural{{EyeState::LeftWink, 0.0, 0.08},
                                            {EyeState::SustainedClosure, 1.0, 1.2}};
        const std::vector<StateRun> left{{EyeState::LeftWink, 0.0, 0.5},
                                         {EyeState::LeftWink, 1.0, 1.45},
                                         {EyeState::LeftWink, 2.0, 2.55},
                                         {EyeState::RightWink, 3.0, 3.9}};
        const std::vector<StateRun> right{{EyeState::RightWink, 0.0, 0.4},
                                          {EyeState::RightWink, 1.0, 1.5}};
        const auto result = recommend_eye_timing(natural, left, right, GestureSettings{});
        require(result.recommendation.has_value(), result.error);
        close_to(result.recommendation->wink_onset_seconds, 0.2075, 1e-9, "wink onset");
        close_to(result.recommendation->closure_onset_seconds, 1.0, 1e-9, "closure onset");
        require(!result.recommendation->summary.empty(), "summary");
    });
    run_test("eye_timing_rejects_unusable_recordings", [] {
        const std::vector<StateRun> natural{{EyeState::RightWink, 0.0, 0.4}};
        const std::vector<StateRun> winks{{EyeState::LeftWink, 0.0, 0.45}};
        const std::vector<StateRun> right{{EyeState::RightWink, 0.0, 0.45}};
        require(!recommend_eye_timing(natural, winks, right, GestureSettings{}).recommendation,
                "overlapping blink and wink durations accepted");
        require(!recommend_eye_timing({}, {}, right, GestureSettings{}).recommendation,
                "missing left winks accepted");
    });

    // ================================================================ physical guard
    run_test("physical_input_guard", [] {
        PhysicalInputGuard guard(1.5);
        require(!guard.yielding(10.0), "yielding before any input");
        guard.on_physical_input(10.0);
        require(guard.yielding(10.0) && guard.yielding(11.4), "not yielding after input");
        require(!guard.yielding(11.6), "still yielding after the timeout");
        require(!guard.yielding(9.0), "yielding for a time before the input");
    });

#if defined(OSI_HAVE_WIN32)
    // ================================================================ win32 (no input is sent)
    run_test("win32_input_structures", [] {
        const INPUT move = win32::make_move_input({1234, 65535});
        require(move.type == INPUT_MOUSE && move.mi.dx == 1234 && move.mi.dy == 65535, "move coords");
        require(move.mi.dwFlags == (MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK),
                "move flags");
        require(move.mi.dwExtraInfo == win32::kInjectedInputTag, "move tag");
        const struct {
            MouseButton button;
            ButtonAction action;
            DWORD flag;
        } cases[] = {{MouseButton::Left, ButtonAction::Down, MOUSEEVENTF_LEFTDOWN},
                     {MouseButton::Left, ButtonAction::Up, MOUSEEVENTF_LEFTUP},
                     {MouseButton::Right, ButtonAction::Down, MOUSEEVENTF_RIGHTDOWN},
                     {MouseButton::Right, ButtonAction::Up, MOUSEEVENTF_RIGHTUP},
                     {MouseButton::Middle, ButtonAction::Down, MOUSEEVENTF_MIDDLEDOWN},
                     {MouseButton::Middle, ButtonAction::Up, MOUSEEVENTF_MIDDLEUP}};
        for (const auto& c : cases) {
            const INPUT input = win32::make_button_input(c.button, c.action);
            require(input.mi.dwFlags == c.flag && input.mi.dwExtraInfo == win32::kInjectedInputTag,
                    "button flags");
        }
    });
    run_test("win32_display_geometry", [] {
        win32::enable_per_monitor_dpi_awareness();
        const auto monitors = win32::enumerate_monitors();
        if (monitors.empty()) {
            std::cout << "SKIP no interactive display\n";
            return;
        }
        const ScreenRect desk = win32::virtual_desktop_rect();
        const ScreenRect primary = win32::primary_monitor_rect();
        require(is_valid_rect(desk) && is_valid_rect(primary), "empty rectangles");
        require(contains(desk, {primary.left, primary.top}) &&
                    contains(desk, {primary.left + primary.width - 1, primary.top + primary.height - 1}),
                "primary monitor outside the virtual desktop");
        require(std::count_if(monitors.begin(), monitors.end(),
                              [](const win32::MonitorInfo& m) { return m.primary; }) == 1,
                "expected exactly one primary monitor");
        ScreenMapper mapper(win32::query_screen_geometry(ScreenTarget::PrimaryMonitor));
        std::cout << "DISPLAY monitors=" << monitors.size() << " desktop=" << desk.width << 'x'
                  << desk.height << " primary=" << primary.width << 'x' << primary.height << '\n';
    });
#endif

    std::cout << "SUMMARY passed=" << passes << " failed=" << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
