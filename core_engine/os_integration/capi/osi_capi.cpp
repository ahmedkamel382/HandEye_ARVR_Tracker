#include "osi_capi.h"

#include "os_integration/os_controller.hpp"
#include "os_integration/settings.hpp"
#include "os_integration/win32_input.hpp"

#if defined(OSI_CAPI_WITH_DSP)
#include "one_euro_filter.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <optional>
#include <string>

namespace {

using namespace osi;

constexpr std::size_t kMaxRecordedEvents = 4096;

void write_text(const std::string& text, char* buffer, int32_t capacity) {
    if (buffer == nullptr || capacity <= 0) {
        return;
    }
    const std::size_t count = std::min<std::size_t>(text.size(), static_cast<std::size_t>(capacity) - 1);
    std::memcpy(buffer, text.data(), count);
    buffer[count] = '\0';
}

// Records every event for the tester's log, and forwards to Windows when the
// real mouse is enabled.
class TeeSink final : public IInputSink {
public:
    explicit TeeSink(bool real_mouse) : real_mouse_(real_mouse) {}

    bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept override {
        record({OSI_EVENT_MOVE, 0, pixel.x, pixel.y});
        return !real_mouse_ || injector_.move_to(pixel, absolute);
    }
    bool button(MouseButton which, ButtonAction action) noexcept override {
        record({action == ButtonAction::Down ? OSI_EVENT_DOWN : OSI_EVENT_UP, static_cast<int32_t>(which), 0, 0});
        return !real_mouse_ || injector_.button(which, action);
    }
    ClickResult click(MouseButton which) noexcept override {
        record({OSI_EVENT_CLICK, static_cast<int32_t>(which), 0, 0});
        return real_mouse_ ? injector_.click(which) : ClickResult::Clicked;
    }

    int32_t drain(osi_event* out, int32_t capacity) noexcept {
        int32_t count = 0;
        while (count < capacity && !events_.empty()) {
            out[count++] = events_.front();
            events_.pop_front();
        }
        return count;
    }

private:
    void record(const osi_event& event) noexcept {
        try {
            if (events_.size() == kMaxRecordedEvents) {
                events_.pop_front();
            }
            events_.push_back(event);
        } catch (...) {
        }
    }

    bool real_mouse_;
    win32::InputInjector injector_;
    std::deque<osi_event> events_;
};

struct Handle {
    Settings settings;
    std::unique_ptr<TeeSink> sink;
    std::unique_ptr<OsController> controller;
};

// Parse for the tester: a malformed line or a bad value is an error (so the
// tester can show it); a key this version does not know is skipped, so a
// settings file from another version still loads.
bool parse_strict(const char* text, Settings& out, std::string& error) {
    if (text == nullptr || *text == '\0') {
        out = Settings{};
        return true;
    }
    SettingsLoadResult loaded = parse_settings(text);
    if (!loaded.error.empty()) {
        error = loaded.error;
        return false;
    }
    out = loaded.settings;
    return true;
}

ScreenTarget target_from(int32_t screen_target) noexcept {
    return screen_target == 1 ? ScreenTarget::VirtualDesktop : ScreenTarget::PrimaryMonitor;
}

TrackerSample to_sample(const osi_tracker* tracker) noexcept {
    TrackerSample sample;
    if (tracker == nullptr) {
        sample.tracking = TrackerState::Lost;
        return sample;
    }
    const bool has_valid_point = tracker->has_point != 0
                                 && std::isfinite(tracker->x)
                                 && std::isfinite(tracker->y);
    if (has_valid_point) {
        sample.point = NormalizedPoint{tracker->x, tracker->y};
    }
    sample.state = tracker->state;
    sample.tracking = (tracker->has_point != 0 && !has_valid_point) ? TrackerState::Lost
                      : tracker->tracking == OSI_HELD             ? TrackerState::Held
                      : tracker->tracking == OSI_LOST             ? TrackerState::Lost
                                                                  : TrackerState::Tracked;
    return sample;
}

// Headless machines (CI runners, remote sessions without a display) can
// report an empty screen; fall back to 1920 x 1080 so the controller still works.
ScreenGeometry geometry_for(ScreenTarget target) noexcept {
    ScreenGeometry geometry = win32::query_screen_geometry(target);
    if (!is_valid_rect(geometry.virtual_desktop) || !is_valid_rect(geometry.target)) {
        geometry = {{0, 0, 1920, 1080}, {0, 0, 1920, 1080}};
    }
    return geometry;
}

osi_rect to_rect(const ScreenRect& r) noexcept {
    return {r.left, r.top, r.width, r.height};
}

#if defined(OSI_CAPI_WITH_DSP)
struct Filter {
    explicit Filter(const dsp::FilterConfig& config) : filter(config, config) {}
    dsp::TrackingFilter2D filter;
};
#endif

} // namespace

extern "C" {

int32_t osi_version(void) {
    return OSI_CAPI_VERSION;
}

int32_t osi_default_settings_ini(char* buffer, int32_t capacity) {
    try {
        const std::string text = serialize_settings(Settings{});
        if (buffer == nullptr || capacity <= static_cast<int32_t>(text.size())) {
            return static_cast<int32_t>(text.size());
        }
        write_text(text, buffer, capacity);
        return static_cast<int32_t>(text.size());
    } catch (...) {
        return 0;
    }
}

void* osi_create(const char* settings_ini, int32_t screen_target, int32_t output_mode, char* error,
                 int32_t error_capacity) {
    try {
        auto handle = std::make_unique<Handle>();
        std::string problem;
        if (!parse_strict(settings_ini, handle->settings, problem)) {
            write_text("settings: " + problem, error, error_capacity);
            return nullptr;
        }
        handle->settings.mapping.screen_target = target_from(screen_target);
        handle->sink = std::make_unique<TeeSink>(output_mode == OSI_OUTPUT_MOUSE);
        handle->controller = std::make_unique<OsController>(
            handle->settings, geometry_for(handle->settings.mapping.screen_target), *handle->sink);
        return handle.release();
    } catch (const std::exception& e) {
        write_text(e.what(), error, error_capacity);
    } catch (...) {
        write_text("unknown error", error, error_capacity);
    }
    return nullptr;
}

void osi_destroy(void* handle) {
    delete static_cast<Handle*>(handle); // ~OsController releases held buttons
}

int32_t osi_apply_settings_ini(void* handle, const char* settings_ini, char* error, int32_t error_capacity) {
    auto* h = static_cast<Handle*>(handle);
    if (h == nullptr) {
        write_text("no controller", error, error_capacity);
        return 0;
    }
    try {
        Settings next;
        std::string problem;
        if (!parse_strict(settings_ini, next, problem)) {
            write_text(problem, error, error_capacity);
            return 0;
        }
        next.mapping.screen_target = h->settings.mapping.screen_target; // fixed per handle
        h->controller->apply_settings(next);
        h->settings = next;
        return 1;
    } catch (const std::exception& e) {
        write_text(e.what(), error, error_capacity);
    } catch (...) {
        write_text("unknown error", error, error_capacity);
    }
    return 0;
}

void osi_set_active(void* handle, int32_t active) {
    if (auto* h = static_cast<Handle*>(handle)) {
        h->controller->set_active(active != 0);
    }
}

void osi_set_inhibited(void* handle, int32_t inhibited) {
    if (auto* h = static_cast<Handle*>(handle)) {
        h->controller->set_inhibited(inhibited != 0);
    }
}

void osi_set_clicks_enabled(void* handle, int32_t enabled) {
    if (auto* h = static_cast<Handle*>(handle)) {
        h->controller->set_clicks_enabled(enabled != 0);
    }
}

int32_t osi_release_all(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    return h == nullptr || h->controller->release_all_buttons() ? 1 : 0;
}

int32_t osi_update(void* handle, double timestamp_seconds, const osi_tracker* hand, const osi_tracker* gaze,
                   osi_report* report) {
    auto* h = static_cast<Handle*>(handle);
    if (h == nullptr) {
        return 0;
    }
    FramePayload frame;
    frame.timestamp_seconds = timestamp_seconds;
    frame.hand = to_sample(hand);
    frame.gaze = to_sample(gaze);
    const FrameReport r = h->controller->update(frame);
    const int32_t processed = r.status == FrameStatus::Processed ? 1 : 0;
    if (report != nullptr) {
        osi_report out{};
        out.processed = processed;
        out.source = static_cast<int32_t>(r.source);
        out.has_cursor = r.cursor.has_value() ? 1 : 0;
        if (r.cursor) {
            out.cursor_x = r.cursor->x;
            out.cursor_y = r.cursor->y;
        }
        out.moved = r.moved ? 1 : 0;
        out.action_count = static_cast<int32_t>(std::min<std::size_t>(r.executed.size(), OSI_MAX_ACTIONS));
        for (int32_t i = 0; i < out.action_count; ++i) {
            out.action_type[i] = static_cast<int32_t>(r.executed[static_cast<std::size_t>(i)].type);
            out.action_button[i] = static_cast<int32_t>(r.executed[static_cast<std::size_t>(i)].button);
        }
        out.pause_toggled = r.pause_toggled ? 1 : 0;
        out.active = r.active ? 1 : 0;
        out.inhibited = r.inhibited ? 1 : 0;
        out.injection_failed = r.injection_failed ? 1 : 0;
        out.release_pending = r.release_pending ? 1 : 0;
        out.gaze_calibrating = r.gaze_calibrating ? 1 : 0;
        out.hand_held = r.hand_held ? 1 : 0;
        out.unknown_states = static_cast<int32_t>(r.unknown_states);
        out.confirmed_hand = static_cast<int32_t>(h->controller->confirmed_hand());
        out.confirmed_eye = static_cast<int32_t>(h->controller->confirmed_eye());
        *report = out;
    }
    return processed;
}

int32_t osi_drain_events(void* handle, osi_event* events, int32_t capacity) {
    auto* h = static_cast<Handle*>(handle);
    if (h == nullptr || events == nullptr || capacity <= 0) {
        return 0;
    }
    return h->sink->drain(events, capacity);
}

int32_t osi_screen_geometry(void* handle, osi_rect* target, osi_rect* desktop) {
    auto* h = static_cast<Handle*>(handle);
    if (h == nullptr) {
        return 0;
    }
    const ScreenGeometry& g = h->controller->mapper().geometry();
    if (target != nullptr) {
        *target = to_rect(g.target);
    }
    if (desktop != nullptr) {
        *desktop = to_rect(g.virtual_desktop);
    }
    return 1;
}

int32_t osi_cursor_position(int32_t* x, int32_t* y) {
    const auto position = win32::cursor_position();
    if (!position) {
        return 0;
    }
    if (x != nullptr) {
        *x = position->x;
    }
    if (y != nullptr) {
        *y = position->y;
    }
    return 1;
}

void* osi_filter_create(double min_cutoff_hz, double beta, double derivative_cutoff_hz) {
#if defined(OSI_CAPI_WITH_DSP)
    try {
        return new Filter(dsp::FilterConfig{min_cutoff_hz, beta, derivative_cutoff_hz});
    } catch (...) {
        return nullptr;
    }
#else
    (void)min_cutoff_hz;
    (void)beta;
    (void)derivative_cutoff_hz;
    return nullptr;
#endif
}

void osi_filter_destroy(void* filter) {
#if defined(OSI_CAPI_WITH_DSP)
    delete static_cast<Filter*>(filter);
#else
    (void)filter;
#endif
}

int32_t osi_filter_update(void* filter, double* x, double* y, double timestamp_seconds) {
#if defined(OSI_CAPI_WITH_DSP)
    auto* f = static_cast<Filter*>(filter);
    if (f == nullptr || x == nullptr || y == nullptr) {
        return OSI_FILTER_INVALID;
    }
    // Check before clamping: std::clamp would turn infinity into a screen edge.
    if (!std::isfinite(*x) || !std::isfinite(*y)) {
        f->filter.reset();
        return OSI_FILTER_INVALID;
    }
    const auto result = f->filter.update(dsp::Point2D{std::clamp(*x, 0.0, 1.0), std::clamp(*y, 0.0, 1.0)},
                                         timestamp_seconds, true);
    if (result.point) {
        *x = result.point->x;
        *y = result.point->y;
        return OSI_FILTER_OK;
    }
    return result.status == dsp::FilterStatus::RejectedTimestamp ? OSI_FILTER_REJECTED : OSI_FILTER_INVALID;
#else
    (void)filter;
    (void)x;
    (void)y;
    (void)timestamp_seconds;
    return OSI_FILTER_INVALID;
#endif
}

void osi_filter_reset(void* filter) {
#if defined(OSI_CAPI_WITH_DSP)
    if (auto* f = static_cast<Filter*>(filter)) {
        f->filter.reset();
    }
#else
    (void)filter;
#endif
}

int32_t osi_filter_apply(void* filter, double* x, double* y, double timestamp_seconds, int32_t valid) {
    if (valid == 0) {
        osi_filter_reset(filter);
        return 0;
    }
    return osi_filter_update(filter, x, y, timestamp_seconds) == OSI_FILTER_OK ? 1 : 0;
}

} // extern "C"
