// Tests for the C interface used by the HandEye tester. Record mode only:
// these tests never move or click the real mouse.

#include "osi_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int passes = 0;
int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Test>
void run_test(const char* name, Test&& test) {
    try {
        test();
        ++passes;
        std::printf("PASS %s\n", name);
    } catch (const std::exception& error) {
        ++failures;
        std::printf("FAIL %s: %s\n", name, error.what());
    }
}

constexpr double kDt = 1.0 / 32.0;

osi_tracker hand(double x, double y, int state, int tracking = OSI_TRACKED) {
    return osi_tracker{1, x, y, state, tracking};
}

osi_tracker none() {
    return osi_tracker{0, 0.0, 0.0, 0, OSI_LOST};
}

std::string default_ini() {
    const int32_t size = osi_default_settings_ini(nullptr, 0);
    std::string text(static_cast<std::size_t>(size) + 1, '\0');
    osi_default_settings_ini(text.data(), size + 1);
    text.resize(static_cast<std::size_t>(size));
    return text;
}

struct Controller {
    explicit Controller(const char* ini = nullptr) {
        char error[256] = {};
        handle = osi_create(ini, 0, OSI_OUTPUT_RECORD, error, sizeof(error));
        require(handle != nullptr, std::string("create failed: ") + error);
    }
    ~Controller() { osi_destroy(handle); }
    void* handle;
};

} // namespace

int main() {
    run_test("version_and_default_settings", [] {
        require(osi_version() == OSI_CAPI_VERSION, "version mismatch");
        const std::string ini = default_ini();
        require(ini.find("[gestures]") != std::string::npos && ini.find("pinch_onset_seconds") != std::string::npos,
                "default INI incomplete");
        char tiny[4];
        require(osi_default_settings_ini(tiny, sizeof(tiny)) == static_cast<int32_t>(ini.size()),
                "small buffer must report the needed size");
    });
    run_test("create_rejects_bad_settings", [] {
        char error[256] = {};
        void* h = osi_create("[gestures]\npinch_onset_seconds = soon\n", 0, OSI_OUTPUT_RECORD, error, sizeof(error));
        require(h == nullptr && std::strlen(error) > 0, "bad settings accepted");
        // A key from another version is skipped, so an older settings file still loads.
        h = osi_create("[nope]\nx = 1\n[app]\nudp_port = 50505\n", 0, OSI_OUTPUT_RECORD, error, sizeof(error));
        require(h != nullptr, std::string("unknown key rejected: ") + error);
        osi_destroy(h);
    });
    run_test("starts_paused_then_moves_and_records", [] {
        Controller c;
        osi_report report{};
        const osi_tracker h = hand(0.5, 0.5, 0);
        const osi_tracker g = none();
        osi_update(c.handle, 0.0, &h, &g, &report);
        require(report.processed == 1 && report.active == 0 && report.has_cursor == 1, "paused report");
        osi_event events[16];
        require(osi_drain_events(c.handle, events, 16) == 0, "paused controller produced events");
        osi_set_active(c.handle, 1);
        osi_update(c.handle, kDt, &h, &g, &report);
        require(report.moved == 1 && report.source == 1, "no hand move");
        const int32_t n = osi_drain_events(c.handle, events, 16);
        require(n == 1 && events[0].kind == OSI_EVENT_MOVE, "move not recorded");
        require(osi_drain_events(c.handle, events, 16) == 0, "drain did not clear");
    });
    run_test("pinch_reports_actions_and_confirmed_state", [] {
        Controller c;
        osi_set_active(c.handle, 1);
        osi_report report{};
        const osi_tracker g = none();
        bool down = false;
        for (int i = 0; i < 8; ++i) {
            const osi_tracker h = hand(0.5, 0.5, 1);
            osi_update(c.handle, i * kDt, &h, &g, &report);
            for (int a = 0; a < report.action_count; ++a) {
                down |= report.action_type[a] == OSI_ACTION_BUTTON_DOWN && report.action_button[a] == 0;
            }
        }
        require(down && report.confirmed_hand == 1, "pinch not reported");
        require(osi_release_all(c.handle) == 1, "release failed in record mode");
    });
    run_test("apply_settings_live", [] {
        Controller c;
        char error[256] = {};
        require(osi_apply_settings_ini(c.handle, "[gestures]\nfist_action = none\n", error, sizeof(error)) == 1,
                std::string("valid settings refused: ") + error);
        require(osi_apply_settings_ini(c.handle, "[mapping]\nhand_region = 1 2\n", error, sizeof(error)) == 0 &&
                    std::strlen(error) > 0,
                "invalid settings accepted");
        osi_set_active(c.handle, 1);
        osi_report report{};
        const osi_tracker g = none();
        int clicks = 0;
        for (int i = 0; i < 30; ++i) {
            const osi_tracker h = hand(0.5, 0.5, 2);
            osi_update(c.handle, i * kDt, &h, &g, &report);
            clicks += report.action_count;
        }
        require(clicks == 0, "fist_action = none still clicked");
    });
    run_test("geometry_and_null_safety", [] {
        Controller c;
        osi_rect target{}, desktop{};
        require(osi_screen_geometry(c.handle, &target, &desktop) == 1 && target.width > 0 && desktop.width > 0,
                "geometry");
        osi_report report{};
        require(osi_update(nullptr, 0.0, nullptr, nullptr, &report) == 0, "null handle update");
        osi_set_active(nullptr, 1);
        osi_destroy(nullptr);
        require(osi_drain_events(nullptr, nullptr, 0) == 0, "null drain");
    });
    run_test("smoothing_filter", [] {
        void* f = osi_filter_create(1.2, 4.0, 1.0);
        if (f == nullptr) {
            std::printf("SKIP built without dsp_filter\n");
            return;
        }
        double x = 0.2, y = 0.3;
        require(osi_filter_apply(f, &x, &y, 0.0, 1) == 1 && x == 0.2 && y == 0.3, "first sample passes through");
        x = 0.8;
        y = 0.3;
        require(osi_filter_apply(f, &x, &y, 0.02, 1) == 1 && x > 0.2 && x < 0.8, "second sample not smoothed");
        require(osi_filter_apply(f, &x, &y, 0.04, 0) == 0, "lost target produced output");
        osi_filter_destroy(f);
    });

    run_test("smoothing_filter_rejects_non_finite", [] {
        void* f = osi_filter_create(1.2, 4.0, 1.0);
        if (f == nullptr) {
            std::printf("SKIP built without dsp_filter\n");
            return;
        }
        const double inf = std::numeric_limits<double>::infinity();
        const double bad[] = {inf, -inf, std::numeric_limits<double>::quiet_NaN()};
        double t = 0.0;
        for (const double value : bad) {
            double x = value;
            double y = 0.4;
            require(osi_filter_update(f, &x, &y, t += kDt) == OSI_FILTER_INVALID, "non-finite x accepted");
            x = 0.4;
            y = value;
            require(osi_filter_update(f, &x, &y, t += kDt) == OSI_FILTER_INVALID, "non-finite y accepted");
            x = value;
            y = 0.4;
            require(osi_filter_apply(f, &x, &y, t += kDt, 1) == 0, "osi_filter_apply accepted non-finite x");
        }
        osi_filter_destroy(f);
    });
    run_test("smoothing_filter_reports_rejected_timestamp", [] {
        void* f = osi_filter_create(1.2, 4.0, 1.0);
        if (f == nullptr) {
            std::printf("SKIP built without dsp_filter\n");
            return;
        }
        double x = 0.2, y = 0.3;
        require(osi_filter_update(f, &x, &y, 1.0) == OSI_FILTER_OK, "first sample refused");
        const double stale_times[] = {1.0, 0.5};
        for (const double t : stale_times) {
            x = 0.9;
            y = 0.8;
            require(osi_filter_update(f, &x, &y, t) == OSI_FILTER_REJECTED, "stale timestamp not rejected");
            require(x == 0.9 && y == 0.8, "rejected update changed its arguments");
        }
        require(osi_filter_update(f, &x, &y, 1.02) == OSI_FILTER_OK && x < 0.9, "history lost after a rejection");
        osi_filter_reset(f);
        x = 0.7;
        y = 0.6;
        require(osi_filter_update(f, &x, &y, 1.04) == OSI_FILTER_OK && x == 0.7 && y == 0.6, "reset kept history");
        osi_filter_destroy(f);
    });
    run_test("clicks_disabled_keeps_gaze_lock_and_toggle_pause", [] {
        std::string ini = default_ini();
        ini += "\n[mapping]\ncursor_source = gaze_only\n[gestures]\nclosure_toggles_pause = true\n";
        Controller c(ini.c_str());
        osi_set_active(c.handle, 1);
        osi_set_clicks_enabled(c.handle, 0);

        osi_report report{};
        const osi_tracker h = none();
        double t = 0.0;
        for (int i = 0; i < 6; ++i, t += kDt) {
            const osi_tracker g{1, 0.50, 0.77, 0, OSI_TRACKED};
            osi_update(c.handle, t, &h, &g, &report);
        }
        const int32_t base_x = report.cursor_x;
        const int32_t base_y = report.cursor_y;

        // Left wink with clicks disabled: cursor must stay locked (not jump to 0.90,0.05) and emit zero clicks.
        for (int i = 0; i < 12; ++i, t += kDt) {
            const osi_tracker g{1, 0.90, 0.05, 1, OSI_TRACKED};
            osi_update(c.handle, t, &h, &g, &report);
            require(report.action_count == 0, "clicks-disabled controller emitted a click action");
            require(report.cursor_x == base_x && report.cursor_y == base_y,
                    "cursor moved during wink with clicks disabled");
        }
        for (int i = 0; i < 3; ++i, t += kDt) {
            const osi_tracker g{1, 0.50, 0.77, 0, OSI_TRACKED};
            osi_update(c.handle, t, &h, &g, &report);
        }

        // Eye closure pause toggle must remain active even with clicks disabled.
        for (int i = 0; i < 40; ++i, t += kDt) {
            const osi_tracker g{1, 0.50, 0.77, 3, OSI_TRACKED};
            osi_update(c.handle, t, &h, &g, &report);
        }
        require(report.active == 0, "TogglePause did not fire when clicks disabled");
    });

    std::printf("SUMMARY passed=%d failed=%d\n", passes, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
