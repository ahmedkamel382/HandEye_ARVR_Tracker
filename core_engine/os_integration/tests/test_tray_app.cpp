// Deterministic TrayApp tests: a test sink, no window, no icon, no hook, no real mouse.

#include "app_log.hpp"
#include "tray_app.hpp"

#include "os_integration/input_sink.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace osi::app {

struct TrayAppTestAccess {
    explicit TrayAppTestAccess(TrayApp& app) : app_(app) {
        auto owned = std::make_unique<RecordingInputSink>();
        sink = owned.get();
        app_.sink_ = std::move(owned);
        app_.controller_ = std::make_unique<OsController>(
            app_.settings_, ScreenGeometry{{0, 0, 1920, 1080}, {0, 0, 1920, 1080}}, *app_.sink_);
        app_.controller_->set_active(true);
        app_.ready_ = true;
    }
    void frames(HandGesture gesture, int count) {
        for (int i = 0; i < count; ++i) {
            FramePayload frame;
            frame.timestamp_seconds = t_;
            frame.hand = {NormalizedPoint{0.5, 0.5}, static_cast<std::int32_t>(gesture)};
            app_.submit_frame(frame);
            t_ += 1.0 / 32.0;
        }
    }
    void stall() {
        app_.last_frame_at_ = 0.0; // long ago on the steady clock
        app_.stale_reported_ = false;
        app_.check_stale_input();
    }
    void timer_retry() { app_.retry_stuck_buttons(); }
    void use_settings_file(const std::filesystem::path& path) {
        app_.options_.settings_path = path;
        app_.settings_ = load_settings_file(path).settings;
    }
    void menu_smoothing() {
        app_.change_settings([](Settings& s) { apply_smoothing_preset(s, SmoothingPreset::Smooth); }, "smoothing");
    }
    void reload() { app_.reload_settings(); }
    double wink() const { return app_.settings_.gestures.wink_onset_seconds; }
    bool event_posted() const { return app_.release_event_posted_.load(); }
    bool active() { return app_.controller_->active(); }
    void pause() { app_.controller_->set_active(false); }
    void eyes(EyeState eye, int count) {
        for (int i = 0; i < count; ++i) {
            FramePayload frame;
            frame.timestamp_seconds = t_;
            frame.gaze = {NormalizedPoint{0.5, 0.5}, static_cast<std::int32_t>(eye)};
            app_.submit_frame(frame);
            t_ += 1.0 / 32.0;
        }
    }
    bool pending() const { return app_.controller_->release_pending(); }

    RecordingInputSink* sink = nullptr;

private:
    TrayApp& app_;
    double t_ = 0.0;
};

} // namespace osi::app

namespace {

using namespace osi;
using namespace osi::app;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const std::filesystem::path& test_log() {
    static const std::filesystem::path path = [] {
        auto file = std::filesystem::temp_directory_path() /
                    ("handeye_tray_test_" + std::to_string(GetCurrentProcessId()) + ".log");
        AppLog::open(file);
        return file;
    }();
    return path;
}

std::string log_text() {
    std::ifstream in(test_log(), std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// What one stale-input check wrote to the log.
std::string stall_and_read(TrayAppTestAccess& access) {
    const std::size_t before = log_text().size();
    access.stall();
    return log_text().substr(before);
}

std::filesystem::path settings_test_file(const char* name) {
    return std::filesystem::temp_directory_path() /
           ("handeye_tray_" + std::to_string(GetCurrentProcessId()) + "_" + name + ".ini");
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

void remove_with_backup(const std::filesystem::path& path) {
    auto backup = path;
    backup += ".bak";
    std::filesystem::remove(path);
    std::filesystem::remove(backup);
}

} // namespace

namespace osi_tray_tests {

void menu_change_keeps_external_edits() {
    test_log();
    const auto path = settings_test_file("merge");
    write_text_file(path, serialize_settings(Settings{}));
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.use_settings_file(path);
    Settings external;
    external.gestures.wink_onset_seconds = 0.41; // the tester saves
    write_text_file(path, serialize_settings(external));
    access.menu_smoothing();
    const auto saved = load_settings_file(path).settings;
    remove_with_backup(path);
    require(saved.gestures.wink_onset_seconds == 0.41, "menu change overwrote the external edit");
    require(detect_smoothing_preset(saved) == SmoothingPreset::Smooth, "menu change not saved");
}

void invalid_file_is_never_overwritten() {
    test_log();
    const auto path = settings_test_file("invalid");
    const std::string broken = "[mapping]\ngaze_region = 0.71 0.33 0.29 0.08\n[hand_filter]\nbeta = oops\n";
    write_text_file(path, broken);
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.use_settings_file(path);
    access.menu_smoothing();
    const std::string after = read_text_file(path);
    remove_with_backup(path);
    require(after == broken, "a file with errors was overwritten");
}

void invalid_reload_keeps_running_settings() {
    test_log();
    const auto path = settings_test_file("reload");
    Settings good;
    good.gestures.wink_onset_seconds = 0.41;
    write_text_file(path, serialize_settings(good));
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.use_settings_file(path);
    const std::string broken = "[gestures]\nwink_onset_seconds = oops\n";
    write_text_file(path, broken);
    access.reload();
    const std::string after = read_text_file(path);
    remove_with_backup(path);
    require(access.wink() == 0.41 && after == broken, "a bad reload replaced good settings");
}

void stale_log_reports_refused_release() {
    test_log();
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.frames(HandGesture::Pinch, 6);
    access.sink->fail_releases = true;
    const std::string text = stall_and_read(access);
    require(access.pending(), "setup: release not pending");
    require(text.find("refused") != std::string::npos, "refusal not logged: " + text);
    require(text.find("was released") == std::string::npos, "claimed a release that failed: " + text);
    access.sink->fail_releases = false;
    access.timer_retry();
    require(!access.pending(), "timer did not retry");
}

void stale_log_reports_successful_release() {
    test_log();
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.frames(HandGesture::Pinch, 6);
    const std::string text = stall_and_read(access);
    require(text.find("held button was released") != std::string::npos, "release not logged: " + text);
}

void stale_log_reports_nothing_held() {
    test_log();
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.frames(HandGesture::Neutral, 2);
    const std::string text = stall_and_read(access);
    require(text.find("no button was held") != std::string::npos, "wrong message: " + text);
}

void release_event_rearms_for_each_stuck_episode() {
    test_log();
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    for (int episode = 0; episode < 2; ++episode) {
        access.frames(HandGesture::Pinch, 8);
        require(!access.event_posted(), "flag set before anything was stuck");
        access.sink->fail_releases = true;
        access.frames(HandGesture::Neutral, 8);
        require(access.pending() && access.event_posted(), "stuck episode not flagged");
        access.timer_retry();
        require(access.event_posted(), "flag cleared while still stuck");
        access.sink->fail_releases = false;
        access.timer_retry();
        require(!access.pending() && !access.event_posted(), "flag not cleared after the release");
    }
}

void eye_closure_never_resumes_the_tray() {
    test_log();
    TrayApp app(GetModuleHandleW(nullptr), TrayOptions{});
    TrayAppTestAccess access(app);
    access.pause();
    for (int round = 0; round < 3; ++round) {
        access.eyes(EyeState::SustainedClosure, 45);
        access.eyes(EyeState::Neutral, 10);
    }
    require(!access.active(), "eye closure resumed the tray");
}

} // namespace osi_tray_tests
