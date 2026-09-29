#pragma once

#include "calibration_wizard.hpp"

#include "os_integration/os_controller.hpp"
#include "os_integration/physical_input_guard.hpp"
#include "os_integration/settings.hpp"

#include <windows.h>
#include <shellapi.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace osi::app {

struct TrayOptions {
    bool dry_run = false;      // log mouse events instead of sending them
    bool start_active = false; // overrides [app] start_active
    std::filesystem::path settings_path;
    std::filesystem::path log_path;
    std::optional<WizardKind> calibrate_on_start;
    // Called on the UI thread after the user changes settings (menu, wizard,
    // reload), so the host can rebuild its DSP filters from hand_filter /
    // gaze_filter.
    std::function<void(const Settings&)> on_settings_changed;
};

// System-tray host for the OS layer: drives OsController with the frames
// the core engine submits, and offers pause/resume, settings and the
// calibration wizard from the tray menu.
//
// Threads: run() owns every window and blocks in the message loop on its
// thread. The core engine calls submit_frame() from its own loop thread.
// `mutex_` guards settings_ and controller_.
class TrayApp {
public:
    TrayApp(HINSTANCE instance, TrayOptions options);
    ~TrayApp();
    TrayApp(const TrayApp&) = delete;
    TrayApp& operator=(const TrayApp&) = delete;

    // Blocks until the user chooses Exit or request_exit() is called.
    int run();

    // Phase 4 hand-off from the core loop: one stabilized frame per new
    // observation. Safe to call from any thread; ignored until run() is ready.
    void submit_frame(const FramePayload& frame) noexcept;

    // Asks run() to return. Safe to call from any thread.
    void request_exit() noexcept;

private:
    enum class Status : int { Waiting, Paused, Active, Yielding, Calibrating, Count };

    static LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
    static LRESULT CALLBACK mouse_hook(int code, WPARAM wparam, LPARAM lparam);
    LRESULT handle(UINT message, WPARAM wparam, LPARAM lparam);

    bool initialize();
    void shutdown();

    // UI thread.
    void show_menu();
    void on_command(UINT id);
    void set_active(bool active, const char* reason);
    void refresh_status();
    void check_stale_input();
    void post_event(WPARAM event) noexcept;
    Status current_status();
    void notify(const std::wstring& title, const std::wstring& text, bool warning = false);
    bool add_tray_icon();
    void change_settings(const std::function<void(Settings&)>& edit, const char* what);
    void reload_settings();
    void start_wizard(WizardKind kind);
    void on_wizard_done();

    HINSTANCE instance_;
    TrayOptions options_;
    HWND hwnd_ = nullptr;
    NOTIFYICONDATAW icon_data_{};
    std::array<HICON, static_cast<std::size_t>(Status::Count)> icons_{};
    Status shown_status_ = Status::Count;
    std::wstring shown_tip_;
    UINT taskbar_created_ = 0;
    HHOOK hook_ = nullptr;
    bool hotkey_registered_ = false;
    bool shut_down_ = false;

    std::mutex mutex_;
    Settings settings_;
    std::unique_ptr<IInputSink> sink_;
    std::unique_ptr<OsController> controller_;
    std::optional<double> last_frame_timestamp_;

    std::atomic<bool> ready_{false};
    std::atomic<HWND> post_target_{nullptr}; // hwnd_ for other threads; null once shut down
    std::atomic<bool> stale_reported_{false};
    PhysicalInputGuard guard_{1.5};
    SampleInbox inbox_;
    std::atomic<bool> wizard_running_{false};
    std::atomic<double> last_frame_at_{-1.0}; // steady seconds; -1 = never
    std::atomic<bool> first_frame_seen_{false};
    std::atomic<double> last_block_notice_{-100.0};
    std::atomic<bool> unknown_state_logged_{false};

    std::unique_ptr<CalibrationWizard> wizard_;
    WizardKind wizard_kind_ = WizardKind::HandRange;
    std::optional<CalibrationWizard::Outcome> wizard_outcome_;
};

} // namespace osi::app
