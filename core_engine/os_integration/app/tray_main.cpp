// handeye_tray: Engineer 5's user-facing application.
//
//   handeye_tray.exe [--dry-run] [--active] [--simulate] [--calibrate hand|gaze|eyes]
//                    [--settings PATH] [--log PATH]
//
// --dry-run   log every mouse action instead of sending it to Windows
// --active    start with mouse control enabled (default: paused)
// --simulate  feed a synthetic hand cursor (a slow circle, never a gesture),
//             to demonstrate the app before the core engine feeds real frames
// --calibrate open the calibration wizard right away
//
// In the finished system the core engine (Engineer 1) owns a TrayApp and
// calls submit_frame() from its loop; --simulate shows exactly that call.

#include "app_log.hpp"
#include "tray_app.hpp"

#include "os_integration/win32_input.hpp"

#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <thread>

namespace {

std::filesystem::path known_folder(REFKNOWNFOLDERID id) {
    PWSTR path = nullptr;
    std::filesystem::path result;
    if (SUCCEEDED(SHGetKnownFolderPath(id, 0, nullptr, &path))) {
        result = path;
    }
    CoTaskMemFree(path);
    return result.empty() ? std::filesystem::temp_directory_path() : result;
}

// Stand-in for the core loop: 30 FPS, hand point moving on a circle.
void simulate(osi::app::TrayApp& app, const std::atomic<bool>& running) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto next = start;
    while (running) {
        const double t = std::chrono::duration<double>(Clock::now() - start).count();
        const double phase = 2.0 * 3.141592653589793 * t / 4.0;
        osi::FramePayload frame;
        frame.timestamp_seconds = t;
        frame.hand.point = osi::NormalizedPoint{0.5 + 0.25 * std::cos(phase), 0.5 + 0.25 * std::sin(phase)};
        app.submit_frame(frame);
        next += std::chrono::milliseconds(33);
        std::this_thread::sleep_until(next);
    }
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    osi::win32::enable_per_monitor_dpi_awareness();

    osi::app::TrayOptions options;
    options.settings_path = known_folder(FOLDERID_RoamingAppData) / L"HandEyeTracker" / L"os_integration.ini";
    options.log_path = known_folder(FOLDERID_LocalAppData) / L"HandEyeTracker" / L"handeye_tray.log";
    bool simulate_frames = false;

    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; argv != nullptr && i < argc; ++i) {
        const std::wstring arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == L"--dry-run") {
            options.dry_run = true;
        } else if (arg == L"--active") {
            options.start_active = true;
        } else if (arg == L"--simulate") {
            simulate_frames = true;
        } else if (arg == L"--calibrate" && has_value) {
            const std::wstring kind = argv[++i];
            options.calibrate_on_start = kind == L"gaze"   ? osi::app::WizardKind::GazeRange
                                         : kind == L"eyes" ? osi::app::WizardKind::EyeTiming
                                                           : osi::app::WizardKind::HandRange;
        } else if (arg == L"--settings" && has_value) {
            options.settings_path = argv[++i];
        } else if (arg == L"--log" && has_value) {
            options.log_path = argv[++i];
        } else {
            MessageBoxW(nullptr,
                        L"Usage: handeye_tray.exe [--dry-run] [--active] [--simulate] "
                        L"[--calibrate hand|gaze|eyes] [--settings PATH] [--log PATH]",
                        L"HandEye Tracker", MB_ICONINFORMATION);
            LocalFree(argv);
            return 2;
        }
    }
    LocalFree(argv);

    // One instance per user session: two would fight over the cursor.
    HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\HandEyeTrackerTray");
    if (single == nullptr || GetLastError() == ERROR_ALREADY_EXISTS) {
        MessageBoxW(nullptr, L"HandEye Tracker is already running (see the notification area).",
                    L"HandEye Tracker", MB_ICONINFORMATION);
        if (single != nullptr) {
            CloseHandle(single);
        }
        return 0;
    }

    int exit_code = 0;
    {
        osi::app::TrayApp app(instance, options);
        std::atomic<bool> running{true};
        std::thread feeder;
        if (simulate_frames) {
            feeder = std::thread([&app, &running] { simulate(app, running); });
        }
        exit_code = app.run();
        running = false;
        if (feeder.joinable()) {
            feeder.join();
        }
    }
    ReleaseMutex(single);
    CloseHandle(single);
    return exit_code;
}
