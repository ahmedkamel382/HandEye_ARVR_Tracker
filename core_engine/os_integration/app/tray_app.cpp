#include "tray_app.hpp"

#include "app_log.hpp"

#include "os_integration/win32_input.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace osi::app {
namespace {

constexpr wchar_t kWindowClass[] = L"HandEyeTrayWindow";
constexpr UINT kTrayMessage = WM_APP + 1;
constexpr UINT kEventMessage = WM_APP + 2;
constexpr UINT kWizardDoneMessage = WM_APP + 3;
constexpr UINT_PTR kStatusTimer = 1;
constexpr int kToggleHotkey = 1;
constexpr UINT kTrayIconId = 1;

enum Event : WPARAM { FirstFrame = 1, GesturePause, GestureResume, InjectionBlocked };

enum MenuId : UINT {
    IdStatus = 1,
    IdToggle = 10,
    IdSourceHandPreferred = 20,
    IdSourceHandOnly,
    IdSourceGazeOnly,
    IdScreenPrimary = 30,
    IdScreenAll,
    IdSmoothResponsive = 40,
    IdSmoothBalanced,
    IdSmoothSmooth,
    IdCalibrateHand = 50,
    IdCalibrateGaze,
    IdCalibrateEyes,
    IdOpenSettings = 60,
    IdReloadSettings,
    IdOpenLog,
    IdExit = 99
};

TrayApp* g_app = nullptr;

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Round status icon drawn in code (no .ico/.rc resources needed): a filled
// disc, optionally hollow, optionally with a white "pupil".
HICON make_icon(COLORREF color, bool hollow, bool pupil) {
    const int size = GetSystemMetrics(SM_CXSMICON);
    BITMAPV5HEADER header{};
    header.bV5Size = sizeof(header);
    header.bV5Width = size;
    header.bV5Height = -size;
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5RedMask = 0x00FF0000;
    header.bV5GreenMask = 0x0000FF00;
    header.bV5BlueMask = 0x000000FF;
    header.bV5AlphaMask = 0xFF000000;
    void* bits = nullptr;
    HDC screen = GetDC(nullptr);
    HBITMAP color_bitmap = CreateDIBSection(screen, reinterpret_cast<BITMAPINFO*>(&header),
                                            DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, screen);
    if (color_bitmap == nullptr) {
        return LoadIconW(nullptr, IDI_APPLICATION);
    }
    auto* pixels = static_cast<std::uint32_t*>(bits);
    const double centre = size / 2.0;
    const double outer = size * 0.44;
    const double inner = outer - std::max(2.0, size * 0.16);
    const double pupil_radius = size * 0.15;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double d = std::hypot(x + 0.5 - centre, y + 0.5 - centre);
            double coverage = std::clamp(outer - d + 0.5, 0.0, 1.0);
            if (hollow) {
                coverage *= std::clamp(d - inner + 0.5, 0.0, 1.0);
            }
            double r = GetRValue(color);
            double g = GetGValue(color);
            double b = GetBValue(color);
            if (pupil) {
                const double white = std::clamp(pupil_radius - d + 0.5, 0.0, 1.0);
                r += (255.0 - r) * white;
                g += (255.0 - g) * white;
                b += (255.0 - b) * white;
            }
            const auto a = static_cast<std::uint32_t>(coverage * 255.0 + 0.5);
            const auto pr = static_cast<std::uint32_t>(r * coverage + 0.5);
            const auto pg = static_cast<std::uint32_t>(g * coverage + 0.5);
            const auto pb = static_cast<std::uint32_t>(b * coverage + 0.5);
            pixels[y * size + x] = (a << 24) | (pr << 16) | (pg << 8) | pb;
        }
    }
    HBITMAP mask = CreateBitmap(size, size, 1, 1, nullptr);
    ICONINFO info{};
    info.fIcon = TRUE;
    info.hbmMask = mask;
    info.hbmColor = color_bitmap;
    HICON icon = CreateIconIndirect(&info);
    DeleteObject(mask);
    DeleteObject(color_bitmap);
    return icon != nullptr ? icon : LoadIconW(nullptr, IDI_APPLICATION);
}

void copy_text(wchar_t* destination, std::size_t capacity, const std::wstring& text) {
    const std::size_t count = std::min(text.size(), capacity - 1);
    std::copy_n(text.data(), count, destination);
    destination[count] = L'\0';
}

// Dry-run sink: every button event and a throttled trace of moves go to the log.
class LoggingInputSink final : public IInputSink {
public:
    bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept override {
        const double now = now_seconds();
        if (now - last_move_log_ >= 0.5) {
            last_move_log_ = now;
            log("move to (" + std::to_string(pixel.x) + ", " + std::to_string(pixel.y) + ") abs (" +
                std::to_string(absolute.x) + ", " + std::to_string(absolute.y) + ")");
        }
        return true;
    }
    bool button(MouseButton which, ButtonAction action) noexcept override {
        log(std::string(name(which)) + (action == ButtonAction::Down ? " down" : " up"));
        return true;
    }
    bool click(MouseButton which) noexcept override {
        log(std::string(name(which)) + " click");
        return true;
    }

private:
    static const char* name(MouseButton b) {
        return b == MouseButton::Left ? "left" : b == MouseButton::Right ? "right" : "middle";
    }
    static void log(const std::string& text) noexcept {
        try {
            AppLog::info("[dry run] " + text);
        } catch (...) {
        }
    }
    double last_move_log_ = -1.0;
};

} // namespace

TrayApp::TrayApp(HINSTANCE instance, TrayOptions options)
    : instance_(instance), options_(std::move(options)) {
    g_app = this;
}

TrayApp::~TrayApp() {
    shutdown();
    for (HICON icon : icons_) {
        if (icon != nullptr) {
            DestroyIcon(icon);
        }
    }
    g_app = nullptr;
}

int TrayApp::run() {
    if (!initialize()) {
        shutdown();
        return 1;
    }
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    shutdown();
    return static_cast<int>(msg.wParam);
}

bool TrayApp::initialize() {
    AppLog::open(options_.log_path);
    AppLog::info("---- HandEye tray starting" + std::string(options_.dry_run ? " (dry run)" : ""));

    auto loaded = load_settings_file(options_.settings_path);
    for (const auto& warning : loaded.warnings) {
        AppLog::warn("settings: " + warning);
    }
    settings_ = loaded.settings;
    if (!std::filesystem::exists(options_.settings_path)) {
        std::string error;
        if (save_settings_file(options_.settings_path, settings_, &error)) {
            AppLog::info("created default settings at " + to_utf8(options_.settings_path.wstring()));
        } else {
            AppLog::warn("could not create settings file: " + error);
        }
    }
    guard_.set_yield_seconds(settings_.app.physical_mouse_yield_seconds);

    if (options_.dry_run) {
        sink_ = std::make_unique<LoggingInputSink>();
    } else {
        sink_ = std::make_unique<win32::InputInjector>();
    }
    try {
        controller_ = std::make_unique<OsController>(
            settings_, win32::query_screen_geometry(settings_.mapping.screen_target), *sink_);
    } catch (const std::exception& error) {
        AppLog::error(std::string("controller: ") + error.what());
        MessageBoxW(nullptr, to_wide(error.what()).c_str(), L"HandEye Tracker", MB_ICONERROR);
        return false;
    }
    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance_;
    wc.lpszClassName = kWindowClass;
    RegisterClassW(&wc);
    // A hidden top-level window (not message-only) so it receives broadcast
    // messages such as TaskbarCreated and WM_DISPLAYCHANGE.
    hwnd_ = CreateWindowExW(WS_EX_TOOLWINDOW, kWindowClass, L"HandEye Tracker", WS_POPUP, 0, 0, 0, 0,
                            nullptr, nullptr, instance_, this);
    if (hwnd_ == nullptr) {
        AppLog::error("could not create the tray window");
        return false;
    }
    taskbar_created_ = RegisterWindowMessageW(L"TaskbarCreated");
    post_target_ = hwnd_;

    icons_[static_cast<std::size_t>(Status::Waiting)] = make_icon(RGB(140, 146, 156), true, false);
    icons_[static_cast<std::size_t>(Status::Paused)] = make_icon(RGB(235, 160, 40), false, false);
    icons_[static_cast<std::size_t>(Status::Active)] = make_icon(RGB(60, 190, 110), false, true);
    icons_[static_cast<std::size_t>(Status::Yielding)] = make_icon(RGB(70, 150, 240), false, false);
    icons_[static_cast<std::size_t>(Status::Calibrating)] = make_icon(RGB(150, 110, 240), true, false);
    if (!add_tray_icon()) {
        AppLog::error("could not add the tray icon");
        return false;
    }

    hotkey_registered_ =
        RegisterHotKey(hwnd_, kToggleHotkey, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 'P') != FALSE;
    if (!hotkey_registered_) {
        AppLog::warn("Ctrl+Alt+P is taken by another program; use the tray menu to pause/resume");
    }
    if (settings_.app.physical_mouse_yield_seconds > 0.0) {
        hook_ = SetWindowsHookExW(WH_MOUSE_LL, mouse_hook, instance_, 0);
        if (hook_ == nullptr) {
            AppLog::warn("physical mouse override unavailable (low-level hook refused)");
        }
    }

    if (options_.start_active || settings_.app.start_active) {
        set_active(true, "startup");
    }
    SetTimer(hwnd_, kStatusTimer, 250, nullptr);
    refresh_status();
    if (options_.calibrate_on_start) {
        const WizardKind kind = *options_.calibrate_on_start;
        PostMessageW(hwnd_, WM_COMMAND,
                     kind == WizardKind::HandRange   ? IdCalibrateHand
                     : kind == WizardKind::GazeRange ? IdCalibrateGaze
                                                     : IdCalibrateEyes,
                     0);
    }
    notify(L"HandEye Tracker is running",
           std::wstring(controller_->active() ? L"Control is ON." : L"Control is paused.") +
               L" Waiting for tracking data from the core engine; Ctrl+Alt+P toggles control." +
               (loaded.warnings.empty() ? L"" : L"\nSome settings were invalid; see the log.") +
               (options_.dry_run ? L"\nDry run: no real mouse input will be sent." : L""));
    AppLog::info("ready; waiting for frames from the core engine (TrayApp::submit_frame)");
    ready_ = true;
    return true;
}

void TrayApp::shutdown() {
    if (shut_down_) {
        return;
    }
    shut_down_ = true;
    ready_ = false;
    post_target_ = nullptr;
    wizard_.reset();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (controller_) {
            controller_->release_all_buttons();
        }
    }
    if (hook_ != nullptr) {
        UnhookWindowsHookEx(hook_);
        hook_ = nullptr;
    }
    if (hwnd_ != nullptr) {
        if (hotkey_registered_) {
            UnregisterHotKey(hwnd_, kToggleHotkey);
        }
        KillTimer(hwnd_, kStatusTimer);
        Shell_NotifyIconW(NIM_DELETE, &icon_data_);
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    AppLog::info("---- HandEye tray stopped");
}

// ---------------------------------------------------------------- frames

void TrayApp::submit_frame(const FramePayload& frame) noexcept {
    if (!ready_.load()) {
        return;
    }
    const double now = now_seconds();
    FrameReport report;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_.load()) {
            return; // shut down while we waited for the lock
        }
        if (last_frame_timestamp_ && frame.timestamp_seconds < *last_frame_timestamp_ - 0.5) {
            // The host restarted its clock: start from a clean gesture state.
            try {
                AppLog::info("frame clock restarted; resetting gesture state");
                controller_->apply_settings(settings_); // already validated; resets state
            } catch (...) {
            }
        }
        last_frame_timestamp_ = frame.timestamp_seconds;

        controller_->set_inhibited(wizard_running_.load() || guard_.yielding(now));
        report = controller_->update(frame);

        if (wizard_running_.load()) {
            WizardSample sample;
            sample.timestamp = frame.timestamp_seconds;
            sample.hand = frame.hand.point;
            sample.gaze = frame.gaze.point;
            sample.eye = eye_state_from_int(frame.gaze.state).value_or(EyeState::Neutral);
            try {
                inbox_.push(sample);
            } catch (...) {
            }
        }
    }
    last_frame_at_.store(now);
    stale_reported_.store(false);

    if (!first_frame_seen_.exchange(true)) {
        post_event(FirstFrame);
    }
    if (report.pause_toggled) {
        post_event(report.active ? GestureResume : GesturePause);
    }
    if (report.injection_failed && now - last_block_notice_.load() > 10.0) {
        last_block_notice_.store(now);
        post_event(InjectionBlocked);
    }
    if (report.unknown_states != 0 && !unknown_state_logged_.exchange(true)) {
        try {
            AppLog::warn("received state integers outside the documented contract; treated as neutral");
        } catch (...) {
        }
    }
}

void TrayApp::request_exit() noexcept {
    if (const HWND target = post_target_.load()) {
        PostMessageW(target, WM_CLOSE, 0, 0);
    }
}

void TrayApp::post_event(WPARAM event) noexcept {
    if (const HWND target = post_target_.load()) {
        PostMessageW(target, kEventMessage, event, 0);
    }
}

// Safety: if the core stops sending frames (crash, camera unplugged) while
// a pinch holds the left button, release it rather than leave it stuck.
void TrayApp::check_stale_input() {
    const double last = last_frame_at_.load();
    if (last < 0.0 || now_seconds() - last <= settings_.app.stale_input_seconds ||
        stale_reported_.exchange(true)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        controller_->release_all_buttons();
    }
    AppLog::warn("tracking frames stopped; any held button was released");
}

// ---------------------------------------------------------------- window

LRESULT CALLBACK TrayApp::window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_NCCREATE) {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }
    auto* self = reinterpret_cast<TrayApp*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (self == nullptr || self->hwnd_ == nullptr) {
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }
    return self->handle(message, wparam, lparam);
}

LRESULT CALLBACK TrayApp::mouse_hook(int code, WPARAM wparam, LPARAM lparam) {
    if (code == HC_ACTION && g_app != nullptr) {
        const auto* info = reinterpret_cast<const MSLLHOOKSTRUCT*>(lparam);
        if ((info->flags & LLMHF_INJECTED) == 0) {
            g_app->guard_.on_physical_input(now_seconds());
        }
    }
    return CallNextHookEx(nullptr, code, wparam, lparam);
}

LRESULT TrayApp::handle(UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == taskbar_created_ && taskbar_created_ != 0) {
        add_tray_icon(); // Explorer restarted
        shown_status_ = Status::Count;
        refresh_status();
        return 0;
    }
    switch (message) {
    case kTrayMessage:
        switch (LOWORD(lparam)) {
        case WM_CONTEXTMENU:
        case NIN_SELECT:
        case NIN_KEYSELECT:
            show_menu();
            break;
        default:
            break;
        }
        return 0;
    case WM_COMMAND:
        on_command(LOWORD(wparam));
        return 0;
    case WM_HOTKEY:
        if (wparam == kToggleHotkey) {
            bool active = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active = controller_->active();
            }
            set_active(!active, "hotkey");
        }
        return 0;
    case WM_TIMER:
        if (wparam == kStatusTimer) {
            check_stale_input();
            refresh_status();
        }
        return 0;
    case kEventMessage:
        switch (wparam) {
        case FirstFrame:
            AppLog::info("first tracker frame received");
            notify(L"Tracking data received", L"Press Ctrl+Alt+P to toggle mouse control.");
            break;
        case GesturePause:
            AppLog::info("control paused by sustained eye closure");
            notify(L"Control paused", L"Eyes closed. Close them again for a second to resume.");
            break;
        case GestureResume:
            AppLog::info("control resumed by sustained eye closure");
            notify(L"Control resumed", L"Eyes closed. Close them again for a second to pause.");
            break;
        case InjectionBlocked:
            AppLog::warn("Windows refused injected input (UIPI: an elevated window has focus?)");
            notify(L"Windows blocked the virtual mouse",
                   L"An administrator window probably has focus. Switch windows, or run the tray "
                   L"app as administrator to control elevated apps.",
                   true);
            break;
        default:
            break;
        }
        refresh_status();
        return 0;
    case kWizardDoneMessage:
        on_wizard_done();
        return 0;
    case WM_DISPLAYCHANGE: {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            controller_->set_screen_geometry(win32::query_screen_geometry(settings_.mapping.screen_target));
            AppLog::info("display configuration changed; screen geometry updated");
        } catch (const std::exception& error) {
            AppLog::error(std::string("display change: ") + error.what());
        }
        return 0;
    }
    case WM_QUERYENDSESSION:
        return TRUE;
    case WM_ENDSESSION:
        if (wparam) {
            shutdown();
        }
        return 0;
    case WM_CLOSE:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd_, message, wparam, lparam);
    }
}

bool TrayApp::add_tray_icon() {
    icon_data_ = {};
    icon_data_.cbSize = sizeof(icon_data_);
    icon_data_.hWnd = hwnd_;
    icon_data_.uID = kTrayIconId;
    icon_data_.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    icon_data_.uCallbackMessage = kTrayMessage;
    icon_data_.hIcon = icons_[static_cast<std::size_t>(Status::Waiting)];
    copy_text(icon_data_.szTip, std::size(icon_data_.szTip), L"HandEye Tracker");
    if (!Shell_NotifyIconW(NIM_ADD, &icon_data_)) {
        return false;
    }
    icon_data_.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &icon_data_);
    return true;
}

TrayApp::Status TrayApp::current_status() {
    if (wizard_running_.load()) {
        return Status::Calibrating;
    }
    const double last = last_frame_at_.load();
    const double stale = settings_.app.stale_input_seconds;
    if (last < 0.0 || now_seconds() - last > std::max(1.0, stale)) {
        return Status::Waiting;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!controller_->active()) {
        return Status::Paused;
    }
    return guard_.yielding(now_seconds()) ? Status::Yielding : Status::Active;
}

void TrayApp::refresh_status() {
    if (hwnd_ == nullptr) {
        return;
    }
    const Status status = current_status();
    bool active = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active = controller_->active();
    }
    std::wstring tip = L"HandEye Tracker — ";
    switch (status) {
    case Status::Waiting:
        tip += active ? L"on, waiting for tracking data" : L"paused, waiting for tracking data";
        break;
    case Status::Paused:
        tip += L"paused (Ctrl+Alt+P)";
        break;
    case Status::Active:
        tip += L"controlling the mouse";
        break;
    case Status::Yielding:
        tip += L"yielding to the physical mouse";
        break;
    case Status::Calibrating:
        tip += L"calibrating";
        break;
    case Status::Count:
        break;
    }
    if (options_.dry_run) {
        tip += L" [dry run]";
    }
    if (status == shown_status_ && tip == shown_tip_) {
        return;
    }
    shown_status_ = status;
    shown_tip_ = tip;
    icon_data_.uFlags = NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    icon_data_.hIcon = icons_[static_cast<std::size_t>(status)];
    copy_text(icon_data_.szTip, std::size(icon_data_.szTip), tip);
    Shell_NotifyIconW(NIM_MODIFY, &icon_data_);
}

void TrayApp::notify(const std::wstring& title, const std::wstring& text, bool warning) {
    NOTIFYICONDATAW data = icon_data_;
    data.uFlags = NIF_INFO;
    data.dwInfoFlags = warning ? NIIF_WARNING : NIIF_INFO;
    copy_text(data.szInfoTitle, std::size(data.szInfoTitle), title);
    copy_text(data.szInfo, std::size(data.szInfo), text);
    Shell_NotifyIconW(NIM_MODIFY, &data);
}

void TrayApp::set_active(bool active, const char* reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (controller_->active() == active) {
            return;
        }
        controller_->set_active(active);
    }
    AppLog::info(std::string("control ") + (active ? "enabled" : "paused") + " (" + reason + ")");
    refresh_status();
}

// ---------------------------------------------------------------- menu

void TrayApp::show_menu() {
    Settings settings;
    bool active = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settings = settings_;
        active = controller_->active();
    }
    const auto check = [](bool on) { return static_cast<UINT>(on ? MF_CHECKED : MF_UNCHECKED); };

    HMENU source_menu = CreatePopupMenu();
    AppendMenuW(source_menu, MF_STRING | check(settings.mapping.cursor_source == CursorSourceMode::HandPreferred),
                IdSourceHandPreferred, L"Hand, gaze when no hand is visible");
    AppendMenuW(source_menu, MF_STRING | check(settings.mapping.cursor_source == CursorSourceMode::HandOnly),
                IdSourceHandOnly, L"Hand only");
    AppendMenuW(source_menu, MF_STRING | check(settings.mapping.cursor_source == CursorSourceMode::GazeOnly),
                IdSourceGazeOnly, L"Gaze only");

    HMENU screen_menu = CreatePopupMenu();
    AppendMenuW(screen_menu, MF_STRING | check(settings.mapping.screen_target == ScreenTarget::PrimaryMonitor),
                IdScreenPrimary, L"Primary monitor");
    AppendMenuW(screen_menu, MF_STRING | check(settings.mapping.screen_target == ScreenTarget::VirtualDesktop),
                IdScreenAll, L"All monitors");

    const auto preset = detect_smoothing_preset(settings);
    HMENU smoothing_menu = CreatePopupMenu();
    AppendMenuW(smoothing_menu, MF_STRING | check(preset == SmoothingPreset::Responsive),
                IdSmoothResponsive, L"Responsive (less lag, more jitter)");
    AppendMenuW(smoothing_menu, MF_STRING | check(preset == SmoothingPreset::Balanced),
                IdSmoothBalanced, L"Balanced (Engineer 4 defaults)");
    AppendMenuW(smoothing_menu, MF_STRING | check(preset == SmoothingPreset::Smooth), IdSmoothSmooth,
                L"Smooth (steadier, more lag)");

    HMENU calibrate_menu = CreatePopupMenu();
    AppendMenuW(calibrate_menu, MF_STRING, IdCalibrateHand, L"Hand range…");
    AppendMenuW(calibrate_menu, MF_STRING, IdCalibrateGaze, L"Gaze range…");
    AppendMenuW(calibrate_menu, MF_STRING, IdCalibrateEyes, L"Blink && wink timing…");

    HMENU menu = CreatePopupMenu();
    AppendMenuW(menu, MF_STRING | MF_GRAYED, IdStatus, shown_tip_.c_str());
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING | check(active), IdToggle,
                hotkey_registered_ ? L"Mouse control\tCtrl+Alt+P" : L"Mouse control");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(source_menu), L"Cursor follows");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(screen_menu), L"Screen");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(smoothing_menu), L"Smoothing");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(calibrate_menu), L"Calibrate");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, IdOpenSettings, L"Open settings file");
    AppendMenuW(menu, MF_STRING, IdReloadSettings, L"Reload settings file");
    AppendMenuW(menu, MF_STRING, IdOpenLog, L"Open log");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, IdExit, L"Exit");

    POINT cursor;
    GetCursorPos(&cursor);
    SetForegroundWindow(hwnd_); // required so the menu closes when clicking elsewhere
    TrackPopupMenuEx(menu, TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, cursor.x, cursor.y, hwnd_, nullptr);
    PostMessageW(hwnd_, WM_NULL, 0, 0);
    DestroyMenu(menu); // destroys the submenus too
}

void TrayApp::on_command(UINT id) {
    switch (id) {
    case IdToggle: {
        bool active = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active = controller_->active();
        }
        set_active(!active, "menu");
        break;
    }
    case IdSourceHandPreferred:
    case IdSourceHandOnly:
    case IdSourceGazeOnly: {
        const CursorSourceMode mode = id == IdSourceHandOnly   ? CursorSourceMode::HandOnly
                                      : id == IdSourceGazeOnly ? CursorSourceMode::GazeOnly
                                                               : CursorSourceMode::HandPreferred;
        change_settings([mode](Settings& s) { s.mapping.cursor_source = mode; }, "cursor source");
        break;
    }
    case IdScreenPrimary:
    case IdScreenAll: {
        const ScreenTarget target = id == IdScreenAll ? ScreenTarget::VirtualDesktop : ScreenTarget::PrimaryMonitor;
        change_settings([target](Settings& s) { s.mapping.screen_target = target; }, "screen");
        break;
    }
    case IdSmoothResponsive:
    case IdSmoothBalanced:
    case IdSmoothSmooth: {
        const SmoothingPreset preset = id == IdSmoothResponsive ? SmoothingPreset::Responsive
                                       : id == IdSmoothSmooth   ? SmoothingPreset::Smooth
                                                                : SmoothingPreset::Balanced;
        change_settings([preset](Settings& s) { apply_smoothing_preset(s, preset); }, "smoothing");
        break;
    }
    case IdCalibrateHand:
        start_wizard(WizardKind::HandRange);
        break;
    case IdCalibrateGaze:
        start_wizard(WizardKind::GazeRange);
        break;
    case IdCalibrateEyes:
        start_wizard(WizardKind::EyeTiming);
        break;
    case IdOpenSettings:
        ShellExecuteW(nullptr, L"open", L"notepad.exe", options_.settings_path.c_str(), nullptr, SW_SHOWNORMAL);
        break;
    case IdReloadSettings:
        reload_settings();
        break;
    case IdOpenLog:
        ShellExecuteW(nullptr, L"open", L"notepad.exe", AppLog::path().c_str(), nullptr, SW_SHOWNORMAL);
        break;
    case IdExit:
        PostQuitMessage(0);
        break;
    default:
        break;
    }
}

// ---------------------------------------------------------------- settings

void TrayApp::change_settings(const std::function<void(Settings&)>& edit, const char* what) {
    Settings next;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        next = settings_;
    }
    edit(next);
    const auto problems = validate(next);
    if (!problems.empty()) {
        AppLog::error(std::string("rejected ") + what + " change: " + problems.front());
        notify(L"Setting not applied", to_wide(problems.front()), true);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            if (next.mapping.screen_target != settings_.mapping.screen_target) {
                controller_->set_screen_geometry(win32::query_screen_geometry(next.mapping.screen_target));
            }
            controller_->apply_settings(next);
        } catch (const std::exception& error) {
            AppLog::error(std::string("applying ") + what + ": " + error.what());
            notify(L"Setting not applied", to_wide(error.what()), true);
            return;
        }
        guard_.set_yield_seconds(next.app.physical_mouse_yield_seconds);
        settings_ = next;
    }
    if (options_.on_settings_changed) {
        options_.on_settings_changed(next);
    }
    std::string error;
    if (save_settings_file(options_.settings_path, next, &error)) {
        AppLog::info(std::string("settings changed: ") + what);
    } else {
        AppLog::error("saving settings: " + error);
        notify(L"Settings could not be saved", to_wide(error), true);
    }
    refresh_status();
}

void TrayApp::reload_settings() {
    auto loaded = load_settings_file(options_.settings_path);
    for (const auto& warning : loaded.warnings) {
        AppLog::warn("settings: " + warning);
    }
    change_settings([&loaded](Settings& s) { s = loaded.settings; }, "reload");
    notify(L"Settings reloaded",
           loaded.warnings.empty() ? L"All values applied."
                                   : L"Some values were invalid and kept their defaults; see the log.",
           !loaded.warnings.empty());
}

// ---------------------------------------------------------------- wizard

void TrayApp::start_wizard(WizardKind kind) {
    if (wizard_) {
        return;
    }
    Settings settings;
    ScreenRect area{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settings = settings_;
        area = controller_->mapper().geometry().target;
    }
    wizard_kind_ = kind;
    wizard_outcome_.reset();
    wizard_running_ = true; // inhibits the controller: no clicks while calibrating
    wizard_ = std::make_unique<CalibrationWizard>(
        instance_, kind, area, settings, inbox_, [this](const CalibrationWizard::Outcome& outcome) {
            wizard_outcome_ = outcome;
            PostMessageW(hwnd_, kWizardDoneMessage, 0, 0); // destroy the wizard outside its own window procedure
        });
    if (!wizard_->open()) {
        wizard_.reset();
        wizard_running_ = false;
        notify(L"Calibration unavailable", L"The calibration window could not be opened.", true);
        return;
    }
    AppLog::info("calibration started");
    refresh_status();
}

void TrayApp::on_wizard_done() {
    wizard_.reset();
    wizard_running_ = false;
    if (!wizard_outcome_) {
        return;
    }
    const CalibrationWizard::Outcome outcome = *wizard_outcome_;
    wizard_outcome_.reset();
    if (outcome.saved) {
        const WizardKind kind = wizard_kind_;
        change_settings([&outcome, kind](Settings& s) {
            switch (kind) {
            case WizardKind::HandRange:
                s.mapping.hand_region = outcome.settings.mapping.hand_region;
                break;
            case WizardKind::GazeRange:
                s.mapping.gaze_region = outcome.settings.mapping.gaze_region;
                break;
            case WizardKind::EyeTiming:
                s.gestures.wink_onset_seconds = outcome.settings.gestures.wink_onset_seconds;
                s.gestures.closure_onset_seconds = outcome.settings.gestures.closure_onset_seconds;
                break;
            }
        }, "calibration");
        notify(L"Calibration saved", outcome.message);
    } else {
        AppLog::info("calibration closed without saving");
    }
    refresh_status();
}

} // namespace osi::app
