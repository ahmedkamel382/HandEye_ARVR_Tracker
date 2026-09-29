#include "os_integration/win32_input.hpp"

#include <array>

namespace osi::win32 {
namespace {

DWORD button_flag(MouseButton button, ButtonAction action) noexcept {
    const bool down = action == ButtonAction::Down;
    switch (button) {
    case MouseButton::Left:
        return down ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
    case MouseButton::Right:
        return down ? MOUSEEVENTF_RIGHTDOWN : MOUSEEVENTF_RIGHTUP;
    case MouseButton::Middle:
        return down ? MOUSEEVENTF_MIDDLEDOWN : MOUSEEVENTF_MIDDLEUP;
    }
    return 0;
}

ScreenRect to_rect(const RECT& r) noexcept {
    return {r.left, r.top, r.right - r.left, r.bottom - r.top};
}

BOOL CALLBACK collect_monitor(HMONITOR monitor, HDC, LPRECT, LPARAM data) {
    auto* monitors = reinterpret_cast<std::vector<MonitorInfo>*>(data);
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    if (GetMonitorInfoW(monitor, &info)) {
        monitors->push_back({to_rect(info.rcMonitor), to_rect(info.rcWork),
                             (info.dwFlags & MONITORINFOF_PRIMARY) != 0, info.szDevice});
    }
    return TRUE;
}

} // namespace

INPUT make_move_input(AbsolutePoint absolute) noexcept {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = absolute.x;
    input.mi.dy = absolute.y;
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
    input.mi.dwExtraInfo = kInjectedInputTag;
    return input;
}

INPUT make_button_input(MouseButton button, ButtonAction action) noexcept {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = button_flag(button, action);
    input.mi.dwExtraInfo = kInjectedInputTag;
    return input;
}

bool InputInjector::move_to(PixelPoint, AbsolutePoint absolute) noexcept {
    INPUT input = make_move_input(absolute);
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool InputInjector::button(MouseButton which, ButtonAction action) noexcept {
    INPUT input = make_button_input(which, action);
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool InputInjector::click(MouseButton which) noexcept {
    std::array<INPUT, 2> inputs{make_button_input(which, ButtonAction::Down),
                                make_button_input(which, ButtonAction::Up)};
    return SendInput(static_cast<UINT>(inputs.size()), inputs.data(), sizeof(INPUT)) == inputs.size();
}

void enable_per_monitor_dpi_awareness() noexcept {
    // Windows 10 1703+. Falls back silently on older systems or when a
    // manifest already set the awareness.
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        SetProcessDPIAware();
    }
}

std::vector<MonitorInfo> enumerate_monitors() {
    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(nullptr, nullptr, collect_monitor, reinterpret_cast<LPARAM>(&monitors));
    return monitors;
}

ScreenRect virtual_desktop_rect() noexcept {
    return {GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN),
            GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN)};
}

ScreenRect primary_monitor_rect() noexcept {
    const HMONITOR monitor = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO info{};
    info.cbSize = sizeof(info);
    if (GetMonitorInfoW(monitor, &info)) {
        return to_rect(info.rcMonitor);
    }
    return {0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
}

ScreenGeometry query_screen_geometry(ScreenTarget target) noexcept {
    const ScreenRect desktop = virtual_desktop_rect();
    return {desktop, target == ScreenTarget::VirtualDesktop ? desktop : primary_monitor_rect()};
}

std::optional<PixelPoint> cursor_position() noexcept {
    POINT point{};
    if (!GetCursorPos(&point)) {
        return std::nullopt;
    }
    return PixelPoint{point.x, point.y};
}

std::wstring last_error_message(DWORD error) {
    wchar_t* buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, error, 0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::wstring message = length != 0 ? std::wstring(buffer, length) : L"unknown error";
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    while (!message.empty() && (message.back() == L'\n' || message.back() == L'\r')) {
        message.pop_back();
    }
    return message + L" (" + std::to_wstring(error) + L")";
}

} // namespace osi::win32
