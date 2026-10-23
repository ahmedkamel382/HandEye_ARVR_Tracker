#pragma once

// Windows-only. Build with WIN32_LEAN_AND_MEAN and NOMINMAX (the CMake
// target sets both).
#include "os_integration/input_sink.hpp"
#include "os_integration/settings.hpp"
#include "os_integration/types.hpp"

#include <windows.h>

#include <optional>
#include <string>
#include <vector>

namespace osi::win32 {

// Stamped into INPUT::mi.dwExtraInfo so our own events can be recognised
// (e.g. by a low-level mouse hook). "HEO5": HandEye OS, Engineer 5.
inline constexpr ULONG_PTR kInjectedInputTag = 0x48454F35;

// Absolute move across the whole virtual desktop.
INPUT make_move_input(AbsolutePoint absolute) noexcept;
INPUT make_button_input(MouseButton button, ButtonAction action) noexcept;

// Sends mouse input through SendInput. SendInput fails (returns 0) when
// User Interface Privilege Isolation blocks it, e.g. while an elevated
// window has focus and this process is not elevated.
class InputInjector final : public IInputSink {
public:
    bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept override;
    bool button(MouseButton button, ButtonAction action) noexcept override;
    // Down and up in a single SendInput call so no other input interleaves.
    bool click(MouseButton button) noexcept override;
};

// ------------------------------------------------------------ displays

struct MonitorInfo {
    ScreenRect bounds;
    ScreenRect work_area; // excludes the taskbar
    bool primary = false;
    std::wstring device_name;
};

// Call once, before any window or geometry query, so all coordinates are
// physical pixels on every monitor (otherwise Windows scales them per DPI).
void enable_per_monitor_dpi_awareness() noexcept;

std::vector<MonitorInfo> enumerate_monitors();
ScreenRect virtual_desktop_rect() noexcept;
ScreenRect primary_monitor_rect() noexcept;
ScreenGeometry query_screen_geometry(ScreenTarget target) noexcept;

std::optional<PixelPoint> cursor_position() noexcept;

// FormatMessage text for GetLastError() values.
std::wstring last_error_message(DWORD error);

} // namespace osi::win32
