#pragma once

#include "os_integration/cursor_stabilizer.hpp"
#include "os_integration/gesture_interpreter.hpp"
#include "os_integration/screen_mapper.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace osi {

enum class ScreenTarget : std::uint8_t { PrimaryMonitor, VirtualDesktop };

// Mirrors dsp::FilterConfig so this module does not depend on dsp_filter.
// The defaults are Engineer 4's kInitialHandConfig / kInitialGazeConfig.
struct FilterSettings {
    double min_cutoff_hz;
    double beta;
    double derivative_cutoff_hz;
};

struct MappingSettings {
    // Hand: the central 70% of the frame covers the whole screen.
    InputRegion hand_region{0.15, 0.15, 0.85, 0.85};
    // Gaze: placeholder band until the calibration wizard measures the user.
    InputRegion gaze_region{0.30, 0.30, 0.70, 0.70};
    CursorSourceMode cursor_source = CursorSourceMode::HandPreferred;
    ScreenTarget screen_target = ScreenTarget::PrimaryMonitor;
    double source_switch_seconds = 0.20;
};

struct StabilizationSettings {
    double anchor_lookback_seconds = 0.10;
    double post_click_hold_seconds = 0.15;
    double drag_threshold_px = 30.0;
};

struct AppSettings {
    bool start_active = false; // tray app starts paused until the user enables it
    double physical_mouse_yield_seconds = 1.5;
    double stale_input_seconds = 0.5; // release buttons when frames stop arriving
};

struct Settings {
    MappingSettings mapping;
    StabilizationSettings stabilization;
    GestureSettings gestures;
    FilterSettings hand_filter{1.2, 4.0, 1.0};
    FilterSettings gaze_filter{0.8, 6.0, 1.5};
    AppSettings app;
};

// Every problem found, as human-readable messages; empty means valid.
std::vector<std::string> validate(const Settings& settings);

struct SettingsLoadResult {
    Settings settings;
    std::vector<std::string> warnings; // unknown keys, malformed or out-of-range values
};

// INI-style "key = value" lines grouped in [sections]; '#' and ';' start
// comments. Never throws: a bad value keeps its default and adds a warning.
SettingsLoadResult parse_settings(std::string_view text);
std::string serialize_settings(const Settings& settings);

// A missing file yields defaults and no warnings.
SettingsLoadResult load_settings_file(const std::filesystem::path& path);
// Writes to a temporary file first, then replaces the target.
bool save_settings_file(const std::filesystem::path& path, const Settings& settings,
                        std::string* error = nullptr);

// Smoothing presets offered by the tray menu. Balanced keeps Engineer 4's
// initial values; Smooth halves the minimum cutoff (steadier, more lag),
// Responsive doubles it (less lag, more jitter).
enum class SmoothingPreset : std::uint8_t { Responsive, Balanced, Smooth };
void apply_smoothing_preset(Settings& settings, SmoothingPreset preset) noexcept;
// The preset the current filter values match, if any.
std::optional<SmoothingPreset> detect_smoothing_preset(const Settings& settings) noexcept;

} // namespace osi
