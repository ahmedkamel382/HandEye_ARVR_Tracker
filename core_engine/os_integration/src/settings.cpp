#include "os_integration/settings.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <utility>

namespace osi {
namespace {

// ---------------------------------------------------------------- text utils

std::string_view trim(std::string_view text) noexcept {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool parse_double(std::string_view text, double& out) noexcept {
    text = trim(text);
    if (text.empty()) {
        return false;
    }
    if (text.front() == '+') {
        text.remove_prefix(1);
    }
    double value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

bool parse_bool(std::string_view text, bool& out) noexcept {
    text = trim(text);
    if (text == "true" || text == "1" || text == "yes" || text == "on") {
        out = true;
        return true;
    }
    if (text == "false" || text == "0" || text == "no" || text == "off") {
        out = false;
        return true;
    }
    return false;
}

std::string format_double(double value) {
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string(buffer.data(), result.ptr);
}

// ------------------------------------------------------------- field table

struct Field {
    const char* section;
    const char* key;
    // Parses into settings; returns an error message or empty on success.
    std::function<std::string(Settings&, std::string_view)> parse;
    std::function<std::string(const Settings&)> format;
    // Range/consistency check; empty when valid.
    std::function<std::string(const Settings&)> check;
};

template <typename Access>
Field double_field(const char* section, const char* key, Access access, double min, double max,
                   bool min_exclusive = false) {
    auto check = [=](const Settings& s) -> std::string {
        const double v = access(const_cast<Settings&>(s));
        const bool low_ok = min_exclusive ? v > min : v >= min;
        if (!std::isfinite(v) || !low_ok || v > max) {
            return std::string(key) + " must be " + (min_exclusive ? "> " : ">= ") +
                   format_double(min) + " and <= " + format_double(max);
        }
        return {};
    };
    return {section, key,
            [=](Settings& s, std::string_view text) -> std::string {
                double value = 0.0;
                if (!parse_double(text, value)) {
                    return "expected a number";
                }
                access(s) = value;
                return {};
            },
            [=](const Settings& s) { return format_double(access(const_cast<Settings&>(s))); },
            check};
}

template <typename Access>
Field seconds_field(const char* section, const char* key, Access access, double max = 10.0) {
    return double_field(section, key, access, 0.0, max);
}

template <typename Access>
Field bool_field(const char* section, const char* key, Access access) {
    return {section, key,
            [=](Settings& s, std::string_view text) -> std::string {
                bool value = false;
                if (!parse_bool(text, value)) {
                    return "expected true or false";
                }
                access(s) = value;
                return {};
            },
            [=](const Settings& s) {
                return std::string(access(const_cast<Settings&>(s)) ? "true" : "false");
            },
            [](const Settings&) { return std::string{}; }};
}

template <typename Access>
Field region_field(const char* section, const char* key, Access access) {
    return {section, key,
            [=](Settings& s, std::string_view text) -> std::string {
                std::array<double, 4> values{};
                std::size_t count = 0;
                std::size_t pos = 0;
                while (pos < text.size()) {
                    const auto start = text.find_first_not_of(" \t,", pos);
                    if (start == std::string_view::npos) {
                        break;
                    }
                    const auto end = text.find_first_of(" \t,", start);
                    const auto token = text.substr(start, end == std::string_view::npos
                                                              ? std::string_view::npos
                                                              : end - start);
                    if (count == values.size() || !parse_double(token, values[count])) {
                        return "expected four numbers: x_min y_min x_max y_max";
                    }
                    ++count;
                    pos = end == std::string_view::npos ? text.size() : end;
                }
                if (count != values.size()) {
                    return "expected four numbers: x_min y_min x_max y_max";
                }
                access(s) = InputRegion{values[0], values[1], values[2], values[3]};
                return {};
            },
            [=](const Settings& s) {
                const InputRegion& r = access(const_cast<Settings&>(s));
                return format_double(r.x_min) + " " + format_double(r.y_min) + " " +
                       format_double(r.x_max) + " " + format_double(r.y_max);
            },
            [=](const Settings& s) -> std::string {
                if (!is_valid_region(access(const_cast<Settings&>(s)))) {
                    return std::string(key) + " needs finite corners in [-1, 2] and each axis "
                                              "spanning at least 0.02";
                }
                return {};
            }};
}

template <typename Enum, std::size_t N, typename Access>
Field enum_field(const char* section, const char* key, Access access,
                 std::array<std::pair<Enum, const char*>, N> names) {
    return {section, key,
            [=](Settings& s, std::string_view text) -> std::string {
                text = trim(text);
                for (const auto& [value, name] : names) {
                    if (text == name) {
                        access(s) = value;
                        return {};
                    }
                }
                std::string allowed;
                for (const auto& entry : names) {
                    allowed += allowed.empty() ? "" : ", ";
                    allowed += entry.second;
                }
                return "expected one of: " + allowed;
            },
            [=](const Settings& s) {
                const Enum current = access(const_cast<Settings&>(s));
                for (const auto& [value, name] : names) {
                    if (value == current) {
                        return std::string(name);
                    }
                }
                return std::string(names[0].second);
            },
            [](const Settings&) { return std::string{}; }};
}

const std::vector<Field>& fields() {
    static const std::vector<Field> table = [] {
        std::vector<Field> f;
        f.push_back(region_field("mapping", "hand_region",
                                 [](Settings& s) -> InputRegion& { return s.mapping.hand_region; }));
        f.push_back(region_field("mapping", "gaze_region",
                                 [](Settings& s) -> InputRegion& { return s.mapping.gaze_region; }));
        f.push_back(enum_field(
            "mapping", "cursor_source",
            [](Settings& s) -> CursorSourceMode& { return s.mapping.cursor_source; },
            std::array<std::pair<CursorSourceMode, const char*>, 3>{
                {{CursorSourceMode::HandPreferred, "hand_preferred"},
                 {CursorSourceMode::HandOnly, "hand_only"},
                 {CursorSourceMode::GazeOnly, "gaze_only"}}}));
        f.push_back(enum_field(
            "mapping", "screen_target",
            [](Settings& s) -> ScreenTarget& { return s.mapping.screen_target; },
            std::array<std::pair<ScreenTarget, const char*>, 2>{
                {{ScreenTarget::PrimaryMonitor, "primary_monitor"},
                 {ScreenTarget::VirtualDesktop, "all_monitors"}}}));
        f.push_back(seconds_field("mapping", "source_switch_seconds", [](Settings& s) -> double& {
            return s.mapping.source_switch_seconds;
        }));

        f.push_back(seconds_field("stabilization", "anchor_lookback_seconds",
                                  [](Settings& s) -> double& {
                                      return s.stabilization.anchor_lookback_seconds;
                                  }, 1.0));
        f.push_back(seconds_field("stabilization", "post_click_hold_seconds",
                                  [](Settings& s) -> double& {
                                      return s.stabilization.post_click_hold_seconds;
                                  }, 2.0));
        f.push_back(double_field("stabilization", "drag_threshold_px",
                                 [](Settings& s) -> double& {
                                     return s.stabilization.drag_threshold_px;
                                 }, 0.0, 10000.0));

        f.push_back(seconds_field("gestures", "glitch_tolerance_seconds",
                                  [](Settings& s) -> double& {
                                      return s.gestures.glitch_tolerance_seconds;
                                  }, 1.0));
        f.push_back(double_field("gestures", "max_gap_seconds",
                                 [](Settings& s) -> double& { return s.gestures.max_gap_seconds; },
                                 0.0, 10.0, true));
        f.push_back(seconds_field("gestures", "pinch_onset_seconds", [](Settings& s) -> double& {
            return s.gestures.pinch_onset_seconds;
        }));
        f.push_back(seconds_field("gestures", "pinch_release_seconds", [](Settings& s) -> double& {
            return s.gestures.pinch_release_seconds;
        }));
        f.push_back(seconds_field("gestures", "fist_onset_seconds", [](Settings& s) -> double& {
            return s.gestures.fist_onset_seconds;
        }));
        f.push_back(enum_field(
            "gestures", "fist_action",
            [](Settings& s) -> FistAction& { return s.gestures.fist_action; },
            std::array<std::pair<FistAction, const char*>, 4>{
                {{FistAction::None, "none"},
                 {FistAction::RightClick, "right_click"},
                 {FistAction::MiddleClick, "middle_click"},
                 {FistAction::TogglePause, "toggle_pause"}}}));
        f.push_back(seconds_field("gestures", "wink_onset_seconds", [](Settings& s) -> double& {
            return s.gestures.wink_onset_seconds;
        }));
        f.push_back(seconds_field("gestures", "closure_onset_seconds", [](Settings& s) -> double& {
            return s.gestures.closure_onset_seconds;
        }));
        f.push_back(bool_field("gestures", "closure_toggles_pause", [](Settings& s) -> bool& {
            return s.gestures.closure_toggles_pause;
        }));

        for (auto [section, member] :
             {std::pair{"hand_filter", &Settings::hand_filter},
              std::pair{"gaze_filter", &Settings::gaze_filter}}) {
            f.push_back(double_field(section, "min_cutoff_hz",
                                     [member = member](Settings& s) -> double& {
                                         return (s.*member).min_cutoff_hz;
                                     }, 0.0, 1000.0, true));
            f.push_back(double_field(section, "beta",
                                     [member = member](Settings& s) -> double& {
                                         return (s.*member).beta;
                                     }, 0.0, 1000.0));
            f.push_back(double_field(section, "derivative_cutoff_hz",
                                     [member = member](Settings& s) -> double& {
                                         return (s.*member).derivative_cutoff_hz;
                                     }, 0.0, 1000.0, true));
        }

        f.push_back(bool_field("app", "start_active",
                               [](Settings& s) -> bool& { return s.app.start_active; }));
        f.push_back(seconds_field("app", "physical_mouse_yield_seconds",
                                  [](Settings& s) -> double& {
                                      return s.app.physical_mouse_yield_seconds;
                                  }, 60.0));
        f.push_back(double_field("app", "stale_input_seconds",
                                 [](Settings& s) -> double& { return s.app.stale_input_seconds; },
                                 0.0, 10.0, true));
        return f;
    }();
    return table;
}

const Field* find_field(std::string_view section, std::string_view key) {
    for (const Field& field : fields()) {
        if (section == field.section && key == field.key) {
            return &field;
        }
    }
    return nullptr;
}

constexpr double kBalancedHandMinCutoff = 1.2; // dsp::kInitialHandConfig.min_cutoff_hz
constexpr double kBalancedGazeMinCutoff = 0.8; // dsp::kInitialGazeConfig.min_cutoff_hz

double preset_factor(SmoothingPreset preset) noexcept {
    switch (preset) {
    case SmoothingPreset::Responsive:
        return 2.0;
    case SmoothingPreset::Smooth:
        return 0.5;
    case SmoothingPreset::Balanced:
        break;
    }
    return 1.0;
}

} // namespace

std::vector<std::string> validate(const Settings& settings) {
    std::vector<std::string> problems;
    for (const Field& field : fields()) {
        std::string problem = field.check(settings);
        if (!problem.empty()) {
            problems.push_back("[" + std::string(field.section) + "] " + problem);
        }
    }
    return problems;
}

SettingsLoadResult parse_settings(std::string_view text) {
    SettingsLoadResult result;
    std::string section;
    std::size_t line_number = 0;
    while (!text.empty()) {
        ++line_number;
        const auto newline = text.find('\n');
        std::string_view line = text.substr(0, newline);
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);

        const auto comment = line.find_first_of("#;");
        line = trim(line.substr(0, comment));
        if (line.empty()) {
            continue;
        }
        const std::string where = "line " + std::to_string(line_number) + ": ";
        if (line.front() == '[') {
            if (line.back() != ']') {
                result.warnings.push_back(where + "malformed section header");
                continue;
            }
            section = std::string(trim(line.substr(1, line.size() - 2)));
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string_view::npos) {
            result.warnings.push_back(where + "expected key = value");
            continue;
        }
        const std::string_view key = trim(line.substr(0, equals));
        const std::string_view value = trim(line.substr(equals + 1));
        const Field* field = find_field(section, key);
        if (field == nullptr) {
            result.warnings.push_back(where + "unknown setting [" + section + "] " +
                                      std::string(key));
            continue;
        }
        Settings candidate = result.settings;
        std::string problem = field->parse(candidate, value);
        if (problem.empty()) {
            problem = field->check(candidate);
        }
        if (!problem.empty()) {
            result.warnings.push_back(where + std::string(key) + ": " + problem +
                                      " (keeping " + field->format(result.settings) + ")");
            continue;
        }
        result.settings = candidate;
    }
    return result;
}

std::string serialize_settings(const Settings& settings) {
    std::ostringstream out;
    out << "# HandEye AR/VR Tracker - OS integration settings (Engineer 5).\n"
        << "# Regions are x_min y_min x_max y_max in normalized camera space;\n"
        << "# min > max flips that axis. Times are in seconds.\n";
    std::string section;
    for (const Field& field : fields()) {
        if (section != field.section) {
            section = field.section;
            out << "\n[" << section << "]\n";
        }
        out << field.key << " = " << field.format(settings) << '\n';
    }
    return out.str();
}

SettingsLoadResult load_settings_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parse_settings(buffer.str());
}

bool save_settings_file(const std::filesystem::path& path, const Settings& settings,
                        std::string* error) {
    const auto fail = [error](std::string message) {
        if (error != nullptr) {
            *error = std::move(message);
        }
        return false;
    };
    const auto problems = validate(settings);
    if (!problems.empty()) {
        return fail("refusing to save invalid settings: " + problems.front());
    }
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            return fail("cannot create " + path.parent_path().string() + ": " + ec.message());
        }
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) {
            return fail("cannot write " + temporary.string());
        }
        out << serialize_settings(settings);
        if (!out.flush()) {
            return fail("write failed for " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        return fail("cannot replace " + path.string());
    }
    return true;
}

void apply_smoothing_preset(Settings& settings, SmoothingPreset preset) noexcept {
    const double factor = preset_factor(preset);
    settings.hand_filter.min_cutoff_hz = kBalancedHandMinCutoff * factor;
    settings.gaze_filter.min_cutoff_hz = kBalancedGazeMinCutoff * factor;
}

std::optional<SmoothingPreset> detect_smoothing_preset(const Settings& settings) noexcept {
    for (SmoothingPreset preset :
         {SmoothingPreset::Responsive, SmoothingPreset::Balanced, SmoothingPreset::Smooth}) {
        const double factor = preset_factor(preset);
        if (std::abs(settings.hand_filter.min_cutoff_hz - kBalancedHandMinCutoff * factor) < 1e-9 &&
            std::abs(settings.gaze_filter.min_cutoff_hz - kBalancedGazeMinCutoff * factor) < 1e-9) {
            return preset;
        }
    }
    return std::nullopt;
}

} // namespace osi
