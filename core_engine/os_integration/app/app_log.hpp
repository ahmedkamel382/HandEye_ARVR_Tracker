#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace osi::app {

// Minimal thread-safe append-only log file. Every line is flushed so the
// log survives a crash.
class AppLog {
public:
    static void open(const std::filesystem::path& path);
    static void info(std::string_view message);
    static void warn(std::string_view message);
    static void error(std::string_view message);
    static std::filesystem::path path();
};

std::string to_utf8(std::wstring_view text);
std::wstring to_wide(std::string_view text);

} // namespace osi::app
