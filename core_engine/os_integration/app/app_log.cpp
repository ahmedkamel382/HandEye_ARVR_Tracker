#include "app_log.hpp"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <mutex>

namespace osi::app {
namespace {

std::mutex g_mutex;
std::ofstream g_file;
std::filesystem::path g_path;

void write(const char* level, std::string_view message) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count() % 1000;
    std::tm local{};
    localtime_s(&local, &seconds);
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d %02d:%02d:%02d.%03d", local.tm_year + 1900,
                  local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec,
                  static_cast<int>(millis));

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file.is_open()) {
        g_file << stamp << ' ' << level << ' ' << message << '\n';
        g_file.flush();
    }
#if !defined(NDEBUG)
    OutputDebugStringA((std::string(level) + ' ' + std::string(message) + '\n').c_str());
#endif
}

} // namespace

void AppLog::open(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_path = path;
    g_file.open(path, std::ios::app);
}

void AppLog::info(std::string_view message) { write("INFO ", message); }
void AppLog::warn(std::string_view message) { write("WARN ", message); }
void AppLog::error(std::string_view message) { write("ERROR", message); }

std::filesystem::path AppLog::path() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_path;
}

std::string to_utf8(std::wstring_view text) {
    if (text.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                                         nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size,
                        nullptr, nullptr);
    return result;
}

std::wstring to_wide(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                                         nullptr, 0);
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size);
    return result;
}

} // namespace osi::app
