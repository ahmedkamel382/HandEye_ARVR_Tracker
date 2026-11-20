// HandEyeTester.exe: double-click launcher for the HandEye tester.
//
// Finds the repository (walking up from this executable), the project's
// Python environment (.venv, created by setup.bat) and the tester script,
// then starts the tester without a console window. If something is missing
// or the tester fails to start, it says what to do in plain words.

#include <windows.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr wchar_t kTitle[] = L"HandEye Tester";
const fs::path kScript = fs::path(L"core_engine") / L"os_integration" / L"tester" / L"handeye_tester.py";

fs::path executable_dir() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length < buffer.size()) {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return fs::path(buffer).parent_path();
}

// The repository is the first ancestor that contains the tester script.
fs::path find_repository(const fs::path& start) {
    std::error_code ec;
    for (fs::path dir = start; !dir.empty(); dir = dir.parent_path()) {
        if (fs::exists(dir / kScript, ec)) {
            return dir;
        }
        if (dir == dir.root_path()) {
            break;
        }
    }
    return {};
}

fs::path log_path() {
    wchar_t buffer[MAX_PATH] = {};
    const DWORD length = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, MAX_PATH);
    const fs::path base = length > 0 && length < MAX_PATH ? fs::path(buffer) : fs::temp_directory_path();
    return base / L"HandEyeTracker" / L"tester.log";
}

std::wstring last_log_lines(const fs::path& path, std::size_t lines) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::vector<std::string> all;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        all.push_back(line);
    }
    std::string text;
    for (std::size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); ++i) {
        text += all[i] + "\n";
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
    return wide;
}

int fail(const std::wstring& message) {
    MessageBoxW(nullptr, message.c_str(), kTitle, MB_ICONERROR | MB_OK);
    return 1;
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    const fs::path here = executable_dir();
    const fs::path repo = find_repository(here);
    if (repo.empty()) {
        return fail(L"Could not find the HandEye project around this program.\n\n"
                    L"HandEyeTester.exe must stay inside the project folder (it is built into "
                    L"out\\build\\...). Looked upward from:\n" + here.wstring());
    }

    std::error_code ec;
    fs::path python = repo / L".venv" / L"Scripts" / L"pythonw.exe";
    if (!fs::exists(python, ec)) {
        python = repo / L".venv" / L"Scripts" / L"python.exe";
    }
    if (!fs::exists(python, ec)) {
        return fail(L"The project's Python environment was not found:\n" + (repo / L".venv").wstring() +
                    L"\n\nRun setup.bat in the project folder once, then start the tester again.");
    }

    // Tell the tester where things are; it also works without these.
    SetEnvironmentVariableW(L"AI_PIPELINE_ROOT", (repo / L"ai_pipeline").c_str());
    const fs::path dll = here / L"os_integration_capi.dll";
    if (fs::exists(dll, ec)) {
        SetEnvironmentVariableW(L"OSI_CAPI_DLL", dll.c_str());
    }
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");

    std::wstring command = L"\"" + python.wstring() + L"\" \"" + (repo / kScript).wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                        repo.c_str(), &startup, &process)) {
        return fail(L"Python could not be started:\n" + python.wstring());
    }
    CloseHandle(process.hThread);

    // If the tester dies straight away (missing package, broken environment),
    // show why instead of failing silently.
    DWORD exit_code = STILL_ACTIVE;
    if (WaitForSingleObject(process.hProcess, 6000) == WAIT_OBJECT_0 &&
        GetExitCodeProcess(process.hProcess, &exit_code) && exit_code != 0) {
        std::wostringstream message;
        message << L"The tester stopped while starting (exit code " << exit_code << L").\n\n"
                << last_log_lines(log_path(), 12) << L"\nFull log: " << log_path().wstring()
                << L"\n\nIf a package is missing, run setup.bat in the project folder.";
        CloseHandle(process.hProcess);
        return fail(message.str());
    }
    CloseHandle(process.hProcess);
    return 0;
}
