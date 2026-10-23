// os_demo: Phase 1 proof that the OS layer controls the Windows cursor.
//
//   os_demo list                         monitors and the geometry in use
//   os_demo trace [options]              drive the cursor along a shape through
//                                        the full OsController pipeline and
//                                        measure accuracy and latency
//   os_demo selftest                     open a private target window, move and
//                                        click inside it, and verify every event
//                                        arrived at the exact pixel
//
// trace options:
//   --pattern circle|square|sweep   (default circle)
//   --seconds N                     (default 5)
//   --fps N                         (default 60)
//   --screen primary|all            (default primary)
//   --dry-run                       compute everything, send nothing
//
// Esc stops trace and selftest immediately; so does moving the real mouse.

#include "os_integration/os_controller.hpp"
#include "os_integration/win32_input.hpp"

#include <windows.h>
#include <timeapi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace osi;
using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.141592653589793238462643383279502884;

bool escape_pressed() {
    return (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0;
}

void print_rect(const char* label, const ScreenRect& r) {
    std::printf("%-18s x=%d y=%d  %dx%d\n", label, r.left, r.top, r.width, r.height);
}

int list_monitors() {
    const auto monitors = win32::enumerate_monitors();
    std::printf("%zu monitor(s)\n", monitors.size());
    for (const auto& m : monitors) {
        std::printf("  %ls%s\n", m.device_name.c_str(), m.primary ? "  (primary)" : "");
        print_rect("    bounds", m.bounds);
        print_rect("    work area", m.work_area);
    }
    print_rect("virtual desktop", win32::virtual_desktop_rect());
    return 0;
}

// Point on the shape at phase p in [0, 1), in normalized camera space.
NormalizedPoint pattern_point(const std::string& pattern, double p) {
    if (pattern == "square") {
        const double side = std::fmod(p * 4.0, 4.0);
        const double f = side - std::floor(side);
        const double lo = 0.1;
        const double hi = 0.9;
        switch (static_cast<int>(side)) {
        case 0:
            return {lo + (hi - lo) * f, lo};
        case 1:
            return {hi, lo + (hi - lo) * f};
        case 2:
            return {hi - (hi - lo) * f, hi};
        default:
            return {lo, hi - (hi - lo) * f};
        }
    }
    if (pattern == "sweep") {
        return {0.5 + 0.45 * std::sin(2.0 * kPi * p), 0.5 + 0.35 * std::sin(6.0 * kPi * p)};
    }
    return {0.5 + 0.35 * std::cos(2.0 * kPi * p), 0.5 + 0.35 * std::sin(2.0 * kPi * p)};
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

int trace(int argc, char** argv) {
    std::string pattern = "circle";
    double seconds = 5.0;
    double fps = 60.0;
    ScreenTarget target = ScreenTarget::PrimaryMonitor;
    bool dry_run = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--pattern" && has_value) {
            pattern = argv[++i];
        } else if (arg == "--seconds" && has_value) {
            seconds = std::atof(argv[++i]);
        } else if (arg == "--fps" && has_value) {
            fps = std::atof(argv[++i]);
        } else if (arg == "--screen" && has_value) {
            target = std::string(argv[++i]) == "all" ? ScreenTarget::VirtualDesktop
                                                      : ScreenTarget::PrimaryMonitor;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!(seconds > 0.0 && seconds <= 600.0) || !(fps >= 1.0 && fps <= 1000.0) ||
        (pattern != "circle" && pattern != "square" && pattern != "sweep")) {
        std::fprintf(stderr, "invalid --pattern, --seconds or --fps\n");
        return 2;
    }

    Settings settings;
    settings.mapping.hand_region = {0.0, 0.0, 1.0, 1.0}; // the pattern already spans the screen
    settings.mapping.cursor_source = CursorSourceMode::HandOnly;
    settings.mapping.screen_target = target;
    const ScreenGeometry geometry = win32::query_screen_geometry(target);

    win32::InputInjector injector;
    RecordingInputSink recorder;
    IInputSink& sink = dry_run ? static_cast<IInputSink&>(recorder) : injector;
    OsController controller(settings, geometry, sink);
    controller.set_active(true);

    print_rect("virtual desktop", geometry.virtual_desktop);
    print_rect("cursor target", controller.mapper().geometry().target);
    std::printf("pattern=%s seconds=%.1f fps=%.0f %s\n", pattern.c_str(), seconds, fps,
                dry_run ? "(dry run: nothing is sent)" : "- press Esc or move the mouse to stop");

    const auto original = win32::cursor_position();
    const auto period = std::chrono::duration<double>(1.0 / fps);
    const auto start = Clock::now();
    std::vector<double> update_us;
    std::vector<double> error_px;
    std::optional<PixelPoint> expected;
    std::size_t frames = 0;
    std::size_t refused = 0;
    const char* stop_reason = "finished";

    timeBeginPeriod(1);
    for (auto next = start;; next += std::chrono::duration_cast<Clock::duration>(period)) {
        std::this_thread::sleep_until(next);
        const double t = std::chrono::duration<double>(Clock::now() - start).count();
        if (t >= seconds) {
            break;
        }
        if (escape_pressed()) {
            stop_reason = "Esc pressed";
            break;
        }
        if (!dry_run && expected) {
            // Where Windows put the cursor for the previous frame.
            const auto actual = win32::cursor_position();
            if (actual) {
                const double error = std::hypot(static_cast<double>(actual->x) - expected->x,
                                                static_cast<double>(actual->y) - expected->y);
                if (error > 3.0) {
                    stop_reason = "physical mouse movement detected";
                    break;
                }
                error_px.push_back(error);
            }
        }

        FramePayload frame;
        frame.timestamp_seconds = t;
        frame.hand.point = pattern_point(pattern, std::fmod(t / 3.0, 1.0));
        const auto before = Clock::now();
        const FrameReport report = controller.update(frame);
        update_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - before).count());
        refused += report.injection_failed ? 1 : 0;
        if (report.moved) {
            expected = report.cursor;
        }
        ++frames;
    }
    timeEndPeriod(1);

    if (!dry_run && original && stop_reason != std::string("physical mouse movement detected")) {
        injector.move_to(*original, controller.mapper().to_absolute(*original));
    }

    std::printf("\nstopped: %s\n", stop_reason);
    std::printf("frames=%zu  refused_by_windows=%zu\n", frames, refused);
    std::printf("OsController::update incl. SendInput (us): mean=%.1f p50=%.1f p99=%.1f max=%.1f\n",
                mean(update_us), percentile(update_us, 0.5), percentile(update_us, 0.99), percentile(update_us, 1.0));
    if (dry_run) {
        std::printf("recorded events=%zu (first: %d,%d)\n", recorder.events.size(),
                    recorder.events.empty() ? 0 : recorder.events.front().pixel.x,
                    recorder.events.empty() ? 0 : recorder.events.front().pixel.y);
    } else {
        std::printf("cursor position error (px): samples=%zu p50=%.2f max=%.2f\n", error_px.size(),
                    percentile(error_px, 0.5), percentile(error_px, 1.0));
    }
    return refused == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- selftest

struct ReceivedEvent {
    UINT message;
    POINT client;
};

std::vector<ReceivedEvent> g_received;

LRESULT CALLBACK target_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
        g_received.push_back({message, {static_cast<short>(LOWORD(lparam)),
                                        static_cast<short>(HIWORD(lparam))}});
        return 0;
    case WM_MOUSEMOVE:
        if ((wparam & MK_LBUTTON) != 0) {
            g_received.push_back({message, {static_cast<short>(LOWORD(lparam)),
                                            static_cast<short>(HIWORD(lparam))}});
        }
        return 0;
    case WM_MOUSEACTIVATE:
        return MA_NOACTIVATE;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        RECT rc;
        GetClientRect(hwnd, &rc);
        FillRect(dc, &rc, static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH)));
        SetBkMode(dc, TRANSPARENT);
        DrawTextW(dc, L"os_demo self-test: automated clicks land here. Esc aborts.", -1, &rc,
                  DT_CENTER | DT_TOP | DT_SINGLELINE);
        EndPaint(hwnd, &ps);
        return 0;
    }
    default:
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }
}

void pump(int milliseconds) {
    const auto until = Clock::now() + std::chrono::milliseconds(milliseconds);
    while (Clock::now() < until) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int selftest() {
    const HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = target_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_CROSS);
    wc.lpszClassName = L"HandEyeOsDemoTarget";
    RegisterClassW(&wc);

    const ScreenRect primary = win32::primary_monitor_rect();
    const int width = 640;
    const int height = 400;
    HWND hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, wc.lpszClassName,
                                L"os_demo self-test", WS_POPUP | WS_BORDER | WS_VISIBLE,
                                primary.left + (primary.width - width) / 2,
                                primary.top + (primary.height - height) / 2, width, height, nullptr,
                                nullptr, instance, nullptr);
    if (hwnd == nullptr) {
        std::fprintf(stderr, "CreateWindow failed: %ls\n", win32::last_error_message(GetLastError()).c_str());
        return 1;
    }
    pump(300);

    const ScreenMapper mapper(win32::query_screen_geometry(ScreenTarget::PrimaryMonitor));
    win32::InputInjector injector;
    const auto original = win32::cursor_position();
    RECT client;
    GetClientRect(hwnd, &client);

    struct Step {
        const char* name;
        POINT client;
        enum { LeftClick, RightClick, MiddleClick, DragTo } kind;
        POINT drag_end;
    };
    const Step steps[] = {
        {"left click, top-left", {40, 40}, Step::LeftClick, {}},
        {"left click, top-right", {client.right - 41, 40}, Step::LeftClick, {}},
        {"left click, bottom-right", {client.right - 41, client.bottom - 41}, Step::LeftClick, {}},
        {"right click, bottom-left", {40, client.bottom - 41}, Step::RightClick, {}},
        {"middle click, centre", {client.right / 2, client.bottom / 2}, Step::MiddleClick, {}},
        {"drag", {100, 200}, Step::DragTo, {500, 250}},
    };

    const auto move_to_client = [&](POINT p) {
        POINT screen = p;
        ClientToScreen(hwnd, &screen);
        const PixelPoint pixel{screen.x, screen.y};
        return injector.move_to(pixel, mapper.to_absolute(pixel));
    };
    const auto same = [](POINT a, POINT b) { return std::abs(a.x - b.x) <= 1 && std::abs(a.y - b.y) <= 1; };
    const auto received = [&](UINT message, POINT at) {
        return std::any_of(g_received.begin(), g_received.end(), [&](const ReceivedEvent& e) {
            return e.message == message && same(e.client, at);
        });
    };

    int failed = 0;
    for (const Step& step : steps) {
        if (escape_pressed()) {
            std::printf("aborted: Esc pressed\n");
            failed = -1;
            break;
        }
        g_received.clear();
        bool sent = move_to_client(step.client);
        pump(60);
        bool ok = false;
        switch (step.kind) {
        case Step::LeftClick:
            sent &= injector.click(MouseButton::Left);
            pump(80);
            ok = received(WM_LBUTTONDOWN, step.client) && received(WM_LBUTTONUP, step.client);
            break;
        case Step::RightClick:
            sent &= injector.click(MouseButton::Right);
            pump(80);
            ok = received(WM_RBUTTONDOWN, step.client) && received(WM_RBUTTONUP, step.client);
            break;
        case Step::MiddleClick:
            sent &= injector.click(MouseButton::Middle);
            pump(80);
            ok = received(WM_MBUTTONDOWN, step.client) && received(WM_MBUTTONUP, step.client);
            break;
        case Step::DragTo:
            sent &= injector.button(MouseButton::Left, ButtonAction::Down);
            pump(60);
            for (int i = 1; i <= 10; ++i) {
                sent &= move_to_client({step.client.x + (step.drag_end.x - step.client.x) * i / 10,
                                        step.client.y + (step.drag_end.y - step.client.y) * i / 10});
                pump(15);
            }
            sent &= injector.button(MouseButton::Left, ButtonAction::Up);
            pump(80);
            ok = received(WM_LBUTTONDOWN, step.client) && received(WM_MOUSEMOVE, step.drag_end) &&
                 received(WM_LBUTTONUP, step.drag_end);
            break;
        }
        ok = ok && sent;
        failed += ok ? 0 : 1;
        std::printf("%s  %-26s at (%ld, %ld)%s\n", ok ? "PASS" : "FAIL", step.name, step.client.x,
                    step.client.y, sent ? "" : "  [SendInput refused]");
    }

    if (original) {
        injector.move_to(*original, mapper.to_absolute(*original));
    }
    DestroyWindow(hwnd);
    pump(50);
    if (failed < 0) {
        return 1;
    }
    std::printf("SELFTEST %s (%d failed)\n", failed == 0 ? "PASSED" : "FAILED", failed);
    return failed == 0 ? 0 : 1;
}

void usage() {
    std::printf("usage: os_demo list | trace [--pattern circle|square|sweep] [--seconds N] "
                "[--fps N] [--screen primary|all] [--dry-run] | selftest\n");
}

} // namespace

int main(int argc, char** argv) {
    win32::enable_per_monitor_dpi_awareness();
    const std::string command = argc > 1 ? argv[1] : "";
    if (command == "list") {
        return list_monitors();
    }
    if (command == "trace") {
        return trace(argc, argv);
    }
    if (command == "selftest") {
        return selftest();
    }
    usage();
    return command.empty() ? 0 : 2;
}
