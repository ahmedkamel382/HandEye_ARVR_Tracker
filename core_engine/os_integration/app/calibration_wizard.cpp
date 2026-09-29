#include "calibration_wizard.hpp"

#include "app_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace osi::app {
namespace {

constexpr wchar_t kWindowClass[] = L"HandEyeCalibrationWizard";
constexpr UINT_PTR kTimerId = 1;

constexpr double kSettleSeconds = 0.9;
constexpr double kCollectSeconds = 1.2;
constexpr std::size_t kMinCollectSamples = 12;
constexpr double kTargetTimeoutSeconds = 8.0;
constexpr std::size_t kReadyFrames = 15;
constexpr double kReadyTimeoutSeconds = 20.0;
constexpr double kPrepSeconds = 2.5;
constexpr double kTrackingIndicatorSeconds = 0.3;

constexpr COLORREF kBackground = RGB(16, 20, 24);
constexpr COLORREF kText = RGB(235, 238, 242);
constexpr COLORREF kMuted = RGB(150, 158, 168);
constexpr COLORREF kAccent = RGB(76, 194, 255);
constexpr COLORREF kGood = RGB(80, 200, 120);
constexpr COLORREF kWarn = RGB(255, 170, 60);
constexpr COLORREF kBad = RGB(255, 110, 110);

constexpr std::size_t kMaxInbox = 4096;

double record_seconds(int phase) {
    return phase == 0 ? 6.0 : 8.0;
}

HFONT make_font(int points, int dpi, int weight) {
    return CreateFontW(-MulDiv(points, dpi, 72), 0, 0, 0, weight, FALSE, FALSE, FALSE,
                       DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                       DEFAULT_PITCH, L"Segoe UI");
}

void draw_text(HDC dc, const std::wstring& text, RECT rect, HFONT font, COLORREF color,
               UINT format = DT_CENTER | DT_WORDBREAK) {
    SelectObject(dc, font);
    SetTextColor(dc, color);
    DrawTextW(dc, text.c_str(), -1, &rect, format | DT_NOPREFIX);
}

void fill_circle(HDC dc, int x, int y, int radius, COLORREF color) {
    HBRUSH brush = CreateSolidBrush(color);
    HGDIOBJ old_brush = SelectObject(dc, brush);
    HGDIOBJ old_pen = SelectObject(dc, GetStockObject(NULL_PEN));
    Ellipse(dc, x - radius, y - radius, x + radius + 1, y + radius + 1);
    SelectObject(dc, old_pen);
    SelectObject(dc, old_brush);
    DeleteObject(brush);
}

void ring(HDC dc, int x, int y, int radius, int width, COLORREF color, double fraction = 1.0) {
    HPEN pen = CreatePen(PS_SOLID, width, color);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
    if (fraction >= 1.0) {
        Ellipse(dc, x - radius, y - radius, x + radius + 1, y + radius + 1);
    } else if (fraction > 0.0) {
        MoveToEx(dc, x, y - radius, nullptr);
        AngleArc(dc, x, y, static_cast<DWORD>(radius), 90.0f, static_cast<float>(-360.0 * fraction));
    }
    SelectObject(dc, old_brush);
    SelectObject(dc, old_pen);
    DeleteObject(pen);
}

void bar(HDC dc, RECT rect, double fraction, COLORREF color) {
    HBRUSH track = CreateSolidBrush(RGB(40, 46, 54));
    FillRect(dc, &rect, track);
    DeleteObject(track);
    RECT filled = rect;
    filled.right = rect.left + static_cast<LONG>((rect.right - rect.left) * std::clamp(fraction, 0.0, 1.0));
    HBRUSH fill = CreateSolidBrush(color);
    FillRect(dc, &filled, fill);
    DeleteObject(fill);
}

std::size_t deliberate_runs(const EyeStateRecorder& recorder, EyeState state) {
    const auto runs = recorder.runs();
    return static_cast<std::size_t>(std::count_if(runs.begin(), runs.end(), [state](const StateRun& r) {
        return r.state == state && r.duration() >= 0.05;
    }));
}

} // namespace

// ---------------------------------------------------------------- inbox

void SampleInbox::push(const WizardSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.size() < kMaxInbox) {
        samples_.push_back(sample);
    }
}

std::vector<WizardSample> SampleInbox::drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<WizardSample> result;
    result.swap(samples_);
    return result;
}

// ---------------------------------------------------------------- wizard

CalibrationWizard::CalibrationWizard(HINSTANCE instance, WizardKind kind, const ScreenRect& area,
                                     const Settings& current, SampleInbox& inbox, Finished finished)
    : instance_(instance),
      kind_(kind),
      area_(area),
      settings_(current),
      inbox_(inbox),
      finished_(std::move(finished)),
      natural_(current.gestures.glitch_tolerance_seconds),
      left_(current.gestures.glitch_tolerance_seconds),
      right_(current.gestures.glitch_tolerance_seconds),
      result_settings_(current) {}

CalibrationWizard::~CalibrationWizard() {
    if (hwnd_ != nullptr) {
        KillTimer(hwnd_, kTimerId);
        SetWindowLongPtrW(hwnd_, GWLP_USERDATA, 0);
        DestroyWindow(hwnd_);
    }
}

bool CalibrationWizard::open() {
    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance_;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = kWindowClass;
    RegisterClassW(&wc); // fails harmlessly if already registered

    hwnd_ = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kWindowClass, L"HandEye calibration",
                            WS_POPUP, area_.left, area_.top, area_.width, area_.height, nullptr,
                            nullptr, instance_, this);
    if (hwnd_ == nullptr) {
        AppLog::error("calibration window could not be created");
        return false;
    }
    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
    SetTimer(hwnd_, kTimerId, 16, nullptr);
    inbox_.drain();
    enter(Stage::Intro);
    return true;
}

LRESULT CALLBACK CalibrationWizard::window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_NCCREATE) {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
        auto* owner = static_cast<CalibrationWizard*>(create->lpCreateParams);
        owner->hwnd_ = hwnd; // CreateWindowExW has not returned yet
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(owner));
    }
    auto* self = reinterpret_cast<CalibrationWizard*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (self == nullptr) {
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }
    return self->handle(message, wparam, lparam);
}

LRESULT CalibrationWizard::handle(UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
    case WM_TIMER:
        if (wparam == kTimerId) {
            tick();
        }
        return 0;
    case WM_KEYDOWN:
        if (wparam == VK_ESCAPE) {
            finish(false);
        } else if (wparam == VK_SPACE || wparam == VK_RETURN) {
            on_continue();
        }
        return 0;
    case WM_RBUTTONUP:
        finish(false);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd_, &ps);
        RECT client;
        GetClientRect(hwnd_, &client);
        HDC memory = CreateCompatibleDC(dc);
        HBITMAP bitmap = CreateCompatibleBitmap(dc, client.right, client.bottom);
        HGDIOBJ old = SelectObject(memory, bitmap);
        paint(memory, client);
        BitBlt(dc, 0, 0, client.right, client.bottom, memory, 0, 0, SRCCOPY);
        SelectObject(memory, old);
        DeleteObject(bitmap);
        DeleteDC(memory);
        EndPaint(hwnd_, &ps);
        return 0;
    }
    case WM_CLOSE:
        finish(false);
        return 0;
    default:
        return DefWindowProcW(hwnd_, message, wparam, lparam);
    }
}

double CalibrationWizard::stage_seconds() const {
    return std::chrono::duration<double>(Clock::now() - stage_start_).count();
}

std::optional<NormalizedPoint> CalibrationWizard::channel(const WizardSample& sample) const noexcept {
    return kind_ == WizardKind::HandRange ? sample.hand : sample.gaze;
}

void CalibrationWizard::enter(Stage stage) {
    stage_ = stage;
    stage_start_ = Clock::now();
    if (hwnd_ != nullptr) {
        InvalidateRect(hwnd_, nullptr, FALSE);
    }
}

void CalibrationWizard::fail(std::wstring reason) {
    result_text_ = std::move(reason);
    AppLog::warn("calibration failed: " + to_utf8(result_text_));
    enter(Stage::Failed);
}

void CalibrationWizard::on_continue() {
    switch (stage_) {
    case Stage::Intro:
    case Stage::Failed:
        observations_.clear();
        target_index_ = 0;
        ready_streak_ = 0;
        natural_.clear();
        left_.clear();
        right_.clear();
        eye_phase_ = EyePhase::Natural;
        inbox_.drain();
        enter(is_range() ? Stage::Settle : Stage::EyeReady);
        break;
    case Stage::Result:
        finish(true);
        break;
    default:
        break; // ignore key presses while recording
    }
}

void CalibrationWizard::finish(bool saved) {
    if (done_) {
        return;
    }
    done_ = true;
    KillTimer(hwnd_, kTimerId);
    ShowWindow(hwnd_, SW_HIDE);
    Outcome outcome;
    outcome.saved = saved && stage_ == Stage::Result;
    outcome.settings = outcome.saved ? result_settings_ : settings_;
    outcome.message = outcome.saved ? result_text_ : L"Calibration cancelled; settings unchanged.";
    finished_(outcome); // the owner defers our destruction until after this returns
}

void CalibrationWizard::tick() {
    if (done_) {
        return;
    }
    const auto samples = inbox_.drain();
    for (const WizardSample& sample : samples) {
        if ((is_range() && channel(sample)) ||
            (!is_range() && sample.gaze && sample.eye != EyeState::Calibrating)) {
            last_tracked_ = Clock::now();
        }
    }
    const double elapsed = stage_seconds();

    switch (stage_) {
    case Stage::Settle:
        if (elapsed >= kSettleSeconds) {
            observations_.push_back({default_calibration_targets()[target_index_], {}});
            enter(Stage::Collect);
        }
        break;
    case Stage::Collect: {
        auto& collected = observations_.back().samples;
        for (const WizardSample& sample : samples) {
            if (const auto point = channel(sample)) {
                collected.push_back(*point);
            }
        }
        if (elapsed >= kCollectSeconds && collected.size() >= kMinCollectSamples) {
            if (++target_index_ == default_calibration_targets().size()) {
                complete_range();
            } else {
                enter(Stage::Settle);
            }
        } else if (elapsed >= kTargetTimeoutSeconds) {
            fail(std::wstring(L"Target ") + std::to_wstring(target_index_ + 1) + L" received only " +
                 std::to_wstring(collected.size()) + L" tracked frames.\n" +
                 (kind_ == WizardKind::HandRange
                      ? L"Keep your hand inside the camera view."
                      : L"Keep your face visible and well lit.") +
                 L"\nIs the core engine sending tracking data?");
        }
        break;
    }
    case Stage::EyeReady:
        for (const WizardSample& sample : samples) {
            const bool ready = sample.gaze && sample.eye != EyeState::Calibrating;
            ready_streak_ = ready ? ready_streak_ + 1 : 0;
        }
        if (ready_streak_ >= kReadyFrames) {
            eye_phase_ = EyePhase::Natural;
            enter(Stage::EyePrep);
        } else if (elapsed >= kReadyTimeoutSeconds) {
            fail(L"No face was tracked, or the gaze model never finished its own calibration.\n"
                 L"Face the camera in good light. Is the core engine sending gaze data?");
        }
        break;
    case Stage::EyePrep:
        if (elapsed >= kPrepSeconds) {
            enter(Stage::EyeRecord);
        }
        break;
    case Stage::EyeRecord: {
        EyeStateRecorder& recorder = eye_phase_ == EyePhase::Natural ? natural_
                                     : eye_phase_ == EyePhase::Left  ? left_
                                                                     : right_;
        for (const WizardSample& sample : samples) {
            recorder.add(sample.gaze ? sample.eye : EyeState::Neutral, sample.timestamp);
        }
        if (elapsed >= record_seconds(static_cast<int>(eye_phase_))) {
            if (eye_phase_ == EyePhase::Right) {
                complete_eye_timing();
            } else {
                eye_phase_ = eye_phase_ == EyePhase::Natural ? EyePhase::Left : EyePhase::Right;
                enter(Stage::EyePrep);
            }
        }
        break;
    }
    default:
        break;
    }
    InvalidateRect(hwnd_, nullptr, FALSE);
}

void CalibrationWizard::complete_range() {
    const RegionFitResult result = fit_input_region(observations_);
    if (!result.fit) {
        fail(to_wide(result.error));
        return;
    }
    result_settings_ = settings_;
    InputRegion& region = kind_ == WizardKind::HandRange ? result_settings_.mapping.hand_region
                                                         : result_settings_.mapping.gaze_region;
    region = result.fit->region;
    wchar_t text[512];
    std::swprintf(text, 512,
                  L"%ls range measured.\nAverage error: %.1f%% of the screen.\n"
                  L"Input region: x %.3f → %.3f, y %.3f → %.3f",
                  kind_ == WizardKind::HandRange ? L"Hand" : L"Gaze", result.fit->rms_error * 100.0,
                  region.x_min, region.x_max, region.y_min, region.y_max);
    result_text_ = text;
    AppLog::info("calibration fit: " + to_utf8(result_text_));
    enter(Stage::Result);
}

void CalibrationWizard::complete_eye_timing() {
    const EyeTimingResult result =
        recommend_eye_timing(natural_.runs(), left_.runs(), right_.runs(), settings_.gestures);
    if (!result.recommendation) {
        fail(to_wide(result.error));
        return;
    }
    result_settings_ = settings_;
    result_settings_.gestures.wink_onset_seconds = result.recommendation->wink_onset_seconds;
    result_settings_.gestures.closure_onset_seconds = result.recommendation->closure_onset_seconds;
    result_text_ = to_wide(result.recommendation->summary);
    AppLog::info("eye timing: " + result.recommendation->summary);
    enter(Stage::Result);
}

void CalibrationWizard::paint(HDC dc, const RECT& client) {
    const int dpi = static_cast<int>(GetDpiForWindow(hwnd_));
    const double scale = dpi / 96.0;
    HFONT title = make_font(26, dpi, FW_SEMIBOLD);
    HFONT body = make_font(15, dpi, FW_NORMAL);
    HFONT small = make_font(11, dpi, FW_NORMAL);

    HBRUSH background = CreateSolidBrush(kBackground);
    FillRect(dc, &client, background);
    DeleteObject(background);
    SetBkMode(dc, TRANSPARENT);

    const LONG w = client.right;
    const LONG h = client.bottom;
    const RECT title_rect{w / 6, h / 5, w - w / 6, h / 5 + static_cast<LONG>(60 * scale)};
    const RECT body_rect{w / 6, title_rect.bottom + static_cast<LONG>(12 * scale), w - w / 6,
                         h * 11 / 20};
    const RECT footer{w / 6, h - static_cast<LONG>(70 * scale), w - w / 6, h - static_cast<LONG>(20 * scale)};
    const std::wstring hand_or_gaze = kind_ == WizardKind::HandRange ? L"Hand" : L"Gaze";
    const bool tracked = std::chrono::duration<double>(Clock::now() - last_tracked_).count() <
                         kTrackingIndicatorSeconds;

    switch (stage_) {
    case Stage::Intro:
        if (is_range()) {
            draw_text(dc, hand_or_gaze + L" range calibration", title_rect, title, kText);
            draw_text(dc,
                      kind_ == WizardKind::HandRange
                          ? L"Five dots will appear one at a time.\nPoint your index finger at each dot "
                            L"and hold still until its ring fills.\nUse a comfortable, relaxed range of "
                            L"motion — that range will cover the whole screen."
                          : L"Five dots will appear one at a time.\nLook at each dot and keep your head "
                            L"still until its ring fills.",
                      body_rect, body, kMuted);
        } else {
            draw_text(dc, L"Blink & wink timing", title_rect, title, kText);
            draw_text(dc,
                      L"This teaches the system the difference between your natural blinks and "
                      L"deliberate winks.\nYou will blink normally for a few seconds, then wink the "
                      L"eye you use for LEFT click, then the eye you use for RIGHT click.",
                      body_rect, body, kMuted);
        }
        draw_text(dc, L"SPACE to start  ·  ESC to cancel", footer, body, kAccent);
        break;

    case Stage::Settle:
    case Stage::Collect: {
        const auto target = default_calibration_targets()[target_index_];
        const int x = static_cast<int>(std::lround(target.u * (w - 1)));
        const int y = static_cast<int>(std::lround(target.v * (h - 1)));
        const int radius = static_cast<int>(28 * scale);
        if (stage_ == Stage::Settle) {
            const double shrink = std::clamp(stage_seconds() / kSettleSeconds, 0.0, 1.0);
            ring(dc, x, y, static_cast<int>(radius * (2.2 - 1.2 * shrink)), static_cast<int>(3 * scale),
                 kMuted);
        } else {
            const auto& collected = observations_.back().samples;
            const double progress = std::min(stage_seconds() / kCollectSeconds,
                                             static_cast<double>(collected.size()) / kMinCollectSamples);
            ring(dc, x, y, radius, static_cast<int>(2 * scale), RGB(60, 68, 78));
            ring(dc, x, y, radius, static_cast<int>(5 * scale), kAccent, progress);
        }
        fill_circle(dc, x, y, static_cast<int>(7 * scale), kText);

        const RECT hint{w / 4, h * 3 / 10, w - w / 4, h * 3 / 10 + static_cast<LONG>(80 * scale)};
        draw_text(dc,
                  (kind_ == WizardKind::HandRange ? L"Point at the dot and hold still"
                                                  : L"Look at the dot") +
                      std::wstring(L"\nTarget ") + std::to_wstring(target_index_ + 1) + L" of " +
                      std::to_wstring(default_calibration_targets().size()),
                  hint, body, kMuted);
        draw_text(dc,
                  tracked ? hand_or_gaze + L" tracked"
                          : (kind_ == WizardKind::HandRange ? L"No hand detected" : L"No face detected") +
                                std::wstring(L" — is the core engine sending tracking data?"),
                  footer, small, tracked ? kGood : kWarn);
        break;
    }

    case Stage::EyeReady:
        draw_text(dc, L"Look at the screen", title_rect, title, kText);
        draw_text(dc, L"Keep your eyes open and your face toward the camera while the eye model "
                      L"settles.",
                  body_rect, body, kMuted);
        draw_text(dc, tracked ? L"Face tracked" : L"Waiting for a tracked face…", footer, small,
                  tracked ? kGood : kWarn);
        break;

    case Stage::EyePrep:
    case Stage::EyeRecord: {
        static const wchar_t* const titles[] = {L"Blink naturally", L"Wink for LEFT click",
                                                L"Wink for RIGHT click"};
        static const wchar_t* const bodies[] = {
            L"Just look at the screen and blink the way you normally do.",
            L"Close the eye you use for LEFT click for about half a second, then open it.\n"
            L"Do this three times.",
            L"Close the eye you use for RIGHT click for about half a second, then open it.\n"
            L"Do this three times."};
        const auto phase = static_cast<std::size_t>(eye_phase_);
        draw_text(dc, titles[phase], title_rect, title, kText);
        draw_text(dc, bodies[phase], body_rect, body, kMuted);
        const RECT progress{w / 4, h * 13 / 20, w - w / 4, h * 13 / 20 + static_cast<LONG>(10 * scale)};
        const RECT status{w / 6, progress.bottom + static_cast<LONG>(16 * scale), w - w / 6,
                          progress.bottom + static_cast<LONG>(60 * scale)};
        if (stage_ == Stage::EyePrep) {
            bar(dc, progress, stage_seconds() / kPrepSeconds, kMuted);
            draw_text(dc, L"Get ready…", status, body, kMuted);
        } else {
            bar(dc, progress, stage_seconds() / record_seconds(static_cast<int>(phase)), kAccent);
            if (eye_phase_ != EyePhase::Natural) {
                const EyeState wanted = eye_phase_ == EyePhase::Left ? EyeState::LeftWink : EyeState::RightWink;
                const std::size_t seen = deliberate_runs(eye_phase_ == EyePhase::Left ? left_ : right_, wanted);
                draw_text(dc, L"Winks recognised: " + std::to_wstring(seen), status, body,
                          seen > 0 ? kGood : kMuted);
            } else {
                draw_text(dc, L"Recording…", status, body, kMuted);
            }
        }
        draw_text(dc, tracked ? L"Face tracked" : L"Face not tracked", footer, small,
                  tracked ? kGood : kWarn);
        break;
    }

    case Stage::Result:
        draw_text(dc, L"Calibration complete", title_rect, title, kGood);
        draw_text(dc, result_text_, body_rect, body, kText);
        draw_text(dc, L"SPACE to save  ·  ESC to discard", footer, body, kAccent);
        break;

    case Stage::Failed:
        draw_text(dc, L"Calibration did not succeed", title_rect, title, kBad);
        draw_text(dc, result_text_, body_rect, body, kText);
        draw_text(dc, L"SPACE to try again  ·  ESC to close", footer, body, kAccent);
        break;
    }

    DeleteObject(title);
    DeleteObject(body);
    DeleteObject(small);
}

} // namespace osi::app
