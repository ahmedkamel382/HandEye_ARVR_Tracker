// The bridge between the engine and Engineer 5's OS integration (core_engine/os_integration).
//
//   TrackingResult (raw, from the AI thread)
//      -> DSP smoothing (Eng 4's 1-Euro filter, hand and gaze each get their own)
//      -> osi::FramePayload { stabilised point + raw state, per tracker }
//      -> osi::app::TrayApp (Eng 5): screen mapping, gestures, clicks, SendInput, tray icon, calibration wizard
//
// Everything after the DSP filter (pointer choice, margins, pinch/wink/fist handling, pause, mouse output)
// now lives in Eng 5's module. This file only converts and forwards.
//
// Windows only (CORE_HAVE_OS_INTEGRATION, set by CMake). Elsewhere a no-op stub keeps the engine compiling.
#pragma once
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dsp_adapter.hpp"
#include "euro_builtin.hpp"
#include "tracking_types.hpp"

#ifdef CORE_HAVE_OS_INTEGRATION
#include "app_log.hpp"
#include "tray_app.hpp"
#include "os_integration/settings.hpp"
#include "os_integration/win32_input.hpp"
#include <shlobj.h>
#endif

namespace core {

struct OsBridgeConfig {
    bool real_mouse   = false;   // false = dry run: Eng 5's module logs the mouse actions instead of sending them
    bool start_active = false;   // false = starts PAUSED (Ctrl+Alt+P, tray menu, or a 1 s eye closure to resume)
    bool use_filter   = true;    // --no-filter
    bool use_dsp      = true;    // Eng 4's filter when compiled in; false = built-in 1-Euro (--filter builtin)
    std::vector<double> hand_filter, gaze_filter;   // optional "min_cutoff,beta,d_cutoff" overrides of the saved settings
};

#ifdef CORE_HAVE_OS_INTEGRATION

class OsBridge {
public:
    // Throws std::runtime_error if another instance is already running (two would fight over the cursor).
    explicit OsBridge(const OsBridgeConfig& cfg) : cfg_(cfg) {
        single_ = CreateMutexW(nullptr, TRUE, L"Local\\HandEyeTrackerTray");
        if (single_ == nullptr || GetLastError() == ERROR_ALREADY_EXISTS) {
            if (single_) CloseHandle(single_);
            single_ = nullptr;
            throw std::runtime_error("HandEye Tracker is already running (see the notification area)");
        }
        osi::win32::enable_per_monitor_dpi_awareness();

        osi::app::TrayOptions opt;
        opt.dry_run = !cfg.real_mouse;
        opt.start_active = cfg.start_active;
        opt.settings_path = knownFolder(FOLDERID_RoamingAppData) / L"HandEyeTracker" / L"os_integration.ini";
        opt.log_path = knownFolder(FOLDERID_LocalAppData) / L"HandEyeTracker" / L"handeye_tray.log";
        // The tray's Smoothing menu changes the filter settings: pick them up here.
        opt.on_settings_changed = [this](const osi::Settings& s) {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_hand_ = s.hand_filter;
            pending_gaze_ = s.gaze_filter;
            pending_.store(true);
        };

        // Initial filter tuning: command line > saved settings > defaults.
        const osi::Settings saved = osi::load_settings_file(opt.settings_path).settings;
        osi::FilterSettings h = saved.hand_filter, g = saved.gaze_filter;
        if (cfg.hand_filter.size() == 3) h = {cfg.hand_filter[0], cfg.hand_filter[1], cfg.hand_filter[2]};
        if (cfg.gaze_filter.size() == 3) g = {cfg.gaze_filter[0], cfg.gaze_filter[1], cfg.gaze_filter[2]};
        buildFilters(h, g);          // throws std::invalid_argument on a bad configuration

        tray_.reset(new osi::app::TrayApp(GetModuleHandleW(nullptr), opt));
        ui_ = std::thread([this] { tray_->run(); ui_done_.store(true); });     // tray icon, menu, wizard
    }

    ~OsBridge() { stop(); if (single_) CloseHandle(single_); }
    OsBridge(const OsBridge&) = delete;
    OsBridge& operator=(const OsBridge&) = delete;

    // One call per new TrackingResult, from the main loop thread.
    void update(const TrackingResult& r) {
        applyPendingFilters();
        const double t = double(r.ts_ms) / 1000.0;
        osi::FramePayload f;
        f.timestamp_seconds = t;
        convert(r.hand, hand_flt_, hand_had_, f.hand, t);
        convert(r.gaze, gaze_flt_, gaze_had_, f.gaze, t);
        tray_->submit_frame(f);
        ++sent_;
    }

    // True once the user chose Exit in the tray menu (or the tray failed to start): the engine should stop.
    bool exited() const { return ui_done_.load(); }

    // Releases any held button and closes the tray. Safe to call twice.
    void stop() {
        if (!ui_.joinable()) return;
        while (!ui_done_.load()) { tray_->request_exit(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        ui_.join();
    }

    const char* name() const { return cfg_.real_mouse ? "OS integration (REAL MOUSE)" : "OS integration (dry run, mouse untouched)"; }
    const std::string& filterDescription() const { return filter_desc_; }
    uint64_t framesSent() const { return sent_; }

private:
    static std::filesystem::path knownFolder(REFKNOWNFOLDERID id) {
        PWSTR p = nullptr;
        std::filesystem::path out;
        if (SUCCEEDED(SHGetKnownFolderPath(id, 0, nullptr, &p))) out = p;
        CoTaskMemFree(p);
        return out.empty() ? std::filesystem::temp_directory_path() : out;
    }

    std::unique_ptr<IPointFilter> makeFilter(const osi::FilterSettings& s) const {
#ifdef CORE_HAVE_DSP_FILTER
        if (cfg_.use_dsp) return std::unique_ptr<IPointFilter>(new DspFilterAdapter(dsp::FilterConfig{s.min_cutoff_hz, s.beta, s.derivative_cutoff_hz}));
#endif
        return std::unique_ptr<IPointFilter>(new OneEuroPoint(s.min_cutoff_hz, s.beta, s.derivative_cutoff_hz));
    }

    void buildFilters(const osi::FilterSettings& h, const osi::FilterSettings& g) {
        std::unique_ptr<IPointFilter> nh = makeFilter(h), ng = makeFilter(g);     // build both first: all or nothing
        hand_flt_ = std::move(nh); gaze_flt_ = std::move(ng);
        hand_had_ = gaze_had_ = false;
        char buf[200];
#ifdef CORE_HAVE_DSP_FILTER
        const char* kind = cfg_.use_dsp ? "dsp_filter (Eng 4)" : "built-in 1-Euro";
#else
        const char* kind = "built-in 1-Euro";
#endif
        std::snprintf(buf, sizeof buf, "%s hand %.3g/%.3g/%.3g gaze %.3g/%.3g/%.3g", kind,
                      h.min_cutoff_hz, h.beta, h.derivative_cutoff_hz, g.min_cutoff_hz, g.beta, g.derivative_cutoff_hz);
        filter_desc_ = buf;
    }

    void applyPendingFilters() {
        if (!pending_.exchange(false)) return;
        osi::FilterSettings h{}, g{};
        { std::lock_guard<std::mutex> lk(pending_mu_); h = pending_hand_; g = pending_gaze_; }
        try { buildFilters(h, g); }
        catch (const std::exception& e) { osi::app::AppLog::error(std::string("filter settings not applied: ") + e.what()); }
    }

    // Detected -> smoothed point. Lost -> no point, and the filter forgets its history (no glide across the gap).
    // The state integer is forwarded raw: states bypass the DSP stage.
    void convert(const ModuleOut& m, std::unique_ptr<IPointFilter>& flt, bool& had, osi::TrackerSample& out, double t) {
        out.state = m.ok ? m.state : 0;
        if (!m.detected()) {
            if (had) { flt->reset(); had = false; }
            return;
        }
        double x = m.x, y = m.y;
        if (cfg_.use_filter) flt->filter(x, y, t);
        out.point = osi::NormalizedPoint{std::min(1.0, std::max(0.0, x)), std::min(1.0, std::max(0.0, y))};
        had = true;
    }

    OsBridgeConfig cfg_;
    HANDLE single_ = nullptr;
    std::unique_ptr<osi::app::TrayApp> tray_;
    std::thread ui_;
    std::atomic<bool> ui_done_{false};

    std::unique_ptr<IPointFilter> hand_flt_, gaze_flt_;
    bool hand_had_ = false, gaze_had_ = false;
    std::string filter_desc_;
    uint64_t sent_ = 0;

    std::mutex pending_mu_;
    std::atomic<bool> pending_{false};
    osi::FilterSettings pending_hand_{}, pending_gaze_{};
};

#else   // not Windows: nothing to drive. Tracking still runs; no mouse output.

class OsBridge {
public:
    explicit OsBridge(const OsBridgeConfig&) {
        std::fprintf(stderr, "[os] OS integration is Windows-only; tracking runs but no mouse output is produced.\n");
    }
    void update(const TrackingResult&) { ++sent_; }
    bool exited() const { return false; }
    void stop() {}
    const char* name() const { return "none (OS integration is Windows-only)"; }
    std::string filterDescription() const { return "n/a"; }
    uint64_t framesSent() const { return sent_; }
private:
    uint64_t sent_ = 0;
};

#endif

} // namespace core
