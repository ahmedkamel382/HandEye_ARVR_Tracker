#pragma once

#include "os_integration/calibration.hpp"
#include "os_integration/settings.hpp"
#include "os_integration/types.hpp"

#include <windows.h>

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace osi::app {

enum class WizardKind { HandRange, GazeRange, EyeTiming };

// One stabilized frame as the wizard sees it.
struct WizardSample {
    double timestamp = 0.0; // frame time, seconds
    std::optional<NormalizedPoint> hand;
    std::optional<NormalizedPoint> gaze;
    EyeState eye = EyeState::Neutral;
};

// Hand-off from the frame thread to the wizard's UI thread.
class SampleInbox {
public:
    void push(const WizardSample& sample);
    std::vector<WizardSample> drain();

private:
    std::mutex mutex_;
    std::vector<WizardSample> samples_;
};

// Full-screen, topmost calibration overlay ("Look here", "Blink now").
// Keyboard: SPACE / Enter continues, Esc (or right-click) cancels.
class CalibrationWizard {
public:
    struct Outcome {
        bool saved = false;
        Settings settings;    // updated copy when saved
        std::wstring message; // shown to the user by the tray app
    };
    using Finished = std::function<void(const Outcome&)>;

    // `area` is the mapping target (a monitor or the whole desktop), so
    // targets land exactly where the mapper will later put the cursor.
    CalibrationWizard(HINSTANCE instance, WizardKind kind, const ScreenRect& area,
                      const Settings& current, SampleInbox& inbox, Finished finished);
    ~CalibrationWizard();
    CalibrationWizard(const CalibrationWizard&) = delete;
    CalibrationWizard& operator=(const CalibrationWizard&) = delete;

    bool open();

private:
    using Clock = std::chrono::steady_clock;

    enum class Stage {
        Intro,
        Settle,       // range: target shown, waiting for the user to get there
        Collect,      // range: recording samples at the target
        EyeReady,     // eyes: waiting for a face and gaze_intent's own calibration
        EyePrep,      // eyes: instruction for the next recording
        EyeRecord,    // eyes: recording states
        Result,
        Failed
    };
    enum class EyePhase { Natural, Left, Right };

    static LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
    LRESULT handle(UINT message, WPARAM wparam, LPARAM lparam);

    void tick();
    void on_continue();
    void finish(bool saved);
    void enter(Stage stage);
    void fail(std::wstring reason);
    void complete_range();
    void complete_eye_timing();
    void paint(HDC dc, const RECT& client);
    double stage_seconds() const;
    bool is_range() const noexcept { return kind_ != WizardKind::EyeTiming; }
    std::optional<NormalizedPoint> channel(const WizardSample& sample) const noexcept;

    HINSTANCE instance_;
    WizardKind kind_;
    ScreenRect area_;
    Settings settings_;
    SampleInbox& inbox_;
    Finished finished_;
    HWND hwnd_ = nullptr;
    bool done_ = false;

    Stage stage_ = Stage::Intro;
    Clock::time_point stage_start_ = Clock::now();
    Clock::time_point last_tracked_ = Clock::time_point{};

    // Range calibration
    std::size_t target_index_ = 0;
    std::vector<TargetObservation> observations_;

    // Eye timing
    EyePhase eye_phase_ = EyePhase::Natural;
    std::size_t ready_streak_ = 0;
    EyeStateRecorder natural_;
    EyeStateRecorder left_;
    EyeStateRecorder right_;

    std::wstring result_text_;
    Settings result_settings_;
};

} // namespace osi::app
