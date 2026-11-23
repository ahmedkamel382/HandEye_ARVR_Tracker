#pragma once

#include "os_integration/cursor_stabilizer.hpp"
#include "os_integration/gesture_interpreter.hpp"
#include "os_integration/input_sink.hpp"
#include "os_integration/screen_mapper.hpp"
#include "os_integration/settings.hpp"

#include <array>
#include <optional>

namespace osi {

enum class FrameStatus : std::uint8_t {
    Processed,
    RejectedTimestamp // non-finite or not newer than the previous frame; nothing changed
};

struct FrameReport {
    FrameStatus status = FrameStatus::Processed;
    CursorSource source = CursorSource::None;
    std::optional<PixelPoint> cursor; // where the cursor is (or would be, when paused)
    bool moved = false;               // Windows accepted a cursor move this frame
    ActionList executed;              // button actions Windows accepted this frame
    bool pause_toggled = false;       // a gesture toggled pause this frame
    bool active = false;              // state after this frame
    bool inhibited = false;
    bool injection_failed = false;    // Windows refused at least one event
    bool release_pending = false;     // a button is still down in Windows because its release was refused
    bool gaze_calibrating = false;
    bool hand_held = false;           // the hand point used this frame is a cached HELD point, not a new observation
    std::uint32_t unknown_states = 0; // state integers outside the contract, treated as neutral
};

// Phase 4 of the frame's journey: stabilized X/Y + raw states in, OS mouse
// events out. Call update() once per new observation from the core loop.
//
//   osi::RecordingInputSink / osi::win32::InputInjector sink;
//   osi::OsController controller(settings, geometry, sink);
//   controller.set_active(true);
//   auto report = controller.update(payload);
//
// Starts paused. While paused or inhibited, gestures are still tracked (so a
// sustained eye closure can resume control) but nothing reaches the OS, and
// any held button is released. Not thread-safe: one owning thread, or
// external locking.
class OsController {
public:
    // Throws std::invalid_argument for invalid settings or geometry.
    OsController(const Settings& settings, const ScreenGeometry& geometry, IInputSink& sink);
    ~OsController();

    OsController(const OsController&) = delete;
    OsController& operator=(const OsController&) = delete;

    FrameReport update(const FramePayload& frame) noexcept;

    // User-level on/off (tray menu, hotkey, sustained eye closure).
    void set_active(bool active) noexcept;
    [[nodiscard]] bool active() const noexcept { return active_; }

    // Temporary suppression that gestures cannot override: the calibration
    // wizard is running, the physical mouse is in use, or input went stale.
    void set_inhibited(bool inhibited) noexcept;
    [[nodiscard]] bool inhibited() const noexcept { return inhibited_; }

    // Enables or suppresses mouse button clicks/holds while keeping gesture
    // tracking, TogglePause (eye closure), and gaze blink/wink cursor lock active.
    void set_clicks_enabled(bool enabled) noexcept;
    [[nodiscard]] bool clicks_enabled() const noexcept { return clicks_enabled_; }

    // Sends button-up for every button this controller is holding. A button
    // stays tracked as held until Windows accepts its release; refused
    // releases are retried by retry_pending_releases() and on every update().
    // Returns true when no button is left down.
    bool release_all_buttons() noexcept;
    // Retries refused releases. Call it periodically even when no frames
    // arrive (the tray app does, every 250 ms). Returns true when none remain.
    bool retry_pending_releases() noexcept;
    [[nodiscard]] bool release_pending() const noexcept;
    [[nodiscard]] bool button_held(MouseButton button) const noexcept;

    // Both throw std::invalid_argument and leave the controller unchanged on
    // bad input. Gesture and lock state is reset; held buttons are released.
    void apply_settings(const Settings& settings);
    void set_screen_geometry(const ScreenGeometry& geometry);

    [[nodiscard]] const Settings& settings() const noexcept { return settings_; }
    // Debounced gesture state after the last update (for testers and diagnostics).
    [[nodiscard]] HandGesture confirmed_hand() const noexcept { return gestures_.confirmed_hand(); }
    [[nodiscard]] EyeState confirmed_eye() const noexcept { return gestures_.confirmed_eye(); }
    [[nodiscard]] const ScreenMapper& mapper() const noexcept { return mapper_; }

private:
    enum class Outcome : std::uint8_t { Sent, Skipped, Failed };

    bool can_output() const noexcept { return active_ && !inhibited_; }
    Outcome execute(const Action& action) noexcept;
    bool release(MouseButton button) noexcept;
    void reset_tracking_state() noexcept;

    Settings settings_;
    ScreenMapper mapper_;
    IInputSink& sink_;
    GestureInterpreter gestures_;
    SourceSelector selector_;
    CursorLock hand_lock_;
    CursorLock gaze_lock_;
    std::array<bool, kMouseButtonCount> held_{};            // down in Windows, as far as we know
    std::array<bool, kMouseButtonCount> release_pending_{}; // held, and a release was refused
    // A button-up waiting a moment for the move to where the drag ended.
    struct PendingDrop {
        MouseButton button;
        PixelPoint target;
        double deadline;
    };
    std::optional<PixelPoint> last_sent_;
    std::optional<PixelPoint> last_cursor_;
    std::optional<PixelPoint> refused_target_; // set while the real cursor is not where the user aimed
    std::optional<PendingDrop> drop_;
    double last_timestamp_ = 0.0;
    bool has_timestamp_ = false;
    bool active_ = false;
    bool inhibited_ = false;
    bool clicks_enabled_ = true;
};

} // namespace osi
