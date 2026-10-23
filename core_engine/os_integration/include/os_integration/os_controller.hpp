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
    bool moved = false;               // a move was sent this frame
    ActionList executed;              // button actions actually sent this frame
    bool pause_toggled = false;       // a gesture toggled pause this frame
    bool active = false;              // state after this frame
    bool inhibited = false;
    bool injection_failed = false;    // Windows refused at least one event
    bool gaze_calibrating = false;
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

    // Sends button-up for every button this controller is holding.
    void release_all_buttons() noexcept;
    [[nodiscard]] bool button_held(MouseButton button) const noexcept;

    // Both throw std::invalid_argument and leave the controller unchanged on
    // bad input. Gesture and lock state is reset; held buttons are released.
    void apply_settings(const Settings& settings);
    void set_screen_geometry(const ScreenGeometry& geometry);

    [[nodiscard]] const Settings& settings() const noexcept { return settings_; }
    [[nodiscard]] const ScreenMapper& mapper() const noexcept { return mapper_; }

private:
    enum class Outcome : std::uint8_t { Sent, Skipped, Failed };

    bool can_output() const noexcept { return active_ && !inhibited_; }
    Outcome execute(const Action& action) noexcept;
    void reset_tracking_state() noexcept;

    Settings settings_;
    ScreenMapper mapper_;
    IInputSink& sink_;
    GestureInterpreter gestures_;
    SourceSelector selector_;
    CursorLock hand_lock_;
    CursorLock gaze_lock_;
    std::array<bool, kMouseButtonCount> held_{};
    std::optional<PixelPoint> last_sent_;
    std::optional<PixelPoint> last_cursor_;
    double last_timestamp_ = 0.0;
    bool has_timestamp_ = false;
    bool active_ = false;
    bool inhibited_ = false;
};

} // namespace osi
