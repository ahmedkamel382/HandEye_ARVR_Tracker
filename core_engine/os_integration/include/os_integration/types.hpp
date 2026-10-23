#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace osi {

// Camera-space point delivered by the core loop (Python AI -> DSP). The
// current project contract is normalized X/Y in [0, 1].
struct NormalizedPoint {
    double x;
    double y;
};

// Pixel position on the Windows virtual desktop. Left/top of secondary
// monitors can be negative.
struct PixelPoint {
    std::int32_t x;
    std::int32_t y;
};

inline bool operator==(PixelPoint a, PixelPoint b) noexcept { return a.x == b.x && a.y == b.y; }
inline bool operator!=(PixelPoint a, PixelPoint b) noexcept { return !(a == b); }

// Win32 absolute mouse coordinates: 0..65535 spanning the virtual desktop
// (MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK).
struct AbsolutePoint {
    std::int32_t x;
    std::int32_t y;
};

inline bool operator==(AbsolutePoint a, AbsolutePoint b) noexcept { return a.x == b.x && a.y == b.y; }

// Rectangle in virtual-desktop pixels.
struct ScreenRect {
    std::int32_t left;
    std::int32_t top;
    std::int32_t width;
    std::int32_t height;
};

struct ScreenGeometry {
    ScreenRect virtual_desktop; // the space SendInput absolute coordinates cover
    ScreenRect target;          // where the cursor may go: one monitor or the whole desktop
};

// State integers as returned by ai_pipeline/hand_kinematics/hand_tracker.py.
enum class HandGesture : std::int32_t { Neutral = 0, Pinch = 1, Fist = 2 };

// State integers as returned by ai_pipeline/gaze_intent/gaze_intent.py.
enum class EyeState : std::int32_t {
    Calibrating = -1,
    Neutral = 0,
    LeftWink = 1,
    RightWink = 2,
    SustainedClosure = 3
};

// Unknown integers yield an empty optional; callers treat them as Neutral.
std::optional<HandGesture> hand_gesture_from_int(std::int32_t value) noexcept;
std::optional<EyeState> eye_state_from_int(std::int32_t value) noexcept;

// One tracker's output for one frame. `point` is empty whenever the tracker
// lost its target (Python's (-1, -1)) or DSP produced no stabilized value.
struct TrackerSample {
    std::optional<NormalizedPoint> point;
    std::int32_t state = 0;
};

// Phase 4 hand-off: everything the core loop gives the OS layer per frame.
struct FramePayload {
    double timestamp_seconds = 0.0; // monotonic, same clock the DSP filters use
    TrackerSample hand;
    TrackerSample gaze;
};

enum class MouseButton : std::uint8_t { Left = 0, Right = 1, Middle = 2 };
inline constexpr std::size_t kMouseButtonCount = 3;

enum class ButtonAction : std::uint8_t { Down, Up };

enum class ActionType : std::uint8_t { ButtonDown, ButtonUp, Click, TogglePause };

struct Action {
    ActionType type = ActionType::Click;
    MouseButton button = MouseButton::Left;
};

inline bool operator==(Action a, Action b) noexcept {
    return a.type == b.type && (a.type == ActionType::TogglePause || a.button == b.button);
}

// Fixed-capacity list so the per-frame path never allocates.
class ActionList {
public:
    static constexpr std::size_t kCapacity = 8;

    bool push(Action action) noexcept {
        if (size_ == kCapacity) {
            return false;
        }
        items_[size_++] = action;
        return true;
    }
    void clear() noexcept { size_ = 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    const Action& operator[](std::size_t index) const noexcept { return items_[index]; }
    const Action* begin() const noexcept { return items_.data(); }
    const Action* end() const noexcept { return items_.data() + size_; }

private:
    std::array<Action, kCapacity> items_{};
    std::size_t size_ = 0;
};

// Finite and inside [0, 1] on both axes.
bool is_normalized(NormalizedPoint point) noexcept;

} // namespace osi
