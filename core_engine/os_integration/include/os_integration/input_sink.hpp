#pragma once

#include "os_integration/types.hpp"

#include <vector>

namespace osi {

// Where mouse commands go. Win32InputInjector sends them to Windows; the
// recording sink captures them for tests and dry runs. Methods return false
// when the operating system refused the input.
class IInputSink {
public:
    virtual ~IInputSink() = default;

    virtual bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept = 0;
    virtual bool button(MouseButton button, ButtonAction action) noexcept = 0;

    // Press and release. Implementations may submit both events atomically.
    virtual bool click(MouseButton which) noexcept {
        const bool down = button(which, ButtonAction::Down);
        const bool up = button(which, ButtonAction::Up); // always attempt the release
        return down && up;
    }
};

struct SinkEvent {
    enum class Kind { Move, Down, Up, Click };
    Kind kind;
    PixelPoint pixel{0, 0};
    AbsolutePoint absolute{0, 0};
    MouseButton button = MouseButton::Left;
};

class RecordingInputSink final : public IInputSink {
public:
    bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept override {
        return record({SinkEvent::Kind::Move, pixel, absolute, MouseButton::Left});
    }
    bool button(MouseButton which, ButtonAction action) noexcept override {
        return record({action == ButtonAction::Down ? SinkEvent::Kind::Down : SinkEvent::Kind::Up,
                       {0, 0}, {0, 0}, which});
    }
    bool click(MouseButton which) noexcept override {
        return record({SinkEvent::Kind::Click, {0, 0}, {0, 0}, which});
    }

    std::vector<SinkEvent> events;
    bool fail = false; // simulate Windows rejecting input (e.g. UIPI)

private:
    bool record(const SinkEvent& event) noexcept {
        try {
            events.push_back(event);
        } catch (...) {
            return false;
        }
        return !fail;
    }
};

} // namespace osi
