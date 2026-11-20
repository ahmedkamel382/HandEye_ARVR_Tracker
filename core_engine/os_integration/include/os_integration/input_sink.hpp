#pragma once

#include "os_integration/types.hpp"

#include <cstdint>
#include <vector>

namespace osi {

// What happened to a press-and-release. PressedOnly is the dangerous case:
// Windows accepted the press but refused the release, so the button is down
// in the OS and must be released later.
enum class ClickResult : std::uint8_t { Clicked, Refused, PressedOnly };

// Where mouse commands go. Win32InputInjector sends them to Windows; the
// recording sink captures them for tests and dry runs. Methods return false
// when the operating system refused the input.
class IInputSink {
public:
    virtual ~IInputSink() = default;

    virtual bool move_to(PixelPoint pixel, AbsolutePoint absolute) noexcept = 0;
    virtual bool button(MouseButton button, ButtonAction action) noexcept = 0;

    // Press and release. Implementations may submit both events atomically.
    virtual ClickResult click(MouseButton which) noexcept {
        if (!button(which, ButtonAction::Down)) {
            return ClickResult::Refused;
        }
        return button(which, ButtonAction::Up) ? ClickResult::Clicked : ClickResult::PressedOnly;
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
        const bool up = action == ButtonAction::Up;
        return record({up ? SinkEvent::Kind::Up : SinkEvent::Kind::Down, {0, 0}, {0, 0}, which}) &&
               !(up && fail_releases);
    }
    ClickResult click(MouseButton which) noexcept override {
        if (!record({SinkEvent::Kind::Click, {0, 0}, {0, 0}, which})) {
            return ClickResult::Refused;
        }
        return fail_releases ? ClickResult::PressedOnly : ClickResult::Clicked;
    }

    std::vector<SinkEvent> events;
    bool fail = false;          // simulate Windows rejecting all input (e.g. UIPI)
    bool fail_releases = false; // simulate Windows accepting presses but refusing releases

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
