// Per-frame cost of OsController::update (mapping, gestures, locks, source
// selection) without the operating system: the sink only counts events.
// Informational; not a CTest timing assertion.

#include "os_integration/os_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

class CountingSink final : public osi::IInputSink {
public:
    bool move_to(osi::PixelPoint, osi::AbsolutePoint) noexcept override {
        ++moves;
        return true;
    }
    bool button(osi::MouseButton, osi::ButtonAction) noexcept override {
        ++buttons;
        return true;
    }
    long long moves = 0;
    long long buttons = 0;
};

} // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr int kFrames = 1'000'000;
    constexpr int kBatch = 1000;

    CountingSink sink;
    osi::OsController controller(osi::Settings{}, {{0, 0, 1920, 1080}, {0, 0, 1920, 1080}}, sink);
    controller.set_active(true);

    std::vector<double> batch_ns;
    batch_ns.reserve(kFrames / kBatch);
    double t = 0.0;
    osi::FramePayload frame;
    for (int b = 0; b < kFrames / kBatch; ++b) {
        const auto start = Clock::now();
        for (int i = 0; i < kBatch; ++i) {
            t += 1.0 / 60.0;
            const double phase = t * 0.7;
            frame.timestamp_seconds = t;
            frame.hand.point = osi::NormalizedPoint{0.5 + 0.3 * std::cos(phase), 0.5 + 0.3 * std::sin(phase)};
            // A pinch for 0.5 s every 3 s exercises locks, drags and button events.
            frame.hand.state = std::fmod(t, 3.0) < 0.5 ? 1 : 0;
            frame.gaze.point = osi::NormalizedPoint{0.5, 0.5};
            frame.gaze.state = 0;
            controller.update(frame);
        }
        batch_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count() / kBatch);
    }
    std::sort(batch_ns.begin(), batch_ns.end());
    std::printf("frames=%d moves=%lld button_events=%lld\n", kFrames, sink.moves, sink.buttons);
    std::printf("ns/frame  p50=%.1f  p99=%.1f  max=%.1f\n", batch_ns[batch_ns.size() / 2],
                batch_ns[batch_ns.size() * 99 / 100], batch_ns.back());
    return 0;
}
