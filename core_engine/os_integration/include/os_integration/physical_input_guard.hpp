#pragma once

#include <atomic>

namespace osi {

// Gives the real mouse priority: after any physical mouse input, virtual
// control yields for `yield_seconds`. Written from the input-hook thread,
// read from the frame thread; times are seconds on one monotonic clock.
class PhysicalInputGuard {
public:
    explicit PhysicalInputGuard(double yield_seconds) noexcept : yield_seconds_(yield_seconds) {}

    void on_physical_input(double now_seconds) noexcept {
        last_input_.store(now_seconds, std::memory_order_relaxed);
        seen_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool yielding(double now_seconds) const noexcept {
        if (!seen_.load(std::memory_order_acquire)) {
            return false;
        }
        const double since = now_seconds - last_input_.load(std::memory_order_relaxed);
        return since >= 0.0 && since < yield_seconds_.load(std::memory_order_relaxed);
    }

    void set_yield_seconds(double seconds) noexcept {
        yield_seconds_.store(seconds, std::memory_order_relaxed);
    }

private:
    std::atomic<double> yield_seconds_;
    std::atomic<double> last_input_{0.0};
    std::atomic<bool> seen_{false};
};

} // namespace osi
