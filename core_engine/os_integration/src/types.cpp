#include "os_integration/types.hpp"

#include <cmath>

namespace osi {

std::optional<HandGesture> hand_gesture_from_int(std::int32_t value) noexcept {
    if (value >= 0 && value <= 2) {
        return static_cast<HandGesture>(value);
    }
    return std::nullopt;
}

std::optional<EyeState> eye_state_from_int(std::int32_t value) noexcept {
    if (value >= -1 && value <= 3) {
        return static_cast<EyeState>(value);
    }
    return std::nullopt;
}

bool is_normalized(NormalizedPoint point) noexcept {
    return std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0.0 &&
           point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0;
}

} // namespace osi
