#include "one_euro_filter.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace dsp {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kInverseTwoPi = 1.0 / (2.0 * kPi);

bool positive_finite(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

// Equivalent to 1 / (1 + 1 / (2*pi*cutoff*dt)). Ratios are kept <= 1
// rather than forming a potentially overflowing cutoff*dt product.
bool smoothing_alpha(double cutoff, double dt, double& alpha) noexcept {
    if (!positive_finite(cutoff)) {
        return false;
    }
    if (cutoff < kInverseTwoPi / std::numeric_limits<double>::max()) {
        return false; // The time constant cannot be represented as a double.
    }
    const double tau = kInverseTwoPi / cutoff;
    if (!positive_finite(tau)) {
        return false;
    }
    if (dt >= tau) {
        alpha = 1.0 / (1.0 + tau / dt);
    } else {
        const double ratio = dt / tau;
        alpha = ratio / (1.0 + ratio);
    }
    return positive_finite(alpha) && alpha <= 1.0;
}

// Convex interpolation without subtracting extreme opposite-signed values.
double low_pass(double current, double previous, double alpha) noexcept {
    if (alpha == 1.0 || current == previous) {
        return current;
    }
    if (std::signbit(current) == std::signbit(previous)) {
        return previous + alpha * (current - previous);
    }
    return alpha * current + (1.0 - alpha) * previous;
}

bool normalized(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

} // namespace

OneEuroFilter1D::OneEuroFilter1D(FilterConfig config, TimingConfig timing)
    : config_(config), timing_(timing) {
    if (!positive_finite(config.min_cutoff_hz) ||
        !positive_finite(config.derivative_cutoff_hz) ||
        !std::isfinite(config.beta) || config.beta < 0.0 ||
        !positive_finite(timing.min_dt_seconds) ||
        !positive_finite(timing.reset_gap_seconds) ||
        timing.reset_gap_seconds <= timing.min_dt_seconds) {
        throw std::invalid_argument("Invalid 1 Euro filter or timing configuration");
    }
}

void OneEuroFilter1D::reset() noexcept {
    previous_filtered_value_ = 0.0;
    previous_filtered_derivative_ = 0.0;
    previous_timestamp_ = 0.0;
    initialized_ = false;
}

ScalarResult OneEuroFilter1D::initialize(double value, double timestamp_seconds,
                                        FilterStatus status) noexcept {
    previous_filtered_value_ = value;
    previous_filtered_derivative_ = 0.0;
    previous_timestamp_ = timestamp_seconds;
    initialized_ = true;
    return {value, status};
}

ScalarResult OneEuroFilter1D::update(double value, double timestamp_seconds) noexcept {
    if (!std::isfinite(timestamp_seconds)) {
        return {std::nullopt, FilterStatus::RejectedTimestamp};
    }
    if (!std::isfinite(value)) {
        reset();
        return {std::nullopt, FilterStatus::InvalidInput};
    }
    if (!initialized_) {
        return initialize(value, timestamp_seconds, FilterStatus::Initialized);
    }

    const double dt = timestamp_seconds - previous_timestamp_;
    if (!std::isfinite(dt) || dt <= 0.0 || dt < timing_.min_dt_seconds) {
        return {std::nullopt, FilterStatus::RejectedTimestamp};
    }
    if (dt > timing_.reset_gap_seconds) {
        return initialize(value, timestamp_seconds, FilterStatus::ReinitializedAfterGap);
    }

    // Canonical/current Casiez recurrence: CURRENT RAW minus PREVIOUS FILTERED.
    // Previous raw input is deliberately not stored in this class.
    const double max_value = std::numeric_limits<double>::max();
    if ((value > 0.0 && previous_filtered_value_ < 0.0 &&
         value > max_value + previous_filtered_value_) ||
        (value < 0.0 && previous_filtered_value_ > 0.0 &&
         value < -max_value + previous_filtered_value_)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }
    const double difference = value - previous_filtered_value_;
    if (!std::isfinite(difference) || (dt < 1.0 && std::abs(difference) > max_value * dt)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }
    const double derivative = difference / dt;
    double derivative_alpha = 0.0;
    if (!std::isfinite(difference) || !std::isfinite(derivative) ||
        !smoothing_alpha(config_.derivative_cutoff_hz, dt, derivative_alpha)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }

    const double filtered_derivative = low_pass(
        derivative, previous_filtered_derivative_, derivative_alpha);
    const double speed = std::abs(filtered_derivative);
    if (!std::isfinite(filtered_derivative) ||
        (config_.beta > 1.0 && speed > max_value / config_.beta)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }
    const double adaptive_increment = config_.beta * speed;
    if (adaptive_increment > max_value - config_.min_cutoff_hz) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }
    const double cutoff = config_.min_cutoff_hz + adaptive_increment;
    double signal_alpha = 0.0;
    if (!smoothing_alpha(cutoff, dt, signal_alpha)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }
    const double filtered_value = low_pass(value, previous_filtered_value_, signal_alpha);
    if (!std::isfinite(filtered_value)) {
        reset();
        return {std::nullopt, FilterStatus::NumericalError};
    }

    // Commit only a completely validated update.
    previous_filtered_value_ = filtered_value;
    previous_filtered_derivative_ = filtered_derivative;
    previous_timestamp_ = timestamp_seconds;
    return {filtered_value, FilterStatus::Filtered};
}

TrackingFilter2D::TrackingFilter2D(FilterConfig x_config, FilterConfig y_config,
                                 TimingConfig timing)
    : x_filter_(x_config, timing), y_filter_(y_config, timing) {}

void TrackingFilter2D::reset() noexcept {
    x_filter_.reset();
    y_filter_.reset();
}

TrackingResult2D TrackingFilter2D::update(Point2D point, double timestamp_seconds,
                                        bool tracking_valid) noexcept {
    if (!tracking_valid) {
        reset();
        return {std::nullopt, FilterStatus::TrackingLost};
    }
    if (!normalized(point.x) || !normalized(point.y)) {
        reset();
        return {std::nullopt, FilterStatus::InvalidInput};
    }

    // Fixed-size stack candidates provide transactional updates across axes.
    auto next_x = x_filter_;
    auto next_y = y_filter_;
    const auto x_result = next_x.update(point.x, timestamp_seconds);
    const auto y_result = next_y.update(point.y, timestamp_seconds);
    if (x_result.value && y_result.value) {
        x_filter_ = next_x;
        y_filter_ = next_y;
        return {Point2D{*x_result.value, *y_result.value}, x_result.status};
    }
    if (x_result.status == FilterStatus::RejectedTimestamp &&
        y_result.status == FilterStatus::RejectedTimestamp) {
        return {std::nullopt, FilterStatus::RejectedTimestamp};
    }
    reset();
    return {std::nullopt, FilterStatus::NumericalError};
}

} // namespace dsp
