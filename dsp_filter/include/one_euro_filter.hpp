#pragma once

#include <optional>

namespace dsp {

struct FilterConfig {
    double min_cutoff_hz;
    double beta;
    double derivative_cutoff_hz;
};

struct TimingConfig {
    // INITIAL ENGINEERING POLICY VALUES; configurable, not measured optima.
    double min_dt_seconds = 0.000001;
    double reset_gap_seconds = 0.25;
};

// INITIAL TUNING VALUES ONLY. Validate with representative project recordings.
inline constexpr FilterConfig kInitialHandConfig{1.2, 4.0, 1.0};
inline constexpr FilterConfig kInitialGazeConfig{0.8, 6.0, 1.5};

struct Point2D {
    double x;
    double y;
};

enum class FilterStatus {
    Initialized,
    Filtered,
    ReinitializedAfterGap,
    TrackingLost,
    InvalidInput,
    RejectedTimestamp,
    NumericalError
};

struct ScalarResult {
    std::optional<double> value;
    FilterStatus status;
};

struct TrackingResult2D {
    std::optional<Point2D> point;
    FilterStatus status;
};

// Generic finite scalar signals, in consistent units. No internal clock or I/O.
// One owning update thread per instance; configuration is copied on construction.
class OneEuroFilter1D {
public:
    // Throws std::invalid_argument for invalid configuration.
    explicit OneEuroFilter1D(FilterConfig config, TimingConfig timing = {});

    // timestamp_seconds is an explicit monotonic observation time, not FPS.
    // Rejected timestamps leave accepted history unchanged. Nonfinite values
    // and unrecoverable arithmetic clear history and produce no value.
    [[nodiscard]] ScalarResult update(double value, double timestamp_seconds) noexcept;
    void reset() noexcept;

private:
    FilterConfig config_;
    TimingConfig timing_;
    double previous_filtered_value_ = 0.0;
    double previous_filtered_derivative_ = 0.0;
    double previous_timestamp_ = 0.0;
    bool initialized_ = false;

    ScalarResult initialize(double value, double timestamp_seconds,
                            FilterStatus status) noexcept;
};

// Current project contract: normalized X/Y in [0, 1]. No state integers, Z,
// model-specific logic, coordinate mapping, or interpretation of gestures.
class TrackingFilter2D {
public:
    TrackingFilter2D(FilterConfig x_config, FilterConfig y_config,
                     TimingConfig timing = {});

    // false validity, nonfinite coordinates, or out-of-range coordinates reset
    // BOTH axes. Missing output is never silently substituted with (0, 0).
    // Both candidate updates succeed before either axis is committed.
    [[nodiscard]] TrackingResult2D update(Point2D point, double timestamp_seconds,
                                         bool tracking_valid) noexcept;
    void reset() noexcept;

private:
    OneEuroFilter1D x_filter_;
    OneEuroFilter1D y_filter_;
};

} // namespace dsp
