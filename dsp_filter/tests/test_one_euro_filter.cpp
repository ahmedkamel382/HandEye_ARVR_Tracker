#include "one_euro_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
using dsp::FilterConfig;
using dsp::FilterStatus;
using dsp::OneEuroFilter1D;
using dsp::Point2D;
using dsp::TimingConfig;
using dsp::TrackingFilter2D;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double absolute = 1e-12,
          double relative = 1e-12) {
    require(std::isfinite(actual) && std::isfinite(expected), "nonfinite comparison");
    require(std::abs(actual - expected) <= absolute + relative * std::abs(expected),
            "floating-point comparison failed");
}

double accepted(dsp::ScalarResult result) {
    require(result.value.has_value(), "expected scalar output");
    require(std::isfinite(*result.value), "nonfinite scalar output");
    return *result.value;
}

Point2D accepted(dsp::TrackingResult2D result) {
    require(result.point.has_value(), "expected point output");
    require(std::isfinite(result.point->x) && std::isfinite(result.point->y),
            "nonfinite point output");
    return *result.point;
}

void analytical_reference() {
    OneEuroFilter1D filter({5.0 / kPi, 1.0 / (2.0 * kPi), 5.0 / kPi});
    constexpr std::array<double, 4> times{0.0, 0.1, 0.2, 0.3};
    constexpr std::array<double, 4> inputs{0.0, 1.0, 1.0, 0.0};
    // Independent algebraic constants supplied in the approved design.
    constexpr std::array<double, 4> expected{0.0, 3.0 / 5.0, 41.0 / 49.0, 1640.0 / 4299.0};
    double maximum_error = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double output = accepted(filter.update(inputs[i], times[i]));
        near(output, expected[i]);
        maximum_error = std::max(maximum_error, std::abs(output - expected[i]));
        if (i == 2) {
            require(std::abs(output - 37.0 / 45.0) > 0.01,
                    "previous-raw derivative regression");
        }
    }
    std::cout << "REFERENCE samples=4 max_abs_error=" << std::scientific << maximum_error
              << " tolerance=1e-12+1e-12*abs(expected)\n" << std::defaultfloat;
}

void adaptive_irregular_dt_reference() {
    OneEuroFilter1D filter({5.0 / (2.0 * kPi), 1.0 / (2.0 * kPi), 5.0 / kPi});
    constexpr std::array<double, 3> times{0.0, 0.1, 0.3};
    constexpr std::array<double, 3> inputs{0.0, 1.0, 1.0};
    // Independent fractions: dt=1/10 gives d_hat=5, alpha_x=1/2;
    // dt=1/5 gives dx=5/2, d_hat=10/3, alpha_x=5/8, output=13/16.
    constexpr std::array<double, 3> expected{0.0, 1.0 / 2.0, 13.0 / 16.0};
    double maximum_error = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const auto result = filter.update(inputs[i], times[i]);
        require(result.status == (i == 0 ? FilterStatus::Initialized : FilterStatus::Filtered),
                "unexpected adaptive irregular-dt status");
        const double output = accepted(result);
        near(output, expected[i]);
        maximum_error = std::max(maximum_error, std::abs(output - expected[i]));
    }
    std::cout << "REFERENCE adaptive_irregular_dt samples=3 max_abs_error="
              << std::scientific << maximum_error
              << " tolerance=1e-12+1e-12*abs(expected)\n" << std::defaultfloat;
}

void nondefault_min_dt_reference() {
    OneEuroFilter1D filter({50.0 / kPi, 1.0 / (2.0 * kPi), 50.0 / kPi}, {0.01, 0.25});
    double maximum_error = 0.0;
    const auto check = [&](dsp::ScalarResult result, double expected, FilterStatus status) {
        require(result.status == status, "unexpected nondefault minimum-dt status");
        const double output = accepted(result);
        near(output, expected);
        maximum_error = std::max(maximum_error, std::abs(output - expected));
    };

    check(filter.update(0.0, 0.0), 0.0, FilterStatus::Initialized);
    const auto initial_rejection = filter.update(9.0, 0.005);
    require(!initial_rejection.value && initial_rejection.status == FilterStatus::RejectedTimestamp,
            "nondefault minimum dt was ignored before motion");

    // dt=1/100: dx=100, d_hat=50, alpha_x=3/5. Exact configured boundary.
    check(filter.update(1.0, 0.01), 3.0 / 5.0, FilterStatus::Filtered);
    const auto moving_rejection = filter.update(-2.0, 0.015);
    require(!moving_rejection.value && moving_rejection.status == FilterStatus::RejectedTimestamp,
            "nondefault minimum dt was ignored after motion");

    // Rejection must retain position=3/5, derivative=50, and timestamp=1/100.
    // Next dt=1/100: dx=40, d_hat=45, alpha_x=29/49, output=41/49.
    check(filter.update(1.0, 0.02), 41.0 / 49.0, FilterStatus::Filtered);

    // dt=1/50 exceeds the configured minimum: dx=-2050/49,
    // d_hat=-1895/147, alpha_x=3319/4789, output=1230/4789.
    check(filter.update(0.0, 0.04), 1230.0 / 4789.0, FilterStatus::Filtered);
    std::cout << "REFERENCE nondefault_min_dt accepted_samples=4 rejected_samples=2 max_abs_error="
              << std::scientific << maximum_error
              << " tolerance=1e-12+1e-12*abs(expected)\n" << std::defaultfloat;
}

void timestamp_rejection(double rejected_time) {
    OneEuroFilter1D subject(dsp::kInitialHandConfig);
    OneEuroFilter1D control(dsp::kInitialHandConfig);
    accepted(subject.update(0.2, 0.0));
    accepted(control.update(0.2, 0.0));
    const auto rejected = subject.update(0.9, rejected_time);
    require(!rejected.value && rejected.status == FilterStatus::RejectedTimestamp,
            "timestamp must be rejected");
    near(accepted(subject.update(0.7, 0.05)), accepted(control.update(0.7, 0.05)));
}

void rate_reference(const std::vector<double>& intervals) {
    // Independent beta=0 solution: error_n = error_0 * product(1-alpha_i).
    OneEuroFilter1D filter({1.0, 0.0, 1.0});
    accepted(filter.update(0.0, 0.0));
    double time = 0.0;
    long double residual = 1.0L;
    constexpr long double pi = 3.141592653589793238462643383279502884L;
    for (double interval : intervals) {
        const double next_time = time + interval;
        const long double dt = static_cast<long double>(next_time) - time;
        residual /= 1.0L + 2.0L * pi * dt;
        time = next_time;
        near(accepted(filter.update(100.0, time)),
             static_cast<double>(100.0L * (1.0L - residual)), 2e-11, 2e-12);
    }
}

struct Statistics {
    std::size_t count = 0;
    double sum = 0.0;
    double sum_square = 0.0;
    double error_square = 0.0;
    double maximum_error = 0.0;

    void add(double value, double target) {
        ++count;
        sum += value;
        sum_square += value * value;
        const double error = value - target;
        error_square += error * error;
        maximum_error = std::max(maximum_error, std::abs(error));
    }
    double mean() const { return sum / static_cast<double>(count); }
    double stddev() const {
        return std::sqrt(std::max(0.0, sum_square / static_cast<double>(count) - mean() * mean()));
    }
    double rms() const { return std::sqrt(error_square / static_cast<double>(count)); }
};

// mt19937's specified integer sequence mapped explicitly to [-1, 1).
// Avoid std::uniform_real_distribution/normal_distribution portability differences.
double deterministic_noise(std::mt19937& random) {
    return 2.0 * (static_cast<double>(random()) / 4294967296.0) - 1.0;
}

std::optional<double> jitter_reduction(double raw_rms, double filtered_rms) {
    if (raw_rms == 0.0) return std::nullopt;
    return 100.0 * (1.0 - filtered_rms / raw_rms);
}

void noise_metrics(FilterConfig config, double center, double amplitude, const char* label) {
    OneEuroFilter1D filter(config);
    std::mt19937 random(42);
    Statistics raw;
    Statistics filtered;
    constexpr int frames = 1200;
    constexpr int warmup = 120;
    for (int i = 0; i < frames; ++i) {
        const double input = center + amplitude * deterministic_noise(random);
        const double output = accepted(filter.update(input, static_cast<double>(i) / 60.0));
        if (i >= warmup) {
            raw.add(input, center);
            filtered.add(output, center);
        }
    }
    require(filtered.rms() < raw.rms(), "stationary RMS did not improve");
    require(filtered.stddev() < raw.stddev(), "stationary standard deviation did not improve");
    std::cout << std::setprecision(12) << "NOISE " << label
              << " samples=" << raw.count << " seed=42 fps=60 warmup_s=2"
              << " raw_mean=" << raw.mean() << " filtered_mean=" << filtered.mean()
              << " raw_std=" << raw.stddev() << " filtered_std=" << filtered.stddev()
              << " raw_rms=" << raw.rms() << " filtered_rms=" << filtered.rms()
              << " reduction_percent=";
    const auto reduction = jitter_reduction(raw.rms(), filtered.rms());
    if (reduction) std::cout << *reduction;
    else std::cout << "N/A";
    std::cout << " max_raw_error=" << raw.maximum_error
              << " max_filtered_error=" << filtered.maximum_error << '\n';
}

struct StepMetrics {
    double t10 = -1.0;
    double t50 = -1.0;
    double t90 = -1.0;
    double settling = -1.0;
    double maximum_error = 0.0;
};

StepMetrics step_metrics(FilterConfig config, double start, double end,
                         const char* label, bool report = true) {
    OneEuroFilter1D filter(config);
    accepted(filter.update(start, 0.0));
    StepMetrics metrics;
    int last_outside = -1;
    constexpr int frames = 600;
    for (int i = 0; i < frames; ++i) {
        const double output = accepted(filter.update(end, static_cast<double>(i + 1) / 60.0));
        const double elapsed = static_cast<double>(i) / 60.0; // First changed input is time zero.
        const double progress = (output - start) / (end - start);
        require(output >= std::min(start, end) && output <= std::max(start, end),
                "step overshoot");
        if (metrics.t10 < 0.0 && progress >= 0.1) metrics.t10 = elapsed;
        if (metrics.t50 < 0.0 && progress >= 0.5) metrics.t50 = elapsed;
        if (metrics.t90 < 0.0 && progress >= 0.9) metrics.t90 = elapsed;
        const double error = std::abs(output - end);
        metrics.maximum_error = std::max(metrics.maximum_error, error);
        if (error > 0.05 * std::abs(end - start)) last_outside = i;
        if (i == frames - 1) near(output, end, 1e-9, 1e-9);
    }
    require(metrics.t90 >= 0.0 && last_outside < frames - 1, "step did not settle");
    metrics.settling = static_cast<double>(last_outside + 1) / 60.0;
    if (report) {
        std::cout << std::setprecision(12) << "STEP " << label << " fps=60"
                  << " start=" << start << " end=" << end
                  << " t10_s=" << metrics.t10 << " t50_s=" << metrics.t50
                  << " t90_s=" << metrics.t90 << " rise_10_90_s=" << metrics.t90 - metrics.t10
                  << " settling_5percent_s=" << metrics.settling
                  << " max_error=" << metrics.maximum_error << '\n';
    }
    return metrics;
}

void ramp_metrics(double direction) {
    OneEuroFilter1D filter(dsp::kInitialHandConfig);
    Statistics error;
    constexpr double velocity = 0.2; // normalized units / second
    for (int i = 0; i <= 240; ++i) {
        const double time = static_cast<double>(i) / 60.0;
        const double input = direction > 0.0 ? 0.1 + velocity * time : 0.9 - velocity * time;
        const double output = accepted(filter.update(input, time));
        if (i >= 60) error.add(output - input, 0.0);
    }
    require(error.rms() < 0.05, "large normalized ramp error");
    std::cout << "RAMP " << (direction > 0.0 ? "positive" : "negative")
              << " samples=" << error.count << " velocity=" << direction * velocity
              << " mean_error=" << error.mean() << " rms_error=" << error.rms()
              << " max_error=" << error.maximum_error
              << " approximate_lag_s=" << std::abs(error.mean()) / velocity << '\n';
}

void coordinate_rejection(double invalid) {
    for (bool bad_x : {true, false}) {
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
        accepted(filter.update({0.1, 0.2}, 0.0, true));
        const auto result = filter.update(bad_x ? Point2D{invalid, 0.9} : Point2D{0.9, invalid},
                                          0.02, true);
        require(!result.point && result.status == FilterStatus::InvalidInput,
                "invalid coordinate accepted");
        const auto fresh = filter.update({0.8, 0.7}, 0.04, true);
        require(fresh.status == FilterStatus::Initialized, "invalid coordinate did not reset both axes");
        const auto point = accepted(fresh);
        near(point.x, 0.8);
        near(point.y, 0.7);
    }
    if (!std::isfinite(invalid)) {
        OneEuroFilter1D scalar(dsp::kInitialHandConfig);
        accepted(scalar.update(0.1, 0.0));
        const auto result = scalar.update(invalid, 0.02);
        require(!result.value && result.status == FilterStatus::InvalidInput,
                "nonfinite scalar accepted");
        near(accepted(scalar.update(0.8, 0.04)), 0.8);
    }
}

void interval_boundary(double dt, FilterStatus expected) {
    OneEuroFilter1D filter(dsp::kInitialHandConfig);
    accepted(filter.update(0.0, 0.0));
    const auto result = filter.update(1.0, dt);
    require(result.status == expected, "wrong dt boundary status");
    if (expected == FilterStatus::RejectedTimestamp) {
        require(!result.value, "rejected interval published output");
    } else if (expected == FilterStatus::ReinitializedAfterGap) {
        near(accepted(result), 1.0);
        OneEuroFilter1D fresh(dsp::kInitialHandConfig);
        accepted(fresh.update(1.0, dt));
        near(accepted(filter.update(0.3, dt + 0.02)), accepted(fresh.update(0.3, dt + 0.02)));
    } else {
        const double output = accepted(result);
        require(output > 0.0 && output < 1.0, "normal interval incorrectly reseeded");
    }
}

void numerical_failure(FilterConfig config, double initial, double next, double dt) {
    OneEuroFilter1D filter(config);
    accepted(filter.update(initial, 0.0));
    const auto result = filter.update(next, dt);
    require(!result.value && result.status == FilterStatus::NumericalError,
            "arithmetic error not reported");
    const auto recovered = filter.update(0.25, dt + 0.01);
    require(recovered.status == FilterStatus::Initialized, "numerical error did not clear history");
    near(accepted(recovered), 0.25);
}

void independent_axes() {
    for (bool moving_x : {true, false}) {
        TrackingFilter2D pair(dsp::kInitialHandConfig, dsp::kInitialGazeConfig);
        OneEuroFilter1D scalar_x(dsp::kInitialHandConfig);
        OneEuroFilter1D scalar_y(dsp::kInitialGazeConfig);
        for (int i = 0; i < 100; ++i) {
            const double time = static_cast<double>(i) / 60.0;
            const double moving = 0.5 + 0.3 * std::sin(time * 4.0);
            const Point2D input = moving_x ? Point2D{moving, 0.4} : Point2D{0.4, moving};
            const auto point = accepted(pair.update(input, time, true));
            near(point.x, accepted(scalar_x.update(input.x, time)));
            near(point.y, accepted(scalar_y.update(input.y, time)));
            near(moving_x ? point.y : point.x, 0.4);
        }
    }
}

void independent_streams() {
    TrackingFilter2D hand(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
    TrackingFilter2D gaze(dsp::kInitialGazeConfig, dsp::kInitialGazeConfig);
    TrackingFilter2D control(dsp::kInitialGazeConfig, dsp::kInitialGazeConfig);
    for (int i = 0; i < 200; ++i) {
        const double time = static_cast<double>(i) / 60.0;
        if (i % 3 == 0) {
            hand.reset();
        } else {
            accepted(hand.update({0.1, 0.9}, time + 100.0, true));
        }
        const Point2D input{0.5 + 0.25 * std::sin(time * 3.0), 0.3};
        const auto actual = accepted(gaze.update(input, time, true));
        const auto expected = accepted(control.update(input, time, true));
        near(actual.x, expected.x);
        near(actual.y, expected.y);
    }
    gaze.reset();
    const auto hand_result = hand.update({0.2, 0.8}, 104.0, true);
    require(hand_result.status == FilterStatus::ReinitializedAfterGap,
            "reset of gaze affected hand timestamp history");
}

void configuration_validation() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<FilterConfig, 13> invalid_configs{{
        {0.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}, {nan, 0.0, 1.0}, {inf, 0.0, 1.0},
        {1.0, -1.0, 1.0}, {1.0, nan, 1.0}, {1.0, inf, 1.0},
        {1.0, 0.0, 0.0}, {1.0, 0.0, -1.0}, {1.0, 0.0, nan}, {1.0, 0.0, inf},
        {-inf, 0.0, 1.0}, {1.0, 0.0, -inf}
    }};
    for (auto config : invalid_configs) {
        bool rejected = false;
        try { OneEuroFilter1D filter(config); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid filter configuration accepted");
    }
    const std::array<TimingConfig, 10> invalid_timing{{
        {0.0, 0.25}, {-1.0, 0.25}, {nan, 0.25}, {inf, 0.25},
        {1e-6, 0.0}, {1e-6, -1.0}, {1e-6, nan}, {1e-6, inf},
        {0.25, 0.25}, {0.5, 0.25}
    }};
    for (auto timing : invalid_timing) {
        bool rejected = false;
        try { OneEuroFilter1D filter(dsp::kInitialHandConfig, timing); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid timing configuration accepted");
    }
    OneEuroFilter1D zero_beta({1.0, 0.0, 1.0});
    near(accepted(zero_beta.update(7.0, 0.0)), 7.0);
}

void long_run() {
    TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
    std::mt19937 random(42);
    double time = 0.0;
    std::size_t outputs = 0, losses = 0, rejected_times = 0, gaps = 0, invalid = 0;
    bool expect_initial = true;
    constexpr int samples = 1000000;
    for (int i = 0; i < samples; ++i) {
        time += (i % 3 == 0 ? 1.0 / 30.0 : (i % 3 == 1 ? 1.0 / 60.0 : 0.023));
        const double phase = static_cast<double>(i % 2000) / 1000.0;
        const double trajectory = phase <= 1.0 ? phase : 2.0 - phase;
        const double x = (i % 10000 < 2000 ? 0.5 : 0.2 + 0.6 * trajectory)
                         + 0.002 * deterministic_noise(random);
        const Point2D input{x, 0.4 + 0.001 * deterministic_noise(random)};
        if (i % 8191 == 0) { filter.reset(); expect_initial = true; }
        if (i % 4093 == 0) {
            const auto result = filter.update({-1.0, -1.0}, time, false);
            require(!result.point && result.status == FilterStatus::TrackingLost, "long-run loss failure");
            expect_initial = true; ++losses; continue;
        }
        if (i % 5003 == 0) {
            require(!filter.update({x, 2.0}, time, true).point, "long-run invalid point accepted");
            expect_initial = true; ++invalid; continue;
        }
        if (i % 3001 == 0 && !expect_initial) {
            require(!filter.update(input, time - 0.1, true).point, "long-run backward time accepted");
            ++rejected_times;
        }
        const bool gap = i % 10007 == 0;
        if (gap) time += 0.5;
        const auto result = filter.update(input, time, true);
        const auto point = accepted(result);
        require(point.x >= 0.0 && point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0,
                "long-run normalized bound violation");
        if (expect_initial || gap) {
            near(point.x, input.x); near(point.y, input.y);
            if (gap && !expect_initial) {
                require(result.status == FilterStatus::ReinitializedAfterGap, "long-run gap failure");
                ++gaps;
            }
        }
        expect_initial = false; ++outputs;
    }
    std::cout << "LONG_RUN observations=" << samples << " valid_outputs=" << outputs
              << " losses=" << losses << " invalid_points=" << invalid
              << " rejected_timestamp_calls=" << rejected_times << " gap_resets=" << gaps << '\n';
}

int failures = 0;
int passes = 0;

template <typename Function>
void run_test(const char* name, Function function) {
    try {
        function();
        ++passes;
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

} // namespace

int main() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double maximum = std::numeric_limits<double>::max();
    const double tiny = std::numeric_limits<double>::denorm_min();
    run_test("first_sample", [] {
        for (double value : {0.0, 100.0, -100.0}) {
            OneEuroFilter1D filter(dsp::kInitialHandConfig);
            const auto result = filter.update(value, 12.0);
            require(result.status == FilterStatus::Initialized, "wrong first status");
            near(accepted(result), value);
        }
    });
    run_test("constant_signal", [] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig);
        for (int i = 0; i < 1000; ++i) near(accepted(filter.update(100.0, i / 60.0)), 100.0);
    });
    run_test("analytical_canonical_reference", analytical_reference);
    run_test("adaptive_irregular_dt_canonical_reference", adaptive_irregular_dt_reference);
    run_test("nondefault_min_dt_preserves_history", nondefault_min_dt_reference);
    run_test("beta_zero_low_pass", [] { rate_reference({0.01, 0.03, 0.05, 0.1, 0.02}); });
    run_test("ascending_step", [] { step_metrics({1.0, 0.1, 1.0}, 0.0, 100.0, "scalar_0_to_100"); });
    run_test("descending_step", [] { step_metrics({1.0, 0.1, 1.0}, 100.0, 0.0, "scalar_100_to_0"); });
    run_test("hand_step_metrics", [] { step_metrics(dsp::kInitialHandConfig, 0.2, 0.8, "hand_initial"); });
    run_test("gaze_step_metrics", [] { step_metrics(dsp::kInitialGazeConfig, 0.2, 0.8, "gaze_initial"); });
    run_test("positive_ramp", [] { ramp_metrics(1.0); });
    run_test("negative_ramp", [] { ramp_metrics(-1.0); });
    run_test("noisy_stationary_500", [] { noise_metrics({1.0, 0.0, 1.0}, 500.0, 5.0, "scalar_500"); });
    run_test("noisy_stationary_hand", [] { noise_metrics(dsp::kInitialHandConfig, 0.5, 0.005, "hand_initial"); });
    run_test("noisy_stationary_gaze", [] { noise_metrics(dsp::kInitialGazeConfig, 0.5, 0.005, "gaze_initial"); });
    run_test("zero_raw_rms_reports_not_applicable", [] {
        require(!jitter_reduction(0.0, 0.0), "zero RMS must not be divided");
        std::cout << "NOISE constant raw_rms=0 filtered_rms=0 reduction_percent=N/A\n";
    });
    run_test("rapid_movement_adaptation", [] {
        const auto adaptive = step_metrics(dsp::kInitialHandConfig, 0.2, 0.8, "adaptive", false);
        const auto fixed = step_metrics({1.2, 0.0, 1.0}, 0.2, 0.8, "fixed", false);
        require(adaptive.t90 < fixed.t90, "adaptive cutoff did not improve response");
        std::cout << "ADAPTATION adaptive_t90_s=" << adaptive.t90 << " beta_zero_t90_s=" << fixed.t90 << '\n';
    });
    run_test("direction_reversal", [] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig);
        double last = 0.0;
        for (int i = 0; i <= 120; ++i) {
            const double input = i <= 60 ? i / 60.0 : (120 - i) / 60.0;
            last = accepted(filter.update(input, i / 60.0));
            require(last >= 0.0 && last <= 1.0, "reversal out of bounds");
        }
        require(last < 0.12, "excessive reversal lag");
    });
    run_test("frame_rate_30", [] { rate_reference(std::vector<double>(300, 1.0 / 30.0)); });
    run_test("frame_rate_60", [] { rate_reference(std::vector<double>(600, 1.0 / 60.0)); });
    run_test("irregular_timing", [] { rate_reference({0.016, 0.035, 0.021, 0.043, 0.019, 0.11, 0.027}); });
    run_test("dropped_frame_interval", [] { rate_reference({1.0 / 60.0, 1.0 / 60.0, 0.1, 1.0 / 60.0}); });
    run_test("duplicate_timestamp", [] { timestamp_rejection(0.0); });
    run_test("zero_dt_after_update", [] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig), control(dsp::kInitialHandConfig);
        accepted(filter.update(0.0, 0.0)); accepted(control.update(0.0, 0.0));
        accepted(filter.update(0.5, 0.02)); accepted(control.update(0.5, 0.02));
        require(!filter.update(1.0, 0.02).value, "zero dt accepted");
        near(accepted(filter.update(0.7, 0.04)), accepted(control.update(0.7, 0.04)));
    });
    run_test("negative_dt", [] { timestamp_rejection(-0.01); });
    run_test("tiny_dt_below", [] { timestamp_rejection(std::nextafter(1e-6, 0.0)); });
    run_test("tiny_dt_exact", [] { interval_boundary(1e-6, FilterStatus::Filtered); });
    run_test("tiny_dt_above", [inf] { interval_boundary(std::nextafter(1e-6, inf), FilterStatus::Filtered); });
    run_test("gap_below", [] { interval_boundary(std::nextafter(0.25, 0.0), FilterStatus::Filtered); });
    run_test("gap_exact", [] { interval_boundary(0.25, FilterStatus::Filtered); });
    run_test("gap_above", [inf] { interval_boundary(std::nextafter(0.25, inf), FilterStatus::ReinitializedAfterGap); });
    run_test("nan_timestamp", [nan] { timestamp_rejection(nan); });
    run_test("positive_infinity_timestamp", [inf] { timestamp_rejection(inf); });
    run_test("negative_infinity_timestamp", [inf] { timestamp_rejection(-inf); });
    run_test("invalid_first_timestamp", [nan] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig);
        require(!filter.update(1.0, nan).value, "invalid first timestamp accepted");
        require(filter.update(0.4, 0.0).status == FilterStatus::Initialized, "invalid first timestamp changed state");
    });
    run_test("nan_coordinate", [nan] { coordinate_rejection(nan); });
    run_test("positive_infinity_coordinate", [inf] { coordinate_rejection(inf); });
    run_test("negative_infinity_coordinate", [inf] { coordinate_rejection(-inf); });
    run_test("explicit_reset", [] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig), fresh(dsp::kInitialHandConfig);
        accepted(filter.update(0.1, 1.0)); accepted(filter.update(0.9, 1.02)); filter.reset();
        near(accepted(filter.update(0.7, 0.0)), accepted(fresh.update(0.7, 0.0)));
        near(accepted(filter.update(0.2, 0.02)), accepted(fresh.update(0.2, 0.02)));
    });
    run_test("tracking_loss", [nan] {
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
        accepted(filter.update({0.1, 0.2}, 0.0, true));
        const auto result = filter.update({-1.0, -1.0}, nan, false);
        require(!result.point && result.status == FilterStatus::TrackingLost, "loss must take priority");
        require(!filter.update({0.0, 0.0}, 0.02, false).point, "false validity published a point");
    });
    run_test("reacquisition", [] {
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
        accepted(filter.update({0.1, 0.1}, 0.0, true));
        accepted(filter.update({0.9, 0.9}, 0.02, true));
        require(!filter.update({-1.0, -1.0}, 0.03, false).point, "loss accepted");
        const auto result = filter.update({0.2, 0.8}, 0.04, true);
        require(result.status == FilterStatus::Initialized, "reacquisition not initialized");
        const auto point = accepted(result); near(point.x, 0.2); near(point.y, 0.8);
    });
    run_test("single_axis_invalidity", [] { coordinate_rejection(1.1); });
    run_test("single_axis_numerical_failure", [maximum] {
        TrackingFilter2D filter(dsp::kInitialHandConfig, {1.0, maximum, 1.0});
        accepted(filter.update({0.0, 0.0}, 0.0, true));
        const auto result = filter.update({0.5, 1.0}, 0.01, true);
        require(!result.point && result.status == FilterStatus::NumericalError, "half-updated point published");
        const auto next = filter.update({0.2, 0.8}, 0.02, true);
        require(next.status == FilterStatus::Initialized, "both axes not reset");
        near(accepted(next).x, 0.2); near(accepted(next).y, 0.8);
    });
    run_test("tracking_timestamp_rejection_preserves_both_axes", [nan, inf] {
        for (double bad_time : {0.02, -0.01, 0.0200001, nan, inf, -inf}) {
            TrackingFilter2D subject(dsp::kInitialHandConfig, dsp::kInitialGazeConfig);
            TrackingFilter2D control(dsp::kInitialHandConfig, dsp::kInitialGazeConfig);
            accepted(subject.update({0.1,0.2}, 0.0, true));
            accepted(control.update({0.1,0.2}, 0.0, true));
            accepted(subject.update({0.4,0.6}, 0.02, true));
            accepted(control.update({0.4,0.6}, 0.02, true));
            const auto rejected = subject.update({0.9,0.9}, bad_time, true);
            require(!rejected.point && rejected.status == FilterStatus::RejectedTimestamp,
                    "2D invalid timestamp accepted");
            const auto actual = accepted(subject.update({0.7,0.3}, 0.04, true));
            const auto expected = accepted(control.update({0.7,0.3}, 0.04, true));
            near(actual.x, expected.x); near(actual.y, expected.y);
        }
    });
    run_test("xy_state_independence", independent_axes);
    run_test("hand_gaze_state_independence", independent_streams);
    run_test("configuration_value_ownership", [] {
        auto config = dsp::kInitialHandConfig;
        OneEuroFilter1D subject(config), control(config);
        config.beta = 1000.0;
        accepted(subject.update(0.0, 0.0)); accepted(control.update(0.0, 0.0));
        near(accepted(subject.update(0.5, 0.02)), accepted(control.update(0.5, 0.02)));
    });
    run_test("normalized_boundaries", [] {
        const std::array<Point2D, 8> points{{{0,0},{1,1},{0,1},{1,0},{0,0.5},{1,0.5},{0.5,0},{0.5,1}}};
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
        for (auto point : points) {
            filter.reset(); const auto output = accepted(filter.update(point, 0.0, true));
            near(output.x, point.x); near(output.y, point.y);
        }
        filter.reset();
        for (int i = 0; i < 1000; ++i) {
            const auto point = accepted(filter.update(points[static_cast<std::size_t>(i) % points.size()],
                                                      i / 60.0, true));
            require(point.x >= 0.0 && point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0,
                    "filtered corner transition left normalized domain");
        }
    });
    run_test("invalid_normalized_ranges_and_sentinel", [tiny, inf] {
        for (double value : {-1.0, -tiny, std::nextafter(1.0, inf), 2.0}) coordinate_rejection(value);
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
        require(!filter.update({-1.0, -1.0}, 0.0, true).point, "sentinel entered recurrence");
        const auto zero = accepted(filter.update({0.0, 0.0}, 0.02, true));
        near(zero.x, 0.0); near(zero.y, 0.0);
    });
    run_test("configuration_validation", configuration_validation);
    run_test("extreme_finite_difference", [maximum] { numerical_failure({1,0,1}, -maximum, maximum, 0.1); });
    run_test("derivative_overflow", [maximum] { numerical_failure({1,0,1}, 0, maximum, 1e-6); });
    run_test("cutoff_multiplication_overflow", [maximum] { numerical_failure({1,maximum,1}, 0, 1, 0.01); });
    run_test("cutoff_addition_overflow", [maximum] { numerical_failure({maximum*0.9,maximum*0.1,1}, 0, 1, 0.2); });
    run_test("unrepresentable_alpha", [tiny] {
        numerical_failure({tiny,0,1}, 0, 1, 0.1);
        numerical_failure({1,0,tiny}, 0, 1, 0.1);
    });
    run_test("large_finite_cutoff", [maximum] {
        OneEuroFilter1D filter({maximum, 0.0, maximum});
        accepted(filter.update(0.0, 0.0)); near(accepted(filter.update(1.0, 0.02)), 1.0);
    });
    run_test("alpha_one_preserves_small_observation", [maximum] {
        OneEuroFilter1D filter({maximum, 0.0, maximum}, {1e-6, 10.0});
        accepted(filter.update(maximum, 0.0));
        require(accepted(filter.update(1e-100, 2.0)) == 1e-100,
                "alpha one lost small observation through cancellation");
    });
    run_test("timestamp_difference_overflow", [maximum] {
        OneEuroFilter1D filter(dsp::kInitialHandConfig);
        accepted(filter.update(0.2, -maximum));
        const auto result = filter.update(0.8, maximum);
        require(!result.value && result.status == FilterStatus::RejectedTimestamp, "timestamp overflow accepted");
        require(!filter.update(0.8, -maximum).value, "overflow advanced timestamp history");
    });
    run_test("error_recovery", [nan] {
        TrackingFilter2D filter(dsp::kInitialHandConfig, dsp::kInitialGazeConfig);
        accepted(filter.update({0.1, 0.2}, 0.0, true));
        require(!filter.update({nan, 0.3}, 0.02, true).point, "invalid point accepted");
        require(filter.update({0.7, 0.8}, 0.04, true).status == FilterStatus::Initialized, "recovery failed");
        require(filter.update({0.6, 0.7}, 0.06, true).status == FilterStatus::Filtered, "recovered track unusable");
    });
    run_test("million_observation_stability", long_run);
    std::cout << "PENDING - external Casiez ground-truth dataset not included because its applicable redistribution license could not be confidently established.\n";
    std::cout << "SUMMARY passed=" << passes << " failed=" << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
