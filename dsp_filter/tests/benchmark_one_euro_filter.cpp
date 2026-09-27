#include "one_euro_filter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kWarmup = 10000;
constexpr std::size_t kBatches = 100;
constexpr std::size_t kBatchSize = 10000;
constexpr std::size_t kIterations = kBatches * kBatchSize;
constexpr std::size_t kSignalSize = 4096;

struct Signals {
    double hand_x;
    double hand_y;
    double gaze_x;
    double gaze_y;
};

std::array<Signals, kSignalSize> signals() {
    std::array<Signals, kSignalSize> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double phase = static_cast<double>(i) * 0.017;
        values[i] = {0.5 + 0.3 * std::sin(phase), 0.5 + 0.2 * std::cos(phase),
                     0.5 + 0.1 * std::sin(phase * 1.7), 0.5 + 0.08 * std::cos(phase * 1.3)};
    }
    return values;
}

template <typename Work>
bool measure(const char* name, Work work) {
    double checksum = 0.0;
    for (std::size_t i = 0; i < kWarmup; ++i) checksum += work(i);

    double total_nanoseconds = 0.0;
    double slowest_batch_average = 0.0;
    for (std::size_t batch = 0; batch < kBatches; ++batch) {
        const std::size_t begin_index = kWarmup + batch * kBatchSize;
        const auto begin = Clock::now();
        for (std::size_t i = begin_index; i < begin_index + kBatchSize; ++i) checksum += work(i);
        const auto end = Clock::now();
        const double nanoseconds = std::chrono::duration<double, std::nano>(end - begin).count();
        total_nanoseconds += nanoseconds;
        slowest_batch_average = std::max(slowest_batch_average, nanoseconds / kBatchSize);
    }
    std::cout << std::fixed << std::setprecision(6)
              << "BENCHMARK " << name << " iterations=" << kIterations
              << " average_ns=" << total_nanoseconds / kIterations
              << " average_us=" << total_nanoseconds / kIterations / 1000.0
              << " slowest_batch_average_ns=" << slowest_batch_average
              << " total_ms=" << total_nanoseconds / 1000000.0
              << " checksum=" << checksum << '\n';
    return std::isfinite(checksum) && checksum > 0.0;
}

} // namespace

int main() {
    std::cout << "Compiler: " << DSP_COMPILER << "\nArchitecture: " << DSP_ARCHITECTURE
              << "\nBuild: " << DSP_BUILD_CONFIGURATION
              << "\nClock: std::chrono::steady_clock; warmup=" << kWarmup
              << "; batches=" << kBatches << "; batch_size=" << kBatchSize << '\n';
    std::cout << "Synthetic workload; checksum consumes outputs. Timings include loop/result handling.\n"
                 "Slowest batch average is not worst single-call latency or a real-time guarantee.\n";

    const auto input = signals(); // Precomputed outside all measured loops.
    std::size_t rejected = 0;
    dsp::OneEuroFilter1D scalar(dsp::kInitialHandConfig);
    const bool scalar_ok = measure("scalar", [&](std::size_t i) {
        const auto result = scalar.update(input[i % kSignalSize].hand_x, static_cast<double>(i) / 60.0);
        if (!result.value) { ++rejected; return 0.0; }
        return *result.value;
    });

    dsp::TrackingFilter2D point(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
    const bool point_ok = measure("tracking_2d", [&](std::size_t i) {
        const auto& value = input[i % kSignalSize];
        const auto result = point.update({value.hand_x, value.hand_y}, static_cast<double>(i) / 60.0, true);
        if (!result.point) { ++rejected; return 0.0; }
        return result.point->x + result.point->y;
    });

    dsp::TrackingFilter2D hand(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
    dsp::TrackingFilter2D gaze(dsp::kInitialGazeConfig, dsp::kInitialGazeConfig);
    const bool combined_ok = measure("hand_plus_gaze", [&](std::size_t i) {
        const auto& value = input[i % kSignalSize];
        const double timestamp = static_cast<double>(i) / 60.0;
        const auto h = hand.update({value.hand_x, value.hand_y}, timestamp, true);
        const auto g = gaze.update({value.gaze_x, value.gaze_y}, timestamp, true);
        if (!h.point || !g.point) { ++rejected; return 0.0; }
        return h.point->x + h.point->y + g.point->x + g.point->y;
    });
    std::cout << "Rejected updates: " << rejected << '\n';
    return scalar_ok && point_ok && combined_ok && rejected == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
