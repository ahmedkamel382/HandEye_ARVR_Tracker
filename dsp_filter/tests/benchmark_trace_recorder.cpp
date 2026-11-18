#include "dsp_trace_recorder.hpp"
#include "dsp_adapter.hpp"
#include "freshness_gate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef DSP_BUILD_CONFIGURATION
#define DSP_BUILD_CONFIGURATION "unknown"
#endif
#ifndef DSP_COMPILER
#define DSP_COMPILER "unknown"
#endif
#ifndef DSP_ARCHITECTURE
#define DSP_ARCHITECTURE "unknown"
#endif

namespace {
namespace rec = dsp::recording;
using Clock = std::chrono::steady_clock;
constexpr std::size_t samples_per_batch = rec::kCapacity;
constexpr int measured_pairs = 31;
volatile double output_sink = 0.0;
volatile std::uint64_t trace_sink = 0;

struct Input {
    core::ModuleOut m;
    double seconds;
    std::int64_t ms;
};

std::array<Input, samples_per_batch> make_inputs() {
    std::array<Input, samples_per_batch> inputs{};
    double seconds = 0.001234567;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto& input = inputs[i];
        input.m.x = float(0.5 + 0.3 * std::sin(double(i) * 0.031));
        input.m.y = float(0.5 + 0.2 * std::cos(double(i) * 0.017));
        input.m.ok = true;
        input.m.tracking_state = core::TrackingState::Tracked;
        input.seconds = seconds;
        input.ms = std::int64_t(seconds * 1000.0);
        seconds += 0.060 + double(i % 5) * 0.003;
    }
    return inputs;
}

struct Batch {
    double ns_per_frame;
    double checksum;
};

// Keep a runtime choice of observation mode rather than benchmarking a special
// compile-time replacement of the production adapter/gate.
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
Batch measure(const std::array<Input, samples_per_batch>& inputs, bool recording) {
    const dsp::FilterConfig config{1.2, 4.0, 1.0};
    rec::Observation observation;
    core::DspFilterAdapter filter(config, {}, recording ? &observation : nullptr);
    core::FreshnessGate gate;
    std::unique_ptr<rec::Recorder> recorder;
    if (recording) {
        recorder = std::make_unique<rec::Recorder>(rec::Channel::Hand);
        recorder->set_initial_parameters(config);
    }
    double checksum = 0.0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (recording) observation = {};
        const auto point = gate.update(input.m, filter, true, input.seconds);
        if (point) checksum += point->x + 0.5 * point->y;
        if (recording) {
            rec::FrameRecord frame;
            frame.frame_id = i;
            frame.mediapipe_ms = frame.capture_ms = input.ms;
            frame.dsp_seconds = input.seconds;
            frame.detector_count = 42;
            frame.ok = true;
            frame.tracker_x = input.m.x; frame.tracker_y = input.m.y;
            frame.tracking = 0;
            frame.dsp = observation;
            frame.published = bool(point);
            if (point) {
                frame.published_x = point->x; frame.published_y = point->y;
                frame.published_tracking = point->held ? 1 : 0;
            }
            recorder->append(frame);
        }
    }
    const auto stop = Clock::now();
    output_sink = checksum;

    if (recording) {
        if (!recorder->complete(samples_per_batch)) throw std::runtime_error("benchmark recording incomplete");
        // Consume every retained field AFTER timing; this makes buffer stores
        // observable and prevents dead-store elimination without timing formatting.
        std::ostringstream serialized;
        if (!recorder->write(serialized, samples_per_batch)) throw std::runtime_error("benchmark serialization failed");
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char c : serialized.str()) { hash ^= c; hash *= 1099511628211ULL; }
        trace_sink = hash;
    }
    return {std::chrono::duration<double, std::nano>(stop - start).count() /
            double(samples_per_batch), checksum};
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

void report(const char* mode, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t p95 = std::size_t(std::ceil(0.95 * double(values.size()))) - 1;
    std::cout << mode << " mean_ns_per_frame=" << mean(values)
              << " median_batch_average_ns=" << values[values.size() / 2]
              << " p95_batch_average_ns=" << values[p95]
              << " min_batch_average_ns=" << values.front()
              << " max_batch_average_ns=" << values.back() << '\n';
}
} // namespace

int main() {
    try {
        const auto inputs = make_inputs();
        // Two complete unreported warm-up batches for each mode.
        for (int i = 0; i < 2; ++i) { measure(inputs, false); measure(inputs, true); }
        std::vector<double> disabled, enabled, deltas;
        disabled.reserve(measured_pairs); enabled.reserve(measured_pairs); deltas.reserve(measured_pairs);
        for (int i = 0; i < measured_pairs; ++i) {
            Batch off{}, on{};
            if (i % 2 == 0) { off = measure(inputs, false); on = measure(inputs, true); }
            else { on = measure(inputs, true); off = measure(inputs, false); }
            if (off.checksum != on.checksum) throw std::runtime_error("recording changed filtering checksum");
            disabled.push_back(off.ns_per_frame); enabled.push_back(on.ns_per_frame);
            deltas.push_back(on.ns_per_frame - off.ns_per_frame);
        }
        std::cout << std::setprecision(10)
                  << "RECORDER_MICROBENCHMARK configuration=" << DSP_BUILD_CONFIGURATION
                  << " compiler=" << DSP_COMPILER << " architecture=" << DSP_ARCHITECTURE << '\n'
                  << "samples_per_batch=" << samples_per_batch << " measured_pairs=" << measured_pairs
                  << " warmup_batches_per_mode=2 alternating_order=true\n";
        report("disabled", disabled); report("enabled", enabled);
        std::cout << "paired_mean_increment_ns_per_frame=" << mean(deltas)
                  << " mean_relative_increment_percent=" << 100.0 * (mean(enabled) / mean(disabled) - 1.0) << '\n'
                  << "output_checksum=" << output_sink << " retained_trace_checksum=" << trace_sink << '\n'
                  << "Synthetic production-adapter/gate microbenchmark only; no camera, AI, OS calls or disk IO.\n"
                  << "Preallocation and serialization excluded. Times are machine/build-specific batch averages,\n"
                  << "not individual worst-case update times, live cadence impact, or end-to-end latency.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
