#include "os_integration/calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace osi {
namespace {

double median(std::vector<double> values) {
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    const double upper = values[mid];
    if (values.size() % 2 == 1) {
        return upper;
    }
    const double lower = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lower + upper);
}

struct LineFit {
    double offset;
    double slope;
};

// Least-squares input = offset + slope * screen.
std::optional<LineFit> fit_line(const std::vector<double>& screen, const std::vector<double>& input) {
    const double n = static_cast<double>(screen.size());
    double mean_s = 0.0;
    double mean_i = 0.0;
    for (std::size_t k = 0; k < screen.size(); ++k) {
        mean_s += screen[k];
        mean_i += input[k];
    }
    mean_s /= n;
    mean_i /= n;
    double covariance = 0.0;
    double variance = 0.0;
    for (std::size_t k = 0; k < screen.size(); ++k) {
        covariance += (screen[k] - mean_s) * (input[k] - mean_i);
        variance += (screen[k] - mean_s) * (screen[k] - mean_s);
    }
    if (variance < 1e-12) {
        return std::nullopt;
    }
    const double slope = covariance / variance;
    return LineFit{mean_i - slope * mean_s, slope};
}

std::string format(const char* pattern, double a, double b = 0.0, double c = 0.0, double d = 0.0) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), pattern, a, b, c, d);
    return buffer;
}

double longest(const std::vector<StateRun>& runs, EyeState a, EyeState b) {
    double result = 0.0;
    for (const StateRun& run : runs) {
        if (run.state == a || run.state == b) {
            result = std::max(result, run.duration());
        }
    }
    return result;
}

std::vector<double> deliberate_durations(const std::vector<StateRun>& runs, EyeState state) {
    std::vector<double> durations;
    for (const StateRun& run : runs) {
        if (run.state == state && run.duration() >= 0.05) {
            durations.push_back(run.duration());
        }
    }
    return durations;
}

} // namespace

std::array<CalibrationTarget, 5> default_calibration_targets() noexcept {
    return {{{0.5, 0.5}, {0.12, 0.12}, {0.88, 0.12}, {0.88, 0.88}, {0.12, 0.88}}};
}

RegionFitResult fit_input_region(const std::vector<TargetObservation>& observations,
                                 const RegionFitOptions& options) {
    RegionFitResult result;
    if (observations.size() < 3) {
        result.error = "At least three calibration targets are needed.";
        return result;
    }

    std::vector<double> us, vs, xs, ys;
    for (std::size_t k = 0; k < observations.size(); ++k) {
        std::vector<double> x, y;
        for (const NormalizedPoint& p : observations[k].samples) {
            if (std::isfinite(p.x) && std::isfinite(p.y)) {
                x.push_back(p.x);
                y.push_back(p.y);
            }
        }
        if (x.size() < std::max<std::size_t>(1, options.min_samples_per_target)) {
            result.error = "Target " + std::to_string(k + 1) + " got only " +
                           std::to_string(x.size()) +
                           " tracked frames. Keep your hand or face visible to the camera.";
            return result;
        }
        us.push_back(observations[k].target.u);
        vs.push_back(observations[k].target.v);
        xs.push_back(median(x));
        ys.push_back(median(y));
    }

    const auto fx = fit_line(us, xs);
    const auto fy = fit_line(vs, ys);
    if (!fx || !fy) {
        result.error = "Calibration targets must differ in both directions.";
        return result;
    }
    if (std::abs(fx->slope) < kMinRegionSpan || std::abs(fy->slope) < kMinRegionSpan) {
        result.error = std::abs(fx->slope) < kMinRegionSpan
                           ? "Almost no left-right movement was measured between targets."
                           : "Almost no up-down movement was measured between targets.";
        result.error += " Point or look further toward each target.";
        return result;
    }

    double squared = 0.0;
    for (std::size_t k = 0; k < us.size(); ++k) {
        const double du = (xs[k] - fx->offset) / fx->slope - us[k];
        const double dv = (ys[k] - fy->offset) / fy->slope - vs[k];
        squared += du * du + dv * dv;
    }
    const double rms = std::sqrt(squared / static_cast<double>(us.size()));

    const InputRegion region{fx->offset, fy->offset, fx->offset + fx->slope, fy->offset + fy->slope};
    if (!is_valid_region(region)) {
        result.error = "The measured range is implausible; please try again.";
        return result;
    }
    if (rms > options.max_rms_error) {
        result.error = format("Targets were inconsistent (average error %.0f%% of the screen). "
                              "Hold steady on each target and try again.",
                              rms * 100.0);
        return result;
    }
    result.fit = RegionFit{region, rms};
    return result;
}

void EyeStateRecorder::add(EyeState state, double timestamp_seconds) {
    samples_.push_back({state, timestamp_seconds});
}

// Mirrors StateDebouncer: a different state interrupts a run only when it
// appears more than the glitch tolerance after the run's last frame.
std::vector<StateRun> EyeStateRecorder::runs() const {
    std::vector<StateRun> result;
    std::optional<StateRun> current;
    for (const Sample& sample : samples_) {
        const bool active = sample.state != EyeState::Neutral && sample.state != EyeState::Calibrating;
        if (current) {
            if (sample.state == current->state) {
                current->end = sample.timestamp;
                continue;
            }
            if (sample.timestamp - current->end <= glitch_tolerance_) {
                continue;
            }
            result.push_back(*current);
            current.reset();
        }
        if (active) {
            current = StateRun{sample.state, sample.timestamp, sample.timestamp};
        }
    }
    if (current) {
        result.push_back(*current);
    }
    return result;
}

void EyeStateRecorder::clear() noexcept {
    samples_.clear();
}

EyeTimingResult recommend_eye_timing(const std::vector<StateRun>& natural_blinks,
                                     const std::vector<StateRun>& left_winks,
                                     const std::vector<StateRun>& right_winks,
                                     const GestureSettings& current) {
    EyeTimingResult result;
    const auto left = deliberate_durations(left_winks, EyeState::LeftWink);
    const auto right = deliberate_durations(right_winks, EyeState::RightWink);
    if (left.empty() || right.empty()) {
        result.error = std::string("Your ") + (left.empty() ? "left" : "right") +
                       " wink was never recognised by the eye model. Try a firmer wink, "
                       "improve the lighting, or collect more training data for that wink.";
        return result;
    }

    const double natural_wink = longest(natural_blinks, EyeState::LeftWink, EyeState::RightWink);
    const double natural_closure =
        longest(natural_blinks, EyeState::SustainedClosure, EyeState::SustainedClosure);
    const double left_typical = median(left);
    const double right_typical = median(right);
    const double deliberate = std::min(left_typical, right_typical);

    const double lower = std::max(0.10, natural_wink * 1.25);
    const double upper = deliberate * 0.7;
    if (lower > upper) {
        result.error = format("Your deliberate winks (%.2f s) were not clearly longer than "
                              "blinks the model mistook for winks (%.2f s). Hold each wink "
                              "a little longer and try again.",
                              deliberate, natural_wink);
        return result;
    }
    const double wink = std::max(lower, std::min(0.6, 0.5 * (lower + upper)));
    const double closure = std::min(
        3.0, std::max({GestureSettings{}.closure_onset_seconds, natural_closure * 2.0, wink + 0.3}));

    EyeTimingRecommendation recommendation;
    recommendation.wink_onset_seconds = wink;
    recommendation.closure_onset_seconds = closure;
    recommendation.summary =
        format("Natural blinks: longest misread wink %.2f s, longest closure %.2f s.\n",
               natural_wink, natural_closure) +
        format("Deliberate winks: left %.2f s, right %.2f s.\n", left_typical, right_typical) +
        format("Wink threshold %.2f s -> %.2f s. Pause (eyes closed) %.2f s -> %.2f s.",
               current.wink_onset_seconds, wink, current.closure_onset_seconds, closure);
    result.recommendation = recommendation;
    return result;
}

} // namespace osi
