#pragma once

#include "os_integration/gesture_interpreter.hpp"
#include "os_integration/screen_mapper.hpp"
#include "os_integration/types.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace osi {

// ------------------------------------------------------------ cursor range
//
// The wizard shows targets at known screen fractions (u, v). While the user
// points at / looks at each one, it records the raw normalized input. Fitting
// input = offset + slope * screen per axis (least squares over the per-target
// medians) gives the input value at the screen edges, u = 0 and u = 1, which
// is exactly the InputRegion the mapper needs.

struct CalibrationTarget {
    double u; // horizontal screen fraction
    double v; // vertical screen fraction
};

// Centre first, then the four corners at 12% / 88%.
std::array<CalibrationTarget, 5> default_calibration_targets() noexcept;

struct TargetObservation {
    CalibrationTarget target;
    std::vector<NormalizedPoint> samples;
};

struct RegionFit {
    InputRegion region;
    // Root-mean-square distance between each target and where its median
    // input would map, as a fraction of the screen (0.05 = 5% of the screen).
    double rms_error;
};

// One target after the fit: what was measured, and how far its median lands
// from the target (screen fractions). rms_error is sqrt(mean(error_u^2 + error_v^2)).
struct TargetFit {
    std::size_t index;     // position in the observations (the order the targets were shown)
    CalibrationTarget target;
    std::size_t samples;   // finite samples (0 = no usable sample)
    double median_x;       // median raw input (NaN without samples)
    double median_y;
    double error_u;        // where the median maps minus the target, horizontally (NaN if no line was fitted)
    double error_v;
};

struct RegionFitResult {
    std::optional<RegionFit> fit;   // the accepted fit
    std::string error;              // user-facing reason when fit is empty
    std::vector<TargetFit> targets; // every observed target, also those without usable samples, also for a rejection
    // The region and RMS whenever they could be computed, also when the fit was then rejected (implausible region,
    // RMS above the limit). Empty when the fit never existed (too few targets or samples, no movement).
    std::optional<RegionFit> computed;
};

struct RegionFitOptions {
    std::size_t min_samples_per_target = 5;
    double max_rms_error = 0.12;
};

RegionFitResult fit_input_region(const std::vector<TargetObservation>& observations,
                                 const RegionFitOptions& options = {});

// Every sample of every target as CSV (target_index,target_u,target_v,x,y; full precision), so a fit can be
// recomputed exactly. A target without any sample gets one row with x and y empty, so it is never lost.
std::string observations_csv(const std::vector<TargetObservation>& observations);

// ------------------------------------------------------------ blink timing
//
// The gaze model classifies every frame, so a reflexive blink can briefly
// look like a wink. The wizard records raw eye states while the user first
// blinks naturally, then winks on purpose, and picks thresholds between the
// two populations.

struct StateRun {
    EyeState state;
    double start;
    double end; // timestamp of the last frame in the run
    [[nodiscard]] double duration() const noexcept { return end - start; }
};

// Collapses a per-frame state stream into runs of identical non-neutral
// states. Interruptions no longer than `glitch_tolerance_seconds` are merged.
class EyeStateRecorder {
public:
    explicit EyeStateRecorder(double glitch_tolerance_seconds = 0.05) noexcept
        : glitch_tolerance_(glitch_tolerance_seconds) {}

    void add(EyeState state, double timestamp_seconds);
    [[nodiscard]] std::vector<StateRun> runs() const;
    void clear() noexcept;

private:
    struct Sample {
        EyeState state;
        double timestamp;
    };
    double glitch_tolerance_;
    std::vector<Sample> samples_;
};

struct EyeTimingRecommendation {
    double wink_onset_seconds;
    double closure_onset_seconds;
    std::string summary; // what was measured, for the wizard's result screen
};

struct EyeTimingResult {
    std::optional<EyeTimingRecommendation> recommendation;
    std::string error;
};

EyeTimingResult recommend_eye_timing(const std::vector<StateRun>& natural_blinks,
                                     const std::vector<StateRun>& left_winks,
                                     const std::vector<StateRun>& right_winks,
                                     const GestureSettings& current);

} // namespace osi
