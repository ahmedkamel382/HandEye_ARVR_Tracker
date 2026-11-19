// Adapter: Engineer 4's dsp::TrackingFilter2D (dsp_filter/) behind the engine's IPointFilter interface.
//
// Compiled only when the dsp_filter target exists (CMake defines CORE_HAVE_DSP_FILTER).
// What the adapter adds, and nothing more:
//   * clamps the incoming X/Y to [0,1]    (Eng 4's filter resets itself on out-of-range input)
//   * update() reports WHAT happened (FilterOutcome), so the engine never mistakes a substitute for a smoothed point:
//       Ok        the smoothed point of this observation
//       Rejected  duplicate / not increasing timestamp: history intact, x/y = last good output (or the raw clamped input)
//       Invalid   non-finite input, InvalidInput or NumericalError: the filter is reset, x/y = raw clamped input
//                 (untouched when the input was not finite); there is NO valid output and the engine must not publish it
//   * filter() is the old status-less call: same x/y values as update(), the outcome is dropped.
// Units match the engine: normalised 0..1 coordinates, time in seconds.
#pragma once
#ifdef CORE_HAVE_DSP_FILTER
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "euro_builtin.hpp"        // core::IPointFilter
#include "one_euro_filter.hpp"     // dsp::TrackingFilter2D  (resolved from dsp_filter/include)
#include "dsp_trace_recorder.hpp"  // optional numerical observation; no recording I/O in update()

namespace core {

class DspFilterAdapter : public IPointFilter {
public:
    // Throws std::invalid_argument if the configuration is invalid (Eng 4's validation).
    explicit DspFilterAdapter(dsp::FilterConfig cfg, dsp::TimingConfig timing = {},
                              dsp::recording::Observation* observation = nullptr)
        : f_(cfg, cfg, timing), observation_(observation) {}

    // The caller owns this slot and clears it before each frame. It must outlive the adapter.
    void observe(dsp::recording::Observation* observation) noexcept { observation_ = observation; }

    void reset() override {
        f_.reset(); have_last_ = false;
        if (observation_) ++observation_->resets;
    }

    void filter(double& x, double& y, double t_sec) override { (void)update(x, y, t_sec); }

    FilterOutcome update(double& x, double& y, double t_sec) override {
        if (observation_) {
            const auto resets = observation_->resets;
            *observation_ = {};
            observation_->resets = resets;
            observation_->adapter_called = true;
            observation_->timestamp_seconds = t_sec;
            observation_->input_x = x;
            observation_->input_y = y;
        }
        if (!std::isfinite(x) || !std::isfinite(y)) {                            // never expected; x, y left as they are
            ++errors_;
            f_.reset();                                                          // same as the DSP does for non-finite input
            have_last_ = false;
            if (observation_) ++observation_->resets;
            return observedOutcome(FilterOutcome::Invalid);
        }
        const double rx = std::min(1.0, std::max(0.0, x)), ry = std::min(1.0, std::max(0.0, y));
        const dsp::TrackingResult2D r = f_.update({rx, ry}, t_sec, true);
        if (observation_) {
            observation_->update_called = true;
            observation_->input_x = rx;
            observation_->input_y = ry;
            observation_->status = static_cast<int>(r.status);
            observation_->has_point = r.point.has_value();
            if (r.point) {
                observation_->filtered_x = r.point->x;
                observation_->filtered_y = r.point->y;
            }
        }
        if (r.point) {
            x = last_x_ = r.point->x;
            y = last_y_ = r.point->y;
            have_last_ = true;
            return observedOutcome(FilterOutcome::Ok);
        }
        if (r.status == dsp::FilterStatus::RejectedTimestamp) {
            ++rejected_;
            if (have_last_) { x = last_x_; y = last_y_; } else { x = rx; y = ry; }
            return observedOutcome(FilterOutcome::Rejected);
        }
        ++errors_;                                   // the filter already reset itself
        have_last_ = false;
        x = rx; y = ry;
        return observedOutcome(FilterOutcome::Invalid);
    }

    uint64_t rejectedTimestamps() const { return rejected_; }
    uint64_t errors() const { return errors_; }

private:
    FilterOutcome observedOutcome(FilterOutcome outcome) noexcept {
        if (observation_) observation_->outcome = static_cast<int>(outcome);
        return outcome;
    }
    dsp::TrackingFilter2D f_;
    dsp::recording::Observation* observation_ = nullptr;
    bool have_last_ = false;
    double last_x_ = 0.0, last_y_ = 0.0;
    uint64_t rejected_ = 0, errors_ = 0;
};

} // namespace core
#endif
