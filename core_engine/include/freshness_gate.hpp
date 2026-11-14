// The gate between a tracker result and the DSP filter (CORE_ENGINE_INTEGRATION_REPORT, section 8.3).
//
// Rule: only a result that was really OBSERVED this frame (TrackingState::Tracked) may update the filter.
//   Tracked : filter(x, y, t) is called once, the filtered point is published and cached.
//   Held    : the filter is NOT called (its history must not see a repeated, never-measured sample). The cached FILTERED
//             point is republished for cursor continuity; with no cache nothing is published. The raw retained
//             coordinates are never published.
//   Lost    : the filter is reset (no glide across the gap), the cache is cleared, nothing is published.
//   unset   : a module without freshness info (gaze, disabled module, Python error): the old behaviour, unchanged:
//             detected() -> filter + publish, otherwise reset + nothing.
//
// Held is a Core decision. It is deliberately NOT expressed as filter(..., tracking_valid = false): that resets the DSP
// filter and the adapter would then raw-pass the input (dsp_adapter.hpp), which is the opposite of a hold.
//
// Header-only and free of OS / OpenCV / Python so the unit tests can drive it with a spy filter. One gate per tracker,
// used by one thread (the engine's main loop).
#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>

#include "euro_builtin.hpp"      // core::IPointFilter
#include "tracking_types.hpp"

namespace core {

struct GatedPoint {
    double x = 0.0, y = 0.0;     // normalised 0..1 (clamped), what the OS stage receives
    bool held = false;           // true: the cached point of an earlier observation (HELD), false: a fresh one
};

class FreshnessGate {
public:
    struct Counters {
        uint64_t filter_updates = 0;        // calls into the filter
        uint64_t held_reused = 0;           // HELD frames answered from the cache
        uint64_t held_without_cache = 0;    // HELD frames with nothing to publish (no TRACKED point since the last reset)
        uint64_t resets = 0;                // filter resets caused by LOST
    };

    // t_sec must increase between calls (the engine's frame timestamp). Returns the point to publish, or nothing.
    std::optional<GatedPoint> update(const ModuleOut& m, IPointFilter& flt, bool use_filter, double t_sec) {
        if (!m.tracking_state) return m.detected() ? fresh(m, flt, use_filter, t_sec) : lose(flt);
        switch (*m.tracking_state) {
            case TrackingState::Tracked:
                return m.detected() ? fresh(m, flt, use_filter, t_sec) : lose(flt);   // "tracked" without a position: contract violation
            case TrackingState::Held:
                if (!m.detected()) return lose(flt);                                   // same: never invent a point
                if (cache_) {
                    ++c_.held_reused;
                    GatedPoint p = *cache_;
                    p.held = true;
                    return p;
                }
                ++c_.held_without_cache;
                return std::nullopt;
            case TrackingState::Lost:
                return lose(flt);
        }
        return lose(flt);
    }

    // Forget everything WITHOUT touching a filter: used when the filter object is replaced (new smoothing settings).
    void reset() { had_ = false; cache_.reset(); }

    bool hasCache() const { return cache_.has_value(); }
    const Counters& counters() const { return c_; }

private:
    std::optional<GatedPoint> fresh(const ModuleOut& m, IPointFilter& flt, bool use_filter, double t_sec) {
        double x = m.x, y = m.y;
        if (use_filter) { flt.filter(x, y, t_sec); ++c_.filter_updates; }
        cache_ = GatedPoint{std::min(1.0, std::max(0.0, x)), std::min(1.0, std::max(0.0, y))};
        had_ = true;
        return cache_;
    }

    std::optional<GatedPoint> lose(IPointFilter& flt) {
        if (had_) { flt.reset(); had_ = false; ++c_.resets; }
        cache_.reset();
        return std::nullopt;
    }

    bool had_ = false;                    // the filter holds history from a TRACKED sample
    std::optional<GatedPoint> cache_;     // last published (filtered) TRACKED point
    Counters c_;
};

} // namespace core
