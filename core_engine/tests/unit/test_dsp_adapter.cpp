// Tests for the bridge between the engine and Engineer 4's dsp_filter:
//   * the adapter returns exactly what dsp::TrackingFilter2D returns
//   * it reproduces the jitter-reduction numbers published in Eng 4's README (same generator, same config)
//   * rejected timestamps, resets, gaps, bad values never freeze or jump the pointer
// Needs the dsp_filter target (CMake adds this test only when it exists).
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "dsp_adapter.hpp"
#include "freshness_gate.hpp"
#include "hand_math.hpp"
#include "tracking_types.hpp"

using namespace core;

static int g_checks = 0, g_failed = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_failed; std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECK_NEAR(a, b, eps) do { ++g_checks; const double _a = (a), _b = (b); \
    if (!(std::fabs(_a - _b) <= (eps))) { ++g_failed; std::printf("    FAIL %s:%d  %s = %.9g, expected %.9g (+-%g)\n", \
    __FILE__, __LINE__, #a, _a, _b, double(eps)); } } while (0)

// ---------------------------------------------------------------------------------------------
static void t_matches_dsp_directly() {
    DspFilterAdapter a(dsp::kInitialHandConfig);
    dsp::TrackingFilter2D ref(dsp::kInitialHandConfig, dsp::kInitialHandConfig);
    std::mt19937 g(7);
    std::normal_distribution<double> n(0.0, 0.004);
    double px = 0.5, py = 0.5;
    int identical = 0;
    for (int i = 0; i < 600; ++i) {
        px = std::min(1.0, std::max(0.0, px + n(g) * 3));                 // random walk with jitter
        py = std::min(1.0, std::max(0.0, py + n(g) * 3));
        double x = px + n(g), y = py + n(g);
        x = std::min(1.0, std::max(0.0, x)); y = std::min(1.0, std::max(0.0, y));
        const auto r = ref.update({x, y}, i / 60.0, true);
        a.filter(x, y, i / 60.0);
        if (r.point && x == r.point->x && y == r.point->y) ++identical;
    }
    CHECK(identical == 600);
    CHECK(a.errors() == 0 && a.rejectedTimestamps() == 0);
}

// Eng 4's README: noise = std::mt19937(42) mapped by 2*(u/4294967296.0)-1, 1,200 samples at 60 Hz, first 120 discarded,
// target 0.5 +/- 0.005. Published: raw RMS 0.00290035467949; filtered RMS 0.000791110657601 (hand, 72.7237 %),
// 0.000746262807022 (gaze, 74.2699 %).
static void jitter_case(const dsp::FilterConfig& cfg, double want_filtered_rms, double want_reduction) {
    DspFilterAdapter a(cfg);
    std::mt19937 g(42);
    double raw2 = 0, filt2 = 0;
    int n = 0;
    for (int i = 0; i < 1200; ++i) {
        const double noise = 2.0 * (double(g()) / 4294967296.0) - 1.0;
        double x = 0.5 + 0.005 * noise, y = 0.5;
        const double raw = x;
        a.filter(x, y, i / 60.0);
        if (i >= 120) { raw2 += (raw - 0.5) * (raw - 0.5); filt2 += (x - 0.5) * (x - 0.5); ++n; }
    }
    const double raw_rms = std::sqrt(raw2 / n), filt_rms = std::sqrt(filt2 / n);
    std::printf("    raw RMS %.9g  filtered RMS %.9g  reduction %.4f %%\n", raw_rms, filt_rms, 100.0 * (1.0 - filt_rms / raw_rms));
    CHECK_NEAR(raw_rms, 0.00290035467949, 1e-12);                          // proves the generator matches theirs
    CHECK_NEAR(filt_rms, want_filtered_rms, 1e-11);
    CHECK_NEAR(100.0 * (1.0 - filt_rms / raw_rms), want_reduction, 5e-4);
}
static void t_reproduces_eng4_numbers_hand() { jitter_case(dsp::kInitialHandConfig, 0.000791110657601, 72.7237); }
static void t_reproduces_eng4_numbers_gaze() { jitter_case(dsp::kInitialGazeConfig, 0.000746262807022, 74.2699); }

static void t_rejected_timestamp_holds_last_output() {
    DspFilterAdapter a(dsp::kInitialHandConfig), clean(dsp::kInitialHandConfig);
    double x = 0.2, y = 0.2;
    a.filter(x, y, 0.0); { double cx = 0.2, cy = 0.2; clean.filter(cx, cy, 0.0); }
    x = 0.3; y = 0.3; a.filter(x, y, 1.0 / 60);
    double cx = 0.3, cy = 0.3; clean.filter(cx, cy, 1.0 / 60);
    const double hold_x = x, hold_y = y;
    x = 0.9; y = 0.9; a.filter(x, y, 1.0 / 60);                            // same timestamp again: rejected
    CHECK(x == hold_x && y == hold_y);
    CHECK(a.rejectedTimestamps() == 1);
    x = 0.4; y = 0.4; a.filter(x, y, 2.0 / 60);                            // history intact: equals a filter that never saw the duplicate
    cx = 0.4; cy = 0.4; clean.filter(cx, cy, 2.0 / 60);
    CHECK(x == cx && y == cy);
}

static void t_reset_gap_and_bad_values() {
    DspFilterAdapter a(dsp::kInitialGazeConfig);
    double x = 0.1, y = 0.9;
    a.filter(x, y, 0.0);
    CHECK(x == 0.1 && y == 0.9);                                            // first sample passes through
    x = 0.8; y = 0.2; a.filter(x, y, 5.0);                                  // 5 s gap (> 0.25 s): re-initialised, no glide
    CHECK(x == 0.8 && y == 0.2);
    a.reset();
    x = 0.6; y = 0.4; a.filter(x, y, 5.01);                                 // timestamps may restart after reset()
    CHECK(x == 0.6 && y == 0.4);
    x = 1.7; y = -0.3; a.filter(x, y, 5.02);                                // out of range: clamped, never outside 0..1
    CHECK(x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0);
    x = std::nan(""); y = 0.5; a.filter(x, y, 5.03);                        // NaN: left alone, counted, no crash
    CHECK(a.errors() >= 1);
}

static void t_invalid_config_throws() {
    bool threw = false;
    try { DspFilterAdapter bad(dsp::FilterConfig{0.0, 4.0, 1.0}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { DspFilterAdapter bad(dsp::FilterConfig{1.0, -1.0, 1.0}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------------------------
// update() says WHAT happened, so a substitute value can never pass for a smoothed point (issue "DSP adapter rejection behaviour").
static void t_update_reports_outcome() {
    DspFilterAdapter a(dsp::kInitialHandConfig), ref(dsp::kInitialHandConfig);
    double x = 0.30, y = 0.40, rx = 0.30, ry = 0.40;
    CHECK(a.update(x, y, 0.0) == FilterOutcome::Ok);
    ref.filter(rx, ry, 0.0);
    x = 0.35; y = 0.45; rx = x; ry = y;
    CHECK(a.update(x, y, 1.0 / 60) == FilterOutcome::Ok);
    ref.filter(rx, ry, 1.0 / 60);
    const double good_x = x, good_y = y;                                           // last good output

    x = 0.90; y = 0.90;                                                            // same timestamp again
    CHECK(a.update(x, y, 1.0 / 60) == FilterOutcome::Rejected);
    CHECK(x == good_x && y == good_y && a.rejectedTimestamps() == 1);              // substitute = last good output ...
    x = 0.40; y = 0.50; rx = x; ry = y;                                            // ... and the history is untouched:
    CHECK(a.update(x, y, 2.0 / 60) == FilterOutcome::Ok);
    ref.filter(rx, ry, 2.0 / 60);
    CHECK(x == rx && y == ry);                                                     // identical to a filter that never saw the duplicate

    x = INFINITY; y = 0.5;                                                         // non-finite: no output, filter reset
    CHECK(a.update(x, y, 3.0 / 60) == FilterOutcome::Invalid);
    CHECK(std::isinf(x) && a.errors() == 1);                                       // input left as it was, never "fixed"
    x = 0.60; y = 0.20;
    CHECK(a.update(x, y, 4.0 / 60) == FilterOutcome::Ok);
    CHECK(x == 0.60 && y == 0.20);                                                 // a new track starts from the measurement

    DspFilterAdapter legacy(dsp::kInitialHandConfig);                              // the status-less call: same values as before
    double lx = 0.3, ly = 0.4;
    legacy.filter(lx, ly, 0.0);
    lx = 0.9; ly = 0.9; legacy.filter(lx, ly, 0.0);                                // duplicate -> last good, no outcome
    CHECK(lx == 0.3 && ly == 0.4);

    // the default implementation (filters that cannot fail) reports Ok
    OneEuroPoint builtin(1.0, 8.0, 1.0);
    double bx = 0.5, by = 0.5;
    CHECK(builtin.update(bx, by, 0.0) == FilterOutcome::Ok);
}

static ModuleOut tracked_at(double x, double y) {
    ModuleOut o;
    o.ok = true; o.x = float(x); o.y = float(y);
    o.tracking_state = TrackingState::Tracked;
    return o;
}
// Through the gate with Eng 4's real filter: a rejected timestamp replays the cached point (flagged HELD), a non-finite input
// publishes nothing and starts a new track; in neither case does a substitute value reach the OS.
static void t_gate_handles_adapter_rejection_and_invalid() {
    DspFilterAdapter a(dsp::kInitialHandConfig), ref(dsp::kInitialHandConfig);
    FreshnessGate gate;
    const double t0 = 0.0, t1 = 1.0 / 60, t2 = 2.0 / 60, t3 = 3.0 / 60, t4 = 4.0 / 60;
    auto p0 = gate.update(tracked_at(0.30, 0.30), a, true, t0);
    auto p1 = gate.update(tracked_at(0.32, 0.31), a, true, t1);
    CHECK(p0 && p1 && !p0->held && !p1->held);

    auto dup = gate.update(tracked_at(0.90, 0.90), a, true, t1);                   // duplicate timestamp
    CHECK(dup && dup->held && dup->x == p1->x && dup->y == p1->y);                 // replay of the last real point, not 0.9
    CHECK(gate.counters().rejected == 1 && a.rejectedTimestamps() == 1);

    double rx = float(0.30), ry = float(0.30);                                     // reference: only the accepted observations
    ref.filter(rx, ry, t0);                                                        // (ModuleOut carries floats: same inputs)
    rx = float(0.32); ry = float(0.31); ref.filter(rx, ry, t1);
    rx = float(0.34); ry = float(0.32); ref.filter(rx, ry, t2);
    auto p2 = gate.update(tracked_at(0.34, 0.32), a, true, t2);
    CHECK(p2 && !p2->held && p2->x == rx && p2->y == ry);                          // the duplicate left no trace in the filter

    auto bad = gate.update(tracked_at(INFINITY, 0.5), a, true, t3);                // unusable input
    CHECK(!bad && !gate.hasCache() && gate.counters().invalid == 1);
    auto p4 = gate.update(tracked_at(0.60, 0.20), a, true, t4);                    // new track from the measurement
    CHECK(p4 && !p4->held && std::fabs(p4->x - 0.60) < 1e-6 && std::fabs(p4->y - 0.20) < 1e-6);
}

// ---------------------------------------------------------------------------------------------
// Freshness gate + Eng 4's filter: a HELD frame must not touch the DSP history. Proof: the gated adapter produces EXACTLY the
// outputs of a second adapter that was fed only the real observations; feeding the retained points (the old behaviour) does not.
struct ConstHandModel : IHandModel {
    void classify(const double*, int& c, double& p) override { c = 0; p = 1.0; }
    const char* name() const override { return "const"; }
};
static void t_gate_with_dsp_adapter_matches_observation_only() {
    ConstHandModel model;
    HandNative hand(&model);
    float pk[kHandPacked * 3];
    for (float& v : pk) v = 0.f;
    for (int i = 0; i < 21; ++i) { pk[3 * (21 + i)] = 0.01f * i; pk[3 * (21 + i) + 1] = 0.02f * (i % 5); }
    DspFilterAdapter gated(dsp::kInitialHandConfig), ref(dsp::kInitialHandConfig), naive(dsp::kInitialHandConfig);
    FreshnessGate gate;
    std::optional<GatedPoint> last_pub;
    int mismatches = 0, naive_diff = 0, held = 0, lost = 0, tracked = 0;
    bool after_lost = false;
    for (int i = 0; i < 900; ++i) {
        const double t = i / 60.0;
        const int drop_len = 3 + 3 * ((i / 50) % 3);                           // dropouts of 3, 6, 9 frames
        const bool miss = (i % 50) >= 40 && (i % 50) < 40 + drop_len;
        pk[3 * 8] = float(0.5 + 0.35 * std::sin(i * 0.07));
        pk[3 * 8 + 1] = float(0.5 + 0.25 * std::cos(i * 0.045));
        const ModuleOut r = miss ? hand.process(nullptr, 0) : hand.process(pk, kHandPacked);
        const auto pub = gate.update(r, gated, true, t);

        if (r.tracking_state == TrackingState::Tracked) {
            ++tracked;
            double x = r.x, y = r.y;
            ref.filter(x, y, t);
            if (!pub || pub->x != x || pub->y != y) ++mismatches;
            if (after_lost) { CHECK_NEAR(pub->x, r.x, 1e-9); after_lost = false; }     // new track starts from the measurement: no glide
            last_pub = pub;
        } else if (r.tracking_state == TrackingState::Held) {
            ++held;
            if (!pub || !last_pub || pub->x != last_pub->x || pub->y != last_pub->y) ++mismatches;
        } else {
            ++lost;
            if (pub) ++mismatches;
            ref.reset();
            last_pub.reset();
            after_lost = true;
        }
        if (r.detected()) {                                                     // old behaviour: retained points filtered as fresh
            double x = r.x, y = r.y;
            naive.filter(x, y, t);
            if (r.tracking_state == TrackingState::Tracked && pub && std::fabs(x - pub->x) > 1e-12) ++naive_diff;
        } else naive.reset();
    }
    CHECK(mismatches == 0);
    CHECK(tracked > 0 && held > 0 && lost > 0);
    CHECK(naive_diff > 0);
    CHECK(gate.counters().filter_updates == uint64_t(tracked));
    CHECK(gated.errors() == 0 && gated.rejectedTimestamps() == 0);              // the DSP never saw a bad or repeated timestamp
}

// ---------------------------------------------------------------------------------------------
int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"adapter == dsp::TrackingFilter2D", t_matches_dsp_directly},
        {"reproduces Eng 4's published jitter numbers (hand config)", t_reproduces_eng4_numbers_hand},
        {"reproduces Eng 4's published jitter numbers (gaze config)", t_reproduces_eng4_numbers_gaze},
        {"rejected timestamp keeps last output, history intact", t_rejected_timestamp_holds_last_output},
        {"reset / gap / clamp / NaN", t_reset_gap_and_bad_values},
        {"invalid configuration throws", t_invalid_config_throws},
        {"freshness gate + dsp adapter == observation-only filter", t_gate_with_dsp_adapter_matches_observation_only},
        {"adapter update() reports Ok / Rejected / Invalid", t_update_reports_outcome},
        {"gate: adapter rejection and invalid input", t_gate_handles_adapter_rejection_and_invalid},
    };
    int failed_before = 0;
    for (const auto& t : tests) {
        t.fn();
        std::printf("[%s] %s\n", g_failed == failed_before ? " ok " : "FAIL", t.name);
        failed_before = g_failed;
    }
    std::printf("\n%zu tests, %d checks, %d failed\n", sizeof(tests) / sizeof(tests[0]), g_checks, g_failed);
    return g_failed ? 1 : 0;
}
