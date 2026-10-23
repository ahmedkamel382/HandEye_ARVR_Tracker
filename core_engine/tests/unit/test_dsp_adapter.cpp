// Tests for the bridge between the engine and Engineer 4's dsp_filter:
//   * the adapter returns exactly what dsp::TrackingFilter2D returns
//   * it reproduces the jitter-reduction numbers published in Eng 4's README (same generator, same config)
//   * rejected timestamps, resets, gaps, bad values never freeze or jump the pointer
// Needs the dsp_filter target (CMake adds this test only when it exists).
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "dsp_adapter.hpp"
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
int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"adapter == dsp::TrackingFilter2D", t_matches_dsp_directly},
        {"reproduces Eng 4's published jitter numbers (hand config)", t_reproduces_eng4_numbers_hand},
        {"reproduces Eng 4's published jitter numbers (gaze config)", t_reproduces_eng4_numbers_gaze},
        {"rejected timestamp keeps last output, history intact", t_rejected_timestamp_holds_last_output},
        {"reset / gap / clamp / NaN", t_reset_gap_and_bad_values},
        {"invalid configuration throws", t_invalid_config_throws},
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
