// Unit tests for the header-only parts of the core engine. No camera, no Python, no OpenCV.
//   build + run:  ctest --output-on-failure      (or run core_unit_tests directly)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "build_paths.hpp"
#include "forest.hpp"
#include "frame_clock.hpp"
#include "gaze_math.hpp"
#include "hand_math.hpp"
#include "euro_builtin.hpp"
#include "freshness_gate.hpp"
#include "tl_model.hpp"
#include "tracking_types.hpp"

using namespace core;

static int g_checks = 0, g_failed = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_failed; std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECK_NEAR(a, b, eps) do { ++g_checks; const double _a = (a), _b = (b); \
    if (!(std::fabs(_a - _b) <= (eps))) { ++g_failed; std::printf("    FAIL %s:%d  %s = %.9g, expected %.9g (+-%g)\n", \
    __FILE__, __LINE__, #a, _a, _b, double(eps)); } } while (0)

static double sq(double v) { return v * v; }

// ======================================================================= 1-Euro filter
static void t_filter_calms_jitter() {
    OneEuroPoint f(1.0, 8.0, 1.0);
    std::mt19937 g(1);
    std::normal_distribution<double> noise(0.0, 0.01);
    double raw = 0, filt = 0;
    int n = 0;
    for (int i = 0; i < 900; ++i) {
        double x = 0.5 + noise(g), y = 0.5 + noise(g);
        const double rx = x, ry = y;
        f.filter(x, y, i / 60.0);
        if (i >= 120) { raw += sq(rx - 0.5) + sq(ry - 0.5); filt += sq(x - 0.5) + sq(y - 0.5); ++n; }
    }
    const double reduction = std::sqrt(filt / n) / std::sqrt(raw / n);
    std::printf("    stationary jitter: filtered/raw RMS = %.3f\n", reduction);
    CHECK(reduction < 0.35);                       // jitter at least ~3x smaller when still
}
static void t_filter_follows_fast_motion() {
    OneEuroPoint f(1.0, 8.0, 1.0);
    double lag_at_end = 0;
    for (int i = 0; i <= 60; ++i) {                // 0.2 -> 0.8 in one second (0.6 units/s)
        double x = 0.2 + 0.6 * i / 60.0, y = 0.5;
        const double truth = x;
        f.filter(x, y, i / 60.0);
        lag_at_end = truth - x;
    }
    std::printf("    lag at 0.6 units/s: %.4f units\n", lag_at_end);
    CHECK(lag_at_end >= 0.0);
    CHECK(lag_at_end < 0.06);
}
static void t_filter_reset_and_edges() {
    OneEuroPoint f;
    double x = 0.3, y = 0.7;
    f.filter(x, y, 1.0);
    CHECK_NEAR(x, 0.3, 1e-12);                     // first sample passes through
    x = 0.9; y = 0.1; f.filter(x, y, 1.0);         // same timestamp twice: must not produce NaN/inf
    CHECK(std::isfinite(x) && std::isfinite(y));
    f.reset();
    x = 0.55; y = 0.45; f.filter(x, y, 50.0);
    CHECK_NEAR(x, 0.55, 1e-12);                    // after reset: raw again
}

// ======================================================================= hand maths
// Golden features: world landmarks -> 24 features, computed by the REAL hand_features.extract_features().
struct HandGolden { float world[63]; double feats[24]; };
static const HandGolden kGolden[] = {
    // open
    { { 0.00110675697f, -0.000126171944f, -0.00919817667f, -0.0141807282f, -0.0073402985f, -0.0165296365f, -0.030513458f, -0.019961495f, -0.00282992679f, -0.0475788377f, -0.0269586295f, 0.00412868103f, -0.0606086701f, -0.0380958542f, 0.0105680851f, -0.020182658f, -0.0597104169f, 0.00154663541f, -0.0202187002f, -0.0695094019f, -0.00409329776f, -0.0219956338f, -0.0861557424f, -0.0100512365f, -0.0179338865f, -0.101912737f, 0.00351059809f, -0.000466845057f, -0.0605115779f, 0.0182718318f, -0.00236289366f, -0.0733739212f, 0.000529742218f, -0.000662581762f, -0.0913809463f, -0.00943582412f, 0.00113642623f, -0.100136541f, -0.00829881895f, 0.0166899897f, -0.055318635f, 0.014786968f, 0.0148599464f, -0.0691492185f, 0.00123572291f, 0.0153776025f, -0.0867424458f, -0.0161029156f, 0.0142701864f, -0.102304727f, -0.000352199073f, 0.0272060391f, -0.0527299494f, -0.00984069332f, 0.0244030301f, -0.0682522357f, -0.00885786489f, 0.0289339367f, -0.0814212263f, -0.00682213111f, 0.0247279499f, -0.0951577425f, -0.00924281497f },
      { 121.76450293007612, 161.21982373551467, 162.95014248606944, 136.2349570247485, 168.3737885602184, 117.58134459353332, 101.55231223191684, 153.13225149723351, 143.40311611232858, 109.81268663278381, 173.39818489005131, 90.012710002022033, 143.15703497996313, 150.56080879193419, 139.92887935614345, 1.16178249322065, 1.3493379777879997, 1.4955314663193999, 1.5755405911357667, 1.1318472711697536, 1.5721998713439151, 1.5071790481876377, 1.5582294070655003, 1.4756659033756514 } },
      // pinch
      { { 0.00380237447f, 0.000958236284f, -0.0158729609f, -0.000732940389f, -0.0245043766f, 0.00425710389f, -0.00658590114f, -0.0503646843f, -0.0131975142f, -0.0138670169f, -0.0775868669f, -0.00312814349f, -0.0146623645f, -0.101450101f, 0.000774032844f, -0.0142487083f, -0.0594291799f, -0.00393830892f, -0.0186128747f, -0.0739685073f, -0.00259263394f, -0.0180305708f, -0.0858322755f, 0.0123245558f, -0.0178912189f, -0.100555584f, -0.00457581924f, 0.00173734035f, -0.059077207f, -0.0116170598f, 0.0036532681f, -0.0736036599f, -0.0181085151f, 0.00141988765f, -0.091538243f, -0.00272307079f, 0.00158644177f, -0.103738435f, -0.000425846723f, 0.0150209498f, -0.0599246547f, 0.00654540304f, 0.0171857886f, -0.0739674717f, -0.00189079857f, 0.0150648095f, -0.0903278589f, -0.00562709244f, 0.0139821963f, -0.0988183022f, -0.00879729446f, 0.0264927577f, -0.051553525f, -0.00595554337f, 0.0251718443f, -0.0654118285f, 0.00375587633f, 0.0282479469f, -0.0825550035f, -0.00610315893f, 0.0272379387f, -0.0974131674f, -0.000255912775f },
        { 108.70557678417542, 126.93683558344802, 163.65212480065196, 174.34313340977394, 130.83643341387958, 79.573523381886673, 150.56490643667419, 114.27422163143959, 149.44900595091454, 129.36868653386085, 156.67788762945261, 172.43622903023098, 143.68582575607846, 113.92516718596481, 127.28209944396012, 0.10481991439204144, 0.27320731510105678, 0.5034047068255304, 0.69920060993438915, 1.7499194383524095, 1.7339123010754212, 1.7577317549336526, 1.669565601139924, 1.6991146387336333 } },
        // fist
        { { 0.000966492516f, 0.00205807108f, -0.0160149913f, -0.0167447701f, -0.00798724499f, 0.0200047512f, -0.0262461603f, -0.0180300009f, 0.00649224594f, -0.0470900424f, -0.0240997039f, -0.0108193355f, -0.0642089248f, -0.0371614024f, 0.00908213295f, -0.0156223392f, -0.0590289347f, 0.0147582376f, -0.0137364026f, -0.0499342792f, -0.0242707133f, -0.00911509991f, -0.0366300344f, -0.00377840386f, -0.00923610479f, -0.0290733799f, -0.00416108686f, -0.000190186125f, -0.0617207177f, 0.00649367133f, -0.000164033248f, -0.0455427393f, -0.00160034059f, 0.00254163588f, -0.0395390131f, -0.00267727068f, 0.00233879429f, -0.0373536311f, -0.000393859518f, 0.0167145263f, -0.0617957301f, 0.0162149146f, 0.0148846479f, -0.0490495227f, 0.0194034074f, 0.00979780592f, -0.0411561802f, -0.00379936979f, 0.00915776379f, -0.0319104567f, -0.0105931256f, 0.0268922411f, -0.0483496934f, 0.0276235845f, 0.0242683925f, -0.0437278636f, -0.0123810787f, 0.0175593048f, -0.0345505252f, -0.00552758807f, 0.0130423168f, -0.0312857479f, 0.000587178918f },
          { 74.217902209342085, 156.67744640191503, 96.45845256423047, 50.715223252466274, 47.801038705286189, 119.3173463266907, 7.1961787533708419, 151.33655824774527, 118.58771810898884, 40.395622036428684, 96.212990677067026, 144.39726946124134, 44.848623093230479, 66.436665958806472, 158.48653667036291, 0.84443565056816672, 0.99372099836180749, 1.1256043922132055, 1.15218936958837, 1.1841272502696594, 0.51503864709430902, 0.62705930007845656, 0.52274181656391272, 0.57886819471331419 } }
};
static void t_hand_features_match_python() {
    for (const HandGolden& g : kGolden) {
        double f[kHandFeatures];
        handFeatures(g.world, f);
        for (int i = 0; i < kHandFeatures; ++i) CHECK_NEAR(f[i], g.feats[i], 1e-9 * (1.0 + std::fabs(g.feats[i])));
    }
    float same[63];                                          // degenerate: all points equal -> palm = 1, angles = 0
    for (float& v : same) v = 0.1f;
    double f[kHandFeatures];
    handFeatures(same, f);
    for (int i = 0; i < kHandFeatures; ++i) CHECK_NEAR(f[i], 0.0, 1e-12);
}

// 42 packed landmarks: image points (cursor) then world points (features)
struct Packed {
    float p[kHandPacked * 3];
    Packed() {
        for (float& v : p) v = 0.f; for (int i = 0; i < kHandPacked; ++i) { p[3 * i + 2] = 0.f; }
        set(8, .42f, .25f); set(5, .40f, .55f); set(4, .30f, .55f); for (int i = 0; i < 21; ++i) w(i, 0.01f * i, 0.02f * (i % 5), 0.f);
    }
    void set(int i, float x, float y) { p[3 * i] = x; p[3 * i + 1] = y; }
    void w(int i, float x, float y, float z) { p[3 * (21 + i)] = x; p[3 * (21 + i) + 1] = y; p[3 * (21 + i) + 2] = z; }
};

static HandNative makeHand(IHandModel& m, int& calls, const HandParams& hp = HandParams()) { (void)calls; return HandNative(&m, hp); }

static void t_hand_vote_gate_grace() {
    int cls = 0; double conf = 1.0;
    LambdaHandModel model([&](const double*, int& c, double& p) { c = cls; p = conf; });
    int dummy = 0;
    HandNative h = makeHand(model, dummy);
    Packed pk;

    ModuleOut o = h.process(pk.p, kHandPacked);               // cursor = image landmark 8, clamped
    CHECK(o.ok && o.detected() && o.state == 0);
    CHECK_NEAR(o.x, 0.42, 1e-6); CHECK_NEAR(o.y, 0.25, 1e-6);

    cls = 1; conf = 0.54;                                      // below 0.55: votes Neutral
    for (int i = 0; i < 5; ++i) o = h.process(pk.p, kHandPacked);
    CHECK(o.state == 0);
    conf = 0.55;                                               // exactly at the gate: counts
    o = h.process(pk.p, kHandPacked);  CHECK(o.state == 0);    // window 0,0,0,0,1 -> still neutral
    o = h.process(pk.p, kHandPacked);  CHECK(o.state == 0);    // 0,0,0,1,1
    o = h.process(pk.p, kHandPacked);  CHECK(o.state == 1);    // 0,0,1,1,1 -> majority pinch (3 frames)

    cls = 2;                                                   // tie: window 0,1,1,2,2? -> first seen among equals
    o = h.process(pk.p, kHandPacked);                          // 0,1,1,1,2 -> 1
    CHECK(o.state == 1);
    o = h.process(pk.p, kHandPacked);                          // 1,1,1,2,2 -> 1
    CHECK(o.state == 1);
    o = h.process(pk.p, kHandPacked);                          // 1,1,2,2,2 -> 2
    CHECK(o.state == 2);

    for (int i = 0; i < 6; ++i) { o = h.process(nullptr, 0); CHECK(o.ok && o.detected() && o.state == 2); }   // held
    o = h.process(nullptr, 0);                                 // 7th miss: genuine loss
    CHECK(o.ok && !o.detected() && o.state == 0);
    o = h.process(pk.p, kHandPacked);                          // votes were cleared: one frame decides
    CHECK(o.state == 2);

    CHECK(!h.process(pk.p, 10).ok);                            // bridge error: not ok, no state change
    CHECK(h.process(pk.p, kHandPacked).ok);
}

// Freshness contract (CORE_ENGINE_INTEGRATION_REPORT): TRACKED / HELD / LOST + missed_frames on every hand result.
static void t_hand_tracking_state_transitions() {
    int cls = 1; double conf = 1.0;
    LambdaHandModel model([&](const double*, int& c, double& p) { c = cls; p = conf; });
    HandNative h(&model);
    Packed pk;
    auto is_lost = [](const ModuleOut& o, int missed) {
        return o.ok && !o.detected() && o.tracking_state == TrackingState::Lost && o.missed_frames == missed &&
               o.x == -1.0f && o.y == -1.0f && o.state == 0;
    };

    for (int i = 1; i <= 10; ++i) CHECK(is_lost(h.process(nullptr, 0), i));       // startup: LOST, never HELD
    ModuleOut fresh = h.process(pk.p, kHandPacked);                                // first observation
    CHECK(fresh.ok && fresh.detected() && fresh.tracking_state == TrackingState::Tracked && fresh.missed_frames == 0);
    CHECK_NEAR(fresh.x, 0.42, 1e-6);                                               // coordinates exactly as before the contract

    ModuleOut o;
    for (int i = 1; i <= 6; ++i) {                                                 // misses 1..6 = HELD, same x, y, state
        o = h.process(nullptr, 0);
        CHECK(o.ok && o.detected() && o.tracking_state == TrackingState::Held && o.missed_frames == i);
        CHECK(o.x == fresh.x && o.y == fresh.y && o.state == fresh.state);         // detected() stays true for ROI / preview
    }
    CHECK(is_lost(h.process(nullptr, 0), 7));                                      // 7th miss: LOST, invalid coordinates
    CHECK(is_lost(h.process(nullptr, 0), 8));
    o = h.process(pk.p, kHandPacked);                                              // reacquisition after LOST
    CHECK(o.tracking_state == TrackingState::Tracked && o.missed_frames == 0 && o.detected());

    for (int i = 0; i < 3; ++i) h.process(nullptr, 0);                             // reacquisition after HELD
    o = h.process(pk.p, kHandPacked);
    CHECK(o.tracking_state == TrackingState::Tracked && o.missed_frames == 0 && h.missedFrames() == 0);

    h.process(nullptr, 0); h.process(nullptr, 0);                                  // bridge error in the middle of a dropout:
    ModuleOut e = h.process(pk.p, 10);                                             // no output, and NO state change
    CHECK(!e.ok && !e.detected() && e.tracking_state == TrackingState::Lost);
    o = h.process(nullptr, 0);
    CHECK(o.tracking_state == TrackingState::Held && o.missed_frames == 3);        // counted 1, 2, (error ignored), 3

    HandNative fresh_tracker(&model);                                              // error at startup: still no track
    CHECK(fresh_tracker.process(pk.p, 10).tracking_state == TrackingState::Lost);
    CHECK(is_lost(fresh_tracker.process(nullptr, 0), 1));

    HandParams no_grace; no_grace.grace_frames = 0;                                // grace 0: the first miss is already LOST
    HandNative g(&model, no_grace);
    g.process(pk.p, kHandPacked);
    CHECK(is_lost(g.process(nullptr, 0), 1));

    CHECK(std::string(toString(TrackingState::Held)) == "HELD");
}

static void t_hand_anchor_clamp_and_fallback() {
    LambdaHandModel model([](const double*, int& c, double& p) { c = 0; p = 1.0; });
    Packed pk;  pk.set(8, 1.4f, -0.2f);
    HandNative tip(&model);
    ModuleOut o = tip.process(pk.p, kHandPacked);
    CHECK_NEAR(o.x, 1.0, 1e-9); CHECK_NEAR(o.y, 0.0, 1e-9);
    HandParams knuckle; knuckle.anchor = HandAnchor::Knuckle;
    HandNative k(&model, knuckle);
    o = k.process(pk.p, kHandPacked);
    CHECK_NEAR(o.x, 0.40, 1e-6); CHECK_NEAR(o.y, 0.55, 1e-6);

    GeometricHandModel g;                                      // no trained model: HandTracker._classify fallback
    double f[kHandFeatures];
    for (double& v : f) v = 1.0;
    f[hand_idx::F_THUMB_TO_INDEX] = 0.2;
    int c = -1; double p = 0;
    g.classify(f, c, p);  CHECK(c == 1 && p == 1.0);           // pinch
    f[hand_idx::F_THUMB_TO_INDEX] = 0.9;
    for (int i = 0; i < 3; ++i) f[hand_idx::F_WRIST_TO_INDEX + i] = 1.0;   // 3 tips < 1.30 -> fist
    f[hand_idx::F_WRIST_TO_INDEX + 3] = 2.0;
    g.classify(f, c, p);  CHECK(c == 2);
    f[hand_idx::F_WRIST_TO_INDEX + 2] = 2.0;                   // only 2 -> neutral
    g.classify(f, c, p);  CHECK(c == 0);
}

// E2-B: malformed structure and nonfinite coordinates must be rejected at the native
// boundary. A bridge error leaves the native tracker state untouched and never reaches
// the classifier. Finite out-of-range coordinates keep the existing clamp (E2B-17).
static void t_hand_invalid_payload_rejection() {
    int classify_calls = 0;
    LambdaHandModel model([&](const double*, int& c, double& p) { ++classify_calls; c = 0; p = 1.0; });
    auto bridge_error = [](const ModuleOut& o, int missed) {
        return !o.ok && !o.detected() && o.tracking_state == TrackingState::Lost &&
               o.missed_frames == missed && o.x == -1.0f && o.y == -1.0f;
    };

    // E2B-01: valid 21+21 still becomes a fresh TRACKED result.
    {
        HandNative h(&model);
        Packed pk;
        ModuleOut o = h.process(pk.p, kHandPacked);
        CHECK(o.ok && o.detected() && o.tracking_state == TrackingState::Tracked);
        CHECK_NEAR(o.x, 0.42, 1e-6);
    }

    // E2B-02/03: incorrect total counts are bridge errors. A 20+22 split that also
    // sums to 42 cannot be distinguished here by count; the Python packer rejects it
    // before the bridge (see check_hand_landmark_diag.py).
    {
        HandNative h(&model);
        Packed pk;
        CHECK(bridge_error(h.process(pk.p, kHandPoints + (kHandPoints - 1)), 0));   // 41
        CHECK(bridge_error(h.process(pk.p, kHandPacked + 1), 0));                   // 43
        CHECK(classify_calls == 1);                                                  // only the valid frame above
    }

    // E2B-08..13: NaN / +/-Inf in image X,Y,Z or world coordinates, including Z.
    {
        HandNative h(&model);
        Packed pk;
        auto set_img = [&](int i, int axis, float v) { pk.p[3 * i + axis] = v; };
        auto set_world = [&](int i, int axis, float v) { pk.p[3 * (kHandPoints + i) + axis] = v; };

        set_img(8, 0, NAN);   CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_img(8, 0, 0.42f);
        set_img(8, 1, NAN);   CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_img(8, 1, 0.25f);
        set_img(8, 2, NAN);   CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_img(8, 2, 0.0f);

        set_world(5, 0, NAN); CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_world(6, 1, NAN); CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_world(7, 2, NAN); CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));

        set_img(8, 0, INFINITY);  CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_img(8, 0, -INFINITY); CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        set_img(8, 0, 0.42f);
        set_world(20, 2, INFINITY); CHECK(bridge_error(h.process(pk.p, kHandPacked), 0));
        CHECK(classify_calls == 1);                                                  // no invalid payload classified
    }

    // E2B-14: invalid frame followed by a valid frame recovers normally.
    {
        HandNative h(&model);
        Packed bad; bad.set(8, NAN, 0.25f);
        CHECK(bridge_error(h.process(bad.p, kHandPacked), 0));
        Packed good;
        ModuleOut o = h.process(good.p, kHandPacked);
        CHECK(o.ok && o.detected() && o.tracking_state == TrackingState::Tracked && o.missed_frames == 0);
    }

    // E2B-15: valid then invalid with no dropout must not mutate retained state.
    {
        HandNative h(&model);
        Packed good;
        ModuleOut v = h.process(good.p, kHandPacked);
        Packed bad; bad.set(8, INFINITY, 0.25f);
        CHECK(bridge_error(h.process(bad.p, kHandPacked), h.missedFrames()));
        ModuleOut held = h.process(nullptr, 0);
        CHECK(held.tracking_state == TrackingState::Held && held.missed_frames == 1);
        CHECK(held.x == v.x && held.y == v.y);
        ModuleOut reacq = h.process(good.p, kHandPacked);
        CHECK(reacq.tracking_state == TrackingState::Tracked && reacq.missed_frames == 0);
    }

    // E2B-16: invalid frame while a hand is retained must not extend or clear grace.
    {
        HandNative h(&model);
        Packed good;
        ModuleOut v = h.process(good.p, kHandPacked);
        ModuleOut first_miss = h.process(nullptr, 0);
        CHECK(first_miss.tracking_state == TrackingState::Held && first_miss.missed_frames == 1);
        Packed bad; bad.set(8, NAN, 0.25f);
        CHECK(bridge_error(h.process(bad.p, kHandPacked), h.missedFrames()));
        ModuleOut second_miss = h.process(nullptr, 0);
        CHECK(second_miss.tracking_state == TrackingState::Held && second_miss.missed_frames == 2);
        CHECK(second_miss.x == v.x && second_miss.y == v.y);
    }

    // E2B-17: finite out-of-range image coordinates keep the existing clamp.
    {
        HandNative h(&model);
        Packed pk; pk.set(8, 1.5f, -0.2f);
        ModuleOut o = h.process(pk.p, kHandPacked);
        CHECK(o.ok && o.detected());
        CHECK_NEAR(o.x, 1.0, 1e-9);
        CHECK_NEAR(o.y, 0.0, 1e-9);
    }
}

// ======================================================================= gaze maths
// 478 landmarks (+ the head-pose matrix rows) with both eyes drawn so that the Eye Aspect Ratio equals the requested value.
struct FacePts {
    std::vector<float> p = std::vector<float>(GazeNative::kPacked * 3, 0.f);
    static constexpr int W = 640, H = 480;
    int count() const { return GazeNative::kPacked; }
    void set(int i, double x, double y) { p[3 * i] = float(x); p[3 * i + 1] = float(y); p[3 * i + 2] = 0.f; }
    void setPose(const float(&m)[16]) { for (int i = 0; i < 16; ++i) p[3 * GazeNative::kFacePoints + i] = m[i]; }
    void eye(const int(&i)[6], double x0, double y0, double ear) {
        const double w = 64.0;                             // eye width in pixels
        const double v = ear * w;                          // each vertical gap in pixels
        set(i[0], x0 / W, y0 / H);       set(i[3], (x0 + w) / W, y0 / H);
        set(i[1], (x0 + w / 3) / W, (y0 - v / 2) / H);  set(i[5], (x0 + w / 3) / W, (y0 + v / 2) / H);
        set(i[2], (x0 + 2 * w / 3) / W, (y0 - v / 2) / H); set(i[4], (x0 + 2 * w / 3) / W, (y0 + v / 2) / H);
    }
    FacePts(double l_ear, double r_ear, double scale = 1.0) {
        for (int i = 0; i < 478; ++i) set(i, 0.5 + 0.0004 * (i % 40 - 20) * scale, 0.5 + 0.0005 * (i / 40 - 6) * scale);
        const int L[6] = { 33, 160, 158, 133, 153, 144 }, R[6] = { 362, 385, 387, 263, 373, 380 };
        eye(L, 200, 200, l_ear);
        eye(R, 380, 200, r_ear);
        set(159, 232.0 / W, (200 - l_ear * 64 / 2) / H);   set(145, 232.0 / W, (200 + l_ear * 64 / 2) / H);   // vertical bounds
        set(386, 412.0 / W, (200 - r_ear * 64 / 2) / H);   set(374, 412.0 / W, (200 + r_ear * 64 / 2) / H);
        set(468, 232.0 / W, 200.0 / H);    set(473, 412.0 / W, 200.0 / H);                                    // irises centred
        const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        setPose(identity);
    }
};
// Models for the tests: the wink classifier records its features, the pitch/yaw regressors return constants.
struct GazeRig {
    std::vector<float> wink_in = std::vector<float>(kGazeFeatures, -1.f), pose_in = std::vector<float>(kPoseFeatures, -1.f);
    int wink_calls = 0;
    double pitch_out = 0.0, yaw_out = 0.0;
    LambdaForest wink{ [this](const float* f) { for (int i = 0; i < kGazeFeatures; ++i) wink_in[i] = f[i]; ++wink_calls; return 0; } };
    LambdaRegressor pitch{ [this](const float* f) { for (int i = 0; i < kPoseFeatures; ++i) pose_in[i] = f[i]; return pitch_out; } };
    LambdaRegressor yaw{ [this](const float*) { return yaw_out; } };
    GazeNative g{ &wink, &pitch, &yaw };
    ModuleOut run(const FacePts& f) { return g.process(f.p.data(), f.count(), FacePts::W, FacePts::H); }
    void calibrate(const FacePts& f) { for (int i = 0; i < 30; ++i) run(f); }
};
static bool all_angles_nan(const ModuleOut& o) {
    return std::isnan(o.yaw_deg) && std::isnan(o.pitch_deg) && std::isnan(o.head_yaw_deg) && std::isnan(o.head_pitch_deg);
}
static void t_gaze_calibration_and_features() {
    GazeRig r;
    FacePts open(0.30, 0.30);
    for (int i = 0; i < 29; ++i) {
        ModuleOut o = r.run(open);
        CHECK(!o.detected() && o.state == -1);                             // calibrating: no pointer (x = y = -1) ...
        CHECK(all_angles_nan(o));                                          // ... and NaN angles (the Python's INVALID_ANGLE)
    }
    CHECK(r.wink_calls == 0 && !r.g.calibrated());
    ModuleOut o = r.run(open);                                             // 30th calibration frame still reports -1
    CHECK(o.state == -1 && r.g.calibrated());
    CHECK(o.x == -1.0f && o.y == -1.0f && all_angles_nan(o));              // the completing frame is still "calibrating"
    for (int i = 0; i < 12; ++i) o = r.run(open);
    CHECK(o.state == 0 && r.wink_calls == 12);
    CHECK_NEAR(r.wink_in[0], 1.0, 1e-4);                                   // normalised EAR (open) = 1
    CHECK_NEAR(r.wink_in[1], 1.0, 1e-4);                                   // rolling min
    CHECK_NEAR(r.wink_in[2], 0.0, 1e-8);                                   // rolling variance
    CHECK_NEAR(r.wink_in[3], 1.0, 1e-4);
    CHECK_NEAR(r.pose_in[0], 0.5, 1e-5);                                   // irises centred: x ratio 0.5 ...
    CHECK_NEAR(r.pose_in[1], 0.0, 1e-5);                                   // ... on the corner line: y ratio 0
    CHECK_NEAR(r.pose_in[2], 0.0, 1e-5);                                   // identity matrix: no head rotation
    FacePts wink(0.075, 0.30);                                             // left eye closes to 25 % of its resting EAR
    r.run(wink);
    CHECK_NEAR(r.wink_in[0], 0.25, 1e-3);
    CHECK_NEAR(r.wink_in[3], 1.0, 1e-3);
    CHECK_NEAR(r.wink_in[1], 0.25, 1e-3);                                  // min over the window drops immediately
}
static void t_gaze_no_face_and_recalibrate() {
    GazeRig r;
    ModuleOut none = r.g.process(nullptr, 0, 640, 480);
    CHECK(none.ok && !none.detected() && none.state == 0 && none.x == -1.0f && none.y == -1.0f && all_angles_nan(none));
    std::vector<float> few(100 * 3, 0.f);
    CHECK(!r.g.process(few.data(), 100, 640, 480).detected());             // too few landmarks
    FacePts open(0.3, 0.3);
    CHECK(!r.g.process(open.p.data(), GazeNative::kFacePoints, 640, 480).detected());   // landmarks but no head-pose matrix = no face
    r.calibrate(open);
    CHECK(r.g.calibrated());
    r.g.recalibrate();
    CHECK(!r.g.calibrated());
    CHECK(r.run(open).state == -1);
}
// EarFeatureState.update(): frames whose raw EAR is not above 1e-6 (either eye) do not count towards the 30.
static void t_gaze_calibration_ignores_zero_ear() {
    GazeRig r;
    FacePts zero(0.0, 0.0), half_zero(0.30, 0.0), open(0.30, 0.30);
    for (int i = 0; i < 40; ++i) { r.run(zero); r.run(half_zero); }
    CHECK(!r.g.calibrated());                                              // 80 bad frames: nothing accumulated
    for (int i = 0; i < 29; ++i) r.run(open);
    CHECK(!r.g.calibrated());
    r.run(zero);                                                           // a bad frame in the middle is simply skipped
    CHECK(!r.g.calibrated());
    r.run(open);
    CHECK(r.g.calibrated());                                               // exactly 30 GOOD frames
}
// EarFeatureState.reset_tracking_state(): face lost -> history cleared; unfinished calibration restarts; finished one kept.
static void t_gaze_face_loss_resets_tracking_state() {
    {   // unfinished calibration restarts, and the baseline is built only from the frames after the loss
        GazeRig r;
        FacePts dim(0.15, 0.15), open(0.30, 0.30);
        for (int i = 0; i < 20; ++i) r.run(dim);
        ModuleOut lost = r.g.process(nullptr, 0, 640, 480);
        CHECK(lost.state == 0 && !lost.detected() && all_angles_nan(lost));
        for (int i = 0; i < 29; ++i) r.run(open);
        CHECK(!r.g.calibrated());                                          // the 20 old frames did NOT count
        r.run(open);
        CHECK(r.g.calibrated());
        r.run(open);
        CHECK_NEAR(r.wink_in[0], 1.0, 1e-4);                               // baseline 0.30 only (a mixed baseline would give ~1.5)
    }
    {   // calibrated: the rolling window is cleared, the baseline survives
        GazeRig r;
        FacePts open(0.30, 0.30), wink(0.075, 0.30);
        r.calibrate(open);
        r.run(open);
        r.run(wink);
        CHECK_NEAR(r.wink_in[1], 0.25, 1e-3);                              // window min holds the wink
        r.g.process(nullptr, 0, 640, 480);                                 // face lost
        CHECK(r.g.calibrated());
        r.run(open);
        CHECK_NEAR(r.wink_in[1], 1.0, 1e-4);                               // stale wink gone from the window
        CHECK_NEAR(r.wink_in[2], 0.0, 1e-8);
        CHECK_NEAR(r.wink_in[0], 1.0, 1e-4);                               // same baseline as before the loss
    }
}
// The freshness fields are hand-only: gaze results leave them unset, so nothing downstream reinterprets gaze.
static void t_gaze_has_no_freshness_fields() {
    GazeRig r;
    FacePts open(0.30, 0.30);
    ModuleOut o = r.run(open);                                             // calibrating
    CHECK(!o.tracking_state.has_value() && o.missed_frames == 0);
    r.calibrate(open);
    o = r.run(open);                                                       // running
    CHECK(o.detected() && !o.tracking_state.has_value() && o.missed_frames == 0);
    o = r.g.process(nullptr, 0, 640, 480);                                 // no face
    CHECK(!o.tracking_state.has_value() && o.missed_frames == 0);
}
static void t_gaze_ratio_follows_iris() {
    GazeRig r;
    FacePts f(0.3, 0.3);
    r.calibrate(f);
    f.set(468, 216.0 / FacePts::W, 190.0 / FacePts::H);                    // left iris: a quarter along the eye, 10 px above the corner line
    r.run(f);
    CHECK_NEAR(r.pose_in[0], 0.375, 1e-5);                                 // mean of 0.25 (left) and 0.5 (right)
    CHECK_NEAR(r.pose_in[1], -0.5 * (10.0 / 480.0) / (64.0 / 640.0), 1e-5);   // above the line is negative; measured in NORMALISED units like the Python: mean of that and 0
}
static void t_head_pose_angles() {
    constexpr double D = 3.14159265358979323846 / 180.0;
    double p, y, ro;
    const float id[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    GazeNative::headPose(id, p, y, ro);
    CHECK_NEAR(p, 0.0, 1e-9); CHECK_NEAR(y, 0.0, 1e-9); CHECK_NEAR(ro, 0.0, 1e-9);
    const float c30 = float(std::cos(30 * D)), s30 = float(std::sin(30 * D));        // 30 deg about Y (yaw)
    const float ry[16] = { c30, 0, s30, 0, 0, 1, 0, 0, -s30, 0, c30, 0, 0, 0, 0, 1 };
    GazeNative::headPose(ry, p, y, ro);
    CHECK_NEAR(p, 0.0, 1e-4); CHECK_NEAR(y, 30.0, 1e-4); CHECK_NEAR(ro, 0.0, 1e-4);
    const float c20 = float(std::cos(20 * D)), s20 = float(std::sin(20 * D));        // 20 deg about X (pitch)
    const float rx[16] = { 1, 0, 0, 0, 0, c20, -s20, 0, 0, s20, c20, 0, 0, 0, 0, 1 };
    GazeNative::headPose(rx, p, y, ro);
    CHECK_NEAR(p, 20.0, 1e-4); CHECK_NEAR(y, 0.0, 1e-4); CHECK_NEAR(ro, 0.0, 1e-4);
    const float c10 = float(std::cos(10 * D)), s10 = float(std::sin(10 * D));        // 10 deg about Z (roll)
    const float rz[16] = { c10, -s10, 0, 0, s10, c10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    GazeNative::headPose(rz, p, y, ro);
    CHECK_NEAR(p, 0.0, 1e-4); CHECK_NEAR(y, 0.0, 1e-4); CHECK_NEAR(ro, 10.0, 1e-4);
    const float lock[16] = { 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1 };      // yaw = 90 deg: gimbal lock branch
    GazeNative::headPose(lock, p, y, ro);
    CHECK_NEAR(y, 90.0, 1e-9); CHECK_NEAR(ro, 0.0, 1e-9);
}
static void t_gaze_adds_head_pose_and_outputs_raw() {
    GazeRig r;
    FacePts f(0.3, 0.3);
    const float c30 = float(std::cos(30 * 3.14159265358979323846 / 180)), s30 = float(std::sin(30 * 3.14159265358979323846 / 180));
    const float ry[16] = { c30, 0, s30, 0, 0, 1, 0, 0, -s30, 0, c30, 0, 0, 0, 0, 1 };
    f.setPose(ry);
    r.pitch_out = 5.0; r.yaw_out = -10.0;                                  // eye-in-socket angles from the models
    r.calibrate(f);
    ModuleOut o = r.run(f);                                                // EMA removed: expect exact sum immediately
    CHECK_NEAR(o.head_yaw_deg, 30.0, 1e-4);
    CHECK_NEAR(o.yaw_deg, 20.0, 1e-4);                                     // -10.0 (eye) + 30.0 (head)
    CHECK_NEAR(o.pitch_deg, 5.0, 1e-4);                                    // 5.0 (eye) + 0.0 (head)
    CHECK_NEAR(o.x, 0.5 + o.yaw_deg / GazeNative::kSpanDeg, 1e-6);         // the 0..1 pointer for the OS stage
    for (int i = 0; i < 80; ++i) o = r.run(f);
    CHECK_NEAR(o.yaw_deg, 20.0, 0.01);                                     // remains exactly rel + head
}

static bool double_bits_eq(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
static bool float_bits_eq(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}
static bool module_out_bit_eq(const ModuleOut& a, const ModuleOut& b) {
    return a.ok == b.ok &&
           a.state == b.state &&
           a.tracking_state == b.tracking_state &&
           a.missed_frames == b.missed_frames &&
           float_bits_eq(a.x, b.x) &&
           float_bits_eq(a.y, b.y) &&
           double_bits_eq(a.yaw_deg, b.yaw_deg) &&
           double_bits_eq(a.pitch_deg, b.pitch_deg) &&
           double_bits_eq(a.head_yaw_deg, b.head_yaw_deg) &&
           double_bits_eq(a.head_pitch_deg, b.head_pitch_deg);
}

static double reconstruct_iris_ratio(double c1x, double c1y, double c2x, double c2y, double ix, double iy, bool x_axis) {
    const double bx = c2x - c1x, by = c2y - c1y;
    const double len_sq = bx * bx + by * by;
    if (len_sq < 1e-12) return x_axis ? 0.5 : 0.0;
    const double len = std::sqrt(len_sq);
    const double vx = ix - c1x, vy = iy - c1y;
    const double scale = (vx * bx + vy * by) / len_sq;
    if (x_axis) return scale;
    const double px = vx - scale * bx, py = vy - scale * by;
    const double d = std::sqrt(px * px + py * py) / len;
    return py < 0.0 ? -d : d;
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) cols.push_back(tok);
    return cols;
}

static double parse_csv_num(const std::string& s) {
    if (s == "nan" || s == "NaN" || s == "NAN") return std::numeric_limits<double>::quiet_NaN();
    return std::strtod(s.c_str(), nullptr);
}

static std::optional<std::string> env_var_opt(const char* name) {
#ifdef _WIN32
    char buf[4096];
    SetLastError(0);
    const DWORD n = GetEnvironmentVariableA(name, buf, static_cast<DWORD>(sizeof(buf)));
    if (n == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
    if (n >= sizeof(buf)) {
        std::string big(n, '\0');
        const DWORD n2 = GetEnvironmentVariableA(name, &big[0], n);
        big.resize(n2);
        return big;
    }
    return std::string(buf, n);
#else
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
#endif
}

static std::string resolve_gaze_models_dir() {
    std::string dir;
    if (const auto env_dir = env_var_opt("CORE_GAZE_MODELS_DIR")) {
        dir = *env_dir;
    } else if (core::paths::kAiPipelineDir[0] != '\0') {
        dir = std::string(core::paths::kAiPipelineDir) + "/gaze_intent/models";
    }
    if (dir.empty()) return dir;
    std::error_code ec;
    const std::filesystem::path abs_p = std::filesystem::absolute(std::filesystem::u8path(dir), ec);
    return ec ? dir : abs_p.generic_string();
}

static void run_gaze_diag_off_vs_on_and_csv_reconstruction(
    IForest* wink, IRegressor* pitch, IRegressor* yaw,
    int& out_bit_mismatches, double& out_max_err) {
    GazeNative g_off(wink, pitch, yaw);
    GazeNative g_on(wink, pitch, yaw);

    const auto hdr_cols = split_csv(kGazeDiagCsvHeader);
    CHECK(hdr_cols.size() == 32);
    CHECK(hdr_cols[0] == "frame_id" && hdr_cols[1] == "ts_ms" && hdr_cols[2] == "capture_s" &&
          hdr_cols[3] == "detected" && hdr_cols[4] == "state" && hdr_cols[18] == "norm_x" &&
          hdr_cols[19] == "norm_y" && hdr_cols[31] == "lm473_y");

    int bit_mismatches = 0;
    double overall_max_err = 0.0;

    for (int i = 0; i < 100; ++i) {
        const bool is_loss = (i >= 80 && i < 90);
        const double l_ear = (i == 55 || i == 56) ? 0.08 : 0.30;
        const double r_ear = 0.30;
        FacePts fp(l_ear, r_ear);

        // Sweep iris position and head pose across frames
        const double dx = 6.0 * std::sin(i * 0.13);
        const double dy = 3.0 * std::cos(i * 0.17);
        fp.set(468, (232.0 + dx) / FacePts::W, (200.0 + dy) / FacePts::H);
        fp.set(473, (412.0 + dx) / FacePts::W, (200.0 + dy) / FacePts::H);

        constexpr double D = 3.14159265358979323846 / 180.0;
        const double yaw_rad = (8.0 * std::sin(i * 0.09)) * D;
        const float cy = float(std::cos(yaw_rad)), sy_m = float(std::sin(yaw_rad));
        const float pose[16] = { cy, 0, sy_m, 0,  0, 1, 0, 0,  -sy_m, 0, cy, 0,  0, 0, 0, 1 };
        fp.setPose(pose);

        GazeDiagRow row;
        ModuleOut out_off = is_loss
            ? g_off.process(nullptr, 0, FacePts::W, FacePts::H)
            : g_off.process(fp.p.data(), fp.count(), FacePts::W, FacePts::H);
        ModuleOut out_on = is_loss
            ? g_on.process(nullptr, 0, FacePts::W, FacePts::H, &row)
            : g_on.process(fp.p.data(), fp.count(), FacePts::W, FacePts::H, &row);

        if (!module_out_bit_eq(out_off, out_on)) ++bit_mismatches;

        row.frame_id = uint64_t(i);
        row.ts_ms = int64_t(i * 16 + (i / 3));
        row.capture_s = double(i) * 0.016666667 + 0.000123456;

        const std::string csv_line = formatGazeDiagCsvRow(row);
        const auto c = split_csv(csv_line);
        CHECK(c.size() == 32);

        // Verify capture_s has exact 9 decimal digits after '.'
        const size_t dot = c[2].find('.');
        CHECK(dot != std::string::npos && (c[2].size() - dot - 1) == 9);

        if (is_loss) {
            CHECK(c[3] == "0" && c[4] == "0");
            CHECK(c[18] == "-1" && c[19] == "-1");
            CHECK(std::isnan(parse_csv_num(c[5])) && std::isnan(parse_csv_num(c[14])) && std::isnan(parse_csv_num(c[20])));
        } else if (i < 30) {
            // Calibration frames 0..29
            CHECK(c[3] == "0" && c[4] == "-1");
            CHECK(c[18] == "-1" && c[19] == "-1");
            CHECK(std::isnan(parse_csv_num(c[14])) && std::isnan(parse_csv_num(c[15])) &&
                  std::isnan(parse_csv_num(c[16])) && std::isnan(parse_csv_num(c[17])));
        } else {
            // Active tracked frame: reconstruct all 12 derived columns from the CSV text alone
            CHECK(c[3] == "1");
            const double l_ix_csv = parse_csv_num(c[5]);
            const double r_ix_csv = parse_csv_num(c[6]);
            const double sx_csv   = parse_csv_num(c[7]);
            const double l_iy_csv = parse_csv_num(c[8]);
            const double r_iy_csv = parse_csv_num(c[9]);
            const double sy_csv   = parse_csv_num(c[10]);
            const double hp_csv   = parse_csv_num(c[11]);
            const double hy_csv   = parse_csv_num(c[12]);
            const double hr_csv   = parse_csv_num(c[13]);
            const double rp_csv   = parse_csv_num(c[14]);
            const double ry_csv   = parse_csv_num(c[15]);
            const double ap_csv   = parse_csv_num(c[16]);
            const double ay_csv   = parse_csv_num(c[17]);
            const double nx_csv   = parse_csv_num(c[18]);
            const double ny_csv   = parse_csv_num(c[19]);

            const double lm33_x  = parse_csv_num(c[20]), lm33_y  = parse_csv_num(c[21]);
            const double lm133_x = parse_csv_num(c[22]), lm133_y = parse_csv_num(c[23]);
            const double lm468_x = parse_csv_num(c[24]), lm468_y = parse_csv_num(c[25]);
            const double lm362_x = parse_csv_num(c[26]), lm362_y = parse_csv_num(c[27]);
            const double lm263_x = parse_csv_num(c[28]), lm263_y = parse_csv_num(c[29]);
            const double lm473_x = parse_csv_num(c[30]), lm473_y = parse_csv_num(c[31]);

            const double rec_l_ix = reconstruct_iris_ratio(lm33_x, lm33_y, lm133_x, lm133_y, lm468_x, lm468_y, true);
            const double rec_r_ix = reconstruct_iris_ratio(lm362_x, lm362_y, lm263_x, lm263_y, lm473_x, lm473_y, true);
            const double rec_sx   = 0.5 * (rec_l_ix + rec_r_ix);
            const double rec_l_iy = reconstruct_iris_ratio(lm33_x, lm33_y, lm133_x, lm133_y, lm468_x, lm468_y, false);
            const double rec_r_iy = reconstruct_iris_ratio(lm362_x, lm362_y, lm263_x, lm263_y, lm473_x, lm473_y, false);
            const double rec_sy   = 0.5 * (rec_l_iy + rec_r_iy);

            const float gf[kPoseFeatures] = { float(rec_sx), float(rec_sy), float(hp_csv), float(hy_csv), float(hr_csv) };
            const double rec_rp = pitch->predict(gf);
            const double rec_ry = yaw->predict(gf);
            const double rec_ap = rec_rp + hp_csv;
            const double rec_ay = rec_ry + hy_csv;
            const double rec_nx = double(float(std::max(0.0, std::min(1.0, 0.5 + rec_ay / GazeNative::kSpanDeg))));
            const double rec_ny = double(float(std::max(0.0, std::min(1.0, 0.5 + rec_ap / GazeNative::kSpanDeg))));

            const double errs[12] = {
                std::fabs(l_ix_csv - rec_l_ix), std::fabs(r_ix_csv - rec_r_ix), std::fabs(sx_csv - rec_sx),
                std::fabs(l_iy_csv - rec_l_iy), std::fabs(r_iy_csv - rec_r_iy), std::fabs(sy_csv - rec_sy),
                std::fabs(rp_csv - rec_rp),     std::fabs(ry_csv - rec_ry),
                std::fabs(ap_csv - rec_ap),     std::fabs(ay_csv - rec_ay),
                std::fabs(nx_csv - rec_nx),     std::fabs(ny_csv - rec_ny)
            };
            for (int k = 0; k < 12; ++k) {
                if (errs[k] > overall_max_err) overall_max_err = errs[k];
                CHECK(errs[k] <= 1e-6);
            }
        }
    }
    CHECK(bit_mismatches == 0);
    out_bit_mismatches = bit_mismatches;
    out_max_err = overall_max_err;
}

// Engineer 4 Requirement #9 (Shipped Treelite DLLs):
//   Requires real gesture_model.dll, gaze_pitch.dll, and gaze_yaw.dll from CORE_GAZE_MODELS_DIR (or
//   configured kAiPipelineDir/gaze_intent/models); never silently falls back to stub lambdas.
static void t_gaze_diag_shipped_dlls() {
#ifdef _WIN32
    const std::string dir = resolve_gaze_models_dir();
    CHECK(!dir.empty());

    std::unique_ptr<IForest> wink_owned;
    std::unique_ptr<IRegressor> pitch_owned, yaw_owned;
    bool loaded_real_dlls = false;
    try {
        wink_owned.reset(new TlForest(dir + "/gesture_model.dll"));
        pitch_owned.reset(new TlRegressor(dir + "/gaze_pitch.dll"));
        yaw_owned.reset(new TlRegressor(dir + "/gaze_yaw.dll"));
        loaded_real_dlls = true;
    } catch (const std::exception& e) {
        std::printf("    shipped gaze DLL load failed (dir='%s'): %s\n", dir.c_str(), e.what());
    }
    CHECK(loaded_real_dlls);
    if (!loaded_real_dlls) return;

    int bit_mismatches = 0;
    double max_err = 0.0;
    run_gaze_diag_off_vs_on_and_csv_reconstruction(
        wink_owned.get(), pitch_owned.get(), yaw_owned.get(), bit_mismatches, max_err);
    std::printf("    diag OFF vs ON (shipped DLLs): 100 frames, %d bit mismatches, max_csv_err=%.3g\n",
                bit_mismatches, max_err);
#else
    bool threw = false;
    try {
        TlForest missing("__nonexistent_gaze_models__/gesture_model.dll");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
#endif
}

// Deliberate absent-DLL diagnostic + explicit synthetic fallback verification:
//   (a) Verifies TlForest/TlRegressor throw explicit errors when model DLLs are absent.
//   (b) Exercises synthetic LambdaForest/LambdaRegressor fallback separately without masking shipped DLLs.
static void t_gaze_absent_dll_and_fallback_diag() {
    auto expect_missing_throws = [](auto&& fn) {
        try {
            fn();
            return false;
        } catch (const std::runtime_error& e) {
            return e.what()[0] != '\0';
        }
    };
    CHECK(expect_missing_throws([] { TlForest("__nonexistent_gaze_models__/gesture_model.dll"); }));
    CHECK(expect_missing_throws([] { TlRegressor("__nonexistent_gaze_models__/gaze_pitch.dll"); }));
    CHECK(expect_missing_throws([] { TlRegressor("__nonexistent_gaze_models__/gaze_yaw.dll"); }));

    LambdaForest wink_fb([](const float* f) { return (f[0] < 0.5f && f[3] > 0.7f) ? 1 : 0; });
    LambdaRegressor pitch_fb([](const float* f) { return -10.0 * double(f[1]) - 0.35 * double(f[2]); });
    LambdaRegressor yaw_fb([](const float* f) { return 8.0 * (double(f[0]) - 0.5) - 0.50 * double(f[3]); });

    int bit_mismatches = 0;
    double max_err = 0.0;
    run_gaze_diag_off_vs_on_and_csv_reconstruction(&wink_fb, &pitch_fb, &yaw_fb, bit_mismatches, max_err);
    std::printf("    diag OFF vs ON (explicit fallback): 100 frames, %d bit mismatches, max_csv_err=%.3g\n",
                bit_mismatches, max_err);
}

// ======================================================================= freshness gate (Core -> DSP)
// A filter that records every call, so "HELD never updates the DSP" is provable. Output = 0.5 * input + 0.25 (never equal to the raw value).
struct SpyFilter : IPointFilter {
    struct Call { double x, y, t; };
    std::vector<Call> calls;
    int resets = 0;
    void reset() override { ++resets; }
    void filter(double& x, double& y, double t) override {
        calls.push_back({x, y, t});
        x = 0.5 * x + 0.25;
        y = 0.5 * y + 0.25;
    }
};
static ModuleOut hand_result(TrackingState ts, double x, double y, int missed = 0, int state = 0) {
    ModuleOut o;
    o.ok = true; o.x = float(x); o.y = float(y); o.state = state;
    o.tracking_state = ts; o.missed_frames = missed;
    return o;
}
static bool same_point(const std::optional<GatedPoint>& p, double x, double y) {
    return p && std::fabs(p->x - x) < 1e-12 && std::fabs(p->y - y) < 1e-12;
}

// E2B downstream consequence: an invalid packed payload becomes a bridge error,
// which the freshness gate treats as LOST: no DSP update, cached point cleared.
static void t_e2b_invalid_payload_never_updates_dsp() {
    int classify_calls = 0;
    LambdaHandModel model([&](const double*, int& c, double& p) { ++classify_calls; c = 0; p = 1.0; });
    HandNative hand(&model);
    Packed good, bad;
    bad.set(8, NAN, 0.25f);

    SpyFilter spy;
    FreshnessGate gate;
    ModuleOut valid = hand.process(good.p, kHandPacked);
    auto p0 = gate.update(valid, spy, true, 1.0);
    CHECK(p0 && spy.calls.size() == 1);

    ModuleOut invalid = hand.process(bad.p, kHandPacked);
    CHECK(!invalid.ok && invalid.tracking_state == TrackingState::Lost);
    auto p1 = gate.update(invalid, spy, true, 1.0 + 1.0 / 60.0);
    CHECK(!p1 && !gate.hasCache());
    CHECK(spy.calls.size() == 1);                       // invalid payload never updated DSP
    CHECK(spy.resets == 1);                             // LOST cleared filter history
    CHECK(classify_calls == 1);                         // invalid payload never reached classifier
}

static void t_gate_tracked_held_lost() {
    SpyFilter spy;
    FreshnessGate gate;
    ModuleOut tracked = hand_result(TrackingState::Tracked, 0.5, 0.75);       // spy: (0.5, 0.75) -> (0.5, 0.625)
    auto p = gate.update(tracked, spy, true, 1.0);
    CHECK(same_point(p, 0.5, 0.625) && !p->held);                              // 1: fresh point filtered and published
    CHECK(spy.calls.size() == 1 && spy.calls[0].x == 0.5f && spy.calls[0].t == 1.0);

    for (int i = 1; i <= 6; ++i) {                                             // 2, 3: HELD x 6 = no DSP update, cached FILTERED point
        p = gate.update(hand_result(TrackingState::Held, 0.5, 0.75, i), spy, true, 1.0 + i / 60.0);
        CHECK(same_point(p, 0.5, 0.625) && p->held);                           // not the raw retained 0.75; flagged as a replay
    }
    CHECK(spy.calls.size() == 1 && spy.resets == 0);
    CHECK(gate.counters().held_reused == 6 && gate.counters().filter_updates == 1);

    p = gate.update(hand_result(TrackingState::Lost, -1, -1, 7), spy, true, 1.2);   // 4: LOST = reset, nothing published
    CHECK(!p && spy.resets == 1 && !gate.hasCache() && spy.calls.size() == 1);
    p = gate.update(hand_result(TrackingState::Lost, -1, -1, 8), spy, true, 1.22);
    CHECK(!p && spy.resets == 1);                                              // a second LOST does not reset again

    p = gate.update(hand_result(TrackingState::Tracked, 0.25, 0.25), spy, true, 1.3);   // 7: reacquisition after LOST = new track
    CHECK(same_point(p, 0.375, 0.375) && !p->held && spy.calls.size() == 2 && spy.calls[1].t == 1.3);
}

static void t_gate_startup_error_and_edges() {
    SpyFilter spy;
    FreshnessGate gate;
    CHECK(!gate.update(hand_result(TrackingState::Lost, -1, -1, 1), spy, true, 0.0));   // 5: startup LOST: no point, no update, no reset
    CHECK(spy.calls.empty() && spy.resets == 0);
    // HELD with nothing cached (cannot come from HandNative, but must never publish the raw retained value)
    CHECK(!gate.update(hand_result(TrackingState::Held, 0.5, 0.5, 1), spy, true, 0.1));
    CHECK(spy.calls.empty() && gate.counters().held_without_cache == 1);

    ModuleOut err;                                                             // 8: bridge error (ok = false): nothing reaches the DSP
    err.tracking_state = TrackingState::Lost;
    gate.update(hand_result(TrackingState::Tracked, 0.5, 0.5), spy, true, 0.2);
    CHECK(!gate.update(err, spy, true, 0.3));
    CHECK(spy.calls.size() == 1 && spy.resets == 1);

    // "Tracked" or "Held" without a usable position is a contract violation: treated as lost, never invented
    gate.update(hand_result(TrackingState::Tracked, 0.5, 0.5), spy, true, 0.4);
    CHECK(!gate.update(hand_result(TrackingState::Tracked, -1, -1), spy, true, 0.5));
    gate.update(hand_result(TrackingState::Tracked, 0.5, 0.5), spy, true, 0.6);
    CHECK(!gate.update(hand_result(TrackingState::Held, -1, -1, 1), spy, true, 0.7) && !gate.hasCache());

    // unset tracking_state (gaze, disabled module, Python error) = the old behaviour
    ModuleOut legacy;  legacy.ok = true; legacy.x = 0.5f; legacy.y = 0.5f;
    const size_t n = spy.calls.size();
    CHECK(gate.update(legacy, spy, true, 0.8) && spy.calls.size() == n + 1);
    ModuleOut none;
    const int r = spy.resets;
    CHECK(!gate.update(none, spy, true, 0.9) && spy.resets == r + 1);

    // --no-filter: raw value published (clamped), the filter is never called, HELD still replays the cache
    SpyFilter spy2;
    FreshnessGate g2;
    CHECK(same_point(g2.update(hand_result(TrackingState::Tracked, 1.6, 0.25), spy2, false, 0.0), 1.0, 0.25));
    CHECK(same_point(g2.update(hand_result(TrackingState::Held, 1.6, 0.25, 1), spy2, false, 0.1), 1.0, 0.25));
    CHECK(spy2.calls.empty());

    // the filter object was replaced (new smoothing settings): cache gone, the NEW filter is not reset
    SpyFilter spy3;
    FreshnessGate g3;
    g3.update(hand_result(TrackingState::Tracked, 0.5, 0.5), spy3, true, 0.0);
    g3.reset();
    CHECK(!g3.update(hand_result(TrackingState::Held, 0.5, 0.5, 1), spy3, true, 0.1) && spy3.resets == 0);
}

// ======================================================================= frame timestamps (capture_s vs ts_ms)
static void t_frame_times_exact_and_strictly_increasing() {
    using namespace std::chrono;
    const Clock::time_point t0 = Clock::now();
    const FrameTimes a = frameTimes(t0, t0 + nanoseconds(123456789), -1);
    CHECK(a.ts_ms == 123);                                                     // MediaPipe clock: truncated milliseconds
    CHECK(std::fabs(a.capture_s - 0.123456789) < 1e-12);                       // DSP clock: nanosecond resolution, not truncated

    const FrameTimes b = frameTimes(t0, t0 + microseconds(16667), -1);         // two frames of a 60 fps stream
    const FrameTimes c = frameTimes(t0, t0 + microseconds(33333), b.ts_ms);
    CHECK(c.ts_ms - b.ts_ms == 17 || c.ts_ms - b.ts_ms == 16);                 // whole milliseconds ...
    CHECK(std::fabs((c.capture_s - b.capture_s) - 0.016666) < 1e-12);          // ... versus the true 16.666 ms

    // frames captured within the same millisecond (a camera burst): MediaPipe gets +1 ms steps, the DSP the true times
    int64_t last = -1;
    double prev_s = -1.0;
    const int offsets_us[] = {10000, 10200, 10400, 10600, 10800, 11000};
    for (int us : offsets_us) {
        const FrameTimes f = frameTimes(t0, t0 + microseconds(us), last);
        CHECK(f.ts_ms > last);                                                 // strictly increasing, always
        CHECK(f.capture_s > prev_s);
        CHECK(std::fabs(f.capture_s - us / 1e6) < 1e-12);                      // the bump never leaks into the DSP time base
        last = f.ts_ms;
        prev_s = f.capture_s;
    }
    CHECK(last >= 10 + 5);                                                     // 6 frames in ~1 ms were pushed apart for MediaPipe

    // random capture times: both clocks stay valid, ts_ms never goes backwards, capture_s never decreases
    std::mt19937 g(5);
    std::uniform_int_distribution<int> step(50, 40000);                        // 0.05 .. 40 ms between frames
    int64_t lt = -1;
    double ls = -1.0;
    long long us = 0;
    bool ok = true;
    for (int i = 0; i < 20000; ++i) {
        us += step(g);
        const FrameTimes f = frameTimes(t0, t0 + microseconds(us), lt);
        ok = ok && f.ts_ms > lt && f.capture_s > ls && std::fabs(f.capture_s - us / 1e6) < 1e-9;
        lt = f.ts_ms;
        ls = f.capture_s;
    }
    CHECK(ok);
}

// A filter whose outcomes are scripted. Rejected / Invalid leave a marker value (0.123, 0.456) in x, y: it must never be published.
struct ScriptedFilter : IPointFilter {
    std::vector<FilterOutcome> script;
    size_t next = 0;
    int resets = 0, legacy_calls = 0;
    void reset() override { ++resets; }
    void filter(double&, double&, double) override { ++legacy_calls; }          // the gate must use update(), never this
    FilterOutcome update(double& x, double& y, double) override {
        const FilterOutcome o = next < script.size() ? script[next++] : FilterOutcome::Ok;
        if (o == FilterOutcome::Ok) { x = 0.5 * x; y = 0.5 * y; }              // distinguishable from the raw input
        else { x = 0.123; y = 0.456; }
        return o;
    }
};
static void t_gate_filter_outcomes() {
    using O = FilterOutcome;
    const double ex = 0.5 * double(0.8f), ey = 0.5 * double(0.6f);          // what ScriptedFilter makes of the float inputs
    ScriptedFilter f;
    f.script = {O::Ok, O::Rejected, O::Ok, O::Invalid, O::Ok, O::Rejected};
    FreshnessGate gate;
    ModuleOut m = hand_result(TrackingState::Tracked, 0.8, 0.6);

    auto p = gate.update(m, f, true, 0.00);
    CHECK(same_point(p, ex, ey) && !p->held);                                // Ok: smoothed point, cached
    p = gate.update(m, f, true, 0.01);                                         // Rejected: replay, not the 0.123 marker
    CHECK(same_point(p, ex, ey) && p->held && gate.counters().rejected == 1 && f.resets == 0);
    p = gate.update(m, f, true, 0.02);                                         // history intact: the next Ok is a fresh point
    CHECK(same_point(p, ex, ey) && !p->held);
    p = gate.update(m, f, true, 0.03);                                         // Invalid: nothing published, cache gone ...
    CHECK(!p && !gate.hasCache() && gate.counters().invalid == 1);
    CHECK(f.resets == 0);                                                      // ... and no reset call: the filter reset itself
    CHECK(!gate.update(hand_result(TrackingState::Held, 0.8, 0.6, 1), f, true, 0.04));       // HELD has nothing to replay
    CHECK(!gate.update(hand_result(TrackingState::Lost, -1, -1, 7), f, true, 0.05) && f.resets == 0);   // nothing to reset either
    p = gate.update(m, f, true, 0.06);                                         // new track
    CHECK(same_point(p, ex, ey) && !p->held);
    CHECK(!gate.update(hand_result(TrackingState::Lost, -1, -1, 7), f, true, 0.07) && f.resets == 1);   // LOST resets once

    ScriptedFilter g;                                                          // Rejected with nothing cached: nothing to replay
    g.script = {O::Rejected};
    FreshnessGate g2;
    CHECK(!g2.update(m, g, true, 0.0) && g.legacy_calls == 0);

    // gaze: no tracking_state, the plain path, the same outcome handling
    ScriptedFilter h;
    h.script = {O::Ok, O::Rejected, O::Invalid, O::Ok};
    FreshnessGate g3;
    ModuleOut gz; gz.ok = true; gz.x = 0.8f; gz.y = 0.6f;
    CHECK(same_point(g3.update(gz, h, true, 0.0), ex, ey));
    auto r = g3.update(gz, h, true, 0.1);
    CHECK(same_point(r, ex, ey) && r->held);
    CHECK(!g3.update(gz, h, true, 0.2) && !g3.hasCache());
    r = g3.update(gz, h, true, 0.3);
    CHECK(same_point(r, ex, ey) && !r->held && h.legacy_calls == 0);
}

// Whole chain: HandNative (retention) -> FreshnessGate -> filter. The gated output must equal a filter that was fed ONLY the
// real observations; the old path (filter every detected() frame) must NOT, which is exactly the defect of the report.
static void t_gate_chain_matches_observation_only_filter() {
    LambdaHandModel model([](const double*, int& c, double& p) { c = 0; p = 1.0; });
    HandNative hand(&model);
    Packed pk;
    OneEuroPoint gated_flt(1.0, 8.0, 1.0), ref_flt(1.0, 8.0, 1.0), naive_flt(1.0, 8.0, 1.0);
    FreshnessGate gate;
    std::mt19937 g(3);
    std::normal_distribution<double> noise(0.0, 0.01);
    std::optional<GatedPoint> last_pub;
    int mismatches = 0, naive_diff = 0, held_seen = 0, tracked_seen = 0, lost_seen = 0;
    for (int i = 0; i < 600; ++i) {
        const double t = i / 60.0;
        const int drop_len = 3 + 3 * ((i / 40) % 3);                          // cycle after cycle: dropouts of 3, 6, 9 frames
        const bool miss = (i % 40) >= 30 && (i % 40) < 30 + drop_len;
        const double ox = 0.5 + 0.3 * std::sin(i * 0.05), oy = 0.5 + 0.2 * std::cos(i * 0.03);
        pk.set(8, float(ox + noise(g)), float(oy + noise(g)));
        const ModuleOut r = miss ? hand.process(nullptr, 0) : hand.process(pk.p, kHandPacked);

        const auto pub = gate.update(r, gated_flt, true, t);                  // the engine path

        if (r.tracking_state == TrackingState::Tracked) {                      // reference: observations only
            ++tracked_seen;
            double x = r.x, y = r.y;
            ref_flt.filter(x, y, t);
            if (!pub || pub->x != std::min(1.0, std::max(0.0, x)) || pub->y != std::min(1.0, std::max(0.0, y))) ++mismatches;
            last_pub = pub;
        } else if (r.tracking_state == TrackingState::Held) {
            ++held_seen;
            if (!pub || !last_pub || pub->x != last_pub->x || pub->y != last_pub->y) ++mismatches;   // the last real filtered point
        } else {
            ++lost_seen;
            if (pub) ++mismatches;
            ref_flt.reset();
            last_pub.reset();
        }
        if (r.detected()) {                                                    // old behaviour: every detected() frame is "fresh"
            double x = r.x, y = r.y;
            naive_flt.filter(x, y, t);
            if (r.tracking_state == TrackingState::Tracked && pub && std::fabs(x - pub->x) > 1e-12) ++naive_diff;
        } else naive_flt.reset();
    }
    CHECK(mismatches == 0);
    CHECK(held_seen > 0 && lost_seen > 0 && tracked_seen > 0);                 // the scenario really contains all three states
    CHECK(naive_diff > 0);                                                     // the old path DID corrupt the filter history
    CHECK(gate.counters().filter_updates == uint64_t(tracked_seen));          // one update per observation, none for HELD
}

int main() {
    struct T { const char* name; void (*fn)(); };
    const T tests[] = {
        { "filter: calms jitter when still", t_filter_calms_jitter },
        { "filter: follows fast motion", t_filter_follows_fast_motion },
        { "filter: reset / equal timestamps", t_filter_reset_and_edges },
        { "hand: 24 features = Python (golden)", t_hand_features_match_python },
        { "hand: gate, vote, grace period", t_hand_vote_gate_grace },
        { "hand: TRACKED / HELD / LOST transitions", t_hand_tracking_state_transitions },
        { "hand: anchors, clamp, geometric fallback", t_hand_anchor_clamp_and_fallback },
        { "hand: E2-B malformed / nonfinite rejection", t_hand_invalid_payload_rejection },
        { "gaze: calibration + feature vector", t_gaze_calibration_and_features },
        { "gaze: no face + recalibrate", t_gaze_no_face_and_recalibrate },
        { "gaze: calibration ignores zero EAR", t_gaze_calibration_ignores_zero_ear },
        { "gaze: face loss resets tracking state", t_gaze_face_loss_resets_tracking_state },
        { "gaze: no freshness fields", t_gaze_has_no_freshness_fields },
        { "gate: E2-B invalid payload never updates DSP", t_e2b_invalid_payload_never_updates_dsp },
        { "gate: TRACKED / HELD / LOST -> filter calls", t_gate_tracked_held_lost },
        { "gate: startup, errors, edges, --no-filter", t_gate_startup_error_and_edges },
        { "gate: chain == observation-only filter", t_gate_chain_matches_observation_only_filter },
        { "gate: filter outcomes (Rejected / Invalid)", t_gate_filter_outcomes },
        { "clock: capture_s exact, ts_ms strictly increasing", t_frame_times_exact_and_strictly_increasing },
        { "gaze: ratio follows the iris", t_gaze_ratio_follows_iris },
        { "gaze: head pose angles", t_head_pose_angles },
        { "gaze: head pose added, raw absolute", t_gaze_adds_head_pose_and_outputs_raw },
        { "gaze: diag OFF vs ON + 32-col CSV reconstruction (shipped DLLs)", t_gaze_diag_shipped_dlls },
        { "gaze: absent DLL error + fallback diag OFF vs ON + CSV reconstruction", t_gaze_absent_dll_and_fallback_diag },
    };
    int failed_tests = 0;
    for (const T& t : tests) {
        const int before = g_failed;
        t.fn();
        const bool ok = g_failed == before;
        failed_tests += ok ? 0 : 1;
        std::printf("[%s] %s\n", ok ? " ok " : "FAIL", t.name);
    }
    std::printf("\n%zu tests, %d checks, %d failed\n", sizeof(tests) / sizeof(tests[0]), g_checks, g_failed);
    return failed_tests == 0 ? 0 : 1;
}