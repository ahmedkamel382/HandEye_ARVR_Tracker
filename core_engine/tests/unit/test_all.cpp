// Unit tests for the header-only parts of the core engine. No camera, no Python, no OpenCV.
//   build + run:  ctest --output-on-failure      (or run core_unit_tests directly)
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "forest.hpp"
#include "gaze_math.hpp"
#include "hand_math.hpp"
#include "euro_builtin.hpp"
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
static void t_gaze_calibration_and_features() {
    GazeRig r;
    FacePts open(0.30, 0.30);
    for (int i = 0; i < 29; ++i) {
        ModuleOut o = r.run(open);
        CHECK(o.detected() && o.state == -1);
        CHECK(o.yaw_deg == 0.0 && o.pitch_deg == 0.0);                     // calibrating: gaze is 0, 0 (like the Python)
    }
    CHECK(r.wink_calls == 0 && !r.g.calibrated());
    ModuleOut o = r.run(open);                                             // 30th calibration frame still reports -1
    CHECK(o.state == -1 && r.g.calibrated());
    CHECK_NEAR(o.x, 0.5, 1e-6);
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
    CHECK(none.ok && !none.detected() && none.state == 0 && none.yaw_deg == 0.0);
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

int main() {
    struct T { const char* name; void (*fn)(); };
    const T tests[] = {
        { "filter: calms jitter when still", t_filter_calms_jitter },
        { "filter: follows fast motion", t_filter_follows_fast_motion },
        { "filter: reset / equal timestamps", t_filter_reset_and_edges },
        { "hand: 24 features = Python (golden)", t_hand_features_match_python },
        { "hand: gate, vote, grace period", t_hand_vote_gate_grace },
        { "hand: anchors, clamp, geometric fallback", t_hand_anchor_clamp_and_fallback },
        { "gaze: calibration + feature vector", t_gaze_calibration_and_features },
        { "gaze: no face + recalibrate", t_gaze_no_face_and_recalibrate },
        { "gaze: ratio follows the iris", t_gaze_ratio_follows_iris },
        { "gaze: head pose angles", t_head_pose_angles },
        { "gaze: head pose added, raw absolute", t_gaze_adds_head_pose_and_outputs_raw },
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