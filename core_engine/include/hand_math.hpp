// Hand tracking logic: a line-for-line C++ port of ai_pipeline/hand_kinematics/hand_tracker.py and
// hand_features.py (Eng 2).  The Python files stay the reference; tests/parity/run_parity.py compares
// this code with them frame by frame.
//
//   input  : 42 landmarks per frame, packed by landmark_extractors.HandLandmarkExtractor
//              rows  0..20 = MediaPipe image landmarks  (normalised x, y; used for the cursor)
//              rows 21..41 = MediaPipe WORLD landmarks  (metric 3D; used for the 24 features)
//   step 1 : 24 features (15 joint angles in degrees, 4 thumb-tip distances, 5 wrist-tip distances,
//            distances divided by the palm size)                                    -> handFeatures()
//   step 2 : Random Forest -> (class, confidence); confidence < 0.55 votes Neutral  -> IHandModel
//   step 3 : majority vote over the last 5 frames (ties go to the class seen first in the window)
//   step 4 : no hand for <= 6 frames -> hold the last output; longer -> (-1, -1, 0)
//   output : cursor = index fingertip (clamped to [0,1]) + state 0 neutral / 1 pinch / 2 fist
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <utility>
#include "tracking_types.hpp"

namespace core {

constexpr int kHandPoints   = 21;                 // landmarks of one hand
constexpr int kHandPacked   = 2 * kHandPoints;    // image landmarks followed by world landmarks
constexpr int kHandFeatures = 24;

// Order of hand_features.FEATURE_NAMES (the model file stores the same list and the engine checks it).
inline const char* const* handFeatureNames() {
    static const char* const names[kHandFeatures] = {
        "ang_thumb_cmc", "ang_thumb_mcp", "ang_thumb_ip",
        "ang_index_mcp", "ang_index_pip", "ang_index_dip",
        "ang_middle_mcp", "ang_middle_pip", "ang_middle_dip",
        "ang_ring_mcp", "ang_ring_pip", "ang_ring_dip",
        "ang_pinky_mcp", "ang_pinky_pip", "ang_pinky_dip",
        "thumb_to_index_tip", "thumb_to_middle_tip", "thumb_to_ring_tip", "thumb_to_pinky_tip",
        "wrist_to_thumb_tip", "wrist_to_index_tip", "wrist_to_middle_tip", "wrist_to_ring_tip", "wrist_to_pinky_tip"};
    return names;
}

namespace hand_idx {
constexpr int WRIST = 0, THUMB_TIP = 4, INDEX_MCP = 5, INDEX_TIP = 8, MIDDLE_MCP = 9;
constexpr int F_THUMB_TO_INDEX = 15;              // feature index of the pinch ratio
constexpr int F_WRIST_TO_INDEX = 20;              // ...then middle, ring, pinky (21..23)
}

// Which point of the hand is the pointer. Tip = what hand_tracker.py does (index fingertip).
// Knuckle (index MCP) or Mid (thumb/index midpoint) move less when the fingers pinch.
enum class HandAnchor { Tip, Knuckle, Mid };

struct HandParams {
    HandAnchor anchor = HandAnchor::Tip;
    int    smoothing_frames = 5;       // HandTracker.SMOOTHING_FRAMES
    double min_confidence   = 0.55;    // HandTracker.MIN_CONFIDENCE
    int    grace_frames     = 6;       // HandTracker.GRACE_FRAMES
    double fallback_pinch_ratio = 0.30;   // HandTracker.FALLBACK_PINCH_RATIO (no trained model only)
    double fallback_curl_ratio  = 1.30;   // HandTracker.FALLBACK_CURL_RATIO
};

// ---------------------------------------------------------------- features ----
namespace detail {
struct V3 { double x, y, z; };
inline V3 pt(const float* w, int i) { return {double(w[3 * i]), double(w[3 * i + 1]), double(w[3 * i + 2])}; }
inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
inline double angleDeg(V3 a, V3 b, V3 c) {            // hand_features._angle: angle at b
    const V3 v1 = sub(a, b), v2 = sub(c, b);
    const double denom = norm(v1) * norm(v2);
    if (denom < 1e-9) return 0.0;
    const double cs = std::max(-1.0, std::min(1.0, dot(v1, v2) / denom));
    return std::acos(cs) * (180.0 / 3.14159265358979323846);
}
} // namespace detail

// world: 21 x 3 floats (MediaPipe world landmarks).  out: kHandFeatures doubles.
inline void handFeatures(const float* world, double* out) {
    using namespace detail;
    static const int kTriplets[15][3] = {
        {0, 1, 2}, {1, 2, 3}, {2, 3, 4},          {0, 5, 6},   {5, 6, 7},    {6, 7, 8},
        {0, 9, 10}, {9, 10, 11}, {10, 11, 12},    {0, 13, 14}, {13, 14, 15}, {14, 15, 16},
        {0, 17, 18}, {17, 18, 19}, {18, 19, 20}};
    static const int kFingerTips[4] = {8, 12, 16, 20};

    double palm = norm(sub(pt(world, hand_idx::MIDDLE_MCP), pt(world, hand_idx::WRIST)));
    if (palm < 1e-9) palm = 1.0;
    int k = 0;
    for (const auto& t : kTriplets) out[k++] = angleDeg(pt(world, t[0]), pt(world, t[1]), pt(world, t[2]));
    const V3 thumb = pt(world, hand_idx::THUMB_TIP);
    for (int t : kFingerTips) out[k++] = norm(sub(thumb, pt(world, t))) / palm;
    out[k++] = norm(sub(pt(world, hand_idx::THUMB_TIP), pt(world, hand_idx::WRIST))) / palm;
    for (int t : kFingerTips) out[k++] = norm(sub(pt(world, t), pt(world, hand_idx::WRIST))) / palm;
}

// ------------------------------------------------------------------ models ----
// Maps 24 features to (class, confidence). Backends: Treelite-compiled forest (below), scikit-learn through
// Python (py_bridge.hpp), the geometric fallback (no trained model), a lambda (tests).
class IHandModel {
public:
    virtual ~IHandModel() = default;
    virtual void classify(const double* features, int& cls, double& confidence) = 0;
    virtual const char* name() const = 0;
    virtual uint64_t errors() const { return 0; }
};

class LambdaHandModel : public IHandModel {
public:
    explicit LambdaHandModel(std::function<void(const double*, int&, double&)> fn) : fn_(std::move(fn)) {}
    void classify(const double* f, int& c, double& p) override { fn_(f, c, p); }
    const char* name() const override { return "lambda"; }
private:
    std::function<void(const double*, int&, double&)> fn_;
};

// HandTracker._classify without a trained model ("rough 3D geometric fallback, UNTESTED thresholds").
class GeometricHandModel : public IHandModel {
public:
    explicit GeometricHandModel(const HandParams& p = HandParams()) : p_(p) {}
    void classify(const double* f, int& cls, double& conf) override {
        conf = 1.0;
        if (f[hand_idx::F_THUMB_TO_INDEX] < p_.fallback_pinch_ratio) { cls = 1; return; }
        int curled = 0;
        for (int i = 0; i < 4; ++i) if (f[hand_idx::F_WRIST_TO_INDEX + i] < p_.fallback_curl_ratio) ++curled;
        cls = (curled >= 3) ? 2 : 0;
    }
    const char* name() const override { return "geometric fallback (NO trained model - rough, untested thresholds)"; }
private:
    HandParams p_;
};

#ifdef CORE_HAVE_HAND_FOREST
extern "C" {
int hand_forest_num_features(void);
int hand_forest_num_classes(void);
int hand_forest_proba(const float* features, double* out, int max_out);   // src/forest_shim.c
}

// hand_gesture_rf.pkl compiled to C by Treelite (tools/export_forests.py).
class TreeliteHandModel : public IHandModel {
public:
    TreeliteHandModel() {
        if (hand_forest_num_features() != kHandFeatures)
            throw std::runtime_error("compiled hand forest expects a different number of features than 24");
        if (hand_forest_num_classes() < 1 || hand_forest_num_classes() > kMaxClasses)
            throw std::runtime_error("compiled hand forest has an unsupported number of classes");
    }
    void classify(const double* f, int& cls, double& conf) override {
        float x[kHandFeatures];
        for (int i = 0; i < kHandFeatures; ++i) x[i] = float(f[i]);       // scikit-learn also rounds to float32
        double p[kMaxClasses];
        const int n = hand_forest_proba(x, p, kMaxClasses);
        int k = 0;
        for (int i = 1; i < n; ++i) if (p[i] > p[k]) k = i;               // first maximum, like numpy.argmax
        cls = k;
        conf = p[k];
    }
    const char* name() const override { return "treelite (compiled)"; }
    int numClasses() const { return hand_forest_num_classes(); }
private:
    static constexpr int kMaxClasses = 16;
};
#endif

// ----------------------------------------------------------------- tracker ----
// Stateful, one instance per camera stream, used by the AI thread only.
class HandNative {
public:
    HandNative(IHandModel* model, const HandParams& p = HandParams()) : model_(model), p_(p) {}

    // pts: n x 3 floats. n == 0: no hand. n == kHandPacked: image landmarks then world landmarks.
    // Any other n is a bridge error (ok = false, no state change).
    ModuleOut process(const float* pts, int n) {
        using namespace hand_idx;
        if (n != 0 && (n != kHandPacked || !pts)) return ModuleOut{};
        if (n == 0) {                                          // detection miss
            if (missed_ < 1000000) ++missed_;
            if (missed_ > p_.grace_frames) {                   // genuine hand loss
                votes_.clear();
                confidence_ = 0.0;
                last_ = make(-1.0f, -1.0f, 0);
            }
            return last_;                                      // otherwise: hold the last output
        }
        missed_ = 0;

        double cx = 0.0, cy = 0.0;                             // cursor from the IMAGE landmarks
        switch (p_.anchor) {
            case HandAnchor::Tip:     cx = pts[3 * INDEX_TIP];     cy = pts[3 * INDEX_TIP + 1];     break;
            case HandAnchor::Knuckle: cx = pts[3 * INDEX_MCP];     cy = pts[3 * INDEX_MCP + 1];     break;
            case HandAnchor::Mid:     cx = 0.5 * (double(pts[3 * INDEX_TIP]) + pts[3 * THUMB_TIP]);
                                      cy = 0.5 * (double(pts[3 * INDEX_TIP + 1]) + pts[3 * THUMB_TIP + 1]); break;
        }
        double feats[kHandFeatures];
        handFeatures(pts + 3 * kHandPoints, feats);            // features from the WORLD landmarks
        int cls = 0;
        double conf = 0.0;
        model_->classify(feats, cls, conf);
        confidence_ = conf;
        const int state = (conf >= p_.min_confidence) ? cls : 0;

        votes_.push_back(state);
        if (int(votes_.size()) > p_.smoothing_frames) votes_.pop_front();
        last_ = make(float(std::max(0.0, std::min(1.0, cx))), float(std::max(0.0, std::min(1.0, cy))), vote());
        return last_;
    }

    double lastConfidence() const { return confidence_; }      // debug only
    int missedFrames() const { return missed_; }

private:
    static ModuleOut make(float x, float y, int s) { ModuleOut o; o.ok = true; o.x = x; o.y = y; o.state = s; return o; }

    // Counter(votes).most_common(1)[0][0]: the most frequent class; a tie goes to the class that
    // appears first in the window (Counter keeps first-seen order and max() keeps the first maximum).
    int vote() const {
        int best = 0, best_count = 0;
        for (size_t i = 0; i < votes_.size(); ++i) {
            bool seen = false;
            for (size_t j = 0; j < i; ++j) if (votes_[j] == votes_[i]) { seen = true; break; }
            if (seen) continue;
            int c = 0;
            for (int v : votes_) if (v == votes_[i]) ++c;
            if (c > best_count) { best_count = c; best = votes_[i]; }
        }
        return best;
    }

    IHandModel* model_;
    HandParams  p_;
    std::deque<int> votes_;
    int         missed_ = 0;
    double      confidence_ = 0.0;
    ModuleOut   last_ = make(-1.0f, -1.0f, 0);
};

} // namespace core
