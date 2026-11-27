// Gaze + blink pipeline: a line-for-line C++ port of gaze_intent.py (Eng 3), Hybrid Blink v2.
//   478 face landmarks + the 4x4 facial transformation matrix (+ optional 8 eye-aligned blendshapes)
//      -> 3D EAR (NaN on degenerate eye), iris ratios, head pose (Euler angles from the matrix)
//      -> HybridBlinkDetector v2 (poison-resistant baseline + slow EMA + eyeBlink/eyeLookDown +
//         pitch/sy-compensated EAR fusion + wall-clock EMA + Schmitt-trigger hysteresis) -> state
//         (or explicit legacy gesture_model.dll fallback strictly wrapped by P0 finite guards)
//      -> pitch / yaw models (gaze_pitch.dll, gaze_yaw.dll) = eye-in-socket angles
//      -> + head angles = absolute world gaze (raw, bypassing C++ EMA to let DSP handle filtering)
// Like the Python: during calibration (state -1) and with no face (state 0) all four angles are NaN (the Python's
// INVALID_ANGLE) and the pointer is x = y = -1, so no invalid value can reach the DSP stage.
// All arithmetic is double, like Python's float64; the features are rounded to float32 before the models (what the
// Python does with np.float32). The head-pose products are float32 too, because the matrix is a float32 array.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <string>
#include <vector>
#include "forest.hpp"
#include "tracking_types.hpp"

namespace core {

    // Opt-in per-frame diagnostic telemetry payload (--gaze-diag-log <file.csv>).
    // Unified 32-column schema shared 1:1 between C++ GazeNative and Python GazeIntent.
    // MUST remain exact 32 columns for PR #4 and analyze_gaze_fixation_diag.py compatibility.
    struct GazeDiagRow {
        uint64_t frame_id = 0;
        int64_t  ts_ms = 0;
        double   capture_s = -1.0;
        int      detected = 0;
        int      state = 0;
        double   left_iris_x  = std::numeric_limits<double>::quiet_NaN();
        double   right_iris_x = std::numeric_limits<double>::quiet_NaN();
        double   sx           = std::numeric_limits<double>::quiet_NaN();
        double   left_iris_y  = std::numeric_limits<double>::quiet_NaN();
        double   right_iris_y = std::numeric_limits<double>::quiet_NaN();
        double   sy           = std::numeric_limits<double>::quiet_NaN();
        double   head_pitch_deg = std::numeric_limits<double>::quiet_NaN();
        double   head_yaw_deg   = std::numeric_limits<double>::quiet_NaN();
        double   head_roll_deg  = std::numeric_limits<double>::quiet_NaN();
        double   rel_pitch_deg  = std::numeric_limits<double>::quiet_NaN();
        double   rel_yaw_deg    = std::numeric_limits<double>::quiet_NaN();
        double   abs_pitch_deg  = std::numeric_limits<double>::quiet_NaN();
        double   abs_yaw_deg    = std::numeric_limits<double>::quiet_NaN();
        double   norm_x = -1.0;
        double   norm_y = -1.0;
        double   lm33_x  = std::numeric_limits<double>::quiet_NaN(), lm33_y  = std::numeric_limits<double>::quiet_NaN();
        double   lm133_x = std::numeric_limits<double>::quiet_NaN(), lm133_y = std::numeric_limits<double>::quiet_NaN();
        double   lm468_x = std::numeric_limits<double>::quiet_NaN(), lm468_y = std::numeric_limits<double>::quiet_NaN();
        double   lm362_x = std::numeric_limits<double>::quiet_NaN(), lm362_y = std::numeric_limits<double>::quiet_NaN();
        double   lm263_x = std::numeric_limits<double>::quiet_NaN(), lm263_y = std::numeric_limits<double>::quiet_NaN();
        double   lm473_x = std::numeric_limits<double>::quiet_NaN(), lm473_y = std::numeric_limits<double>::quiet_NaN();
    };

    inline constexpr const char* kGazeDiagCsvHeader =
        "frame_id,ts_ms,capture_s,detected,state,"
        "left_iris_x,right_iris_x,sx,left_iris_y,right_iris_y,sy,"
        "head_pitch_deg,head_yaw_deg,head_roll_deg,"
        "rel_pitch_deg,rel_yaw_deg,abs_pitch_deg,abs_yaw_deg,"
        "norm_x,norm_y,"
        "lm33_x,lm33_y,lm133_x,lm133_y,lm468_x,lm468_y,"
        "lm362_x,lm362_y,lm263_x,lm263_y,lm473_x,lm473_y";

    inline std::string formatGazeDiagCsvRow(const GazeDiagRow& d) {
        auto fmtVal = [](char* b, size_t n, double v) -> const char* {
            if (std::isnan(v)) return "nan";
            if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
            std::snprintf(b, n, "%.9g", v);
            return b;
        };
        char c_cap[32];
        if (std::isnan(d.capture_s)) std::snprintf(c_cap, sizeof c_cap, "nan");
        else if (std::isinf(d.capture_s)) std::snprintf(c_cap, sizeof c_cap, "%s", d.capture_s > 0 ? "inf" : "-inf");
        else std::snprintf(c_cap, sizeof c_cap, "%.9f", d.capture_s);

        char b[27][32];
        const double vals[27] = {
            d.left_iris_x, d.right_iris_x, d.sx, d.left_iris_y, d.right_iris_y, d.sy,
            d.head_pitch_deg, d.head_yaw_deg, d.head_roll_deg,
            d.rel_pitch_deg, d.rel_yaw_deg, d.abs_pitch_deg, d.abs_yaw_deg,
            d.norm_x, d.norm_y,
            d.lm33_x, d.lm33_y, d.lm133_x, d.lm133_y, d.lm468_x, d.lm468_y,
            d.lm362_x, d.lm362_y, d.lm263_x, d.lm263_y, d.lm473_x, d.lm473_y
        };
        const char* s[27];
        for (int i = 0; i < 27; ++i) s[i] = fmtVal(b[i], sizeof b[i], vals[i]);

        char line[1024];
        std::snprintf(
            line, sizeof line,
            "%llu,%lld,%s,%d,%d,"
            "%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,"
            "%s,%s,%s,%s,"
            "%s,%s,"
            "%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s,%s,%s",
            (unsigned long long)d.frame_id, (long long)d.ts_ms, c_cap, d.detected, d.state,
            s[0], s[1], s[2], s[3], s[4], s[5],
            s[6], s[7], s[8],
            s[9], s[10], s[11], s[12],
            s[13], s[14],
            s[15], s[16], s[17], s[18], s[19], s[20],
            s[21], s[22], s[23], s[24], s[25], s[26]
        );
        return std::string(line);
    }

    // Separate opt-in blink telemetry row (--blink-diag-log <file.csv>).
    // Exact 22-column schema matching Python BlinkDiagnosticPayload without touching gaze_diag.csv.
    struct BlinkDiagRow {
        uint64_t    frame_id = 0;
        int64_t     ts_ms = 0;
        double      capture_s = -1.0;
        int         detected = 0;
        int         state = 0;
        const char* detector_mode = "invalid";
        double      raw_ear_l      = std::numeric_limits<double>::quiet_NaN();
        double      raw_ear_r      = std::numeric_limits<double>::quiet_NaN();
        double      base_ear_l     = 1.0;
        double      base_ear_r     = 1.0;
        double      norm_ear_l     = std::numeric_limits<double>::quiet_NaN();
        double      norm_ear_r     = std::numeric_limits<double>::quiet_NaN();
        double      bs_blink_l     = std::numeric_limits<double>::quiet_NaN();
        double      bs_blink_r     = std::numeric_limits<double>::quiet_NaN();
        double      bs_lookdown_l  = std::numeric_limits<double>::quiet_NaN();
        double      bs_lookdown_r  = std::numeric_limits<double>::quiet_NaN();
        double      bs_squint_l    = std::numeric_limits<double>::quiet_NaN();
        double      bs_squint_r    = std::numeric_limits<double>::quiet_NaN();
        double      score_l        = std::numeric_limits<double>::quiet_NaN();
        double      score_r        = std::numeric_limits<double>::quiet_NaN();
        double      sy             = std::numeric_limits<double>::quiet_NaN();
        double      head_pitch_deg = std::numeric_limits<double>::quiet_NaN();
    };

    inline constexpr const char* kBlinkDiagCsvHeader =
        "frame_id,ts_ms,capture_s,detected,state,detector_mode,"
        "raw_ear_l,raw_ear_r,base_ear_l,base_ear_r,norm_ear_l,norm_ear_r,"
        "bs_blink_l,bs_blink_r,bs_lookdown_l,bs_lookdown_r,bs_squint_l,bs_squint_r,"
        "score_l,score_r,sy,head_pitch_deg";

    inline std::string formatBlinkDiagCsvRow(const BlinkDiagRow& d) {
        auto fmtVal = [](char* b, size_t n, double v) -> const char* {
            if (std::isnan(v)) return "nan";
            if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
            std::snprintf(b, n, "%.9g", v);
            return b;
        };
        char c_cap[32];
        if (std::isnan(d.capture_s)) std::snprintf(c_cap, sizeof c_cap, "nan");
        else if (std::isinf(d.capture_s)) std::snprintf(c_cap, sizeof c_cap, "%s", d.capture_s > 0 ? "inf" : "-inf");
        else std::snprintf(c_cap, sizeof c_cap, "%.9f", d.capture_s);

        char b[16][32];
        const double vals[16] = {
            d.raw_ear_l, d.raw_ear_r, d.base_ear_l, d.base_ear_r, d.norm_ear_l, d.norm_ear_r,
            d.bs_blink_l, d.bs_blink_r, d.bs_lookdown_l, d.bs_lookdown_r, d.bs_squint_l, d.bs_squint_r,
            d.score_l, d.score_r, d.sy, d.head_pitch_deg
        };
        const char* s[16];
        for (int i = 0; i < 16; ++i) s[i] = fmtVal(b[i], sizeof b[i], vals[i]);

        char line[768];
        std::snprintf(
            line, sizeof line,
            "%llu,%lld,%s,%d,%d,%s,"
            "%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s",
            (unsigned long long)d.frame_id, (long long)d.ts_ms, c_cap, d.detected, d.state,
            d.detector_mode ? d.detector_mode : "invalid",
            s[0], s[1], s[2], s[3], s[4], s[5],
            s[6], s[7], s[8], s[9], s[10], s[11],
            s[12], s[13], s[14], s[15]
        );
        return std::string(line);
    }

    enum class BlinkMode { Hybrid = 0, LegacyForest = 1 };

    class GazeNative {
    public:
        static constexpr int kFacePoints = 478;
        static constexpr int kPoseRows = 6;                            // 4x4 matrix = 16 floats -> 6 rows of 3
        static constexpr int kPacked = kFacePoints + kPoseRows;        // 484 (legacy landmarks + pose)
        static constexpr int kBlendshapeRows = 3;                      // 8 eye-aligned blendshapes + 1 sentinel -> 3 rows of 3
        static constexpr int kPackedWithBlendshapes = kPacked + kBlendshapeRows;  // 487
        static constexpr double kSpanDeg = 60.0;                       // x, y = 0.5 + angle / kSpanDeg  (+-30 deg covers the unit square)

        GazeNative(IForest* wink, IRegressor* pitch, IRegressor* yaw,
                   int calibration_frames = 30, int window = 10,
                   BlinkMode blink_mode = BlinkMode::Hybrid)
            : wink_(wink), pitch_(pitch), yaw_(yaw), calib_target_(calibration_frames),
              window_(static_cast<size_t>(window)), blink_mode_(blink_mode) {}

        bool calibrated() const { return calibrated_; }
        BlinkMode blinkMode() const { return blink_mode_; }
        double baselineLeftEar() const { return base_l_; }
        double baselineRightEar() const { return base_r_; }
        double baselineBsLeft() const { return base_bs_l_; }
        double baselineBsRight() const { return base_bs_r_; }
        double lastScoreL() const { return score_l_; }
        double lastScoreR() const { return score_r_; }

        void recalibrate() {
            calibrated_ = false;
            calib_frames_ = 0;
            sum_l_ = sum_r_ = 0.0;
            base_l_ = base_r_ = 1.0;
            base_bs_l_ = base_bs_r_ = 0.0;
            base_sy_ = kDefaultBaseSy;
            cal_ear_l_.clear();
            cal_ear_r_.clear();
            cal_bs_l_.clear();
            cal_bs_r_.clear();
            cal_sy_.clear();
            resetTrackingState();
        }

        // lm: n x 3 floats:
        //   - n == kPacked (484): 478 landmarks + 6 pose rows (legacy / EAR-only hybrid path).
        //   - n == kPackedWithBlendshapes (487): 484 rows + 3 rows carrying 8 validated [0, 1] eye-aligned blendshapes.
        //   - n == 0 (or malformed count, or any non-finite/out-of-range input/output) -> failInvalid() (x=y=-1, state=0).
        ModuleOut process(const float* lm, int n, int W, int H,
                          GazeDiagRow* diag = nullptr,
                          int64_t ts_ms = -1,
                          BlinkDiagRow* blink_diag = nullptr) {
            ModuleOut out;
            out.ok = true;
            auto failInvalid = [&]() -> ModuleOut {
                resetTrackingState();
                markInvalid(out);
                out.state = 0;
                if (diag) {
                    *diag = GazeDiagRow{};
                    diag->detected = 0;
                    diag->state = 0;
                    diag->norm_x = -1.0;
                    diag->norm_y = -1.0;
                }
                if (blink_diag) {
                    *blink_diag = BlinkDiagRow{};
                    blink_diag->detected = 0;
                    blink_diag->state = 0;
                    blink_diag->detector_mode = "invalid";
                    blink_diag->base_ear_l = base_l_;
                    blink_diag->base_ear_r = base_r_;
                }
                return out;
            };

            if (!lm || (n != kPacked && n != kPackedWithBlendshapes) || W <= 0 || H <= 0) {
                return failInvalid();
            }
            const float* pose = lm + 3 * kFacePoints;
            if (!allFinite(lm, 3 * kFacePoints) || !allFinite(pose, 16)) {
                return failInvalid();
            }

            bool has_bs = false;
            double bs[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            if (n == kPackedWithBlendshapes) {
                const float* bsp = lm + 3 * kPacked;
                for (int i = 0; i < 8; ++i) {
                    const float v = bsp[i];
                    if (!std::isfinite(v) || v < 0.0f || v > 1.0f) {
                        return failInvalid();
                    }
                    bs[i] = double(v);
                }
                has_bs = true;
            }

            // 1. 3D EAR (degenerate |p1 - p4| <= kEarDistEps returns NaN -> failInvalid)
            const double raw_l = ear(lm, kLeftEye, W, H);
            const double raw_r = ear(lm, kRightEye, W, H);
            if (!std::isfinite(raw_l) || !std::isfinite(raw_r)) {
                return failInvalid();
            }

            // 2. Iris position + head pose (computed & validated finite BEFORE calibration/blink)
            const double l_ix = irisRatio(lm, 33, 133, 468, true);
            const double r_ix = irisRatio(lm, 362, 263, 473, true);
            const double sx = 0.5 * (l_ix + r_ix);
            const double l_iy = irisRatio(lm, 33, 133, 468, false);
            const double r_iy = irisRatio(lm, 362, 263, 473, false);
            const double sy = 0.5 * (l_iy + r_iy);

            double head_pitch, head_yaw, head_roll;
            headPose(pose, head_pitch, head_yaw, head_roll);
            if (!std::isfinite(sx) || !std::isfinite(sy) ||
                !std::isfinite(head_pitch) || !std::isfinite(head_yaw) || !std::isfinite(head_roll)) {
                return failInvalid();
            }

            // 3. Startup calibration phase (returns state == -1 and NaN angles, incl. completing frame)
            if (!calibrated_) {
                if (blink_mode_ == BlinkMode::Hybrid) {
                    updateHybridCalibration(raw_l, raw_r, head_pitch, sy, has_bs, bs);
                } else {
                    if (raw_l > kCalibEarEps && raw_r > kCalibEarEps) {
                        sum_l_ += raw_l;
                        sum_r_ += raw_r;
                        ++calib_frames_;
                        if (calib_frames_ >= calib_target_) {
                            base_l_ = sum_l_ / calib_target_;
                            base_r_ = sum_r_ / calib_target_;
                            calibrated_ = true;
                        }
                    }
                }
                out.state = -1;
                markInvalid(out);
                if (diag) {
                    *diag = GazeDiagRow{};
                    diag->detected = 0;
                    diag->state = -1;
                    diag->left_iris_x = l_ix;
                    diag->right_iris_x = r_ix;
                    diag->sx = sx;
                    diag->left_iris_y = l_iy;
                    diag->right_iris_y = r_iy;
                    diag->sy = sy;
                    diag->head_pitch_deg = head_pitch;
                    diag->head_yaw_deg = head_yaw;
                    diag->head_roll_deg = head_roll;
                    diag->norm_x = -1.0;
                    diag->norm_y = -1.0;
                    fillDiagLandmarks(lm, *diag);
                }
                if (blink_diag) {
                    *blink_diag = BlinkDiagRow{};
                    blink_diag->detected = 0;
                    blink_diag->state = -1;
                    blink_diag->detector_mode = "calibrating";
                    blink_diag->raw_ear_l = raw_l;
                    blink_diag->raw_ear_r = raw_r;
                    blink_diag->base_ear_l = base_l_;
                    blink_diag->base_ear_r = base_r_;
                    if (has_bs) {
                        blink_diag->bs_blink_l = bs[0];
                        blink_diag->bs_blink_r = bs[1];
                        blink_diag->bs_lookdown_l = bs[2];
                        blink_diag->bs_lookdown_r = bs[3];
                        blink_diag->bs_squint_l = bs[4];
                        blink_diag->bs_squint_r = bs[5];
                    }
                    blink_diag->sy = sy;
                    blink_diag->head_pitch_deg = head_pitch;
                }
                return out;
            }

            // 4. Gaze regression (validated finite BEFORE mutating blink state)
            const float gf[kPoseFeatures] = { float(sx), float(sy), float(head_pitch), float(head_yaw), float(head_roll) };
            const double rel_pitch = pitch_ ? pitch_->predict(gf) : 0.0;
            const double rel_yaw = yaw_ ? yaw_->predict(gf) : 0.0;

            // absolute world gaze = eye-in-socket + head (raw absolute values, bypassing EMA to leave filtering to DSP)
            const double abs_pitch = rel_pitch + head_pitch;
            const double abs_yaw = rel_yaw + head_yaw;
            if (!std::isfinite(rel_pitch) || !std::isfinite(rel_yaw) ||
                !std::isfinite(abs_pitch) || !std::isfinite(abs_yaw)) {
                return failInvalid();
            }

            // 5. Blink / Wink / Closure intent state
            double norm_l = std::numeric_limits<double>::quiet_NaN();
            double norm_r = std::numeric_limits<double>::quiet_NaN();
            double out_score_l = std::numeric_limits<double>::quiet_NaN();
            double out_score_r = std::numeric_limits<double>::quiet_NaN();
            const char* det_mode = "legacy_rf";

            if (blink_mode_ == BlinkMode::Hybrid) {
                out.state = stepHybridBlink(raw_l, raw_r, head_pitch, sy, ts_ms, has_bs, bs,
                                            norm_l, norm_r, out_score_l, out_score_r, det_mode);
            } else {
                const double left = base_l_ > 0.0 ? raw_l / base_l_ : 0.0;
                const double right = base_r_ > 0.0 ? raw_r / base_r_ : 0.0;
                norm_l = left;
                norm_r = right;
                push(hist_l_, left);
                push(hist_r_, right);
                double l_min, l_var, r_min, r_var;
                rolling(hist_l_, l_min, l_var);
                rolling(hist_r_, r_min, r_var);
                const float wf[kGazeFeatures] = {
                    float(left), float(l_min), float(l_var),
                    float(right), float(r_min), float(r_var)
                };
                bool wf_ok = true;
                for (int i = 0; i < kGazeFeatures; ++i) {
                    if (!std::isfinite(wf[i])) { wf_ok = false; break; }
                }
                const int raw_state = (wink_ && wf_ok) ? wink_->predict(wf) : 0;
                out.state = (raw_state >= 0 && raw_state <= 3) ? raw_state : 0;
            }

            setPointer(out, abs_yaw, abs_pitch);
            out.head_yaw_deg = head_yaw;
            out.head_pitch_deg = head_pitch;

            if (diag) {
                *diag = GazeDiagRow{};
                diag->detected = out.detected() ? 1 : 0;
                diag->state = out.state;
                diag->left_iris_x = l_ix;
                diag->right_iris_x = r_ix;
                diag->sx = sx;
                diag->left_iris_y = l_iy;
                diag->right_iris_y = r_iy;
                diag->sy = sy;
                diag->head_pitch_deg = head_pitch;
                diag->head_yaw_deg = head_yaw;
                diag->head_roll_deg = head_roll;
                diag->rel_pitch_deg = rel_pitch;
                diag->rel_yaw_deg = rel_yaw;
                diag->abs_pitch_deg = abs_pitch;
                diag->abs_yaw_deg = abs_yaw;
                diag->norm_x = double(out.x);
                diag->norm_y = double(out.y);
                fillDiagLandmarks(lm, *diag);
            }
            if (blink_diag) {
                *blink_diag = BlinkDiagRow{};
                blink_diag->detected = out.detected() ? 1 : 0;
                blink_diag->state = out.state;
                blink_diag->detector_mode = det_mode;
                blink_diag->raw_ear_l = raw_l;
                blink_diag->raw_ear_r = raw_r;
                blink_diag->base_ear_l = base_l_;
                blink_diag->base_ear_r = base_r_;
                blink_diag->norm_ear_l = norm_l;
                blink_diag->norm_ear_r = norm_r;
                if (has_bs) {
                    blink_diag->bs_blink_l = bs[0];
                    blink_diag->bs_blink_r = bs[1];
                    blink_diag->bs_lookdown_l = bs[2];
                    blink_diag->bs_lookdown_r = bs[3];
                    blink_diag->bs_squint_l = bs[4];
                    blink_diag->bs_squint_r = bs[5];
                }
                blink_diag->score_l = out_score_l;
                blink_diag->score_r = out_score_r;
                blink_diag->sy = sy;
                blink_diag->head_pitch_deg = head_pitch;
            }
            return out;
        }

        // Euler angles in degrees from a row-major 4x4 matrix (gaze_intent.get_head_pose_angles).
        static void headPose(const float* m, double& pitch, double& yaw, double& roll) {
            const float r00 = m[0], r10 = m[4], r20 = m[8], r21 = m[9], r22 = m[10], r11 = m[5], r12 = m[6];
            volatile float a = r00 * r00, b = r10 * r10;                // float32 like numpy; volatile = no fused multiply-add
            const float s = a + b;
            const double sy = std::sqrt(double(s));
            constexpr double kDeg = 180.0 / 3.14159265358979323846;
            if (!(sy < 1e-6)) {
                pitch = std::atan2(double(r21), double(r22)) * kDeg;
                yaw = std::atan2(-double(r20), sy) * kDeg;
                roll = std::atan2(double(r10), double(r00)) * kDeg;
            } else {
                pitch = std::atan2(-double(r12), double(r11)) * kDeg;
                yaw = std::atan2(-double(r20), sy) * kDeg;
                roll = 0.0;
            }
        }

    private:
        static constexpr int kLeftEye[6]  = { 33, 160, 158, 133, 153, 144 };
        static constexpr int kRightEye[6] = { 362, 385, 387, 263, 373, 380 };

        // Numerical guards (mirrored in gaze_intent.py)
        static constexpr double kEarDistEps      = 1e-12;
        static constexpr double kCalibEarEps     = 1e-6;
        static constexpr double kIrisBaselineEps = 1e-12;

        // HybridBlinkDetector v2 shared spec constants (mirrored 1:1 in hybrid_blink_spec.json & gaze_intent.py)
        static constexpr double kCalibMinEarBs     = 0.12;
        static constexpr double kCalibMinEarNoBs   = 0.08;
        static constexpr double kCalibMaxAsym      = 0.35;
        static constexpr double kCalibMaxPitchDeg  = 20.0;
        static constexpr double kCalibMaxBsBlink   = 0.35;
        static constexpr double kCalibEarTrimLo    = 0.20;
        static constexpr double kCalibEarTrimHi    = 0.90;
        static constexpr double kCalibEarQuantile  = 0.75;
        static constexpr double kCalibBsTrimLo     = 0.10;
        static constexpr double kCalibBsTrimHi     = 0.80;
        static constexpr double kCalibBsQuantile   = 0.25;
        static constexpr double kMaxBaseBs         = 0.25;
        static constexpr double kDefaultBaseSy     = -0.08;

        static constexpr double kAdaptAlpha        = 0.005;
        static constexpr double kAdaptMaxAsym      = 0.25;
        static constexpr double kAdaptMaxPitchDeg  = 15.0;
        static constexpr double kAdaptMaxRawScore  = 0.22;
        static constexpr double kAdaptMaxBsBlink   = 0.25;
        static constexpr double kAdaptMaxBsLookDown = 0.35;

        static constexpr double kBsLookDownDeadzone = 0.20;
        static constexpr double kBsLookDownGain     = 0.30;
        static constexpr double kBsDenomMin         = 0.35;
        static constexpr double kLdGateEarLo        = 0.34;
        static constexpr double kLdGateEarHi        = 0.48;

        static constexpr double kEarSyGain          = 0.90;
        static constexpr double kEarSyMaxDelta      = 0.20;
        static constexpr double kEarPitchGain       = 0.14;
        static constexpr double kEarPitchMaxDeg     = 25.0;
        static constexpr double kEarLookDownGain    = 0.24;
        static constexpr double kEarMaxDroopComp    = 0.30;
        static constexpr double kEarOpenRef         = 0.92;
        static constexpr double kEarClosedSpan      = 0.55;

        static constexpr double kFusionWBs          = 0.55;
        static constexpr double kFusionWEar         = 0.45;

        static constexpr double kEmaTauAttackS      = 0.015;
        static constexpr double kEmaTauReleaseS     = 0.030;
        static constexpr double kMaxGapS            = 0.250;
        static constexpr double kDefaultDtS         = 1.0 / 30.0;

        static constexpr double kCloseThresh        = 0.48;
        static constexpr double kOpenThresh         = 0.30;
        static constexpr double kWinkAsymEnter      = 0.18;
        static constexpr double kWinkAsymExit       = 0.10;
        static constexpr double kWinkOtherMax       = 0.65;
        static constexpr double kSymBand            = 0.14;

        static double clampRange(double v, double lo, double hi) {
            if (!std::isfinite(v)) return lo;
            return v < lo ? lo : (v > hi ? hi : v);
        }
        static double clampUnit(double v) { return clampRange(v, 0.0, 1.0); }

        static double trimmedQuantile(const std::vector<double>& values, double low_frac, double high_frac, double q) {
            if (values.empty()) return 0.0;
            std::vector<double> s = values;
            std::sort(s.begin(), s.end());
            const int n = static_cast<int>(s.size());
            int lo = static_cast<int>(std::floor(low_frac * n));
            int hi = static_cast<int>(std::floor(high_frac * n));
            if (lo < 0) lo = 0;
            if (hi <= lo) hi = lo + 1;
            if (hi > n) hi = n;
            const int m = hi - lo;
            if (m == 1) return s[size_t(lo)];
            const double pos = q * double(m - 1);
            const int idx = static_cast<int>(std::floor(pos));
            const double frac = pos - double(idx);
            if (idx + 1 >= m) return s[size_t(lo + m - 1)];
            return s[size_t(lo + idx)] * (1.0 - frac) + s[size_t(lo + idx + 1)] * frac;
        }

        void updateHybridCalibration(double raw_l, double raw_r, double head_pitch_deg, double sy,
                                     bool has_bs, const double bs[8]) {
            const double min_ear = has_bs ? kCalibMinEarBs : kCalibMinEarNoBs;
            if (raw_l <= min_ear || raw_r <= min_ear) return;
            const double max_ear = std::max(raw_l, raw_r);
            if (std::fabs(raw_l - raw_r) / max_ear >= kCalibMaxAsym) return;
            if (std::fabs(head_pitch_deg) >= kCalibMaxPitchDeg) return;
            if (has_bs && (bs[0] >= kCalibMaxBsBlink || bs[1] >= kCalibMaxBsBlink)) return;

            cal_ear_l_.push_back(raw_l);
            cal_ear_r_.push_back(raw_r);
            cal_sy_.push_back(sy);
            if (has_bs) {
                cal_bs_l_.push_back(bs[0]);
                cal_bs_r_.push_back(bs[1]);
            }
            ++calib_frames_;

            if (calib_frames_ >= calib_target_) {
                base_l_ = trimmedQuantile(cal_ear_l_, kCalibEarTrimLo, kCalibEarTrimHi, kCalibEarQuantile);
                base_r_ = trimmedQuantile(cal_ear_r_, kCalibEarTrimLo, kCalibEarTrimHi, kCalibEarQuantile);
                base_sy_ = trimmedQuantile(cal_sy_, 0.0, 1.0, 0.5);
                if (!cal_bs_l_.empty() && !cal_bs_r_.empty()) {
                    base_bs_l_ = clampRange(trimmedQuantile(cal_bs_l_, kCalibBsTrimLo, kCalibBsTrimHi, kCalibBsQuantile),
                                            0.0, kMaxBaseBs);
                    base_bs_r_ = clampRange(trimmedQuantile(cal_bs_r_, kCalibBsTrimLo, kCalibBsTrimHi, kCalibBsQuantile),
                                            0.0, kMaxBaseBs);
                } else {
                    base_bs_l_ = 0.0;
                    base_bs_r_ = 0.0;
                }
                calibrated_ = true;
            }
        }

        int stepHybridBlink(double raw_l, double raw_r, double head_pitch_deg, double sy, int64_t ts_ms,
                            bool has_bs, const double bs[8],
                            double& out_norm_l, double& out_norm_r,
                            double& out_score_l, double& out_score_r,
                            const char*& out_mode) {
            const double norm_l = base_l_ > 0.0 ? raw_l / base_l_ : 0.0;
            const double norm_r = base_r_ > 0.0 ? raw_r / base_r_ : 0.0;
            out_norm_l = norm_l;
            out_norm_r = norm_r;
            push(hist_l_, norm_l);
            push(hist_r_, norm_r);

            const double bs_bl_l = has_bs ? bs[0] : 0.0;
            const double bs_bl_r = has_bs ? bs[1] : 0.0;
            const double bs_ld_l = has_bs ? bs[2] : 0.0;
            const double bs_ld_r = has_bs ? bs[3] : 0.0;

            const double sy_down = clampRange(sy - base_sy_, 0.0, kEarSyMaxDelta);
            const double pitch_down = clampRange(head_pitch_deg, 0.0, kEarPitchMaxDeg) / kEarPitchMaxDeg;
            const double base_droop = kEarSyGain * sy_down + kEarPitchGain * pitch_down;
            const double gate_span = kLdGateEarHi - kLdGateEarLo;
            const double open_gate_l = clampUnit(((norm_l + base_droop) - kLdGateEarLo) / gate_span);
            const double open_gate_r = clampUnit(((norm_r + base_droop) - kLdGateEarLo) / gate_span);

            const double ld_excess_l = has_bs ? (std::max(0.0, bs_ld_l - kBsLookDownDeadzone) * open_gate_l) : 0.0;
            const double ld_excess_r = has_bs ? (std::max(0.0, bs_ld_r - kBsLookDownDeadzone) * open_gate_r) : 0.0;

            const double droop_l = clampRange(
                base_droop * open_gate_l + kEarLookDownGain * ld_excess_l,
                0.0, kEarMaxDroopComp);
            const double droop_r = clampRange(
                base_droop * open_gate_r + kEarLookDownGain * ld_excess_r,
                0.0, kEarMaxDroopComp);
            const double norm_l_comp = norm_l + droop_l;
            const double norm_r_comp = norm_r + droop_r;

            const double ear_score_l = clampUnit((kEarOpenRef - norm_l_comp) / kEarClosedSpan);
            const double ear_score_r = clampUnit((kEarOpenRef - norm_r_comp) / kEarClosedSpan);

            double raw_score_l = ear_score_l;
            double raw_score_r = ear_score_r;
            if (has_bs) {
                const double denom_l = std::max(kBsDenomMin, 1.0 - base_bs_l_);
                const double denom_r = std::max(kBsDenomMin, 1.0 - base_bs_r_);
                const double bs_score_l = clampUnit((bs_bl_l - base_bs_l_ - kBsLookDownGain * ld_excess_l) / denom_l);
                const double bs_score_r = clampUnit((bs_bl_r - base_bs_r_ - kBsLookDownGain * ld_excess_r) / denom_r);
                raw_score_l = kFusionWBs * bs_score_l + kFusionWEar * ear_score_l;
                raw_score_r = kFusionWBs * bs_score_r + kFusionWEar * ear_score_r;
                out_mode = "hybrid_bs_ear";
            } else {
                out_mode = "hybrid_ear_only";
            }

            if (!has_smooth_) {
                score_l_ = raw_score_l;
                score_r_ = raw_score_r;
                has_smooth_ = true;
            } else {
                const double dt_s = (ts_ms >= 0 && last_ts_ms_ >= 0)
                                        ? (double(ts_ms - last_ts_ms_) / 1000.0)
                                        : kDefaultDtS;
                if (dt_s <= 0.0 || dt_s > kMaxGapS) {
                    score_l_ = raw_score_l;
                    score_r_ = raw_score_r;
                    eye_closed_l_ = false;
                    eye_closed_r_ = false;
                    prev_state_ = 0;
                } else {
                    const double tau_l = (raw_score_l > score_l_) ? kEmaTauAttackS : kEmaTauReleaseS;
                    const double tau_r = (raw_score_r > score_r_) ? kEmaTauAttackS : kEmaTauReleaseS;
                    const double alpha_l = 1.0 - std::exp(-dt_s / tau_l);
                    const double alpha_r = 1.0 - std::exp(-dt_s / tau_r);
                    score_l_ = score_l_ + alpha_l * (raw_score_l - score_l_);
                    score_r_ = score_r_ + alpha_r * (raw_score_r - score_r_);
                }
            }
            last_ts_ms_ = ts_ms;

            const double c_l = score_l_;
            const double c_r = score_r_;
            eye_closed_l_ = eye_closed_l_ ? (c_l >= kOpenThresh) : (c_l >= kCloseThresh);
            eye_closed_r_ = eye_closed_r_ ? (c_r >= kOpenThresh) : (c_r >= kCloseThresh);

            const double diff = c_l - c_r;
            const double abs_diff = std::fabs(diff);
            const double asym_l = (prev_state_ == 1) ? kWinkAsymExit : kWinkAsymEnter;
            const double asym_r = (prev_state_ == 2) ? kWinkAsymExit : kWinkAsymEnter;

            int state = 0;
            if (eye_closed_l_ && eye_closed_r_) {
                if (c_l >= kCloseThresh && c_r < kWinkOtherMax && diff >= asym_l) {
                    state = 1;
                } else if (c_r >= kCloseThresh && c_l < kWinkOtherMax && (-diff) >= asym_r) {
                    state = 2;
                } else {
                    state = 3;
                }
            } else if (prev_state_ == 3 && c_l >= kOpenThresh && c_r >= kOpenThresh && abs_diff < kSymBand) {
                state = 3;
            } else if (eye_closed_l_ && !eye_closed_r_) {
                if (diff >= asym_l && c_r < kWinkOtherMax) {
                    state = 1;
                } else if (c_l >= kCloseThresh && c_r >= kOpenThresh && abs_diff < kSymBand) {
                    state = 3;
                } else {
                    state = 0;
                }
            } else if (eye_closed_r_ && !eye_closed_l_) {
                if ((-diff) >= asym_r && c_l < kWinkOtherMax) {
                    state = 2;
                } else if (c_r >= kCloseThresh && c_l >= kOpenThresh && abs_diff < kSymBand) {
                    state = 3;
                } else {
                    state = 0;
                }
            } else {
                state = 0;
            }

            prev_state_ = state;
            out_score_l = score_l_;
            out_score_r = score_r_;

            // Slow adaptive open-eye EMA (active only on clearly open, symmetric, neutral-pitch frames)
            const double min_ear = has_bs ? kCalibMinEarBs : kCalibMinEarNoBs;
            if (state == 0 &&
                raw_l > min_ear && raw_r > min_ear &&
                (std::fabs(raw_l - raw_r) / std::max(raw_l, raw_r)) < kAdaptMaxAsym &&
                std::fabs(head_pitch_deg) < kAdaptMaxPitchDeg &&
                raw_score_l < kAdaptMaxRawScore && raw_score_r < kAdaptMaxRawScore &&
                (!has_bs || (bs_bl_l < kAdaptMaxBsBlink && bs_bl_r < kAdaptMaxBsBlink &&
                             bs_ld_l < kAdaptMaxBsLookDown && bs_ld_r < kAdaptMaxBsLookDown))) {
                base_l_ = (1.0 - kAdaptAlpha) * base_l_ + kAdaptAlpha * raw_l;
                base_r_ = (1.0 - kAdaptAlpha) * base_r_ + kAdaptAlpha * raw_r;
                if (has_bs) {
                    base_bs_l_ = clampRange((1.0 - kAdaptAlpha) * base_bs_l_ + kAdaptAlpha * bs_bl_l, 0.0, kMaxBaseBs);
                    base_bs_r_ = clampRange((1.0 - kAdaptAlpha) * base_bs_r_ + kAdaptAlpha * bs_bl_r, 0.0, kMaxBaseBs);
                }
            }
            return state;
        }

        // The Python's INVALID_ANGLE sentinel: NaN (-1 deg is a valid angle). x = y = -1 keeps detected() false.
        static void markInvalid(ModuleOut& o) {
            constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
            o.x = -1.0f;
            o.y = -1.0f;
            o.yaw_deg = o.pitch_deg = o.head_yaw_deg = o.head_pitch_deg = kNaN;
        }
        static void fillDiagLandmarks(const float* lm, GazeDiagRow& d) {
            d.lm33_x  = double(lm[3 * 33]);  d.lm33_y  = double(lm[3 * 33 + 1]);
            d.lm133_x = double(lm[3 * 133]); d.lm133_y = double(lm[3 * 133 + 1]);
            d.lm468_x = double(lm[3 * 468]); d.lm468_y = double(lm[3 * 468 + 1]);
            d.lm362_x = double(lm[3 * 362]); d.lm362_y = double(lm[3 * 362 + 1]);
            d.lm263_x = double(lm[3 * 263]); d.lm263_y = double(lm[3 * 263 + 1]);
            d.lm473_x = double(lm[3 * 473]); d.lm473_y = double(lm[3 * 473 + 1]);
        }
        // Face lost -> stale blink history/hysteresis cleared; unfinished calibration restarts; finished kept.
        void resetTrackingState() {
            hist_l_.clear();
            hist_r_.clear();
            has_smooth_ = false;
            last_ts_ms_ = -1;
            score_l_ = 0.0;
            score_r_ = 0.0;
            eye_closed_l_ = false;
            eye_closed_r_ = false;
            prev_state_ = 0;
            if (!calibrated_) {
                calib_frames_ = 0;
                sum_l_ = 0.0;
                sum_r_ = 0.0;
                cal_ear_l_.clear();
                cal_ear_r_.clear();
                cal_bs_l_.clear();
                cal_bs_r_.clear();
                cal_sy_.clear();
            }
        }

        static bool allFinite(const float* p, int count) {
            if (!p || count < 0) return false;
            for (int i = 0; i < count; ++i) {
                if (!std::isfinite(p[i])) return false;
            }
            return true;
        }
        static double clamp01(double v) {
            if (!std::isfinite(v)) return -1.0;
            return std::max(0.0, std::min(1.0, v));
        }
        static void setPointer(ModuleOut& o, double yaw_deg, double pitch_deg) {
            if (!std::isfinite(yaw_deg) || !std::isfinite(pitch_deg)) {
                markInvalid(o);
                return;
            }
            const double nx = clamp01(0.5 + yaw_deg / kSpanDeg);
            const double ny = clamp01(0.5 + pitch_deg / kSpanDeg);
            if (!std::isfinite(nx) || !std::isfinite(ny) || nx < 0.0 || ny < 0.0) {
                markInvalid(o);
                return;
            }
            o.yaw_deg = yaw_deg;
            o.pitch_deg = pitch_deg;
            o.x = float(nx);
            o.y = float(ny);
        }

        // True 3D distance in pixels (same as Python's get_3d_dist): MediaPipe's z is scaled by frame WIDTH.
        static double d3(const float* lm, int a, int b, int W, int H) {
            const double dx = (double(lm[3 * a]) - lm[3 * b]) * W;
            const double dy = (double(lm[3 * a + 1]) - lm[3 * b + 1]) * H;
            const double dz = (double(lm[3 * a + 2]) - lm[3 * b + 2]) * W;
            if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        static double ear(const float* lm, const int(&i)[6], int W, int H) {
            if (W <= 0 || H <= 0) return std::numeric_limits<double>::quiet_NaN();
            const double v1 = d3(lm, i[1], i[5], W, H);
            const double v2 = d3(lm, i[2], i[4], W, H);
            const double h  = d3(lm, i[0], i[3], W, H);
            if (!std::isfinite(v1) || !std::isfinite(v2) || !std::isfinite(h) || h <= kEarDistEps) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return (v1 + v2) / (2.0 * h);
        }
        // The iris against the line joining the eye corners, in normalised 2D space (gaze_intent.get_iris_ratio).
        static double irisRatio(const float* lm, int c1, int c2, int iris, bool x_axis) {
            const double ax = lm[3 * c1], ay = lm[3 * c1 + 1];
            const double bx = double(lm[3 * c2]) - ax, by = double(lm[3 * c2 + 1]) - ay;
            const double len_sq = bx * bx + by * by;
            if (len_sq < kIrisBaselineEps) return x_axis ? 0.5 : 0.0;
            const double len = std::sqrt(len_sq);
            const double ix = double(lm[3 * iris]) - ax, iy = double(lm[3 * iris + 1]) - ay;
            const double scale = (ix * bx + iy * by) / len_sq;
            if (x_axis) return scale;
            const double px = ix - scale * bx, py = iy - scale * by;
            const double d = std::sqrt(px * px + py * py) / len;
            return py < 0 ? -d : d;
        }
        void push(std::deque<double>& h, double v) { h.push_back(v); if (h.size() > window_) h.pop_front(); }
        static void rolling(const std::deque<double>& h, double& mn, double& var) {
            mn = *std::min_element(h.begin(), h.end());
            double mean = 0.0;
            for (double v : h) mean += v;
            mean /= double(h.size());
            var = 0.0;
            for (double v : h) var += (v - mean) * (v - mean);
            var /= double(h.size());
        }

        IForest* wink_;
        IRegressor* pitch_, * yaw_;
        int calib_target_;
        size_t window_;
        BlinkMode blink_mode_ = BlinkMode::Hybrid;
        bool calibrated_ = false;
        int calib_frames_ = 0;
        double sum_l_ = 0.0, sum_r_ = 0.0;
        double base_l_ = 1.0, base_r_ = 1.0;
        double base_bs_l_ = 0.0, base_bs_r_ = 0.0;
        double base_sy_ = kDefaultBaseSy;
        std::vector<double> cal_ear_l_, cal_ear_r_, cal_bs_l_, cal_bs_r_, cal_sy_;
        bool has_smooth_ = false;
        int64_t last_ts_ms_ = -1;
        double score_l_ = 0.0, score_r_ = 0.0;
        bool eye_closed_l_ = false, eye_closed_r_ = false;
        int prev_state_ = 0;
        std::deque<double> hist_l_, hist_r_;
    };

} // namespace core