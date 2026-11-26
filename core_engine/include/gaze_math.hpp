// Gaze + blink pipeline: a line-for-line C++ port of gaze_intent.py (Eng 3), current version.
//   478 face landmarks + the 4x4 facial transformation matrix
//      -> EAR, 30-frame blink calibration, rolling features -> wink model (gesture_model.dll) -> state
//      -> iris ratios (projected on the eye-corner line) + head pose (Euler angles from the matrix)
//      -> pitch / yaw models (gaze_pitch.dll = direct absolute world pitch, gaze_yaw.dll = eye-in-socket yaw)
//      -> abs_yaw = rel_yaw + head_yaw, rel_pitch = abs_pitch - head_pitch (raw, bypassing C++ EMA for DSP)
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
#include "forest.hpp"
#include "tracking_types.hpp"

namespace core {

    // Opt-in per-frame diagnostic telemetry payload (--gaze-diag-log <file.csv>).
    // Unified 32-column schema shared 1:1 between C++ GazeNative and Python GazeIntent.
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

    class GazeNative {
    public:
        static constexpr int kFacePoints = 478;
        static constexpr int kPoseRows = 6;                // the 4x4 matrix rides behind the landmarks: 16 floats, row-major,
        static constexpr int kPacked = kFacePoints + kPoseRows;  // padded with zeros to 6 rows of 3 (see landmark_extractors.py)
        static constexpr double kSpanDeg = 60.0;                 // x = 0.5 + yaw / kSpanDeg, y = 0.5 - pitch / kSpanDeg  (+-30 deg covers the unit square; +pitch UP -> y toward 0)

        GazeNative(IForest* wink, IRegressor* pitch, IRegressor* yaw, int calibration_frames = 30, int window = 10)
            : wink_(wink), pitch_(pitch), yaw_(yaw), calib_target_(calibration_frames),
            window_(static_cast<size_t>(window)) {}

        bool calibrated() const { return calibrated_; }
        void recalibrate() {
            calibrated_ = false; calib_frames_ = 0; sum_l_ = sum_r_ = 0.0; base_l_ = base_r_ = 1.0;
            hist_l_.clear(); hist_r_.clear();
        }

        // lm: n x 3 floats = kFacePoints landmarks (normalised to the image) + the packed pose matrix (n == kPacked).
        // n == 0 (or no matrix, or any non-finite input/intermediate/output) means "no face" -> x = y = -1, state 0.
        ModuleOut process(const float* lm, int n, int W, int H, GazeDiagRow* diag = nullptr) {
            ModuleOut out;
            out.ok = true;
            auto failInvalid = [&]() -> ModuleOut {
                resetTrackingState();                               // EarFeatureState.reset_tracking_state()
                markInvalid(out);                                   // x = y = -1, all four angles NaN
                out.state = 0;
                if (diag) {
                    *diag = GazeDiagRow{};
                    diag->detected = 0;
                    diag->state = 0;
                    diag->norm_x = -1.0;
                    diag->norm_y = -1.0;
                }
                return out;
            };

            if (!lm || n < kPacked || W <= 0 || H <= 0) {
                return failInvalid();
            }
            const float* pose = lm + 3 * kFacePoints;
            if (!allFinite(lm, 3 * kFacePoints) || !allFinite(pose, 16)) {
                return failInvalid();
            }

            const double raw_l = ear(lm, kLeftEye, W, H);
            const double raw_r = ear(lm, kRightEye, W, H);
            if (!std::isfinite(raw_l) || !std::isfinite(raw_r)) {
                return failInvalid();
            }

            if (!calibrated_) {                                     // 30-frame blink calibration
                if (raw_l > kCalibEarEps && raw_r > kCalibEarEps) { // ignore corrupted / zero EAR frames (EarFeatureState.update)
                    sum_l_ += raw_l;
                    sum_r_ += raw_r;
                    ++calib_frames_;
                    if (calib_frames_ >= calib_target_) {
                        base_l_ = sum_l_ / calib_target_;
                        base_r_ = sum_r_ / calib_target_;
                        calibrated_ = true;
                    }
                }
                out.state = -1;                                     // "calibrating": no pointer (x = y = -1) and NaN angles
                markInvalid(out);
                if (diag) {
                    const double l_ix = irisRatio(lm, 33, 133, 468, true);
                    const double r_ix = irisRatio(lm, 362, 263, 473, true);
                    const double l_iy = irisRatio(lm, 33, 133, 468, false);
                    const double r_iy = irisRatio(lm, 362, 263, 473, false);
                    double hp, hy, hr;
                    headPose(pose, hp, hy, hr);
                    *diag = GazeDiagRow{};
                    diag->detected = 0;
                    diag->state = -1;
                    diag->left_iris_x = l_ix;
                    diag->right_iris_x = r_ix;
                    diag->sx = 0.5 * (l_ix + r_ix);
                    diag->left_iris_y = l_iy;
                    diag->right_iris_y = r_iy;
                    diag->sy = 0.5 * (l_iy + r_iy);
                    diag->head_pitch_deg = hp;
                    diag->head_yaw_deg = hy;
                    diag->head_roll_deg = hr;
                    diag->norm_x = -1.0;
                    diag->norm_y = -1.0;
                    fillDiagLandmarks(lm, *diag);
                }
                return out;
            }

            // iris position + head pose (validated finite before updating/returning tracked state)
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

            const float gf[kPoseFeatures] = { float(sx), float(sy), float(head_pitch), float(head_yaw), float(head_roll) };
            // gaze_pitch.dll predicts direct absolute world pitch; gaze_yaw.dll predicts eye-in-socket rel_yaw.
            const double abs_pitch = pitch_ ? pitch_->predict(gf) : head_pitch;
            const double rel_yaw = yaw_ ? yaw_->predict(gf) : 0.0;

            const double rel_pitch = abs_pitch - head_pitch;
            const double abs_yaw = rel_yaw + head_yaw;
            if (!std::isfinite(rel_pitch) || !std::isfinite(rel_yaw) ||
                !std::isfinite(abs_pitch) || !std::isfinite(abs_yaw)) {
                return failInvalid();
            }

            const double left = base_l_ > 0 ? raw_l / base_l_ : 0.0;
            const double right = base_r_ > 0 ? raw_r / base_r_ : 0.0;
            push(hist_l_, left);
            push(hist_r_, right);
            double l_min, l_var, r_min, r_var;
            rolling(hist_l_, l_min, l_var);                         // over what the window holds so far (np.min / np.var)
            rolling(hist_r_, r_min, r_var);
            const float wf[kGazeFeatures] = { float(left), float(l_min), float(l_var), float(right), float(r_min), float(r_var) };
            out.state = wink_ ? wink_->predict(wf) : 0;

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
            }
            else {
                pitch = std::atan2(-double(r12), double(r11)) * kDeg;
                yaw = std::atan2(-double(r20), sy) * kDeg;
                roll = 0.0;
            }
        }

    private:
        static constexpr int kLeftEye[6] = { 33, 160, 158, 133, 153, 144 };
        static constexpr int kRightEye[6] = { 362, 385, 387, 263, 373, 380 };

        // Numerical guards, the same values as gaze_intent.py (EAR_DIST_EPS, CALIB_EAR_EPS, IRIS_BASELINE_EPS).
        static constexpr double kEarDistEps = 1e-12;        // horizontal eye distance below this -> EAR = 0
        static constexpr double kCalibEarEps = 1e-6;        // calibration ignores frames whose raw EAR is not above this
        static constexpr double kIrisBaselineEps = 1e-12;   // squared eye-corner baseline below this -> neutral iris ratio

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
        // EarFeatureState.reset_tracking_state(): face lost -> stale blink history must not leak into the next track;
        // an unfinished calibration restarts. A finished calibration (the baselines) is kept.
        void resetTrackingState() {
            hist_l_.clear();
            hist_r_.clear();
            if (!calibrated_) { calib_frames_ = 0; sum_l_ = 0.0; sum_r_ = 0.0; }
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
            const double ny = clamp01(0.5 - pitch_deg / kSpanDeg);
            if (!std::isfinite(nx) || !std::isfinite(ny) || nx < 0.0 || ny < 0.0) {
                markInvalid(o);
                return;
            }
            o.yaw_deg = yaw_deg;
            o.pitch_deg = pitch_deg;
            o.x = float(nx);
            o.y = float(ny);
        }

        // True 3D distance in pixels (same as Python's _get_3d_dist): MediaPipe's z is scaled like x, i.e. by the frame WIDTH.
        static double d3(const float* lm, int a, int b, int W, int H) {
            const double dx = (double(lm[3 * a]) - lm[3 * b]) * W;
            const double dy = (double(lm[3 * a + 1]) - lm[3 * b + 1]) * H;
            const double dz = (double(lm[3 * a + 2]) - lm[3 * b + 2]) * W;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        static double ear(const float* lm, const int(&i)[6], int W, int H) {
            const double v1 = d3(lm, i[1], i[5], W, H);
            const double v2 = d3(lm, i[2], i[4], W, H);
            const double h = d3(lm, i[0], i[3], W, H);
            return h > kEarDistEps ? (v1 + v2) / (2.0 * h) : 0.0;
        }
        // The iris against the line joining the eye corners, in normalised 2D space (gaze_intent._get_iris_ratio).
        //    x: how far along the line (0 = corner1, 1 = corner2).  y: signed distance from the line / line length.
        static double irisRatio(const float* lm, int c1, int c2, int iris, bool x_axis) {
            const double ax = lm[3 * c1], ay = lm[3 * c1 + 1];
            const double bx = double(lm[3 * c2]) - ax, by = double(lm[3 * c2 + 1]) - ay;        // baseline vector
            const double len_sq = bx * bx + by * by;
            if (len_sq < kIrisBaselineEps) return x_axis ? 0.5 : 0.0;
            const double len = std::sqrt(len_sq);
            const double ix = double(lm[3 * iris]) - ax, iy = double(lm[3 * iris + 1]) - ay;    // corner1 -> iris
            const double scale = (ix * bx + iy * by) / len_sq;
            if (x_axis) return scale;
            const double px = ix - scale * bx, py = iy - scale * by;                          // perpendicular part
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
            var /= double(h.size());                                // population variance, like np.var
        }

        IForest* wink_;
        IRegressor* pitch_, * yaw_;
        int calib_target_;
        size_t window_;
        bool calibrated_ = false;
        int calib_frames_ = 0;
        double sum_l_ = 0.0, sum_r_ = 0.0, base_l_ = 1.0, base_r_ = 1.0;
        std::deque<double> hist_l_, hist_r_;
    };

} // namespace core