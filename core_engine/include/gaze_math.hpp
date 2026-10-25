// Gaze + blink pipeline: a line-for-line C++ port of gaze_intent.py (Eng 3), current version.
//   478 face landmarks + the 4x4 facial transformation matrix
//      -> EAR, 30-frame blink calibration, rolling features -> wink model (gesture_model.dll) -> state
//      -> iris ratios (projected on the eye-corner line) + head pose (Euler angles from the matrix)
//      -> pitch / yaw models (gaze_pitch.dll, gaze_yaw.dll) = eye-in-socket angles
//      -> + head angles = absolute world gaze (raw, bypassing C++ EMA to let DSP handle filtering)
// Like the Python: during calibration (state -1) the gaze is 0, 0; with no face the answer is (x = -1, y = -1, state 0).
// All arithmetic is double, like Python's float64; the features are rounded to float32 before the models (what the
// Python does with np.float32). The head-pose products are float32 too, because the matrix is a float32 array.
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include "forest.hpp"
#include "tracking_types.hpp"

namespace core {

    class GazeNative {
    public:
        static constexpr int kFacePoints = 478;
        static constexpr int kPoseRows = 6;                // the 4x4 matrix rides behind the landmarks: 16 floats, row-major,
        static constexpr int kPacked = kFacePoints + kPoseRows;  // padded with zeros to 6 rows of 3 (see landmark_extractors.py)
        static constexpr double kSpanDeg = 60.0;                 // x, y = 0.5 + angle / kSpanDeg  (+-30 deg covers the unit square)

        GazeNative(IForest* wink, IRegressor* pitch, IRegressor* yaw, int calibration_frames = 30, int window = 10)
            : wink_(wink), pitch_(pitch), yaw_(yaw), calib_target_(calibration_frames),
            window_(static_cast<size_t>(window)) {}

        bool calibrated() const { return calibrated_; }
        void recalibrate() {
            calibrated_ = false; calib_frames_ = 0; sum_l_ = sum_r_ = 0.0; base_l_ = base_r_ = 1.0;
            hist_l_.clear(); hist_r_.clear();
        }

        // lm: n x 3 floats = kFacePoints landmarks (normalised to the image) + the packed pose matrix (n == kPacked).
        // n == 0 (or no matrix) means "no face" -> x = y = -1, state 0.
        ModuleOut process(const float* lm, int n, int W, int H) {
            ModuleOut out;
            out.ok = true;
            if (!lm || n < kPacked) {
                out.x = -1.0f;
                out.y = -1.0f;
                out.state = 0;
                out.yaw_deg = 0.0;
                out.pitch_deg = 0.0;
                return out;
            }
            const float* pose = lm + 3 * kFacePoints;

            const double raw_l = ear(lm, kLeftEye, W, H);
            const double raw_r = ear(lm, kRightEye, W, H);

            if (!calibrated_) {                                     // 30-frame blink calibration
                sum_l_ += raw_l;
                sum_r_ += raw_r;
                ++calib_frames_;
                if (calib_frames_ >= calib_target_) {
                    base_l_ = sum_l_ / calib_target_;
                    base_r_ = sum_r_ / calib_target_;
                    calibrated_ = true;
                }
                out.state = -1;                                     // "calibrating": gaze is 0, 0 and the OS stage holds still
                setPointer(out, 0.0, 0.0);
                return out;
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

            // iris position, averaged over both eyes
            const double sx = 0.5 * (irisRatio(lm, 33, 133, 468, true) + irisRatio(lm, 362, 263, 473, true));
            const double sy = 0.5 * (irisRatio(lm, 33, 133, 468, false) + irisRatio(lm, 362, 263, 473, false));

            double head_pitch, head_yaw, head_roll;
            headPose(pose, head_pitch, head_yaw, head_roll);

            const float gf[kPoseFeatures] = { float(sx), float(sy), float(head_pitch), float(head_yaw), float(head_roll) };
            const double rel_pitch = pitch_ ? pitch_->predict(gf) : 0.0;
            const double rel_yaw = yaw_ ? yaw_->predict(gf) : 0.0;

            // absolute world gaze = eye-in-socket + head (raw absolute values, bypassing EMA to leave filtering to DSP)
            const double abs_pitch = rel_pitch + head_pitch;
            const double abs_yaw = rel_yaw + head_yaw;

            setPointer(out, abs_yaw, abs_pitch);
            out.head_yaw_deg = head_yaw;
            out.head_pitch_deg = head_pitch;
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

        static double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
        static void setPointer(ModuleOut& o, double yaw_deg, double pitch_deg) {
            o.yaw_deg = yaw_deg;
            o.pitch_deg = pitch_deg;
            o.x = float(clamp01(0.5 + yaw_deg / kSpanDeg));
            o.y = float(clamp01(0.5 + pitch_deg / kSpanDeg));
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
            return h > 0 ? (v1 + v2) / (2.0 * h) : 0.0;
        }
        // The iris against the line joining the eye corners, in normalised 2D space (gaze_intent._get_iris_ratio).
        //    x: how far along the line (0 = corner1, 1 = corner2).  y: signed distance from the line / line length.
        static double irisRatio(const float* lm, int c1, int c2, int iris, bool x_axis) {
            const double ax = lm[3 * c1], ay = lm[3 * c1 + 1];
            const double bx = double(lm[3 * c2]) - ax, by = double(lm[3 * c2 + 1]) - ay;        // baseline vector
            const double len = std::sqrt(bx * bx + by * by);
            if (len == 0) return x_axis ? 0.5 : 0.0;
            const double ix = double(lm[3 * iris]) - ax, iy = double(lm[3 * iris + 1]) - ay;    // corner1 -> iris
            const double scale = (ix * bx + iy * by) / (len * len);
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