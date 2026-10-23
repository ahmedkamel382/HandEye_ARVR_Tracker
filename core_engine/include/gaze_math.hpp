// Gaze + blink pipeline: a line-for-line C++ port of gaze_intent.py (Eng 3).
//   478 face landmarks -> EAR, iris gaze ratios, 30-frame calibration, rolling features -> forest -> state
// All arithmetic is done in double, exactly like the Python (which uses float64), and the 7 features
// are rounded to float32 before the forest, exactly like scikit-learn does internally.
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include "forest.hpp"
#include "tracking_types.hpp"

namespace core {

class GazeNative {
public:
    static constexpr int kMinPoints = 474;      // iris landmarks are 468..477

    explicit GazeNative(IForest* forest, int calibration_frames = 30, int window = 10)
        : forest_(forest), calib_target_(calibration_frames), window_(static_cast<size_t>(window)) {}

    bool calibrated() const { return calibrated_; }
    void recalibrate() {
        calibrated_ = false; calib_frames_ = 0; sum_l_ = sum_r_ = 0.0; base_l_ = base_r_ = 1.0;
        hist_l_.clear(); hist_r_.clear();
    }

    // lm: n x 3 floats normalised to the image. n == 0 means "no face" -> (-1, -1, 0).
    ModuleOut process(const float* lm, int n, int W, int H) {
        ModuleOut out;
        out.ok = true;
        if (!lm || n < kMinPoints) return out;

        const double raw_l = ear(lm, kLeftEye, W, H);
        const double raw_r = ear(lm, kRightEye, W, H);

        // Gaze (runs independently of calibration)
        const double gx = clamp01(0.5 * (ratio(lm, 33, 133, 468, W, H) + ratio(lm, 362, 263, 473, W, H)));
        const double gy = clamp01(0.5 * (ratio(lm, 159, 145, 468, W, H) + ratio(lm, 386, 374, 473, W, H)));
        out.x = float(gx);
        out.y = float(gy);

        if (!calibrated_) {                                       // 30-frame dynamic calibration
            sum_l_ += raw_l;
            sum_r_ += raw_r;
            ++calib_frames_;
            if (calib_frames_ >= calib_target_) {
                base_l_ = sum_l_ / calib_target_;
                base_r_ = sum_r_ / calib_target_;
                calibrated_ = true;
            }
            out.state = -1;                                       // "calibrating": hold the cursor
            return out;
        }

        const double left  = base_l_ > 0 ? raw_l / base_l_ : 0.0;
        const double right = base_r_ > 0 ? raw_r / base_r_ : 0.0;
        push(hist_l_, left);
        push(hist_r_, right);
        double l_min, l_var, r_min, r_var;
        rolling(hist_l_, left, l_min, l_var);
        rolling(hist_r_, right, r_min, r_var);

        // Face bounding-box area in pixels (this is feature #7 the forest was trained on)
        double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
        for (int i = 0; i < n; ++i) {
            const double x = double(lm[3 * i]) * W, y = double(lm[3 * i + 1]) * H;
            minx = std::min(minx, x); maxx = std::max(maxx, x);
            miny = std::min(miny, y); maxy = std::max(maxy, y);
        }
        const double area = (maxx - minx) * (maxy - miny);

        const float f[kGazeFeatures] = { float(left), float(l_min), float(l_var),
                                         float(right), float(r_min), float(r_var), float(area) };
        out.state = forest_ ? forest_->predict(f) : 0;
        return out;
    }

private:
    static constexpr int kLeftEye[6]  = {33, 160, 158, 133, 153, 144};
    static constexpr int kRightEye[6] = {362, 385, 387, 263, 373, 380};

    static double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

    // True 3D distance: x and y in pixels, z as given (same as Python's _get_3d_dist).
    static double d3(const float* lm, int a, int b, int W, int H) {
        return std::hypot(std::hypot((double(lm[3 * a]) - lm[3 * b]) * W,
                                     (double(lm[3 * a + 1]) - lm[3 * b + 1]) * H),
                          double(lm[3 * a + 2]) - lm[3 * b + 2]);
    }
    static double ear(const float* lm, const int (&i)[6], int W, int H) {
        const double v1 = d3(lm, i[1], i[5], W, H);
        const double v2 = d3(lm, i[2], i[4], W, H);
        const double h  = d3(lm, i[0], i[3], W, H);
        return h > 0 ? (v1 + v2) / (2.0 * h) : 0.0;
    }
    static double ratio(const float* lm, int b1, int b2, int iris, int W, int H) {
        const double axis = d3(lm, b1, b2, W, H);
        if (axis == 0) return 0.5;
        return d3(lm, iris, b2, W, H) / axis;
    }
    void push(std::deque<double>& h, double v) { h.push_back(v); if (h.size() > window_) h.pop_front(); }
    void rolling(const std::deque<double>& h, double current, double& mn, double& var) const {
        if (h.size() == window_) {                                // np.min / np.var (population variance)
            mn = *std::min_element(h.begin(), h.end());
            double mean = 0.0;
            for (double v : h) mean += v;
            mean /= double(window_);
            var = 0.0;
            for (double v : h) var += (v - mean) * (v - mean);
            var /= double(window_);
        } else {
            mn = current;
            var = 0.0;
        }
    }

    IForest* forest_;
    int calib_target_;
    size_t window_;
    bool calibrated_ = false;
    int calib_frames_ = 0;
    double sum_l_ = 0.0, sum_r_ = 0.0, base_l_ = 1.0, base_r_ = 1.0;
    std::deque<double> hist_l_, hist_r_;
};

// C++17: out-of-class definitions are implicit for constexpr static members (they are inline).

} // namespace core
