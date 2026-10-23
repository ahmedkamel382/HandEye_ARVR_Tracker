// BUILT-IN 1-Euro filter (Casiez, Roussel, Vogel 2012): smooths jitter when the pointer is still, and
// lets fast movements through with little lag. This is the project's reference implementation;
// Eng 4's version (dsp_filter/) is the default; see dsp_adapter.hpp. This one is the fallback (--filter builtin).
//
// Units: coordinates are normalised camera units (0..1), time is seconds.
//   min_cutoff : Hz. Lower = steadier when still (more lag).     default 1.0
//   beta       : speed coefficient. Higher = less lag when moving.   default 8.0 (for 0..1 units/s)
//   d_cutoff   : Hz for the speed estimate.                       default 1.0
#pragma once
#include <algorithm>
#include <cmath>

namespace core {

class IPointFilter {
public:
    virtual ~IPointFilter() = default;
    virtual void reset() = 0;
    // t_sec must increase between calls. x, y are replaced with the filtered values.
    virtual void filter(double& x, double& y, double t_sec) = 0;
};

class OneEuro {
public:
    OneEuro(double min_cutoff = 1.0, double beta = 8.0, double d_cutoff = 1.0)
        : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}

    void reset() { init_ = false; }

    double filter(double x, double t) {
        if (!init_) {
            init_ = true; x_prev_ = x; dx_prev_ = 0.0; t_prev_ = t;
            return x;
        }
        const double dt = std::max(t - t_prev_, 1e-6);
        t_prev_ = t;
        const double dx = (x - x_prev_) / dt;
        const double a_d = alpha(dt, d_cutoff_);
        const double dx_hat = a_d * dx + (1.0 - a_d) * dx_prev_;
        const double cutoff = min_cutoff_ + beta_ * std::fabs(dx_hat);
        const double a = alpha(dt, cutoff);
        const double x_hat = a * x + (1.0 - a) * x_prev_;
        x_prev_ = x_hat;
        dx_prev_ = dx_hat;
        return x_hat;
    }

private:
    static double alpha(double dt, double cutoff) {
        const double tau = 1.0 / (2.0 * 3.14159265358979323846 * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }
    double min_cutoff_, beta_, d_cutoff_;
    bool init_ = false;
    double x_prev_ = 0.0, dx_prev_ = 0.0, t_prev_ = 0.0;
};

class OneEuroPoint : public IPointFilter {
public:
    OneEuroPoint(double min_cutoff = 1.0, double beta = 8.0, double d_cutoff = 1.0)
        : fx_(min_cutoff, beta, d_cutoff), fy_(min_cutoff, beta, d_cutoff) {}
    void reset() override { fx_.reset(); fy_.reset(); }
    void filter(double& x, double& y, double t_sec) override {
        x = fx_.filter(x, t_sec);
        y = fy_.filter(y, t_sec);
    }
private:
    OneEuro fx_, fy_;
};

} // namespace core
