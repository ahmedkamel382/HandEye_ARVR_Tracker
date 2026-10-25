// The gaze models. Since Eng 3's rewrite there are THREE Treelite/TL2cgen models, all shipped as compiled .dll files in
// ai_pipeline/gaze_intent/models/ and run natively (no Python, see tl_model.hpp):
//   gesture_model.dll  wink/blink classifier   6 features -> class 0..3        (IForest)
//   gaze_pitch.dll     eye-in-socket pitch     5 features -> degrees           (IRegressor)
//   gaze_yaw.dll       eye-in-socket yaw       5 features -> degrees           (IRegressor)
// LambdaForest / LambdaRegressor are for unit tests.
#pragma once
#include <cstdint>
#include <functional>
#include <utility>

namespace core {

constexpr int kGazeFeatures = 6;   // wink model: [left_ear, l_min, l_var, right_ear, r_min, r_var]
constexpr int kPoseFeatures = 5;   // pitch/yaw models: [iris_x, iris_y, head_pitch, head_yaw, head_roll]

class IForest {
public:
    virtual ~IForest() = default;
    // `features` points to kGazeFeatures floats. Returns the class label (0..3).
    virtual int predict(const float* features) = 0;
    virtual const char* name() const = 0;
    virtual uint64_t errors() const { return 0; }
};

class IRegressor {
public:
    virtual ~IRegressor() = default;
    // `features` points to kPoseFeatures floats. Returns degrees.
    virtual double predict(const float* features) = 0;
    virtual const char* name() const = 0;
};

class LambdaForest : public IForest {
public:
    explicit LambdaForest(std::function<int(const float*)> fn) : fn_(std::move(fn)) {}
    int predict(const float* f) override { return fn_(f); }
    const char* name() const override { return "lambda"; }
private:
    std::function<int(const float*)> fn_;
};

class LambdaRegressor : public IRegressor {
public:
    explicit LambdaRegressor(std::function<double(const float*)> fn) : fn_(std::move(fn)) {}
    double predict(const float* f) override { return fn_(f); }
    const char* name() const override { return "lambda"; }
private:
    std::function<double(const float*)> fn_;
};

} // namespace core
