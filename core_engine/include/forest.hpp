// The blink/wink classifier (Random Forest). Three interchangeable backends:
//   TreeliteForest   : the forest compiled to native C by Treelite/TL2cgen (microseconds, no Python).
//                      Only built when core_engine/generated/gaze_forest/{main.c,header.h} exist.
//   PySklearnForest  : loads gesture_model.pkl through the embedded Python (about 5 ms/call). Fallback.
//   LambdaForest     : for unit tests.
#pragma once
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>

namespace core {

constexpr int kGazeFeatures = 7;   // [left_ear, l_min, l_var, right_ear, r_min, r_var, bbox_area]

class IForest {
public:
    virtual ~IForest() = default;
    // `features` points to kGazeFeatures floats. Returns the class label (0..3).
    virtual int predict(const float* features) = 0;
    virtual const char* name() const = 0;
    virtual uint64_t errors() const { return 0; }
};

class LambdaForest : public IForest {
public:
    explicit LambdaForest(std::function<int(const float*)> fn) : fn_(std::move(fn)) {}
    int predict(const float* f) override { return fn_(f); }
    const char* name() const override { return "lambda"; }
private:
    std::function<int(const float*)> fn_;
};

#ifdef CORE_HAVE_TREELITE_FOREST
extern "C" {
int gaze_forest_num_features(void);
int gaze_forest_num_classes(void);
int gaze_forest_predict(const float* features);   // implemented in src/forest_shim.c
}

class TreeliteForest : public IForest {
public:
    TreeliteForest() {
        if (gaze_forest_num_features() != kGazeFeatures)
            throw std::runtime_error("compiled forest expects a different number of features than 7");
    }
    int predict(const float* f) override { return gaze_forest_predict(f); }
    const char* name() const override { return "treelite (compiled)"; }
    int numClasses() const { return gaze_forest_num_classes(); }
};
#endif

} // namespace core
