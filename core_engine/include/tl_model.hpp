// Runs a Treelite/TL2cgen model that was compiled to a Windows .dll (what gaze_intent.py loads with tl2cgen.Predictor),
// but natively: no Python, no GIL, microseconds per call. Same binaries as the Python reference => same answers.
//
// The .dll exports the TL2cgen C API:  predict(union Entry*, int pred_margin, <float|double>* result),
// get_num_feature, get_num_target, get_num_class, get_threshold_type, get_leaf_output_type.
// Entry holds the feature value as float or double, depending on get_threshold_type() (what Predictor does too).
//
// Windows only (it is a .dll). Elsewhere the constructor throws.
#pragma once
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "forest.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace core {

class TlModel {
public:
    TlModel(const std::string& path, int expected_features, bool classifier) : path_(path) {
#ifdef _WIN32
        std::wstring w;
        if (!path.empty()) {
            const int n = MultiByteToWideChar(CP_UTF8, 0, path.data(), int(path.size()), nullptr, 0);
            w.assign(size_t(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, path.data(), int(path.size()), &w[0], n);
        }
        mod_ = LoadLibraryW(w.c_str());
        if (!mod_)
            throw std::runtime_error("gaze model could not be loaded: " + path + " (Windows error " +
                                     std::to_string(GetLastError()) + "). The three .dll files must be in "
                                     "<ai-root>/gaze_intent/models (or pass --gaze-models DIR)");
        predict_ = reinterpret_cast<PredictFn>(GetProcAddress(mod_, "predict"));
        auto numFeature = reinterpret_cast<IntFn>(GetProcAddress(mod_, "get_num_feature"));
        auto numTarget  = reinterpret_cast<IntFn>(GetProcAddress(mod_, "get_num_target"));
        auto numClass   = reinterpret_cast<ClassFn>(GetProcAddress(mod_, "get_num_class"));
        auto thrType    = reinterpret_cast<StrFn>(GetProcAddress(mod_, "get_threshold_type"));
        auto leafType   = reinterpret_cast<StrFn>(GetProcAddress(mod_, "get_leaf_output_type"));
        if (!predict_ || !numFeature || !numTarget || !numClass || !thrType || !leafType)
            throw std::runtime_error("gaze model is not a TL2cgen model (missing exports): " + path);

        thr64_  = std::strcmp(thrType(), "float64") == 0;
        leaf64_ = std::strcmp(leafType(), "float64") == 0;
        n_feat_ = numFeature();
        if (n_feat_ != expected_features)
            throw std::runtime_error("gaze model " + path + " expects " + std::to_string(n_feat_) +
                                     " features, the engine sends " + std::to_string(expected_features));
        const int targets = numTarget();
        int32_t nc[16] = {};
        if (targets < 1 || targets > 16) throw std::runtime_error("unsupported gaze model (targets): " + path);
        numClass(nc);
        n_class_ = nc[0];
        int max_class = 1;
        for (int t = 0; t < targets; ++t) max_class = nc[t] > max_class ? nc[t] : max_class;
        if (classifier ? n_class_ < 2 : n_class_ != 1)
            throw std::runtime_error(std::string("gaze model is not a ") + (classifier ? "classifier: " : "regressor: ") + path);
        entry_.assign(size_t(n_feat_) * (thr64_ ? 8 : 4), 0);
        res_.assign(size_t(targets * max_class + 8) * 8, 0);
#else
        (void)expected_features; (void)classifier;
        throw std::runtime_error("the gaze models are Windows .dll files: " + path);
#endif
    }
    ~TlModel() {
#ifdef _WIN32
        if (mod_) FreeLibrary(mod_);
#endif
    }
    TlModel(const TlModel&) = delete;
    TlModel& operator=(const TlModel&) = delete;

    // Class with the highest score (first one wins a tie, like numpy.argmax).
    int classify(const float* f) { run(f); int best = 0; for (int k = 1; k < n_class_; ++k) if (value(k) > value(best)) best = k; return best; }
    double regress(const float* f) { run(f); return value(0); }
    const std::string& path() const { return path_; }

private:
#ifdef _WIN32
    using PredictFn = void (*)(void*, int, void*);
    using IntFn = int32_t (*)(void);
    using ClassFn = void (*)(int32_t*);
    using StrFn = const char* (*)(void);

    void run(const float* f) {
        for (int i = 0; i < n_feat_; ++i) {
            if (thr64_) { const double d = f[i]; std::memcpy(&entry_[size_t(i) * 8], &d, 8); }
            else        { std::memcpy(&entry_[size_t(i) * 4], &f[i], 4); }
        }
        std::fill(res_.begin(), res_.end(), 0);
        predict_(entry_.data(), 0, res_.data());
    }
    double value(int k) const {
        if (leaf64_) { double d; std::memcpy(&d, &res_[size_t(k) * 8], 8); return d; }
        float v; std::memcpy(&v, &res_[size_t(k) * 4], 4); return double(v);
    }
    HMODULE mod_ = nullptr;
    PredictFn predict_ = nullptr;
#else
    void run(const float*) {}
    double value(int) const { return 0.0; }
#endif
    std::string path_;
    bool thr64_ = false, leaf64_ = false;
    int n_feat_ = 0, n_class_ = 0;
    std::vector<unsigned char> entry_, res_;
};

class TlForest : public IForest {
public:
    explicit TlForest(const std::string& path) : m_(path, kGazeFeatures, true) {}
    int predict(const float* f) override { return m_.classify(f); }
    const char* name() const override { return "treelite .dll (gesture_model.dll)"; }
private:
    TlModel m_;
};

class TlRegressor : public IRegressor {
public:
    explicit TlRegressor(const std::string& path) : m_(path, kPoseFeatures, false) {}
    double predict(const float* f) override { return m_.regress(f); }
    const char* name() const override { return "treelite .dll"; }
private:
    TlModel m_;
};

} // namespace core
