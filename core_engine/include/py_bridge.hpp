// Pybind11 side of the bridge. Python is used ONLY for what has no C++ equivalent here:
//   * MediaPipe landmark extraction (PyLandmarker)
//   * optionally, scikit-learn as a fallback forest (PySklearnForest, PySklearnHandModel)
// Everything runs on the AI thread, which holds the GIL. Python errors never escape:
// they are counted and turned into "nothing detected".
#pragma once
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "forest.hpp"
#include "hand_math.hpp"

namespace py = pybind11;

namespace core {

inline void noopFree(void*) {}

// Zero-copy numpy view (H x W x 3, uint8) over a cv::Mat. Valid only while the Mat is alive:
// the Python side must not keep a reference to it.
inline py::array wrapMat(const cv::Mat& m) {
    std::vector<py::ssize_t> shape   { m.rows, m.cols, 3 };
    std::vector<py::ssize_t> strides { static_cast<py::ssize_t>(m.step[0]),
                                       static_cast<py::ssize_t>(m.step[1]), 1 };
    // Passing a base object tells numpy "someone else owns this memory": no copy, no free.
    return py::array(py::dtype::of<uint8_t>(), shape, strides, m.data, py::capsule(m.data, noopFree));
}

inline void prependSysPath(const std::string& dir) {
    py::list path = py::module_::import("sys").attr("path");
    path.insert(0, dir);
}

// Wraps a Python object with  process_frame(rgb_frame, timestamp_ms) -> (N,3) float32 landmark array
// (N == 0 or an empty array means "nothing detected"; the hand extractor returns N = 42: image + world landmarks).
class PyLandmarker {
public:
    void load(const char* module, const char* cls) {
        py::module_ m = py::module_::import(module);
        inst_ = m.attr(cls)();
        fn_ = inst_.attr("process_frame");
        loaded_ = true;
    }
    bool loaded() const { return loaded_; }

    // Returns the number of landmarks written to `out` (x,y,z triples), 0 = nothing detected, -1 = Python error.
    int call(const cv::Mat& rgb, int64_t ts_ms, std::vector<float>& out) {
        try {
            py::array arr = wrapMat(rgb);
            py::object r = fn_(arr, ts_ms);
            auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(r);
            if (!a) throw std::runtime_error("process_frame must return an (N,3) float array");
            if (a.size() == 0) { out.clear(); return 0; }
            if (a.ndim() != 2 || a.shape(1) != 3) throw std::runtime_error("landmark array must have shape (N,3)");
            const size_t n = static_cast<size_t>(a.shape(0));
            out.assign(a.data(), a.data() + n * 3);
            return static_cast<int>(n);
        } catch (const std::exception& e) {
            ++errors;
            last_error = e.what();
            out.clear();
            return -1;
        }
    }

    // Call with the GIL held, before the interpreter is finalised.
    void shutdown() {
        try { if (loaded_ && py::hasattr(inst_, "shutdown")) inst_.attr("shutdown")(); } catch (const std::exception&) {}
        fn_ = py::object();
        inst_ = py::object();
        loaded_ = false;
    }

    uint64_t errors = 0;
    std::string last_error;

private:
    py::object inst_, fn_;
    bool loaded_ = false;
};

// Fallback forest: scikit-learn through the embedded interpreter (gesture_model.pkl).
class PySklearnForest : public IForest {
public:
    void load(const std::string& pkl_path) {
        py::module_::import("warnings").attr("filterwarnings")(
            "ignore", py::arg("category") = py::module_::import("builtins").attr("UserWarning"));
        model_ = py::module_::import("joblib").attr("load")(pkl_path);
        predict_ = model_.attr("predict");
    }
    int predict(const float* f) override {
        try {
            py::array_t<double> a({static_cast<py::ssize_t>(1), static_cast<py::ssize_t>(kGazeFeatures)});
            double* p = a.mutable_data();
            for (int i = 0; i < kGazeFeatures; ++i) p[i] = f[i];
            py::object r = predict_(a);
            return py::int_(r.cast<py::sequence>()[0]).cast<int>();
        } catch (const std::exception& e) {
            ++errors_;
            last_error_ = e.what();
            return 0;
        }
    }
    const char* name() const override { return "scikit-learn (Python, slow fallback)"; }
    uint64_t errors() const override { return errors_; }
    void shutdown() { predict_ = py::object(); model_ = py::object(); }
    std::string last_error_;
private:
    py::object model_, predict_;
    uint64_t errors_ = 0;
};

// Hand gesture model through scikit-learn (hand_gesture_rf.pkl = {"model": RandomForest, "feature_names": [...]}).
// Used when the compiled forest is not available, or when --hand-model points to another .pkl (tests).
class PySklearnHandModel : public IHandModel {
public:
    void load(const std::string& pkl_path) {
        py::module_::import("warnings").attr("filterwarnings")(
            "ignore", py::arg("category") = py::module_::import("builtins").attr("UserWarning"));
        py::object obj = py::module_::import("joblib").attr("load")(pkl_path);
        py::object model = obj;
        if (py::isinstance<py::dict>(obj) && obj.contains("model")) {
            model = obj["model"];
            if (obj.contains("feature_names")) {
                py::list names = py::list(obj["feature_names"]);
                bool same = static_cast<int>(py::len(names)) == kHandFeatures;
                for (int i = 0; same && i < kHandFeatures; ++i)
                    same = names[i].cast<std::string>() == handFeatureNames()[i];
                if (!same)
                    throw std::runtime_error("the hand model was trained with different features than the engine "
                                             "computes (hand_features.FEATURE_NAMES changed?) - update hand_math.hpp");
            }
        }
        model.attr("n_jobs") = 1;                     // single-row inference: threads only add overhead
        model_ = model;
        proba_ = model_.attr("predict_proba");
        classes_.clear();
        for (py::handle c : model_.attr("classes_")) classes_.push_back(c.cast<int>());
        if (classes_.empty()) throw std::runtime_error("the hand model has no classes");
    }
    void classify(const double* f, int& cls, double& conf) override {
        try {
            py::array_t<double> a({static_cast<py::ssize_t>(1), static_cast<py::ssize_t>(kHandFeatures)});
            double* p = a.mutable_data();
            for (int i = 0; i < kHandFeatures; ++i) p[i] = f[i];
            py::array_t<double, py::array::c_style | py::array::forcecast> r =
                py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(proba_(a));
            if (!r || r.ndim() != 2 || r.shape(0) != 1 || r.shape(1) != static_cast<py::ssize_t>(classes_.size()))
                throw std::runtime_error("predict_proba returned an unexpected shape");
            int k = 0;
            for (py::ssize_t i = 1; i < r.shape(1); ++i) if (r.data()[i] > r.data()[k]) k = static_cast<int>(i);
            cls = classes_[static_cast<size_t>(k)];
            conf = r.data()[k];
        } catch (const std::exception& e) {
            ++errors_;
            last_error_ = e.what();
            cls = 0;
            conf = 0.0;
        }
    }
    const char* name() const override { return "scikit-learn (Python, slow fallback)"; }
    uint64_t errors() const override { return errors_; }
    void shutdown() { proba_ = py::object(); model_ = py::object(); }
    std::string last_error_;
private:
    py::object model_, proba_;
    std::vector<int> classes_;
    uint64_t errors_ = 0;
};

} // namespace core
