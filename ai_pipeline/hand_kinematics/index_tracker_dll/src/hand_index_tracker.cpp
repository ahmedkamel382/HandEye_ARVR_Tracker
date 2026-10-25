#ifndef HAND_INDEX_TRACKER_BUILD
#define HAND_INDEX_TRACKER_BUILD
#endif
#include "hand_index_tracker.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>

namespace {

constexpr const char* kVersion = "1.0.0";

std::mutex g_runtime_mutex;

struct GilGuard {
    GilGuard() : state(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state); }

    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;

    PyGILState_STATE state;
};

struct PyRef {
    explicit PyRef(PyObject* object = nullptr) : object(object) {}
    ~PyRef() { Py_XDECREF(object); }

    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;

    PyRef(PyRef&& other) noexcept : object(other.release()) {}
    PyRef& operator=(PyRef&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(object);
            object = other.release();
        }
        return *this;
    }

    PyObject* get() const { return object; }
    PyObject* release() {
        PyObject* result = object;
        object = nullptr;
        return result;
    }

    PyObject* object;
};

void write_error(char* buffer, size_t capacity, const std::string& message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    const size_t count = (message.size() < capacity - 1) ? message.size() : capacity - 1;
    if (count > 0) {
        std::memcpy(buffer, message.data(), count);
    }
    buffer[count] = '\0';
}

void mark_lost(hand_index_result* result) {
    result->valid = 0;
    result->status = HAND_INDEX_STATUS_LOST;
    result->landmark_index = HAND_INDEX_LANDMARK_INDEX;
    result->x = -1.0f;
    result->y = -1.0f;
    result->camera_x = -1.0f;
    result->camera_y = -1.0f;
}

std::string consume_python_exception(bool* is_model_error) {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);

    if (is_model_error != nullptr) {
        *is_model_error = (type != nullptr) &&
            (PyErr_GivenExceptionMatches(type, PyExc_FileNotFoundError) != 0);
    }

    PyErr_NormalizeException(&type, &value, &traceback);
    PyRef type_ref(type);
    PyRef value_ref(value);
    PyRef traceback_ref(traceback);

    std::string message;
    if (type_ref.get() != nullptr) {
        PyRef type_name(PyObject_GetAttrString(type_ref.get(), "__name__"));
        const char* type_text = nullptr;
        if (type_name.get() != nullptr) {
            type_text = PyUnicode_AsUTF8(type_name.get());
        }
        if (type_text != nullptr) {
            message = type_text;
        }
    }
    if (message.empty()) {
        message = "PythonError";
    }

    if (value_ref.get() != nullptr) {
        PyRef value_text(PyObject_Str(value_ref.get()));
        const char* text = nullptr;
        if (value_text.get() != nullptr) {
            text = PyUnicode_AsUTF8(value_text.get());
        }
        if (text != nullptr && text[0] != '\0') {
            message += ": ";
            message += text;
        }
    }

    PyErr_Clear();
    return message;
}

bool ensure_python(std::string& error) {
    if (Py_IsInitialized()) {
        return true;
    }

    Py_InitializeEx(0);
    if (!Py_IsInitialized()) {
        error = "Py_InitializeEx failed";
        return false;
    }

    PyEval_SaveThread();
    return true;
}

bool add_sys_path(const char* path, std::string& error) {
    if (path == nullptr || path[0] == '\0') {
        return true;
    }

    PyObject* sys_path = PySys_GetObject("path");
    if (sys_path == nullptr || !PyList_Check(sys_path)) {
        error = "Python sys.path is unavailable";
        return false;
    }

    PyRef path_object(PyUnicode_DecodeFSDefault(path));
    if (path_object.get() == nullptr) {
        bool ignored_model_error = false;
        error = consume_python_exception(&ignored_model_error);
        return false;
    }

    const int contains = PySequence_Contains(sys_path, path_object.get());
    if (contains < 0) {
        bool ignored_model_error = false;
        error = consume_python_exception(&ignored_model_error);
        return false;
    }
    if (contains == 0 && PyList_Insert(sys_path, 0, path_object.get()) < 0) {
        bool ignored_model_error = false;
        error = consume_python_exception(&ignored_model_error);
        return false;
    }
    return true;
}

bool set_float_kwarg(PyObject* kwargs, const char* name, double value, std::string& error) {
    PyRef object(PyFloat_FromDouble(value));
    if (object.get() == nullptr) {
        bool ignored_model_error = false;
        error = consume_python_exception(&ignored_model_error);
        return false;
    }
    if (PyDict_SetItemString(kwargs, name, object.get()) < 0) {
        bool ignored_model_error = false;
        error = consume_python_exception(&ignored_model_error);
        return false;
    }
    return true;
}

bool is_finite(float value) {
    return std::isfinite(static_cast<double>(value)) != 0;
}

bool validate_config(const hand_index_config& config, std::string& error) {
    if (config.abi_version != HAND_INDEX_ABI_VERSION) {
        error = "Unsupported hand_index_config ABI version";
        return false;
    }

    const float left = config.tracking_region[0];
    const float top = config.tracking_region[1];
    const float right = config.tracking_region[2];
    const float bottom = config.tracking_region[3];
    if (!is_finite(left) || !is_finite(top) || !is_finite(right) || !is_finite(bottom) ||
        !(0.0f <= left && left < right && right <= 1.0f) ||
        !(0.0f <= top && top < bottom && bottom <= 1.0f)) {
        error = "tracking_region must be (left, top, right, bottom) with 0 <= left < right <= 1 and 0 <= top < bottom <= 1";
        return false;
    }

    if (!is_finite(config.smoothing_alpha) ||
        !(config.smoothing_alpha > 0.0f && config.smoothing_alpha <= 1.0f)) {
        error = "smoothing_alpha must satisfy 0 < alpha <= 1";
        return false;
    }

    const float confidences[] = {
        config.min_hand_detection_confidence,
        config.min_hand_presence_confidence,
        config.min_tracking_confidence,
    };
    for (float confidence : confidences) {
        if (!is_finite(confidence) || confidence < 0.0f || confidence > 1.0f) {
            error = "confidence thresholds must be in [0, 1]";
            return false;
        }
    }
    return true;
}

bool read_float_attribute(PyObject* object, const char* name, float& value, std::string& error) {
    PyRef attribute(PyObject_GetAttrString(object, name));
    if (attribute.get() == nullptr) {
        error = consume_python_exception(nullptr);
        return false;
    }
    const double converted = PyFloat_AsDouble(attribute.get());
    if (converted == -1.0 && PyErr_Occurred()) {
        error = consume_python_exception(nullptr);
        return false;
    }
    if (!std::isfinite(converted)) {
        error = std::string("Python returned a non-finite value for ") + name;
        return false;
    }
    value = static_cast<float>(converted);
    return true;
}

bool parse_result(PyObject* object, hand_index_result* result, std::string& error) {
    PyRef valid_attribute(PyObject_GetAttrString(object, "valid"));
    if (valid_attribute.get() == nullptr) {
        error = consume_python_exception(nullptr);
        return false;
    }
    const int valid = PyObject_IsTrue(valid_attribute.get());
    if (valid < 0) {
        error = consume_python_exception(nullptr);
        return false;
    }

    PyRef landmark_attribute(PyObject_GetAttrString(object, "landmark_index"));
    if (landmark_attribute.get() == nullptr) {
        error = consume_python_exception(nullptr);
        return false;
    }
    const long landmark = PyLong_AsLong(landmark_attribute.get());
    if (landmark == -1 && PyErr_Occurred()) {
        error = consume_python_exception(nullptr);
        return false;
    }

    if (valid == 0) {
        return true;
    }

    float x = 0.0f;
    float y = 0.0f;
    float camera_x = 0.0f;
    float camera_y = 0.0f;
    if (!read_float_attribute(object, "x", x, error) ||
        !read_float_attribute(object, "y", y, error) ||
        !read_float_attribute(object, "camera_x", camera_x, error) ||
        !read_float_attribute(object, "camera_y", camera_y, error)) {
        return false;
    }

    result->valid = 1;
    result->status = HAND_INDEX_STATUS_TRACKED;
    result->landmark_index = static_cast<int32_t>(landmark);
    result->x = x;
    result->y = y;
    result->camera_x = camera_x;
    result->camera_y = camera_y;
    return true;
}

PyObject* make_frame(
    PyObject* numpy_module,
    PyObject* numpy_uint8,
    const uint8_t* data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    std::string& error) {
    const size_t row_bytes = static_cast<size_t>(width) * 3u;
    const size_t total_bytes = row_bytes * static_cast<size_t>(height);
    if (total_bytes > static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max())) {
        error = "Frame is too large for Python";
        return nullptr;
    }

    PyObject* buffer_raw =
        PyByteArray_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total_bytes));
    if (buffer_raw == nullptr) {
        error = consume_python_exception(nullptr);
        return nullptr;
    }
    PyRef buffer(buffer_raw);
    char* destination = PyByteArray_AS_STRING(buffer.get());
    for (int32_t row = 0; row < height; ++row) {
        std::memcpy(
            destination + static_cast<size_t>(row) * row_bytes,
            data + static_cast<size_t>(row) * static_cast<size_t>(stride_bytes),
            row_bytes);
    }

    PyRef flat(PyObject_CallMethod(
        numpy_module, "frombuffer", "OO", buffer.get(), numpy_uint8));
    if (flat.get() == nullptr) {
        error = consume_python_exception(nullptr);
        return nullptr;
    }

    PyRef frame(PyObject_CallMethod(
        flat.get(), "reshape", "iii", height, width, 3));
    if (frame.get() == nullptr) {
        error = consume_python_exception(nullptr);
        return nullptr;
    }
    return frame.release();
}

}  // namespace

struct hand_index_tracker {
    PyObject* py_tracker = nullptr;
    PyObject* numpy_module = nullptr;
    PyObject* numpy_uint8 = nullptr;
    std::mutex mutex;
};

extern "C" {

const char* HAND_INDEX_CALL hand_index_version(void) {
    return kVersion;
}

uint32_t HAND_INDEX_CALL hand_index_abi_version(void) {
    return HAND_INDEX_ABI_VERSION;
}

const char* HAND_INDEX_CALL hand_index_error_string(hand_index_error error) {
    switch (error) {
        case HAND_INDEX_OK:
            return "OK";
        case HAND_INDEX_ERROR_INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case HAND_INDEX_ERROR_PYTHON:
            return "PYTHON_ERROR";
        case HAND_INDEX_ERROR_MODEL:
            return "MODEL_ERROR";
        case HAND_INDEX_ERROR_CLOSED:
            return "CLOSED";
        case HAND_INDEX_ERROR_INTERNAL:
            return "INTERNAL_ERROR";
        default:
            return "UNKNOWN_ERROR";
    }
}

void HAND_INDEX_CALL hand_index_config_default(hand_index_config* config) {
    if (config == nullptr) {
        return;
    }
    std::memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(hand_index_config);
    config->abi_version = HAND_INDEX_ABI_VERSION;
    config->tracking_region[0] = 0.0f;
    config->tracking_region[1] = 0.0f;
    config->tracking_region[2] = 1.0f;
    config->tracking_region[3] = 1.0f;
    config->smoothing_alpha = 0.35f;
    config->min_hand_detection_confidence = 0.3f;
    config->min_hand_presence_confidence = 0.3f;
    config->min_tracking_confidence = 0.3f;
}

void HAND_INDEX_CALL hand_index_result_init(hand_index_result* result) {
    if (result == nullptr) {
        return;
    }
    std::memset(result, 0, sizeof(*result));
    result->struct_size = sizeof(hand_index_result);
    mark_lost(result);
}

hand_index_error HAND_INDEX_CALL hand_index_create(
    const hand_index_config* supplied_config,
    hand_index_tracker** out_tracker,
    char* error_message,
    size_t error_message_capacity) {
    if (error_message != nullptr && error_message_capacity > 0) {
        error_message[0] = '\0';
    }
    if (out_tracker == nullptr) {
        write_error(error_message, error_message_capacity, "out_tracker must not be null");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }
    *out_tracker = nullptr;

    hand_index_config default_config;
    hand_index_config_default(&default_config);
    const hand_index_config& config = (supplied_config != nullptr) ? *supplied_config : default_config;

    if (config.struct_size < sizeof(hand_index_config)) {
        write_error(
            error_message,
            error_message_capacity,
            "hand_index_config.struct_size is smaller than the current ABI");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }

    std::string error;
    if (!validate_config(config, error)) {
        write_error(error_message, error_message_capacity, error);
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> runtime_lock(g_runtime_mutex);
    if (!ensure_python(error)) {
        write_error(error_message, error_message_capacity, error);
        return HAND_INDEX_ERROR_PYTHON;
    }

    {
        GilGuard gil;
        PyErr_Clear();

        if (!add_sys_path(config.site_packages_dir, error) ||
            !add_sys_path(config.python_dir, error)) {
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef numpy_module(PyImport_ImportModule("numpy"));
        if (numpy_module.get() == nullptr) {
            bool model_error = false;
            error = consume_python_exception(&model_error);
            write_error(error_message, error_message_capacity, error);
            return model_error ? HAND_INDEX_ERROR_MODEL : HAND_INDEX_ERROR_PYTHON;
        }

        PyRef numpy_uint8(PyObject_GetAttrString(numpy_module.get(), "uint8"));
        if (numpy_uint8.get() == nullptr) {
            error = consume_python_exception(nullptr);
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef module(PyImport_ImportModule("index_tracker"));
        if (module.get() == nullptr) {
            bool model_error = false;
            error = consume_python_exception(&model_error);
            write_error(error_message, error_message_capacity, error);
            return model_error ? HAND_INDEX_ERROR_MODEL : HAND_INDEX_ERROR_PYTHON;
        }

        PyRef tracker_class(PyObject_GetAttrString(module.get(), "IndexFingerTracker"));
        if (tracker_class.get() == nullptr) {
            error = consume_python_exception(nullptr);
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef args(PyTuple_New(0));
        PyRef kwargs(PyDict_New());
        PyRef region(Py_BuildValue(
            "(ffff)",
            config.tracking_region[0],
            config.tracking_region[1],
            config.tracking_region[2],
            config.tracking_region[3]));
        if (args.get() == nullptr || kwargs.get() == nullptr || region.get() == nullptr) {
            error = consume_python_exception(nullptr);
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef model_object;
        PyObject* model_value = Py_None;
        if (config.model_path != nullptr && config.model_path[0] != '\0') {
            model_object = PyRef(PyUnicode_DecodeFSDefault(config.model_path));
            if (model_object.get() == nullptr) {
                error = consume_python_exception(nullptr);
                write_error(error_message, error_message_capacity, error);
                return HAND_INDEX_ERROR_PYTHON;
            }
            model_value = model_object.get();
        }

        if (PyDict_SetItemString(kwargs.get(), "model_path", model_value) < 0 ||
            PyDict_SetItemString(kwargs.get(), "tracking_region", region.get()) < 0 ||
            !set_float_kwarg(kwargs.get(), "smoothing_alpha", config.smoothing_alpha, error) ||
            !set_float_kwarg(
                kwargs.get(),
                "min_hand_detection_confidence",
                config.min_hand_detection_confidence,
                error) ||
            !set_float_kwarg(
                kwargs.get(),
                "min_hand_presence_confidence",
                config.min_hand_presence_confidence,
                error) ||
            !set_float_kwarg(
                kwargs.get(),
                "min_tracking_confidence",
                config.min_tracking_confidence,
                error)) {
            if (error.empty()) {
                error = consume_python_exception(nullptr);
            }
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef tracker_object(PyObject_Call(tracker_class.get(), args.get(), kwargs.get()));
        if (tracker_object.get() == nullptr) {
            bool model_error = false;
            error = consume_python_exception(&model_error);
            write_error(error_message, error_message_capacity, error);
            return model_error ? HAND_INDEX_ERROR_MODEL : HAND_INDEX_ERROR_PYTHON;
        }

        hand_index_tracker* tracker = new (std::nothrow) hand_index_tracker();
        if (tracker == nullptr) {
            write_error(error_message, error_message_capacity, "Out of memory");
            return HAND_INDEX_ERROR_INTERNAL;
        }

        tracker->py_tracker = tracker_object.release();
        tracker->numpy_module = numpy_module.release();
        tracker->numpy_uint8 = numpy_uint8.release();
        *out_tracker = tracker;
    }

    return HAND_INDEX_OK;
}

hand_index_error HAND_INDEX_CALL hand_index_process_bgr(
    hand_index_tracker* tracker,
    const uint8_t* bgr_data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    hand_index_result* out_result,
    char* error_message,
    size_t error_message_capacity) {
    if (error_message != nullptr && error_message_capacity > 0) {
        error_message[0] = '\0';
    }
    if (out_result == nullptr) {
        write_error(error_message, error_message_capacity, "out_result must not be null");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }
    if (out_result->struct_size < sizeof(hand_index_result)) {
        write_error(
            error_message,
            error_message_capacity,
            "hand_index_result.struct_size is smaller than the current ABI");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }
    hand_index_result_init(out_result);

    if (tracker == nullptr) {
        write_error(error_message, error_message_capacity, "tracker must not be null");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }
    if (bgr_data == nullptr) {
        write_error(error_message, error_message_capacity, "bgr_data must not be null");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }
    if (width <= 0 || height <= 0) {
        write_error(error_message, error_message_capacity, "width and height must be positive");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }

    const int64_t minimum_stride = static_cast<int64_t>(width) * 3;
    int64_t effective_stride = (stride_bytes == 0) ? minimum_stride : stride_bytes;
    if (effective_stride < minimum_stride ||
        effective_stride > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        write_error(
            error_message,
            error_message_capacity,
            "stride_bytes must be 0 or at least width * 3");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }

    const uint64_t row_bytes = static_cast<uint64_t>(width) * 3u;
    const uint64_t total_bytes = row_bytes * static_cast<uint64_t>(height);
    if (total_bytes > static_cast<uint64_t>(std::numeric_limits<Py_ssize_t>::max())) {
        write_error(error_message, error_message_capacity, "Frame is too large for Python");
        return HAND_INDEX_ERROR_INVALID_ARGUMENT;
    }

    if (!Py_IsInitialized()) {
        write_error(error_message, error_message_capacity, "Python runtime is not initialized");
        return HAND_INDEX_ERROR_CLOSED;
    }

    std::lock_guard<std::mutex> tracker_lock(tracker->mutex);
    {
        GilGuard gil;
        PyErr_Clear();

        if (tracker->py_tracker == nullptr) {
            write_error(error_message, error_message_capacity, "Tracker is closed");
            return HAND_INDEX_ERROR_CLOSED;
        }

        std::string error;
        PyRef frame(make_frame(
            tracker->numpy_module,
            tracker->numpy_uint8,
            bgr_data,
            width,
            height,
            static_cast<int32_t>(effective_stride),
            error));
        if (frame.get() == nullptr) {
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef track_method(PyObject_GetAttrString(tracker->py_tracker, "track"));
        if (track_method.get() == nullptr) {
            error = consume_python_exception(nullptr);
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }

        PyRef tracking_result(PyObject_CallOneArg(track_method.get(), frame.get()));
        if (tracking_result.get() == nullptr) {
            bool model_error = false;
            error = consume_python_exception(&model_error);
            write_error(error_message, error_message_capacity, error);
            return model_error ? HAND_INDEX_ERROR_MODEL : HAND_INDEX_ERROR_PYTHON;
        }

        if (!parse_result(tracking_result.get(), out_result, error)) {
            mark_lost(out_result);
            write_error(error_message, error_message_capacity, error);
            return HAND_INDEX_ERROR_PYTHON;
        }
    }

    return HAND_INDEX_OK;
}

void HAND_INDEX_CALL hand_index_destroy(hand_index_tracker* tracker) {
    if (tracker == nullptr) {
        return;
    }
    if (!Py_IsInitialized()) {
        delete tracker;
        return;
    }

    {
        std::lock_guard<std::mutex> tracker_lock(tracker->mutex);
        GilGuard gil;
        if (tracker->py_tracker != nullptr) {
            PyRef close_method(PyObject_GetAttrString(tracker->py_tracker, "close"));
            if (close_method.get() != nullptr) {
                PyRef close_result(PyObject_CallNoArgs(close_method.get()));
                if (close_result.get() == nullptr) {
                    PyErr_Clear();
                }
            } else {
                PyErr_Clear();
            }
        }
        Py_XDECREF(tracker->py_tracker);
        Py_XDECREF(tracker->numpy_module);
        Py_XDECREF(tracker->numpy_uint8);
        tracker->py_tracker = nullptr;
        tracker->numpy_module = nullptr;
        tracker->numpy_uint8 = nullptr;
    }

    delete tracker;
}

}  // extern "C"
