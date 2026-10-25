#include "hand_index_tracker.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef HAND_INDEX_TEST_SOURCE_DIR
#define HAND_INDEX_TEST_SOURCE_DIR "."
#endif
#ifndef HAND_INDEX_TEST_REAL_PYTHON_DIR
#define HAND_INDEX_TEST_REAL_PYTHON_DIR "."
#endif
#ifndef HAND_INDEX_TEST_REAL_MODEL_PATH
#define HAND_INDEX_TEST_REAL_MODEL_PATH ""
#endif
#ifndef HAND_INDEX_TEST_SITE_PACKAGES
#define HAND_INDEX_TEST_SITE_PACKAGES ""
#endif

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

bool nearly_equal(float left, float right) {
    return std::fabs(static_cast<double>(left - right)) < 1e-5;
}

std::vector<uint8_t> make_frame(
    int32_t width,
    int32_t height,
    int32_t stride,
    bool padded) {
    std::vector<uint8_t> frame(static_cast<size_t>(stride) * static_cast<size_t>(height), 0xEEu);
    for (int32_t row = 0; row < height; ++row) {
        for (int32_t column = 0; column < width; ++column) {
            uint8_t* pixel = frame.data() + static_cast<size_t>(row) * stride + column * 3;
            pixel[0] = static_cast<uint8_t>(row);
            pixel[1] = static_cast<uint8_t>(column);
            pixel[2] = 42;
        }
    }
    if (!padded) {
        for (int32_t row = 0; row < height; ++row) {
            std::memmove(
                frame.data() + static_cast<size_t>(row) * width * 3,
                frame.data() + static_cast<size_t>(row) * stride,
                static_cast<size_t>(width) * 3);
        }
        frame.resize(static_cast<size_t>(width) * 3 * static_cast<size_t>(height));
    }
    return frame;
}

void print_error(const char* operation, hand_index_error error, const char* message) {
    std::printf(
        "%s failed: %s (%d): %s\n",
        operation,
        hand_index_error_string(error),
        static_cast<int>(error),
        message[0] != '\0' ? message : "(no message)");
}

}  // namespace

int main(int argc, char** argv) {
    bool real_mode = false;
    std::string python_dir = std::string(HAND_INDEX_TEST_SOURCE_DIR) + "/tests/mock_python";
    std::string site_packages = HAND_INDEX_TEST_SITE_PACKAGES;
    std::string model_path;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--real") {
            real_mode = true;
            python_dir = HAND_INDEX_TEST_REAL_PYTHON_DIR;
            model_path = HAND_INDEX_TEST_REAL_MODEL_PATH;
        } else if (argument == "--mock") {
            real_mode = false;
            python_dir = std::string(HAND_INDEX_TEST_SOURCE_DIR) + "/tests/mock_python";
            model_path.clear();
        } else if (argument == "--python-dir" && index + 1 < argc) {
            python_dir = argv[++index];
        } else if (argument == "--site-packages" && index + 1 < argc) {
            site_packages = argv[++index];
        } else if (argument == "--model" && index + 1 < argc) {
            model_path = argv[++index];
        } else {
            std::printf("Unknown argument: %s\n", argument.c_str());
            return 2;
        }
    }

    std::printf("Hand index DLL smoke test (%s mode)\n", real_mode ? "real" : "mock");
    std::printf("version=%s abi=%u\n", hand_index_version(), hand_index_abi_version());
    check(std::strcmp(hand_index_version(), "1.0.0") == 0, "version string");
    check(hand_index_abi_version() == HAND_INDEX_ABI_VERSION, "ABI version");
    check(
        std::strcmp(hand_index_error_string(HAND_INDEX_ERROR_MODEL), "MODEL_ERROR") == 0,
        "error string");

    hand_index_config config;
    hand_index_config_default(&config);
    check(config.struct_size == sizeof(hand_index_config), "config struct size");
    check(config.abi_version == HAND_INDEX_ABI_VERSION, "config ABI version");
    check(nearly_equal(config.smoothing_alpha, 0.35f), "default smoothing alpha");
    check(nearly_equal(config.tracking_region[2], 1.0f), "default full region");

    config.python_dir = python_dir.c_str();
    config.site_packages_dir = site_packages.empty() ? nullptr : site_packages.c_str();
    config.model_path = model_path.empty() ? nullptr : model_path.c_str();
    config.tracking_region[0] = 0.2f;
    config.tracking_region[1] = 0.1f;
    config.tracking_region[2] = 0.8f;
    config.tracking_region[3] = 0.9f;
    config.smoothing_alpha = 0.6f;

    hand_index_config invalid_config = config;
    invalid_config.smoothing_alpha = 0.0f;
    hand_index_tracker* invalid_tracker = nullptr;
    char error[1024] = {0};
    const hand_index_error invalid_config_error =
        hand_index_create(&invalid_config, &invalid_tracker, error, sizeof(error));
    check(invalid_config_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject invalid config");
    check(invalid_tracker == nullptr, "no tracker on invalid config");

    hand_index_tracker* tracker = nullptr;
    error[0] = '\0';
    const hand_index_error create_error =
        hand_index_create(&config, &tracker, error, sizeof(error));
    if (create_error != HAND_INDEX_OK || tracker == nullptr) {
        print_error("hand_index_create", create_error, error);
        return 1;
    }
    check(true, "create tracker");

    const int32_t width = 16;
    const int32_t height = 8;
    const int32_t stride = width * 3 + 5;
    std::vector<uint8_t> padded = make_frame(width, height, stride, true);
    std::vector<uint8_t> packed = make_frame(width, height, width * 3, false);
    hand_index_result result;
    hand_index_result_init(&result);
    check(result.struct_size == sizeof(hand_index_result), "result struct size");
    check(result.valid == 0 && nearly_equal(result.x, -1.0f), "result init is lost");

    error[0] = '\0';
    hand_index_error process_error = hand_index_process_bgr(
        tracker, padded.data(), width, height, stride, &result, error, sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("first process", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }

    if (real_mode) {
        check(result.valid == 0, "real no-hand frame is lost");
        check(result.status == HAND_INDEX_STATUS_LOST, "real lost status");
        check(
            nearly_equal(result.x, -1.0f) && nearly_equal(result.y, -1.0f) &&
                nearly_equal(result.camera_x, -1.0f) && nearly_equal(result.camera_y, -1.0f),
            "real lost coordinates are -1");
    } else {
        check(result.valid == 1, "mock valid result");
        check(result.status == HAND_INDEX_STATUS_TRACKED, "mock tracked status");
        check(nearly_equal(result.x, 0.2f), "region left passed to Python");
        check(nearly_equal(result.y, 0.9f), "region bottom passed to Python");
        check(nearly_equal(result.camera_x, 0.6f), "smoothing alpha passed to Python");
        check(nearly_equal(result.camera_y, 0.5f), "camera y returned");
        check(result.landmark_index == HAND_INDEX_LANDMARK_INDEX, "landmark index 8");
        check(
            nearly_equal(result.x, 0.2f) && nearly_equal(result.y, 0.9f),
            "mapped coordinates returned unchanged");
    }

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, padded.data(), width, height, stride, &result, error, sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("second process", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }

    if (real_mode) {
        check(result.valid == 0, "real second frame is lost");
    } else {
        check(result.valid == 0, "mock lost result");
        check(result.status == HAND_INDEX_STATUS_LOST, "mock lost status");
        check(
            nearly_equal(result.x, -1.0f) && nearly_equal(result.y, -1.0f) &&
                nearly_equal(result.camera_x, -1.0f) && nearly_equal(result.camera_y, -1.0f),
            "lost result has no stale coordinates");
    }

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, packed.data(), width, height, 0, &result, error, sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("packed process", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }
    if (real_mode) {
        check(result.valid == 0, "real packed frame is lost");
    } else {
        check(result.valid == 1, "packed frame accepted with stride 0");
        check(nearly_equal(result.x, 0.2f), "third call is valid again");
    }

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, nullptr, width, height, stride, &result, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject null frame");
    check(result.valid == 0, "null frame leaves lost result");

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, padded.data(), 0, height, stride, &result, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject zero width");

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, padded.data(), width, height, width * 3 - 1, &result, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject short stride");

    hand_index_result_init(&result);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        nullptr, padded.data(), width, height, stride, &result, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject null tracker");

    result.struct_size = 0;
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, padded.data(), width, height, stride, &result, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject old result ABI");

    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker, padded.data(), width, height, stride, nullptr, error, sizeof(error));
    check(process_error == HAND_INDEX_ERROR_INVALID_ARGUMENT, "reject null result");

    hand_index_destroy(tracker);
    hand_index_destroy(nullptr);
    check(true, "destroy tracker");

    if (failures != 0) {
        std::printf("%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("All DLL interface checks passed\n");
    return 0;
}
