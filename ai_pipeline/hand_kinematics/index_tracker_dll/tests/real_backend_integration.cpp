#include "hand_index_tracker.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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

using Microsoft::WRL::ComPtr;

int failures = 0;

struct Image {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> bgr;
};

struct ComScope {
    HRESULT result = E_FAIL;

    ComScope() {
        result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    }

    ~ComScope() {
        if (SUCCEEDED(result)) {
            CoUninitialize();
        }
    }
};

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

bool finite_in_unit_interval(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

bool nearly_equal(float left, float right, float tolerance = 1e-4f) {
    return std::fabs(left - right) <= tolerance;
}

bool nearly_equal_xy(
    const hand_index_result& left,
    const hand_index_result& right,
    float tolerance) {
    return nearly_equal(left.x, right.x, tolerance) &&
           nearly_equal(left.y, right.y, tolerance);
}

std::string wide_to_utf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (required <= 0) {
        return {};
    }
    std::string result(static_cast<size_t>(required), '\0');
    if (WideCharToMultiByte(
            CP_UTF8, 0, value.c_str(), -1, &result[0], required, nullptr, nullptr) <= 0) {
        return {};
    }
    result.resize(static_cast<size_t>(required - 1));
    return result;
}

std::string hresult_text(const char* operation, HRESULT result) {
    char buffer[256];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%s failed (HRESULT 0x%08lX)",
        operation,
        static_cast<unsigned long>(result));
    return buffer;
}

bool load_image_bgr(const std::wstring& path, Image& image, std::string& error) {
    ComScope com;
    if (FAILED(com.result)) {
        error = hresult_text("CoInitializeEx", com.result);
        return false;
    }

    ComPtr<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(factory.GetAddressOf()));
    if (FAILED(result)) {
        error = hresult_text("CoCreateInstance(WIC)", result);
        return false;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    result = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnDemand,
        decoder.GetAddressOf());
    if (FAILED(result)) {
        error = hresult_text("CreateDecoderFromFilename", result);
        return false;
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    result = decoder->GetFrame(0, frame.GetAddressOf());
    if (FAILED(result)) {
        error = hresult_text("GetFrame", result);
        return false;
    }

    UINT width = 0;
    UINT height = 0;
    result = frame->GetSize(&width, &height);
    if (FAILED(result) || width == 0 || height == 0) {
        error = "image has invalid dimensions";
        return false;
    }
    if (width > static_cast<UINT>(std::numeric_limits<int32_t>::max() / 3) ||
        height > static_cast<UINT>(std::numeric_limits<int32_t>::max())) {
        error = "image dimensions exceed the DLL ABI limits";
        return false;
    }

    ComPtr<IWICFormatConverter> converter;
    result = factory->CreateFormatConverter(converter.GetAddressOf());
    if (FAILED(result)) {
        error = hresult_text("CreateFormatConverter", result);
        return false;
    }

    result = converter->Initialize(
        frame.Get(),
        GUID_WICPixelFormat24bppBGR,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);
    if (FAILED(result)) {
        error = hresult_text("IWICFormatConverter::Initialize", result);
        return false;
    }

    const uint64_t stride64 = static_cast<uint64_t>(width) * 3u;
    const uint64_t size64 = stride64 * static_cast<uint64_t>(height);
    if (stride64 > std::numeric_limits<UINT>::max() ||
        size64 > std::numeric_limits<UINT>::max() ||
        size64 > std::numeric_limits<size_t>::max()) {
        error = "image byte size exceeds the decoder limits";
        return false;
    }

    image.width = static_cast<int32_t>(width);
    image.height = static_cast<int32_t>(height);
    image.bgr.resize(static_cast<size_t>(size64));

    result = converter->CopyPixels(
        nullptr,
        static_cast<UINT>(stride64),
        static_cast<UINT>(size64),
        image.bgr.data());
    if (FAILED(result)) {
        error = hresult_text("IWICFormatConverter::CopyPixels", result);
        image = {};
        return false;
    }
    return true;
}

std::vector<uint8_t> make_padded_frame(const Image& image, int32_t padding) {
    const int32_t stride = image.width * 3 + padding;
    std::vector<uint8_t> padded(
        static_cast<size_t>(stride) * static_cast<size_t>(image.height),
        0x7Fu);
    for (int32_t row = 0; row < image.height; ++row) {
        std::memcpy(
            padded.data() + static_cast<size_t>(row) * static_cast<size_t>(stride),
            image.bgr.data() + static_cast<size_t>(row) * static_cast<size_t>(image.width) * 3,
            static_cast<size_t>(image.width) * 3);
    }
    return padded;
}

void print_result(const char* label, const hand_index_result& result) {
    std::printf(
        "%s: valid=%d status=%d landmark=%d x=%.6f y=%.6f camera=(%.6f, %.6f)\n",
        label,
        result.valid,
        result.status,
        result.landmark_index,
        result.x,
        result.y,
        result.camera_x,
        result.camera_y);
}

void print_error(const char* operation, hand_index_error error, const char* message) {
    std::printf(
        "%s failed: %s (%d): %s\n",
        operation,
        hand_index_error_string(error),
        static_cast<int>(error),
        message[0] != '\0' ? message : "(no message)");
}

void print_usage() {
    std::printf(
        "Usage: hand_index_real_backend_integration <image> [options]\n"
        "  --python-dir <dir>       Directory containing index_tracker.py\n"
        "  --site-packages <dir>    Python site-packages directory\n"
        "  --model <file>           hand_landmarker.task path\n"
        "  --help                   Show this help\n"
        "\n"
        "The image must contain a visible hand. The test uses the actual embedded\n"
        "Python/MediaPipe backend and never falls back to the mock tracker.\n");
}

void check_tracked_result(
    const hand_index_result& result,
    const char* prefix) {
    check(result.valid == 1, prefix);
    check(result.status == HAND_INDEX_STATUS_TRACKED, "status is TRACKED");
    check(result.landmark_index == HAND_INDEX_LANDMARK_INDEX, "landmark index is 8");
    check(finite_in_unit_interval(result.x), "x is finite in [0,1]");
    check(finite_in_unit_interval(result.y), "y is finite in [0,1]");
    check(finite_in_unit_interval(result.camera_x), "camera_x is finite in [0,1]");
    check(finite_in_unit_interval(result.camera_y), "camera_y is finite in [0,1]");
    check(nearly_equal(result.x, result.camera_x), "x equals camera_x at full region/alpha=1");
    check(nearly_equal(result.y, result.camera_y), "y equals camera_y at full region/alpha=1");
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    std::wstring image_path;
    std::wstring python_dir = L"" HAND_INDEX_TEST_REAL_PYTHON_DIR;
    std::wstring site_packages = L"" HAND_INDEX_TEST_SITE_PACKAGES;
    std::wstring model_path = L"" HAND_INDEX_TEST_REAL_MODEL_PATH;

    for (int index = 1; index < argc; ++index) {
        const std::wstring argument = argv[index];
        if (argument == L"--help" || argument == L"-h") {
            print_usage();
            return 0;
        }
        if (argument == L"--python-dir" && index + 1 < argc) {
            python_dir = argv[++index];
        } else if (argument == L"--site-packages" && index + 1 < argc) {
            site_packages = argv[++index];
        } else if (argument == L"--model" && index + 1 < argc) {
            model_path = argv[++index];
        } else if (!argument.empty() && argument[0] == L'-') {
            std::printf("Unknown argument: %ls\n", argument.c_str());
            return 2;
        } else if (image_path.empty()) {
            image_path = argument;
        } else {
            std::printf("Unexpected extra argument: %ls\n", argument.c_str());
            return 2;
        }
    }

    if (image_path.empty()) {
        print_usage();
        std::printf("ERROR: no hand image supplied\n");
        return 2;
    }
    if (model_path.empty()) {
        std::printf("ERROR: no model path was compiled or supplied\n");
        return 2;
    }

    std::printf("Hand index real-backend integration test\n");
    std::printf("DLL version=%s abi=%u\n", hand_index_version(), hand_index_abi_version());
    check(std::strcmp(hand_index_version(), "1.0.0") == 0, "version string");
    check(hand_index_abi_version() == HAND_INDEX_ABI_VERSION, "ABI version");

    Image image;
    std::string load_error;
    if (!load_image_bgr(image_path, image, load_error)) {
        std::printf("ERROR: cannot load image: %s\n", load_error.c_str());
        return 1;
    }
    std::printf("Image: %dx%d\n", image.width, image.height);

    const std::string python_dir_utf8 = wide_to_utf8(python_dir);
    const std::string site_packages_utf8 = wide_to_utf8(site_packages);
    const std::string model_path_utf8 = wide_to_utf8(model_path);
    if (python_dir_utf8.empty() || model_path_utf8.empty()) {
        std::printf("ERROR: Python directory or model path is not valid UTF-8\n");
        return 2;
    }

    hand_index_config config;
    hand_index_config_default(&config);
    config.python_dir = python_dir_utf8.c_str();
    config.site_packages_dir =
        site_packages_utf8.empty() ? nullptr : site_packages_utf8.c_str();
    config.model_path = model_path_utf8.c_str();
    config.smoothing_alpha = 1.0f;

    char error[1024] = {0};
    hand_index_tracker* tracker = nullptr;
    const hand_index_error create_error =
        hand_index_create(&config, &tracker, error, sizeof(error));
    if (create_error != HAND_INDEX_OK || tracker == nullptr) {
        print_error("hand_index_create", create_error, error);
        return 1;
    }
    check(true, "create real tracker");

    const std::vector<uint8_t> padded = make_padded_frame(image, 7);
    const int32_t padded_stride = image.width * 3 + 7;
    hand_index_result first;
    hand_index_result_init(&first);
    error[0] = '\0';
    hand_index_error process_error = hand_index_process_bgr(
        tracker,
        image.bgr.data(),
        image.width,
        image.height,
        0,
        &first,
        error,
        sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("first hand image", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }
    print_result("first hand image", first);
    check_tracked_result(first, "hand image produces a valid result");

    hand_index_result second;
    hand_index_result_init(&second);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker,
        padded.data(),
        image.width,
        image.height,
        padded_stride,
        &second,
        error,
        sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("padded hand image", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }
    print_result("padded hand image", second);
    check_tracked_result(second, "padded hand image produces a valid result");
    check(
        nearly_equal_xy(first, second, 0.03f),
        "padded frame agrees with packed frame");

    std::vector<uint8_t> blank(
        static_cast<size_t>(image.width) * 3u * static_cast<size_t>(image.height),
        0);
    hand_index_result lost;
    hand_index_result_init(&lost);
    lost.x = 1234.0f;
    lost.y = 5678.0f;
    lost.camera_x = 9012.0f;
    lost.camera_y = 3456.0f;
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker,
        blank.data(),
        image.width,
        image.height,
        0,
        &lost,
        error,
        sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("blank frame", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }
    print_result("blank frame", lost);
    check(lost.valid == 0, "blank frame is lost");
    check(lost.status == HAND_INDEX_STATUS_LOST, "blank frame status is LOST");
    check(
        nearly_equal(lost.x, -1.0f) && nearly_equal(lost.y, -1.0f) &&
            nearly_equal(lost.camera_x, -1.0f) && nearly_equal(lost.camera_y, -1.0f),
        "loss clears all coordinates");

    hand_index_result reacquired;
    hand_index_result_init(&reacquired);
    error[0] = '\0';
    process_error = hand_index_process_bgr(
        tracker,
        image.bgr.data(),
        image.width,
        image.height,
        0,
        &reacquired,
        error,
        sizeof(error));
    if (process_error != HAND_INDEX_OK) {
        print_error("reacquired hand image", process_error, error);
        hand_index_destroy(tracker);
        return 1;
    }
    print_result("reacquired hand image", reacquired);
    check_tracked_result(reacquired, "hand is reacquired after loss");
    check(
        nearly_equal_xy(first, reacquired, 0.03f),
        "reacquired coordinates agree with the first detection");

    hand_index_destroy(tracker);
    check(true, "destroy real tracker");

    if (failures != 0) {
        std::printf("%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("All real-backend integration checks passed\n");
    return 0;
}
