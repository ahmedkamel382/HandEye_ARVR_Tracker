#include "hand_index_tracker.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf(
            "Usage: %s <python_dir> <site_packages_dir> <hand_landmarker.task> [width height]\n",
            argv[0]);
        return 2;
    }

    const int32_t width = (argc > 4) ? std::stoi(argv[4]) : 640;
    const int32_t height = (argc > 5) ? std::stoi(argv[5]) : 480;

    hand_index_config config;
    hand_index_config_default(&config);
    config.python_dir = argv[1];
    config.site_packages_dir = argv[2];
    config.model_path = argv[3];

    char error[1024] = {0};
    hand_index_tracker* tracker = nullptr;
    hand_index_error result = hand_index_create(&config, &tracker, error, sizeof(error));
    if (result != HAND_INDEX_OK) {
        std::printf("create failed: %s: %s\n", hand_index_error_string(result), error);
        return 1;
    }

    // Replace this buffer with the camera frame (cv::Mat::data, frame.cols,
    // frame.rows, and frame.step for an OpenCV BGR Mat).
    std::vector<uint8_t> bgr(static_cast<size_t>(width) * height * 3, 0);

    hand_index_result tracking;
    hand_index_result_init(&tracking);
    result = hand_index_process_bgr(
        tracker,
        bgr.data(),
        width,
        height,
        width * 3,
        &tracking,
        error,
        sizeof(error));

    if (result != HAND_INDEX_OK) {
        std::printf("process failed: %s: %s\n", hand_index_error_string(result), error);
    } else if (tracking.valid) {
        std::printf(
            "tracked pointer=(%.6f, %.6f) camera=(%.6f, %.6f) landmark=%d\n",
            tracking.x,
            tracking.y,
            tracking.camera_x,
            tracking.camera_y,
            tracking.landmark_index);
    } else {
        std::printf("no hand: pointer=(-1, -1)\n");
    }

    hand_index_destroy(tracker);
    return result == HAND_INDEX_OK ? 0 : 1;
}
