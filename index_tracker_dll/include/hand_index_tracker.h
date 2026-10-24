#ifndef HAND_INDEX_TRACKER_H
#define HAND_INDEX_TRACKER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HAND_INDEX_ABI_VERSION 1u
#define HAND_INDEX_LANDMARK_INDEX 8

#if defined(_WIN32)
#  if defined(HAND_INDEX_TRACKER_BUILD)
#    define HAND_INDEX_API __declspec(dllexport)
#  else
#    define HAND_INDEX_API __declspec(dllimport)
#  endif
#  define HAND_INDEX_CALL __cdecl
#else
#  if defined(HAND_INDEX_TRACKER_BUILD)
#    define HAND_INDEX_API __attribute__((visibility("default")))
#  else
#    define HAND_INDEX_API
#  endif
#  define HAND_INDEX_CALL
#endif

typedef enum hand_index_error {
    HAND_INDEX_OK = 0,
    HAND_INDEX_ERROR_INVALID_ARGUMENT = 1,
    HAND_INDEX_ERROR_PYTHON = 2,
    HAND_INDEX_ERROR_MODEL = 3,
    HAND_INDEX_ERROR_CLOSED = 4,
    HAND_INDEX_ERROR_INTERNAL = 5
} hand_index_error;

typedef enum hand_index_tracking_status {
    HAND_INDEX_STATUS_LOST = 0,
    HAND_INDEX_STATUS_TRACKED = 1
} hand_index_tracking_status;

/*
 * Configuration for one tracker instance.
 *
 * Call hand_index_config_default() before changing fields. All strings are
 * UTF-8 and are copied by hand_index_create(); the caller keeps ownership.
 *
 * python_dir:
 *   Directory containing index_tracker.py and hand_features.py. NULL uses the
 *   interpreter's normal module search path.
 * site_packages_dir:
 *   Optional directory containing numpy, opencv, and mediapipe (normally a
 *   virtual environment's Lib\site-packages). NULL uses the interpreter's
 *   normal module search path.
 * model_path:
 *   Path to hand_landmarker.task. NULL lets index_tracker.py resolve its
 *   normal development path. Deployment packages should pass an explicit path.
 * tracking_region:
 *   (left, top, right, bottom) in normalized camera coordinates. Defaults to
 *   (0, 0, 1, 1).
 * smoothing_alpha:
 *   IndexFingerTracker EMA alpha. Defaults to 0.35; valid range (0, 1].
 */
typedef struct hand_index_config {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* python_dir;
    const char* site_packages_dir;
    const char* model_path;
    float tracking_region[4];
    float smoothing_alpha;
    float min_hand_detection_confidence;
    float min_hand_presence_confidence;
    float min_tracking_confidence;
} hand_index_config;

/*
 * One frame result.
 *
 * Set struct_size to sizeof(hand_index_result) before calling
 * hand_index_process_bgr(), or initialize the structure with
 * hand_index_result_init().
 *
 * valid / status:
 *   1 / HAND_INDEX_STATUS_TRACKED when landmark 8 was observed on this frame.
 *   0 / HAND_INDEX_STATUS_LOST when no detection was available. Lost results
 *   never contain stale coordinates.
 *
 * x, y:
 *   Pointer coordinates normalized to [0, 1], with region mapping and EMA
 *   smoothing applied. X increases left-to-right in the internally mirrored
 *   frame; Y increases top-to-bottom.
 *
 * camera_x, camera_y:
 *   Normalized, clamped landmark-8 coordinates in the internally mirrored
 *   frame, before region mapping and before EMA smoothing.
 *
 * On loss, x, y, camera_x, and camera_y are all -1.0f.
 */
typedef struct hand_index_result {
    uint32_t struct_size;
    int32_t valid;
    int32_t status;
    int32_t landmark_index;
    float x;
    float y;
    float camera_x;
    float camera_y;
} hand_index_result;

typedef struct hand_index_tracker hand_index_tracker;

HAND_INDEX_API const char* HAND_INDEX_CALL hand_index_version(void);
HAND_INDEX_API uint32_t HAND_INDEX_CALL hand_index_abi_version(void);
HAND_INDEX_API const char* HAND_INDEX_CALL hand_index_error_string(hand_index_error error);

HAND_INDEX_API void HAND_INDEX_CALL hand_index_config_default(hand_index_config* config);
HAND_INDEX_API void HAND_INDEX_CALL hand_index_result_init(hand_index_result* result);

/*
 * Create a tracker. The embedded Python runtime and MediaPipe model are loaded
 * here. This call can take several seconds on first use.
 *
 * On failure, *out_tracker is set to NULL and error_message receives a
 * NUL-terminated UTF-8 diagnostic when capacity is nonzero.
 */
HAND_INDEX_API hand_index_error HAND_INDEX_CALL hand_index_create(
    const hand_index_config* config,
    hand_index_tracker** out_tracker,
    char* error_message,
    size_t error_message_capacity);

/*
 * Process one tightly packed or strided BGR8 frame.
 *
 * bgr_data points to width * height BGR pixels. stride_bytes is the byte
 * distance between rows; pass 0 for width * 3. The DLL copies the frame for
 * the duration of the call and never retains the caller's pointer.
 *
 * A frame with no hand returns HAND_INDEX_OK and a LOST result. Null pointers,
 * non-positive dimensions, or a stride smaller than width * 3 return
 * HAND_INDEX_ERROR_INVALID_ARGUMENT.
 *
 * Calls for the same tracker are serialized internally. Do not call destroy
 * concurrently with create/process on the same instance.
 */
HAND_INDEX_API hand_index_error HAND_INDEX_CALL hand_index_process_bgr(
    hand_index_tracker* tracker,
    const uint8_t* bgr_data,
    int32_t width,
    int32_t height,
    int32_t stride_bytes,
    hand_index_result* out_result,
    char* error_message,
    size_t error_message_capacity);

/*
 * Close the MediaPipe detector and release the tracker. Safe for NULL.
 *
 * The embedded Python interpreter is intentionally not finalized here: the
 * host application may own it or may still use other Python modules. Python
 * and its modules remain loaded until process exit.
 */
HAND_INDEX_API void HAND_INDEX_CALL hand_index_destroy(hand_index_tracker* tracker);

#ifdef __cplusplus
}
#endif

#endif
