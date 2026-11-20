/* C interface to the Engineer 5 OS-integration library, for the HandEye
 * tester (tester/handeye_tester.py, loaded with ctypes) and any other tool
 * that cannot link C++ directly.
 *
 * It wraps the same osi::OsController the engine uses, so a test drives the
 * real mapping, gesture, click-stabilisation and SendInput code.
 * Plain C types only; every function is safe to call with a null handle.
 * One handle per thread (the controller is not thread-safe).
 */
#ifndef OSI_CAPI_H
#define OSI_CAPI_H

#include <stdint.h>

#ifdef _WIN32
#define OSI_API __declspec(dllexport)
#else
#define OSI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OSI_CAPI_VERSION 1
#define OSI_MAX_ACTIONS 8
#define OSI_MAX_EVENTS 256

/* Output modes for osi_create. */
#define OSI_OUTPUT_RECORD 0 /* events are only recorded (dry run): the mouse is never touched */
#define OSI_OUTPUT_MOUSE 1  /* events go to Windows through SendInput, and are recorded too */

/* Tracker freshness, same meaning as osi::TrackerState. */
#define OSI_TRACKED 0
#define OSI_HELD 1
#define OSI_LOST 2

/* Action types in osi_report.actions (osi::ActionType). */
#define OSI_ACTION_BUTTON_DOWN 0
#define OSI_ACTION_BUTTON_UP 1
#define OSI_ACTION_CLICK 2
#define OSI_ACTION_TOGGLE_PAUSE 3

/* Event kinds from osi_drain_events (osi::SinkEvent::Kind). */
#define OSI_EVENT_MOVE 0
#define OSI_EVENT_DOWN 1
#define OSI_EVENT_UP 2
#define OSI_EVENT_CLICK 3

typedef struct osi_tracker {
    int32_t has_point; /* 0 = no point this frame */
    double x;          /* normalised 0..1 */
    double y;
    int32_t state;     /* raw tracker state integer */
    int32_t tracking;  /* OSI_TRACKED / OSI_HELD / OSI_LOST */
} osi_tracker;

typedef struct osi_report {
    int32_t processed;        /* 0 = timestamp rejected */
    int32_t source;           /* 0 none, 1 hand, 2 gaze */
    int32_t has_cursor;
    int32_t cursor_x;         /* virtual-desktop pixels */
    int32_t cursor_y;
    int32_t moved;
    int32_t action_count;
    int32_t action_type[OSI_MAX_ACTIONS];
    int32_t action_button[OSI_MAX_ACTIONS]; /* 0 left, 1 right, 2 middle */
    int32_t pause_toggled;
    int32_t active;
    int32_t inhibited;
    int32_t injection_failed;
    int32_t release_pending;
    int32_t gaze_calibrating;
    int32_t hand_held;
    int32_t unknown_states;
    int32_t confirmed_hand;   /* debounced gesture after this frame */
    int32_t confirmed_eye;
} osi_report;

typedef struct osi_event {
    int32_t kind;   /* OSI_EVENT_* */
    int32_t button; /* 0 left, 1 right, 2 middle */
    int32_t x;      /* pixel, for moves */
    int32_t y;
} osi_event;

typedef struct osi_rect {
    int32_t left, top, width, height;
} osi_rect;

OSI_API int32_t osi_version(void);

/* Default settings as INI text. Returns the length written (without the
 * terminator), or the length needed when `capacity` is too small. */
OSI_API int32_t osi_default_settings_ini(char* buffer, int32_t capacity);

/* Creates a controller. `settings_ini` may be NULL or empty for defaults.
 * screen_target: 0 primary monitor, 1 all monitors. Returns NULL on failure
 * and writes the reason to `error` (UTF-8) when given. */
OSI_API void* osi_create(const char* settings_ini, int32_t screen_target, int32_t output_mode,
                         char* error, int32_t error_capacity);
OSI_API void osi_destroy(void* handle);

/* Replaces the settings (INI text). Returns 1 on success; on failure the
 * controller is unchanged and the reason goes to `error`. Unknown keys and
 * bad values are reported as errors so the tester can show them. */
OSI_API int32_t osi_apply_settings_ini(void* handle, const char* settings_ini, char* error,
                                       int32_t error_capacity);

OSI_API void osi_set_active(void* handle, int32_t active);
OSI_API void osi_set_inhibited(void* handle, int32_t inhibited);
OSI_API int32_t osi_release_all(void* handle); /* 1 when no button is left down */

/* One frame. Fills `report` (may be NULL). Returns `report->processed`. */
OSI_API int32_t osi_update(void* handle, double timestamp_seconds, const osi_tracker* hand,
                           const osi_tracker* gaze, osi_report* report);

/* Copies and clears the recorded mouse events (oldest first). Returns the count. */
OSI_API int32_t osi_drain_events(void* handle, osi_event* events, int32_t capacity);

/* Mapping target and the whole virtual desktop, in pixels. */
OSI_API int32_t osi_screen_geometry(void* handle, osi_rect* target, osi_rect* desktop);

/* Current Windows cursor position. Returns 0 if unavailable. */
OSI_API int32_t osi_cursor_position(int32_t* x, int32_t* y);

/* Optional 1-Euro smoothing (Engineer 4's dsp_filter, used read-only), so the
 * tester can feed OsController the same kind of stabilised points as the
 * engine. Returns NULL when the library was built without dsp_filter. */
OSI_API void* osi_filter_create(double min_cutoff_hz, double beta, double derivative_cutoff_hz);
OSI_API void osi_filter_destroy(void* filter);
/* Filters one point in place. `valid` = 0 resets the filter (target lost).
 * Returns 1 when (x, y) holds a smoothed point, 0 when there is no output. */
OSI_API int32_t osi_filter_apply(void* filter, double* x, double* y, double timestamp_seconds,
                                 int32_t valid);

#ifdef __cplusplus
}
#endif

#endif /* OSI_CAPI_H */
