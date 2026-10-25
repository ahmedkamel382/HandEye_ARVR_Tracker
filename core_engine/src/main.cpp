// ============================================================================
//  GradProject - C++ core engine ("Traffic Cop", Engineer 1)
//
//  Whole chain in one program:
//
//   webcam -> [capture thread] -> lock-free queue -> [AI thread] -> results -> [main thread]
//                                                          |                      |
//     BGR -> mirror + RGB ONCE, into a pre-allocated slot   |                      +-- 1-Euro filter
//                                                          |                      +-- OS integration (Eng 5): screen map, gestures, mouse, tray
//     AI thread (the only one that touches Python + GIL):   |                      +-- OS output (log, or real mouse)
//       MediaPipe face landmarks  (Python, returns raw points)
//       MediaPipe hand landmarks  (Python, returns raw points)
//       gaze math + 3 models      (native C++: wink classifier + pitch/yaw regressors = Treelite-compiled .dll files)
//       hand features + gesture forest + vote + dropout grace  (native C++, forest = Treelite-compiled C)
//
//  Rules
//   * Pixels are written once (capture thread) and read in place (AI thread): no copies between
//     threads, no per-frame allocation, memory bounded by --pool.
//   * Mirror + BGR->RGB happen exactly once, here. Python never flips or converts.
//   * A Python exception never reaches the loop: that call becomes "nothing detected".
//   * Capture never blocks. If the AI is slower than the camera, stale frames are skipped (counted).
//   * Idle threads sleep on a semaphore: no spin-waiting, no wasted heat.
//   * The real mouse is NEVER touched unless started with --control (or --os win32).
// ============================================================================
#include "py_bridge.hpp"          // pybind11 first (Python.h must precede system headers)

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <blockingconcurrentqueue.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "build_paths.hpp"
#include "dsp_adapter.hpp"
#include "forest.hpp"
#include "tl_model.hpp"
#include "frame_pool.hpp"
#include "gaze_math.hpp"
#include "hand_math.hpp"
#include "os_bridge.hpp"
#include "tracking_types.hpp"

using namespace std::chrono_literals;
using core::Clock;

namespace {

std::atomic<bool> g_run{true};
void onSignal(int) { g_run.store(false); }

#ifdef _WIN32
std::atomic<bool> g_teardown_done{false};

// Closing the console window, logging off or shutting down must still release a held mouse button:
// ask the main loop to stop and give it up to 4 s (Windows kills the process after ~5 s).
BOOL WINAPI consoleHandler(DWORD type) {
    switch (type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
            g_run.store(false);
            return TRUE;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            g_run.store(false);
            for (int i = 0; i < 40 && !g_teardown_done.load(); ++i) Sleep(100);
            return TRUE;
        default:
            return FALSE;
    }
}

// Windows 10/11 may move a process that is not in the foreground onto efficiency cores / low clocks (EcoQoS).
// This engine is latency-critical while another window has the focus, so opt out of that throttling.
void disablePowerThrottling() {
#ifdef PROCESS_POWER_THROTTLING_CURRENT_VERSION
    PROCESS_POWER_THROTTLING_STATE st = {};
    st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    st.StateMask = 0;                                   // 0 = "do not throttle"
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof(st));   // best effort
#endif
}

std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), &w[0], n);
    return w;
}
#endif

// ---------------------------------------------------------------- config ----
struct Config {
    int         camera_index = 0;
    std::string video_path;                 // read a file instead of a webcam (repeatable tests)
    int         width  = 640;
    int         height = 480;
    double      fps    = 60.0;
    int         pool_size = 8;              // frames in flight (memory bound = pool * W*H*3)
    bool        latest_only = true;         // AI skips stale frames; --process-all disables
    int         seconds = 0;                // 0 = until Ctrl+C / 'q'
    bool        use_gaze = true;
    bool        use_hand = true;
    int         hand_roi = 0;               // 0 = off. N = feed the hand model an NxN crop (experimental)
    bool        preview = false;            // debug window drawn by the AI thread
    std::string ai_root = core::paths::kAiPipelineDir;
    std::string site    = core::paths::kPySiteDir;
    std::string gaze_models;                // folder with gesture_model.dll, gaze_pitch.dll, gaze_yaw.dll (default <ai-root>/gaze_intent/models)
    std::string selftest_hand;              // same, hand forest
    std::string hand_forest = "auto";       // auto | treelite | sklearn
    std::string hand_model;                 // explicit hand .pkl (scikit-learn backend); default assets/hand_gesture_rf.pkl
    bool        check = false;              // --check: diagnose the environment, then exit
    std::string os      = "log";            // log (dry run) | win32 (real mouse)
    bool        start_active = false;       // the OS stage starts paused unless --start-active
    bool        use_filter = true;          // --no-filter
    std::string filter  = "auto";           // auto | dsp (Eng 4) | builtin
    std::vector<double> hand_filter, gaze_filter;   // optional "min_cutoff,beta,d_cutoff" overrides (Hz, -, Hz)
    core::HandParams   hand;
};

struct Stats {
    std::atomic<uint64_t> captured{0}, evicted{0}, skipped{0}, stale{0}, processed{0};
    std::atomic<uint64_t> py_errors{0};
    std::atomic<uint64_t> latency_us_sum{0}, latency_us_max{0};
    std::atomic<uint64_t> gaze_us_sum{0}, hand_us_sum{0};
};

struct Pipeline {
    Pipeline(const Config& c, cv::Size frame_size)
        : cfg(c), pool(size_t(c.pool_size), frame_size),
          queue(size_t(c.pool_size)), results(256) {}

    Config cfg;
    core::FramePool pool;
    moodycamel::BlockingConcurrentQueue<core::Packet>         queue;
    moodycamel::BlockingConcurrentQueue<core::TrackingResult> results;
    Stats stats;
    Clock::time_point t0 = Clock::now();
    std::atomic<bool> ai_ready{false}, ai_failed{false}, ai_done{false}, capture_done{false}, source_lost{false};
};

inline double usBetween(Clock::time_point a, Clock::time_point b) {
    return double(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
}

// ------------------------------------------------------- readable errors ----
// Python errors arrive with a long traceback tail ("At: ..."); show the message, then a fix hint.
std::string shortErr(const char* what) {
    std::string m = what ? what : "";
    const size_t at = m.find("\n\nAt:");
    if (at != std::string::npos) m.resize(at);
    while (!m.empty() && (m.back() == '\n' || m.back() == ' ')) m.pop_back();
    return m;
}

std::string hintFor(const std::string& err, const std::string& site) {
    const size_t k = err.find("No module named '");
    if (k != std::string::npos) {
        const size_t b = k + 17, e = err.find('\'', b);
        std::string mod = err.substr(b, e == std::string::npos ? std::string::npos : e - b);
        if (mod == "landmark_extractors")
            return "ai_pipeline/bridge/landmark_extractors.py was not found: check --ai-root (it must contain bridge/ and gaze_intent/)";
        std::string pkg = mod;
        if (mod == "cv2") pkg = "opencv-python";
        else if (mod == "sklearn") pkg = "scikit-learn";
        return "install it into the venv the engine was configured with:  python -m pip install " + pkg +
               "   (the engine looks in: " + site + ")";
    }
    if (err.find("DLL load failed") != std::string::npos)
        return "install the Microsoft Visual C++ Redistributable 2015-2022 (x64), then retry";
    if (err.find("model missing") != std::string::npos || err.find(".task") != std::string::npos)
        return "the MediaPipe model file is missing: put face_landmarker.task / hand_landmarker.task into the repo's assets/ folder";
    if (err.find("gaze model") != std::string::npos)
        return "the three gaze models (gesture_model.dll, gaze_pitch.dll, gaze_yaw.dll) must be in <ai-root>/gaze_intent/models, "
               "or pass --gaze-models DIR";
    if (err.find("hand_gesture_rf.pkl") != std::string::npos || err.find("hand model") != std::string::npos)
        return "hand_gesture_rf.pkl is expected in the repo's assets/ folder (or pass --hand-model FILE); "
               "without it the engine would only have the rough geometric fallback";
    return std::string();
}

// ---------------------------------------------------------- gaze models ----
// Eng 3's three Treelite models, loaded from their .dll files (no Python involved).
struct GazeModels {
    std::unique_ptr<core::TlForest> wink;
    std::unique_ptr<core::TlRegressor> pitch, yaw;
};

GazeModels makeGazeModels(const Config& cfg) {
    const std::string dir = cfg.gaze_models.empty() ? cfg.ai_root + "/gaze_intent/models" : cfg.gaze_models;
    GazeModels m;
    m.wink.reset(new core::TlForest(dir + "/gesture_model.dll"));
    m.pitch.reset(new core::TlRegressor(dir + "/gaze_pitch.dll"));
    m.yaw.reset(new core::TlRegressor(dir + "/gaze_yaw.dll"));
    return m;
}

// ----------------------------------------------------------- hand model ----
// Must be called with the GIL held (the scikit-learn backend uses Python).
// "auto": --hand-model FILE (scikit-learn) > the compiled forest > scikit-learn on assets/hand_gesture_rf.pkl >
//         the rough geometric fallback (what hand_tracker.py does when the pkl does not exist).
std::unique_ptr<core::IHandModel> makeHandModel(const Config& cfg) {
    const bool explicit_pkl = !cfg.hand_model.empty();
#ifdef CORE_HAVE_HAND_FOREST
    if (!explicit_pkl && cfg.hand_forest != "sklearn") return std::unique_ptr<core::IHandModel>(new core::TreeliteHandModel());
#else
    if (cfg.hand_forest == "treelite")
        throw std::runtime_error("this build has no compiled hand forest: run tools/export_forests.py, "
                                 "then delete the build folder and re-configure");
#endif
    const std::string pkl = explicit_pkl ? cfg.hand_model : cfg.ai_root + "/../assets/hand_gesture_rf.pkl";
    if (!std::ifstream(pkl)) {
        if (cfg.hand_forest == "auto" && !explicit_pkl) {
            std::fprintf(stderr, "[warning] hand_gesture_rf.pkl not found (%s) -> rough geometric fallback with "
                                 "untested thresholds. Train the hand model first.\n", pkl.c_str());
            return std::unique_ptr<core::IHandModel>(new core::GeometricHandModel(cfg.hand));
        }
        throw std::runtime_error("hand model file not found: " + pkl);
    }
    std::unique_ptr<core::PySklearnHandModel> m(new core::PySklearnHandModel());
    m->load(pkl);
    return std::unique_ptr<core::IHandModel>(m.release());
}

// Compares the active hand model with scikit-learn's own answers (hand_forest/parity_cases.csv):
// rows are "f0..f23,class,confidence". The class must match exactly, the confidence within 1e-9.
int runHandSelfTest(const Config& cfg) {
    std::ifstream in(cfg.selftest_hand);
    if (!in) { std::fprintf(stderr, "Cannot open %s\n", cfg.selftest_hand.c_str()); return 1; }
    std::unique_ptr<core::IHandModel> model;
    try { model = makeHandModel(cfg); }
    catch (const std::exception& e) { std::fprintf(stderr, "[FATAL] %s\n", shortErr(e.what()).c_str()); return 2; }

    size_t n = 0, ok = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tok;
        double f[core::kHandFeatures];
        int i = 0, label = -1;
        double conf = 0.0;
        while (std::getline(ss, tok, ',')) {
            if (i < core::kHandFeatures) f[i] = double(std::strtof(tok.c_str(), nullptr));   // the file holds float32 values
            else if (i == core::kHandFeatures) label = std::atoi(tok.c_str());
            else if (i == core::kHandFeatures + 1) conf = std::strtod(tok.c_str(), nullptr);
            ++i;
        }
        if (i != core::kHandFeatures + 2) continue;
        ++n;
        int got = -1;
        double gc = 0.0;
        model->classify(f, got, gc);
        if (got == label && std::fabs(gc - conf) <= 1e-9) ++ok;
        else if (n - ok <= 5) std::printf("  mismatch on row %zu: scikit-learn=%d (%.12g) model=%d (%.12g)\n", n, label, conf, got, gc);
    }
    std::printf("hand model self-test [%s]: %zu/%zu predictions match scikit-learn %s\n",
                model->name(), ok, n, (n > 0 && ok == n) ? "(PASS)" : "(FAIL)");
    if (auto* sk = dynamic_cast<core::PySklearnHandModel*>(model.get())) sk->shutdown();
    model.reset();
    return (n > 0 && ok == n) ? 0 : 3;
}

// ---------------------------------------------------------- capture thread --
void captureThread(cv::VideoCapture& cap, Pipeline& P, bool paced, double fps) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    cv::Mat frame, mirrored;                         // reused every iteration: no reallocation
    uint64_t id = 0;
    auto next_tick = Clock::now();
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));

    while (g_run.load(std::memory_order_relaxed)) {
        if (!cap.read(frame) || frame.empty()) {               // end of the video file, or the camera is gone
            if (!paced && g_run.load()) {
                P.source_lost.store(true);
                std::fprintf(stderr, "[capture] the camera stopped delivering frames (unplugged, taken by another "
                                     "program, or the PC went to sleep). Shutting down cleanly.\n");
            }
            break;
        }
        const auto t_cap = Clock::now();
        if (frame.size() != P.pool.size() || frame.type() != CV_8UC3) {
            std::fprintf(stderr, "[capture] frame size/type changed - stopping\n");
            break;
        }
        P.stats.captured.fetch_add(1, std::memory_order_relaxed);

        uint32_t slot;
        if (!P.pool.acquire(slot)) {
            core::Packet stale;                                // AI is behind: evict the oldest
            if (P.queue.try_dequeue(stale)) {
                P.pool.release(stale.slot);
                P.stats.evicted.fetch_add(1, std::memory_order_relaxed);
            }
            if (!P.pool.acquire(slot)) {                       // AI holds every slot
                P.stats.skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }

        // The one and only mirror + colour conversion in the whole system.
        cv::flip(frame, mirrored, 1);
        cv::cvtColor(mirrored, P.pool.mat(slot), cv::COLOR_BGR2RGB);

        const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_cap - P.t0).count();
        P.queue.enqueue(core::Packet{slot, id++, ts_ms, t_cap});

        if (paced) {                                           // video-file mode only
            next_tick += period;
            std::this_thread::sleep_until(next_tick);
        }
    }
    P.capture_done.store(true, std::memory_order_release);
}

// -------------------------------------------------------------- AI thread ---
void drawPreview(const cv::Mat& rgb, const core::TrackingResult& r, cv::Mat& bgr) {
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (r.gaze.state == -1)
        cv::putText(bgr, "GAZE CALIBRATING - keep both eyes open", {10, 54}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 200, 255}, 2);
    if (r.gaze.detected())
        cv::circle(bgr, {int(r.gaze.x * bgr.cols), int(r.gaze.y * bgr.rows)}, 8, {0, 255, 0}, 2);
    if (r.hand.detected())
        cv::circle(bgr, {int(r.hand.x * bgr.cols), int(r.hand.y * bgr.rows)}, 8, {0, 0, 255}, -1);
    char txt[96];
    std::snprintf(txt, sizeof txt, "gaze state %d | hand state %d", r.gaze.state, r.hand.state);
    cv::putText(bgr, txt, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);
    cv::imshow("core_engine preview (q = quit)", bgr);
}

void aiBody(Pipeline& P) {
    const Config& cfg = P.cfg;
    py::gil_scoped_acquire gil;                      // this thread owns Python for its whole life
    core::PyLandmarker face, hand;
    GazeModels gaze_models;
    std::unique_ptr<core::IHandModel> hand_model;

    try {
        core::prependSysPath(cfg.site);
        core::prependSysPath(cfg.ai_root + "/bridge");
        if (cfg.use_gaze) {
            gaze_models = makeGazeModels(cfg);
            face.load("landmark_extractors", "FaceLandmarkExtractor");
        }
        if (cfg.use_hand) {
            hand_model = makeHandModel(cfg);
            if (cfg.hand_roi > 0) {                      // a moving crop cannot be tracked between frames: use IMAGE mode
                py::object env = py::module_::import("os").attr("environ");
                if (!env.contains("HAND_RUNNING_MODE")) env["HAND_RUNNING_MODE"] = "IMAGE";
            }
            hand.load("landmark_extractors", "HandLandmarkExtractor");
        }
    } catch (const std::exception& e) {
        const std::string msg = shortErr(e.what());
        std::fprintf(stderr, "[FATAL] could not start the AI stage:\n%s\n", msg.c_str());
        const std::string hint = hintFor(msg, cfg.site);
        if (!hint.empty()) std::fprintf(stderr, "  -> %s\n  (run  core_engine --check  for a full diagnosis)\n", hint.c_str());
        P.ai_failed.store(true);
        return;
    }
    std::printf("[ai] ready | gaze: %s | hand: %s%s\n",
                cfg.use_gaze ? (std::string("face landmarks + native math, models = ") + gaze_models.wink->name() + " + pitch/yaw").c_str() : "off",
                cfg.use_hand ? (std::string("hand landmarks + native features, model = ") + hand_model->name()).c_str() : "off",
                cfg.hand_roi > 0 ? " | hand ROI crop ON" : "");
    P.ai_ready.store(true);

    core::GazeNative gaze(gaze_models.wink.get(), gaze_models.pitch.get(), gaze_models.yaw.get());
    core::HandNative handn(hand_model.get(), cfg.hand);
    const cv::Size fs = P.pool.size();
    const int roi = std::min({cfg.hand_roi, fs.width, fs.height});
    cv::Mat hand_crop, preview_bgr;
    if (roi > 0) hand_crop.create(roi, roi, CV_8UC3);
    bool hand_valid = false;                         // ROI state lives here: the AI thread is its only user
    float last_hx = 0.f, last_hy = 0.f;
    std::vector<float> face_pts, hand_pts;           // reused buffers: no per-frame allocation after warm-up
    face_pts.reserve(size_t(core::GazeNative::kPacked) * 3);
    hand_pts.reserve(size_t(core::kHandPacked) * 3);

    int64_t last_ts = -1;
    uint64_t err_printed = 0;
    core::Packet pkt;

    for (;;) {
        if (!P.queue.wait_dequeue_timed(pkt, 5ms)) {          // sleeps while idle
            if (P.capture_done.load(std::memory_order_acquire) && P.queue.size_approx() == 0) break;
            continue;
        }
        if (cfg.latest_only) {                                // jump to the newest queued frame
            core::Packet newer;
            while (P.queue.try_dequeue(newer)) {
                P.pool.release(pkt.slot);
                P.stats.stale.fetch_add(1, std::memory_order_relaxed);
                pkt = newer;
            }
        }

        const int64_t ts = std::max<int64_t>(pkt.ts_ms, last_ts + 1);   // MediaPipe VIDEO mode: strictly increasing
        last_ts = ts;
        const cv::Mat& frame = P.pool.mat(pkt.slot);

        core::TrackingResult r;
        r.frame_id = pkt.frame_id;
        r.ts_ms = ts;
        r.captured_at = pkt.captured_at;

        if (cfg.use_gaze) {
            const auto t_g = Clock::now();
            const int n = face.call(frame, ts, face_pts);
            r.gaze = (n >= 0) ? gaze.process(face_pts.data(), n, fs.width, fs.height) : core::ModuleOut{};
            r.gaze_ms = usBetween(t_g, Clock::now()) / 1000.0;
        }
        if (cfg.use_hand) {
            const auto t_h = Clock::now();
            int n;
            if (roi > 0 && hand_valid) {
                // Crop an roi x roi window centred on the last known pointer, clamped to the frame,
                // then map the landmarks back to full-frame normalised coordinates.
                const int x0 = std::clamp(int(last_hx * fs.width)  - roi / 2, 0, fs.width  - roi);
                const int y0 = std::clamp(int(last_hy * fs.height) - roi / 2, 0, fs.height - roi);
                frame(cv::Rect(x0, y0, roi, roi)).copyTo(hand_crop);
                n = hand.call(hand_crop, ts, hand_pts);
                for (int i = 0; i < std::min(n, core::kHandPoints); ++i) {          // image landmarks only; world ones are metric
                    hand_pts[3 * i]     = (x0 + hand_pts[3 * i]     * roi) / float(fs.width);
                    hand_pts[3 * i + 1] = (y0 + hand_pts[3 * i + 1] * roi) / float(fs.height);
                }
            } else {
                n = hand.call(frame, ts, hand_pts);
            }
            r.hand = (n >= 0) ? handn.process(hand_pts.data(), n) : core::ModuleOut{};
            hand_valid = r.hand.detected();
            if (hand_valid) { last_hx = r.hand.x; last_hy = r.hand.y; }
            r.hand_ms = usBetween(t_h, Clock::now()) / 1000.0;
        }

        if (cfg.preview) {
            drawPreview(frame, r, preview_bgr);
            if ((cv::waitKey(1) & 0xFF) == 'q') g_run.store(false);
        }
        P.pool.release(pkt.slot);                              // slot is reusable immediately

        // Python failures: counted, first few printed, loop keeps going.
        const uint64_t errs = face.errors + hand.errors + (hand_model ? hand_model->errors() : 0);
        P.stats.py_errors.store(errs, std::memory_order_relaxed);
        if (errs > err_printed && err_printed < 5) {
            std::fprintf(stderr, "[ai] python error #%llu: %s\n", (unsigned long long)errs,
                         (hand.errors ? hand.last_error : face.last_error).c_str());
            err_printed = errs;
        }

        const double lat = usBetween(pkt.captured_at, Clock::now());
        P.stats.processed.fetch_add(1, std::memory_order_relaxed);
        P.stats.latency_us_sum.fetch_add(uint64_t(lat), std::memory_order_relaxed);
        P.stats.gaze_us_sum.fetch_add(uint64_t(r.gaze_ms * 1000.0), std::memory_order_relaxed);
        P.stats.hand_us_sum.fetch_add(uint64_t(r.hand_ms * 1000.0), std::memory_order_relaxed);
        uint64_t prev = P.stats.latency_us_max.load(std::memory_order_relaxed);
        while (uint64_t(lat) > prev &&
               !P.stats.latency_us_max.compare_exchange_weak(prev, uint64_t(lat), std::memory_order_relaxed)) {}

        P.results.enqueue(r);
    }

    if (cfg.preview) cv::destroyAllWindows();
    face.shutdown();                                           // releases MediaPipe resources
    hand.shutdown();
    if (auto* sh = dynamic_cast<core::PySklearnHandModel*>(hand_model.get())) sh->shutdown();
    gaze_models = GazeModels();
    hand_model.reset();
}

void aiThread(Pipeline& P) {
    aiBody(P);
    P.ai_done.store(true, std::memory_order_release);
}

// ------------------------------------------------------------------- CLI ----
void usage() {
    std::printf(
"core_engine - camera -> hand/gaze trackers -> filter -> OS output\n\n"
"source:   --camera N | --video FILE | --width W --height H --fps F\n"
"loop:     --pool N (frames in flight, default 8) | --process-all | --seconds N | --preview\n"
"modules:  --no-gaze | --no-hand | --hand-roi N (experimental crop) | --gaze-models DIR (default <ai-root>/gaze_intent/models)\n"
"          --hand-forest auto|treelite|sklearn | --hand-model FILE.pkl (default assets/hand_gesture_rf.pkl)\n"
"hand:     --hand-anchor tip|knuckle|mid   (pointer source, margins and calibration: tray app / os_integration.ini)\n"
"filter:   --filter auto|dsp|builtin (auto = Eng 4's dsp_filter when compiled in) | --no-filter\n"
"          --hand-filter MIN,BETA,DCUT | --gaze-filter MIN,BETA,DCUT   (Hz, -, Hz; tune with real recordings)\n"
"output:   --os log|win32 (default log = dry run) | --control (same as --os win32) | --start-active\n"
"paths:    --ai-root DIR | --site DIR\n"
"checks:   --check (diagnose Python, packages, models, forest and camera; sends nothing to the OS)\n"
"          --selftest-hand-forest hand_forest/parity_cases.csv\n"
"exit codes: 0 ok | 1 bad options / source or filter config | 2 AI stage or Python failed to start | 3 camera lost | 130 interrupted\n"
"hotkeys (Windows): Ctrl+Alt+P = pause/resume (OS integration), F9 = quit.  Ctrl+C also quits.  Tray icon: Calibrate, Exit.\n");
}

// Opens the camera (or the video file). On Windows the camera is tried with DirectShow first, then with the
// default backend. Prints what to try when nothing works.
bool openSource(const Config& cfg, cv::VideoCapture& cap) {
    if (!cfg.video_path.empty()) {
        cap.open(cfg.video_path);
        if (!cap.isOpened()) std::fprintf(stderr, "Could not open the video file: %s\n", cfg.video_path.c_str());
        return cap.isOpened();
    }
    bool ok;
#ifdef _WIN32
    ok = cap.open(cfg.camera_index, cv::CAP_DSHOW);
    if (!ok) {
        std::fprintf(stderr, "[camera] DirectShow could not open camera %d, trying the default backend...\n", cfg.camera_index);
        ok = cap.open(cfg.camera_index);
    }
#else
    ok = cap.open(cfg.camera_index);
#endif
    if (!ok) {
        std::fprintf(stderr,
            "Could not open camera %d.\n"
            "  - Another program may be using it (browser tab, Teams/Zoom, the Python collector scripts): close it.\n"
            "  - Windows: Settings > Privacy & security > Camera must allow desktop apps.\n"
            "  - Try another camera: --camera 1\n", cfg.camera_index);
        return false;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    cap.set(cv::CAP_PROP_FPS,          cfg.fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE,   1);
    return true;
}

// Frames per second the camera really delivers (reads n frames).
double measureFps(cv::VideoCapture& cap, int n = 30) {
    cv::Mat m;
    const auto t0 = Clock::now();
    int got = 0;
    for (int i = 0; i < n; ++i) { if (cap.read(m) && !m.empty()) ++got; else break; }
    const double sec = usBetween(t0, Clock::now()) / 1e6;
    return (got > 1 && sec > 0.0) ? double(got) / sec : 0.0;
}

bool parseTriple(const std::string& s, std::vector<double>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        char* end = nullptr;
        out.push_back(std::strtod(tok.c_str(), &end));
        if (end == tok.c_str() || *end != '\0') { out.clear(); return false; }
    }
    if (out.size() != 3) { out.clear(); return false; }
    return true;
}

Config parseArgs(int argc, char** argv, bool& ok) {
    Config c;
    ok = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--camera")         c.camera_index = std::atoi(next());
        else if (a == "--video")          c.video_path = next();
        else if (a == "--width")          c.width  = std::atoi(next());
        else if (a == "--height")         c.height = std::atoi(next());
        else if (a == "--fps")            c.fps    = std::atof(next());
        else if (a == "--pool")           c.pool_size = std::max(2, std::atoi(next()));
        else if (a == "--seconds")        c.seconds = std::atoi(next());
        else if (a == "--process-all")    c.latest_only = false;
        else if (a == "--no-gaze")        c.use_gaze = false;
        else if (a == "--no-hand")        c.use_hand = false;
        else if (a == "--hand-roi")       c.hand_roi = std::max(0, std::atoi(next()));
        else if (a == "--preview")        c.preview = true;
        else if (a == "--ai-root")        c.ai_root = next();
        else if (a == "--site")           c.site = next();
        else if (a == "--gaze-models")    c.gaze_models = next();
        else if (a == "--selftest-hand-forest") c.selftest_hand = next();
        else if (a == "--hand-forest")    c.hand_forest = next();
        else if (a == "--hand-model")     c.hand_model = next();
        else if (a == "--check")          c.check = true;
        else if (a == "--os")             c.os = next();
        else if (a == "--control")        c.os = "win32";
        else if (a == "--start-active")   c.start_active = true;
        else if (a == "--no-filter")      c.use_filter = false;
        else if (a == "--filter")         c.filter = next();
        else if (a == "--hand-filter")    { if (!parseTriple(next(), c.hand_filter)) { std::fprintf(stderr, "--hand-filter needs MIN,BETA,DCUT\n"); ok = false; } }
        else if (a == "--gaze-filter")    { if (!parseTriple(next(), c.gaze_filter)) { std::fprintf(stderr, "--gaze-filter needs MIN,BETA,DCUT\n"); ok = false; } }
        else if (a == "--hand-anchor") {
            const std::string v = next();
            if (v == "tip") c.hand.anchor = core::HandAnchor::Tip;
            else if (v == "knuckle") c.hand.anchor = core::HandAnchor::Knuckle;
            else if (v == "mid") c.hand.anchor = core::HandAnchor::Mid;
            else { std::fprintf(stderr, "--hand-anchor must be tip, knuckle or mid\n"); ok = false; }
        } else if (a == "--help" || a == "-h") { usage(); ok = false; }
        else { std::fprintf(stderr, "Unknown option: %s (try --help)\n", a.c_str()); ok = false; }
    }
    if (c.hand_forest != "auto" && c.hand_forest != "treelite" && c.hand_forest != "sklearn") {
        std::fprintf(stderr, "--hand-forest must be auto, treelite or sklearn\n"); ok = false;
    }
    if (c.hand_forest == "treelite" && !c.hand_model.empty()) {
        std::fprintf(stderr, "--hand-model needs the scikit-learn backend (drop --hand-forest treelite)\n"); ok = false;
    }
    if (c.os != "log" && c.os != "win32") { std::fprintf(stderr, "--os must be log or win32\n"); ok = false; }
    if (c.filter != "auto" && c.filter != "dsp" && c.filter != "builtin") {
        std::fprintf(stderr, "--filter must be auto, dsp or builtin\n"); ok = false;
    }
#ifndef CORE_HAVE_DSP_FILTER
    if (c.filter == "dsp") { std::fprintf(stderr, "--filter dsp: this build has no dsp_filter target\n"); ok = false; }
#endif
    return c;
}

// --check: one command that tells an integrator what is wrong with the environment, with a fix per line.
// Runs with the GIL held by the main thread; sends nothing to the OS.
int runCheck(const Config& cfg) {
    int failed = 0, warned = 0;
    auto ok   = [&](const char* n, const std::string& d) { std::printf("[ OK ] %-18s %s\n", n, d.c_str()); };
    auto warn = [&](const char* n, const std::string& d) { ++warned; std::printf("[WARN] %-18s %s\n", n, d.c_str()); };
    auto bad  = [&](const char* n, const std::string& d, const std::string& hint) {
        ++failed;
        std::printf("[FAIL] %-18s %s\n", n, d.c_str());
        if (!hint.empty()) std::printf("         -> %s\n", hint.c_str());
    };
    auto firstLine = [](const std::string& m) { return m.substr(0, m.find('\n')); };
    std::printf("core_engine --check   (nothing is sent to the OS)\n\n");

    try {
        py::module_ sys = py::module_::import("sys");
        const std::string ver = py::str(sys.attr("version")).cast<std::string>();
        ok("python", ver.substr(0, ver.find(' ')) + "   prefix " + py::str(sys.attr("prefix")).cast<std::string>());
        core::prependSysPath(cfg.site);
        core::prependSysPath(cfg.ai_root + "/bridge");
    } catch (const std::exception& e) { bad("python", firstLine(shortErr(e.what())), ""); }

    auto pkg = [&](const char* name) {
        try {
            py::module_ m = py::module_::import(name);
            const std::string v = py::hasattr(m, "__version__") ? py::str(m.attr("__version__")).cast<std::string>() : "?";
            ok(name, "version " + v);
        } catch (const std::exception& e) {
            const std::string msg = shortErr(e.what());
            bad(name, firstLine(msg), hintFor(msg, cfg.site));
        }
    };
    pkg("numpy");

    core::PyLandmarker face, hand;
    GazeModels gmodels;
    if (cfg.use_gaze) {
        try {
            gmodels = makeGazeModels(cfg);
            const float wp[core::kGazeFeatures] = {1.f, 1.f, 0.f, 1.f, 1.f, 0.f};        // both eyes open
            const float gp[core::kPoseFeatures] = {0.5f, 0.0f, 0.f, 0.f, 0.f};           // looking straight ahead
            char b[200];
            std::snprintf(b, sizeof b, "wink class %d, pitch %.2f, yaw %.2f deg (test inputs)", gmodels.wink->predict(wp),
                          gmodels.pitch->predict(gp), gmodels.yaw->predict(gp));
            ok("gaze models", b);
        } catch (const std::exception& e) {
            const std::string msg = shortErr(e.what());
            bad("gaze models", firstLine(msg), hintFor(msg, cfg.site));
        }
        try { face.load("landmark_extractors", "FaceLandmarkExtractor"); ok("face landmarker", "loaded"); }
        catch (const std::exception& e) { const std::string msg = shortErr(e.what()); bad("face landmarker", firstLine(msg), hintFor(msg, cfg.site)); }
    }
    std::unique_ptr<core::IHandModel> hmodel;
    if (cfg.use_hand) {
        try {
            hmodel = makeHandModel(cfg);
            double probe[core::kHandFeatures];                       // an open, relaxed hand
            for (int i = 0; i < 15; ++i) probe[i] = 160.0;
            for (int i = 15; i < 19; ++i) probe[i] = 1.2;
            for (int i = 19; i < 24; ++i) probe[i] = 1.8;
            int pc = -1;
            double pp = 0.0;
            hmodel->classify(probe, pc, pp);
            char b[200];
            std::snprintf(b, sizeof b, "%s, test class %d (confidence %.2f)", hmodel->name(), pc, pp);
            if (std::string(hmodel->name()).find("geometric") != std::string::npos)
                warn("hand model", std::string(b) + "  - no hand_gesture_rf.pkl: the pinch/fist decision is a rough guess");
            else ok("hand model", b);
        } catch (const std::exception& e) {
            const std::string msg = shortErr(e.what());
            bad("hand model", firstLine(msg), hintFor(msg, cfg.site));
        }
        try { hand.load("landmark_extractors", "HandLandmarkExtractor"); ok("hand landmarker", "loaded"); }
        catch (const std::exception& e) { const std::string msg = shortErr(e.what()); bad("hand landmarker", firstLine(msg), hintFor(msg, cfg.site)); }
    }

    cv::VideoCapture cap;
    if (openSource(cfg, cap)) {
        cv::Mat m;
        for (int i = 0; i < 30 && m.empty(); ++i) cap.read(m);
        if (m.empty()) bad("video source", "opened, but no frame arrived", "another program may hold the camera");
        else {
            char buf[160];
            if (cfg.video_path.empty()) {
                const double fps = measureFps(cap);
                std::snprintf(buf, sizeof buf, "%dx%d, delivers %.1f FPS (asked for %.0f)", m.cols, m.rows, fps, cfg.fps);
                if (fps > 0 && fps < 50) warn("video source", std::string(buf) + "  - below 60 FPS: low light makes the camera slow its exposure; check the driver/mode");
                else ok("video source", buf);
            } else {
                std::snprintf(buf, sizeof buf, "video file %dx%d", m.cols, m.rows);
                ok("video source", buf);
            }
        }
    } else {
        bad("video source", "cannot open (details above)", "");
    }
    cap.release();

    face.shutdown();
    hand.shutdown();
    if (auto* sh = dynamic_cast<core::PySklearnHandModel*>(hmodel.get())) sh->shutdown();
    hmodel.reset();

    std::printf("\n%s%s\n", failed ? (std::to_string(failed) + " check(s) FAILED").c_str() : "ALL CHECKS PASSED",
                warned ? (" (" + std::to_string(warned) + " warning(s))").c_str() : "");
    return failed ? 1 : 0;
}

} // namespace

// ------------------------------------------------------------------ main ----
int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
    disablePowerThrottling();
#endif
    bool args_ok = true;
    const Config cfg = parseArgs(argc, argv, args_ok);
    if (!args_ok) return 1;
    cv::setNumThreads(1);          // OpenCV's worker pool only adds wake-ups/heat on small images

#ifndef _WIN32
    if (cfg.os == "win32") { std::fprintf(stderr, "--control / --os win32 is only available on Windows\n"); return 1; }
#endif

    // ---- embedded Python (started before the camera so --selftest-hand-forest needs no camera) ----
#ifdef _WIN32
    if (core::paths::kPyHomeDir[0]) {
        // An embedded Python cannot report this itself: a wrong home aborts the process with "Fatal Python error".
        if (!std::ifstream(std::string(core::paths::kPyHomeDir) + "/Lib/os.py")) {
            std::fprintf(stderr, "[FATAL] The Python installation this program was built with is not there any more:\n"
                                 "  %s\nDelete the build folder and configure again with the project's .venv.\n",
                         core::paths::kPyHomeDir);
            return 2;
        }
        _wputenv_s(L"PYTHONHOME", utf8ToWide(core::paths::kPyHomeDir).c_str());   // base install of the venv
    }
#endif
    // (no signal handlers: keep our Ctrl+C handler; no program dir on sys.path)
    py::scoped_interpreter interpreter{false, 0, nullptr, false};
    try { core::prependSysPath(cfg.site); } catch (const std::exception&) {}
    if (!cfg.selftest_hand.empty()) return runHandSelfTest(cfg);
    if (cfg.check) return runCheck(cfg);

    // ---- video source ----
    cv::VideoCapture cap;
    const bool from_file = !cfg.video_path.empty();
    if (!openSource(cfg, cap)) return 1;

    // The first frame tells us the REAL size (drivers sometimes ignore the request).
    cv::Mat first;
    for (int i = 0; i < 30 && first.empty(); ++i) cap.read(first);
    if (first.empty() || first.type() != CV_8UC3) { std::fprintf(stderr, "No usable frames from source\n"); return 1; }
    const double reported_fps = cap.get(cv::CAP_PROP_FPS);
    if (!from_file) {
        const double eff = measureFps(cap);                       // what the driver REALLY delivers
        std::printf("Camera delivers %.1f FPS (asked for %.0f)\n", eff, cfg.fps);
        if (eff > 0 && eff < 50)
            std::printf("[warning] below 60 FPS. Common causes: low light (auto-exposure slows the camera), USB hub, "
                        "a driver mode without MJPG. The engine still works; it will just track at the camera's rate.\n");
    }
    std::printf("Source: %dx%d @ %.1f FPS | pool=%d (%.1f MB) | policy=%s\n", first.cols, first.rows,
                reported_fps, cfg.pool_size, double(cfg.pool_size) * first.total() * 3 / 1048576.0,
                cfg.latest_only ? "latest-only" : "process-every-frame");

    Pipeline P(cfg, first.size());
    // OS stage (Eng 5): DSP smoothing, then screen mapping / gestures / mouse / tray. Starts PAUSED unless --start-active.
    core::OsBridgeConfig oc;
    oc.real_mouse = (cfg.os == "win32");
    oc.start_active = cfg.start_active;
    oc.use_filter = cfg.use_filter;
    oc.use_dsp = (cfg.filter != "builtin");
    oc.hand_filter = cfg.hand_filter;
    oc.gaze_filter = cfg.gaze_filter;
    std::unique_ptr<core::OsBridge> os;
    try { os.reset(new core::OsBridge(oc)); }
    catch (const std::exception& e) { std::fprintf(stderr, "[FATAL] OS integration: %s\n", e.what()); return 1; }
    std::printf("Filter: %s%s\n", os->filterDescription().c_str(), cfg.use_filter ? "" : "  (DISABLED by --no-filter)");
    std::printf("OS: %s. Starts %s (Ctrl+Alt+P or the tray menu to pause/resume).\n", os->name(),
                cfg.start_active ? "ACTIVE" : "PAUSED");

    py::gil_scoped_release release_gil;      // main thread lets go; the AI thread takes over
    std::thread th_ai(aiThread, std::ref(P));
    while (!P.ai_ready.load() && !P.ai_failed.load() && g_run.load()) std::this_thread::sleep_for(20ms);
    if (P.ai_failed.load() || !P.ai_ready.load()) {
        g_run.store(false);
        P.capture_done.store(true);          // capture never started: tell the AI loop to exit, or join() hangs
        th_ai.join();
        return P.ai_failed.load() ? 2 : 130; // 2 = AI stage failed to start, 130 = interrupted by the user
    }

    const double pace_fps = (reported_fps > 1.0 ? reported_fps : cfg.fps);
    std::thread th_cap(captureThread, std::ref(cap), std::ref(P), from_file, pace_fps);

    // ---- main thread: DSP filter + OS stage, and the 1 Hz report ----
    const auto t_start = Clock::now();
    auto next_report = t_start + 1s;
    uint64_t l_cap = 0, l_proc = 0, l_lat = 0, l_gz = 0, l_hd = 0;
    core::TrackingResult r, last;
#ifdef _WIN32
    bool f9_prev = false;
#endif
    while (g_run.load()) {
        if (P.results.wait_dequeue_timed(r, 50ms)) { os->update(r); last = r; }

#ifdef _WIN32
        const bool f9 = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
        if (f9 && !f9_prev) g_run.store(false);
        f9_prev = f9;
#endif
        if (os->exited()) break;                                           // Exit chosen in the tray menu

        if (Clock::now() >= next_report) {
            next_report += 1s;
            const Stats& s = P.stats;
            const uint64_t cap_n = s.captured.load(), proc_n = s.processed.load();
            const uint64_t lat = s.latency_us_sum.load(), gz = s.gaze_us_sum.load(), hd = s.hand_us_sum.load();
            const double dp = double(proc_n - l_proc);
            std::printf("cap %3llu | ai %3llu fps | gaze %5.1f ms hand %5.1f ms | latency avg %5.1f max %5.1f ms | "
                        "evict %llu stale %llu pyerr %llu | slots %zu/%zu | gaze(%.2f,%.2f s=%d) hand(%.2f,%.2f s=%d) | "
                        "os frames %llu\n",
                        (unsigned long long)(cap_n - l_cap), (unsigned long long)(proc_n - l_proc),
                        dp ? double(gz - l_gz) / dp / 1000.0 : 0.0, dp ? double(hd - l_hd) / dp / 1000.0 : 0.0,
                        dp ? double(lat - l_lat) / dp / 1000.0 : 0.0, double(s.latency_us_max.load()) / 1000.0,
                        (unsigned long long)s.evicted.load(), (unsigned long long)s.stale.load(),
                        (unsigned long long)s.py_errors.load(), P.pool.free_count(), P.pool.capacity(),
                        last.gaze.x, last.gaze.y, last.gaze.state, last.hand.x, last.hand.y, last.hand.state,
                        (unsigned long long)os->framesSent());
            std::fflush(stdout);
            l_cap = cap_n; l_proc = proc_n; l_lat = lat; l_gz = gz; l_hd = hd;
        }
        if (P.ai_done.load()) break;                                       // source finished, all drained
        if (cfg.seconds > 0 && Clock::now() - t_start >= std::chrono::seconds(cfg.seconds)) break;
    }

    // ---- teardown (order matters) ----
    g_run.store(false);
    th_cap.join();                  // capture stops, sets capture_done
    th_ai.join();                   // AI drains the queue, shuts the Python modules down
    while (P.results.try_dequeue(r)) os->update(r);                        // results still in flight
    os->stop();                     // closes the tray; Eng 5's controller releases any held mouse button
    cap.release();

    const Stats& s = P.stats;
    std::printf("\n--- summary ---\ncaptured %llu | processed %llu | evicted %llu | skipped %llu | stale %llu | python errors %llu\n",
                (unsigned long long)s.captured.load(), (unsigned long long)s.processed.load(),
                (unsigned long long)s.evicted.load(), (unsigned long long)s.skipped.load(),
                (unsigned long long)s.stale.load(), (unsigned long long)s.py_errors.load());
    std::printf("os stage: %llu frames submitted\n", (unsigned long long)os->framesSent());
    std::printf("pool slots returned: %zu/%zu %s\n", P.pool.free_count(), P.pool.capacity(),
                P.pool.free_count() == P.pool.capacity() ? "(no leaks)" : "(LEAK!)");
    std::printf("Engine safely shut down.\n");
#ifdef _WIN32
    g_teardown_done.store(true);        // lets a pending console-close handler return
#endif
    return P.source_lost.load() ? 3 : 0;     // 3 = the camera disappeared; release_gil reacquires the GIL, then Python finalises
}
