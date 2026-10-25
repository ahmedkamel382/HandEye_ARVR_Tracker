// Runs the C++ gaze + hand ports (the gaze .dll models and the compiled hand forest) on landmark recordings
// written by run_parity.py, and prints one result line per frame:
//   gaze frame: "yaw_deg pitch_deg state"     hand frame: "x y state"
//   usage: core_parity_check <recording.txt> <gaze models dir>
//
// Input text format:
//   S                     start a new gaze sequence (fresh GazeNative: calibration starts over)
//   F W H n  x y z ...    face frame  (n = 484 packed landmarks + head-pose matrix rows, or 0 = no face)
//   H W H n  x y z ...    hand frame  (n = 42 packed image+world landmarks, or 0 = no hand; ONE stateful tracker)
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "forest.hpp"
#include "gaze_math.hpp"
#include "tl_model.hpp"
#include "hand_math.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: core_parity_check <recording.txt> <gaze models dir>\n"); return 2; }
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    const std::string dir = argv[2];
    std::unique_ptr<core::TlForest> wink;
    std::unique_ptr<core::TlRegressor> pitch, yaw;
    try {
        wink.reset(new core::TlForest(dir + "/gesture_model.dll"));
        pitch.reset(new core::TlRegressor(dir + "/gaze_pitch.dll"));
        yaw.reset(new core::TlRegressor(dir + "/gaze_yaw.dll"));
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    auto newGaze = [&] { return std::unique_ptr<core::GazeNative>(new core::GazeNative(wink.get(), pitch.get(), yaw.get())); };
    std::unique_ptr<core::GazeNative> gaze = newGaze();
    core::TreeliteHandModel hand_model;
    core::HandNative hand(&hand_model);
    std::vector<float> pts;
    std::string tag;
    while (in >> tag) {
        if (tag == "S") { gaze = newGaze(); continue; }
        int W = 0, H = 0, n = 0;
        in >> W >> H >> n;
        pts.assign(size_t(n) * 3, 0.f);
        for (auto& v : pts) in >> v;
        if (!in) { std::fprintf(stderr, "truncated recording\n"); return 2; }
        core::ModuleOut o = (tag == "F") ? gaze->process(pts.data(), n, W, H)
                                         : hand.process(pts.data(), n);
        if (tag == "F") std::printf("%.9g %.9g %d\n", o.yaw_deg, o.pitch_deg, o.state);
        else            std::printf("%.9g %.9g %d\n", double(o.x), double(o.y), o.state);
    }
    return 0;
}
