// Runs the C++ gaze + hand ports (and the Treelite-compiled forest) on landmark recordings written by
// run_parity.py, and prints one result line per frame:  "x y state".
//
// Input text format:
//   S                     start a new gaze sequence (fresh GazeNative: calibration starts over)
//   F W H n  x y z ...    face frame  (n landmarks, n may be 0 = no face)
//   H W H n  x y z ...    hand frame  (n = 42 packed image+world landmarks, or 0 = no hand; ONE stateful tracker)
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "forest.hpp"
#include "gaze_math.hpp"
#include "hand_math.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: core_parity_check <recording.txt>\n"); return 2; }
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    core::TreeliteForest forest;
    std::unique_ptr<core::GazeNative> gaze(new core::GazeNative(&forest));
    core::TreeliteHandModel hand_model;
    core::HandNative hand(&hand_model);
    std::vector<float> pts;
    std::string tag;
    while (in >> tag) {
        if (tag == "S") { gaze.reset(new core::GazeNative(&forest)); continue; }
        int W = 0, H = 0, n = 0;
        in >> W >> H >> n;
        pts.assign(size_t(n) * 3, 0.f);
        for (auto& v : pts) in >> v;
        if (!in) { std::fprintf(stderr, "truncated recording\n"); return 2; }
        core::ModuleOut o = (tag == "F") ? gaze->process(pts.data(), n, W, H)
                                         : hand.process(pts.data(), n);
        std::printf("%.9g %.9g %d\n", double(o.x), double(o.y), o.state);
    }
    return 0;
}
