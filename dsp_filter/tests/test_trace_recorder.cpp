#include "dsp_trace_recorder.hpp"
#include "dsp_adapter.hpp"
#include "frame_clock.hpp"
#include "freshness_gate.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Observe ordinary C++ allocations during the intended update/append path. All
// fixtures, recorder storage, strings and output streams are prepared beforehand.
namespace allocation_probe {
bool active = false;
std::size_t count = 0;
}

void* operator new(std::size_t size) {
    if (allocation_probe::active) ++allocation_probe::count;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

namespace rec = dsp::recording;
namespace fs = std::filesystem;
std::size_t checks = 0;
const dsp::FilterConfig hand_config{1.2, 4.0, 1.0};
const dsp::FilterConfig gaze_config{0.8, 6.0, 1.5};

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class Callable>
void must_throw(Callable&& call, const char* message) {
    bool threw = false;
    try { call(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}

bool same_number(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
}

bool same_counters(const core::FreshnessGate::Counters& a,
                   const core::FreshnessGate::Counters& b) {
    return a.filter_updates == b.filter_updates && a.held_reused == b.held_reused &&
        a.held_without_cache == b.held_without_cache && a.resets == b.resets &&
        a.rejected == b.rejected && a.invalid == b.invalid;
}

core::ModuleOut module(float x, float y, int tracking = 0, int state = 0,
                       bool ok = true, int missed = 0) {
    core::ModuleOut m;
    m.x = x; m.y = y; m.state = state; m.ok = ok; m.missed_frames = missed;
    if (tracking >= 0) m.tracking_state = static_cast<core::TrackingState>(tracking);
    return m;
}

struct Step {
    std::int64_t ns;
    core::ModuleOut m;
    int landmarks;
};

std::vector<Step> hand_steps() {
    return {
        {1234567, module(0.4f, 0.5f), 42},
        {67901234, module(0.45f, 0.55f), 42},
        {67901234, module(0.9f, 0.1f), 42}, // Same capture time: rejected, cached output held.
        {134567901, module(0.9f, 0.1f, 1, 0, true, 1), 0},
        {201234568, module(-1.f, -1.f, 2, 0, false), -1}, // Bridge error clears gate.
        {267901235, module(0.9f, 0.1f, 1, 0, true, 2), 0}, // Native grace may survive bridge error.
        {334567902, module(0.6f, 0.4f), 42},
        {401234569, module(std::numeric_limits<float>::infinity(), 0.5f), 42},
        {467901236, module(0.f, 0.f), 42},
        {1201234567, module(1.f, 1.f), 42},
        {1267901234, module(0.7f, 0.8f), 42},
        {1334567901, module(-1.f, -1.f, 2, 0, true, 7), 0},
        {1401234568, module(0.2f, 0.9f), 42},
    };
}

std::vector<Step> gaze_steps() {
    return {
        {1234567, module(-1.f, -1.f, -1, -1), 484}, // Calibration is not a DSP observation.
        {67901234, module(0.25f, 0.75f, -1), 484},
        {134567901, module(0.3f, 0.7f, -1), 484},
        {134567901, module(0.8f, 0.2f, -1), 484},
        {201234568, module(-1.f, -1.f, -1), 0},
        {267901235, module(0.9f, 0.1f, -1), 484},
        {334567902, module(-1.f, -1.f, -1, 0, false), -1},
        {401234569, module(0.f, 0.f, -1), 484},
        {1201234567, module(1.f, 1.f, -1), 484},
        {1267901234, module(0.8f, 0.2f, -1, 1), 484}, // Wink does not change DSP coordinate handling.
    };
}

struct Harness {
    rec::Observation observation;
    core::DspFilterAdapter adapter;
    core::FreshnessGate gate;
    std::int64_t previous_ms = -1;
    core::Clock::time_point epoch{};

    explicit Harness(dsp::FilterConfig cfg, bool observe = true)
        : adapter(cfg, {}, observe ? &observation : nullptr) {}

    rec::FrameRecord step(std::uint64_t frame, const Step& input) {
        observation = {};
        const auto captured = epoch + std::chrono::duration_cast<core::Clock::duration>(
            std::chrono::nanoseconds(input.ns));
        const auto times = core::frameTimes(epoch, captured, previous_ms);
        previous_ms = times.ts_ms;
        const auto point = gate.update(input.m, adapter, true, times.capture_s);
        rec::FrameRecord r;
        r.frame_id = frame;
        r.mediapipe_ms = times.ts_ms;
        r.capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(captured - epoch).count();
        r.dsp_seconds = times.capture_s;
        r.detector_count = input.landmarks;
        r.ok = input.m.ok;
        r.tracker_x = input.m.x; r.tracker_y = input.m.y;
        r.state = input.m.state;
        r.tracking = input.m.tracking_state ? static_cast<int>(*input.m.tracking_state) : -1;
        r.missed = input.m.missed_frames;
        r.dsp = observation;
        r.published = bool(point);
        if (point) {
            r.published_x = point->x; r.published_y = point->y;
            r.published_tracking = point->held ? 1 : 0;
        }
        return r;
    }
};

rec::FrameRecord valid_record() {
    Harness h(hand_config);
    return h.step(0, {1234567, module(0.4f, 0.5f), 42});
}

std::string render(const rec::Recorder& recorder, std::uint64_t submitted) {
    std::ostringstream stream;
    require(recorder.write(stream, submitted), "in-memory trace write failed");
    return stream.str();
}

fs::path unique_test_parent() {
    const auto base = fs::temp_directory_path();
    const auto tick = core::Clock::now().time_since_epoch().count();
    for (int i = 0; i < 100; ++i) {
        const auto candidate = base / ("handeye-recorder-test-" + std::to_string(tick) + "-" + std::to_string(i));
        if (fs::create_directory(candidate)) return candidate;
    }
    throw std::runtime_error("cannot reserve a unique test directory");
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void adapter_equivalence() {
    rec::Observation observation;
    core::DspFilterAdapter plain(hand_config), observed(hand_config, {}, &observation);
    const std::array<std::array<double, 3>, 10> inputs{{
        {{0.2, 0.8, 0.0}}, {{0.4, 0.6, 0.067}}, {{0.9, 0.1, 0.067}},
        {{0.7, 0.3, 0.02}}, {{0.3, 0.7, 0.06700000001}},
        {{0.5, 0.5, 0.134}}, {{-0.3, 1.2, 0.2}},
        {{std::numeric_limits<double>::infinity(), 0.5, 0.267}},
        {{0.1, 0.9, 0.334}}, {{0.8, 0.2, 1.0}},
    }};
    for (const auto& input : inputs) {
        observation = {};
        double ax = input[0], ay = input[1], bx = ax, by = ay;
        const auto a = plain.update(ax, ay, input[2]);
        const auto b = observed.update(bx, by, input[2]);
        require(a == b && same_number(ax, bx) && same_number(ay, by),
                "observer changed the production adapter result");
        require(plain.errors() == observed.errors() &&
                plain.rejectedTimestamps() == observed.rejectedTimestamps(),
                "observer changed adapter counters");
        require(observation.adapter_called && observation.outcome == static_cast<int>(b),
                "adapter snapshot missed the real outcome");
    }
    observation = {};
    plain.reset(); observed.reset();
    require(observation.resets == 1 && !observation.adapter_called,
            "explicit reset was not recorded independently of update");
}

void gate_equivalence() {
    for (const auto channel : {rec::Channel::Hand, rec::Channel::Gaze}) {
        const auto config = channel == rec::Channel::Hand ? hand_config : gaze_config;
        const auto steps = channel == rec::Channel::Hand ? hand_steps() : gaze_steps();
        Harness plain(config, false), observed(config);
        rec::Recorder recorder(channel);
        recorder.set_initial_parameters(config);
        for (std::size_t i = 0; i < steps.size(); ++i) {
            const auto a = plain.step(i, steps[i]);
            const auto b = observed.step(i, steps[i]);
            require(a.published == b.published && a.published_tracking == b.published_tracking &&
                    a.published_x == b.published_x && a.published_y == b.published_y &&
                    a.dsp_seconds == b.dsp_seconds && a.mediapipe_ms == b.mediapipe_ms,
                    "recording changed an actual gate publication or timestamp");
            require(same_counters(plain.gate.counters(), observed.gate.counters()),
                    "recording changed gate counters");
            require(plain.adapter.errors() == observed.adapter.errors() &&
                    plain.adapter.rejectedTimestamps() == observed.adapter.rejectedTimestamps(),
                    "recording changed adapter counters through the gate");
            require(recorder.append(b), "normal recording unexpectedly overflowed");
        }
        require(recorder.complete(steps.size()), "complete gate sequence marked incomplete");
    }
}

void no_stale_diagnostics() {
    Harness h(hand_config);
    const auto sequence = hand_steps();
    std::vector<rec::FrameRecord> records;
    for (std::size_t i = 0; i < sequence.size(); ++i) records.push_back(h.step(i, sequence[i]));
    require(records[0].dsp.update_called && records[0].dsp.status == int(dsp::FilterStatus::Initialized),
            "first production update was not captured");
    require(records[2].dsp.status == int(dsp::FilterStatus::RejectedTimestamp) &&
            records[2].published && records[2].published_tracking == 1,
            "rejection was not distinguished from a held publication");
    require(records[3].published && !records[3].dsp.adapter_called && !records[3].dsp.update_called &&
            records[3].dsp.status == -1 && records[3].dsp.outcome == -1 && !records[3].dsp.has_point,
            "held frame reused a previous adapter diagnostic");
    require(!records[4].published && records[4].dsp.resets == 1 && !records[4].dsp.update_called,
            "loss reset was not captured without a DSP update");
    require(!records[5].published && !records[5].dsp.adapter_called && records[5].dsp.resets == 0,
            "held-without-cache retained a stale reset or point");
    require(records[6].dsp.status == int(dsp::FilterStatus::Initialized) &&
            records[6].published_x == double(sequence[6].m.x), "reacquisition did not initialize directly");
    require(records[7].dsp.adapter_called && !records[7].dsp.update_called &&
            records[7].dsp.outcome == int(core::FilterOutcome::Invalid) && !records[7].published,
            "nonfinite adapter bypass was recorded as a real DSP update");
    require(records[8].dsp.status == int(dsp::FilterStatus::Initialized) &&
            records[8].published && records[8].published_x == 0.0 && records[8].published_y == 0.0,
            "valid origin/reinitialization was lost");
    require(records[9].dsp.status == int(dsp::FilterStatus::ReinitializedAfterGap),
            "actual long-gap status was not captured");
    Harness gaze(gaze_config);
    const auto calibration = gaze.step(0, gaze_steps()[0]);
    require(calibration.state == -1 && !calibration.dsp.adapter_called && !calibration.published,
            "calibration fabricated a DSP observation");
}

void numerical_error_observation() {
    rec::Observation observation;
    const dsp::FilterConfig large{1.0, std::numeric_limits<double>::max(), 1.0};
    core::DspFilterAdapter plain(large), observed(large, {}, &observation);
    double ax = 0.0, ay = 0.0, bx = 0.0, by = 0.0;
    plain.update(ax, ay, 0.0); observed.update(bx, by, 0.0);
    observation = {};
    ax = ay = bx = by = 1.0;
    const auto a = plain.update(ax, ay, 0.01);
    const auto b = observed.update(bx, by, 0.01);
    require(a == core::FilterOutcome::Invalid && a == b && ax == bx && ay == by,
            "numerical-error instrumentation changed fallback behavior");
    require(observation.update_called && !observation.has_point &&
            observation.status == int(dsp::FilterStatus::NumericalError) &&
            observation.outcome == int(core::FilterOutcome::Invalid),
            "numerical error was not captured from the actual return value");
    observation = {};
    ax = bx = 0.3; ay = by = 0.7;
    require(plain.update(ax, ay, 0.02) == observed.update(bx, by, 0.02) && ax == bx && ay == by,
            "numerical-error recovery changed with recording");
    require(observation.status == int(dsp::FilterStatus::Initialized), "old error status leaked into recovery");
}

void independent_channels() {
    Harness hand(hand_config), gaze(gaze_config), hand_reference(hand_config, false), gaze_reference(gaze_config, false);
    for (std::size_t i = 0; i < 128; ++i) {
        const auto ns = std::int64_t(i) * 66666667 + 1234567;
        const Step hs{ns, module(float(i % 17) / 16.f, float(i % 13) / 12.f), 42};
        const Step gs{ns, module(float(i % 11) / 10.f, float(i % 7) / 6.f, -1), 484};
        const auto h = hand.step(i, hs), g = gaze.step(i, gs);
        const auto hr = hand_reference.step(i, hs), gr = gaze_reference.step(i, gs);
        require(h.published_x == hr.published_x && h.published_y == hr.published_y &&
                g.published_x == gr.published_x && g.published_y == gr.published_y,
                "interleaved hand/gaze diagnostics shared state");
        require(h.dsp.input_x == double(hs.m.x) && g.dsp.input_x == double(gs.m.x),
                "channel snapshot overwritten by the other tracker");
    }
}

void precise_timestamps() {
    Harness h(hand_config);
    const auto first = h.step(0, {1234567, module(0.1f, 0.9f), 42});
    const auto second = h.step(1, {1254567, module(0.2f, 0.8f), 42});
    const auto duplicate = h.step(2, {1254567, module(0.3f, 0.7f), 42});
    require(first.capture_ms == 1 && second.capture_ms == 1 && duplicate.capture_ms == 1,
            "original capture milliseconds were not preserved");
    require(first.mediapipe_ms == 1 && second.mediapipe_ms == 2 && duplicate.mediapipe_ms == 3,
            "MediaPipe clock correction was not retained separately");
    require(first.dsp_seconds == first.dsp.timestamp_seconds &&
            second.dsp_seconds == second.dsp.timestamp_seconds &&
            second.dsp_seconds != double(second.mediapipe_ms) / 1000.0,
            "recording rounded the actual DSP clock to MediaPipe milliseconds");
    require(duplicate.dsp.status == int(dsp::FilterStatus::RejectedTimestamp) &&
            duplicate.dsp_seconds == second.dsp_seconds, "duplicate time was manufactured into a fresh time");
}

void capacity_and_completeness() {
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.set_initial_parameters(hand_config);
    require(!recorder.complete(0), "empty recording claimed completion");
    auto frame = valid_record();
    for (std::size_t i = 0; i < rec::kCapacity; ++i) {
        frame.frame_id = i;
        require(recorder.append(frame), "buffer overflowed before its advertised capacity");
    }
    require(recorder.size() == 4096 && recorder.attempted() == 4096 && recorder.overflow() == 0,
            "capacity accounting is incorrect");
    require(recorder.complete(4096) && !recorder.complete(4095) && !recorder.complete(4097),
            "submission count did not participate in completeness");
    require(!recorder.append(frame) && recorder.size() == 4096 && recorder.attempted() == 4097 &&
            recorder.overflow() == 1 && !recorder.complete(4097),
            "overflow overwrote data or claimed completion");
    const auto text = render(recorder, 4097);
    require(text.find("overflow=1") != std::string::npos && text.find("complete=0") != std::string::npos,
            "overflow was not explicit in the exported trace");
}

void no_update_allocations() {
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.set_initial_parameters(hand_config);
    Harness h(hand_config);
    allocation_probe::count = 0;
    allocation_probe::active = true;
    for (std::size_t i = 0; i < rec::kCapacity; ++i) {
        const Step input{std::int64_t(i) * 66666667 + 1234567,
                         module(float(i % 17) / 16.f, float(i % 13) / 12.f), 42};
        recorder.append(h.step(i, input));
    }
    allocation_probe::active = false;
    require(allocation_probe::count == 0, "intended production update/append path allocated memory");
    require(recorder.complete(rec::kCapacity), "allocation test did not exercise a full buffer");
}

struct CommaPunctuation : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};

void serialization_precision() {
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.set_initial_parameters(hand_config);
    auto frame = valid_record();
    frame.dsp_seconds = std::nextafter(0.1, 1.0);
    frame.dsp.timestamp_seconds = frame.dsp_seconds;
    frame.dsp.filtered_x = std::nextafter(0.5, 1.0);
    frame.published_x = frame.dsp.filtered_x;
    recorder.append(frame);
    std::ostringstream out;
    out.imbue(std::locale(std::locale::classic(), new CommaPunctuation));
    out << std::hex << std::boolalpha << std::showbase << std::showpos << std::uppercase;
    require(recorder.write(out, 1), "locale-independent serialization failed");
    const auto text = out.str();
    require(text.find("version=2 channel=hand capacity=4096") != std::string::npos,
            "version/channel/capacity not recorded");
    require(text.find("t=0.10000000000000002") != std::string::npos &&
            text.find("dx=0.50000000000000011") != std::string::npos,
            "binary64 values lost round-trip precision");
    require(text.find("complete=1") != std::string::npos, "complete footer missing");
    must_throw([&] { recorder.set_initial_parameters(gaze_config); },
               "initial parameters changed after capture started");
}

void settings_change_invalidates() {
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.set_initial_parameters(hand_config);
    auto frame = valid_record();
    recorder.append(frame);
    frame.frame_id = 1; frame.generation = 2;
    recorder.append(frame);
    require(recorder.config_changes() > 0 && !recorder.complete(2),
            "mixed filter configurations silently claimed completion");
    rec::Recorder changed_before_first(rec::Channel::Hand);
    changed_before_first.append(frame);
    require(changed_before_first.config_changes() > 0 && !changed_before_first.complete(1),
            "configuration replacement before the first observed frame went unnoticed");
}

struct FailBuffer : std::streambuf {
    int_type overflow(int_type) override { return traits_type::eof(); }
    std::streamsize xsputn(const char*, std::streamsize) override { return 0; }
};

void stream_failure() {
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.append(valid_record());
    FailBuffer sink;
    std::ostream failing(&sink);
    require(!recorder.write(failing, 1), "failed stream write reported success");
}

void file_session_success_and_no_overwrite() {
    const auto parent = unique_test_parent();
    const auto directory = parent / "recording";
    rec::OutputSession session(directory);
    require(fs::exists(directory / "INCOMPLETE"), "incomplete marker not established before capture");
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.set_initial_parameters(hand_config);
    recorder.append(valid_record());
    const auto result = session.finish(recorder, 1);
    require(result.complete && fs::exists(result.trace_path) && fs::exists(directory / "COMPLETE") &&
            !fs::exists(directory / "INCOMPLETE"), "successful capture was not finalized explicitly");
    const auto original = read_text(result.trace_path);
    must_throw([&] { rec::OutputSession duplicate(directory); }, "existing experiment directory was accepted");
    bool refused = false;
    try { refused = !session.finish(recorder, 1).complete; } catch (const std::exception&) { refused = true; }
    require(refused && read_text(result.trace_path) == original, "second finish changed completed data");
    const auto file = parent / "existing-file";
    { std::ofstream stream(file); stream << "preserve me"; }
    must_throw([&] { rec::OutputSession duplicate(file); }, "existing file accepted as output directory");
    require(read_text(file) == "preserve me", "existing file overwritten");
}

void file_session_failure_and_overflow() {
    const auto parent = unique_test_parent();
    const auto directory = parent / "write-failure";
    rec::OutputSession session(directory);
    // A conflicting path is a portable deterministic write failure; no ACL changes.
    fs::create_directory(directory / "trace.partial");
    rec::Recorder recorder(rec::Channel::Hand);
    recorder.append(valid_record());
    const auto failure = session.finish(recorder, 1);
    require(!failure.complete && !failure.error.empty() && fs::exists(directory / "INCOMPLETE") &&
            !fs::exists(directory / "COMPLETE"), "write failure was not retained as incomplete");
    const auto conflict_dir = parent / "final-file-conflict";
    rec::OutputSession conflict_session(conflict_dir);
    { std::ofstream existing(conflict_dir / "trace.txt"); existing << "existing experiment data"; }
    const auto conflict = conflict_session.finish(recorder, 1);
    require(!conflict.complete && read_text(conflict_dir / "trace.txt") == "existing experiment data" &&
            fs::exists(conflict_dir / "INCOMPLETE"), "final-file collision overwrote an existing trace");
    const auto overflow_dir = parent / "overflow";
    rec::OutputSession overflow_session(overflow_dir);
    rec::Recorder full(rec::Channel::Hand);
    const auto frame = valid_record();
    for (std::size_t i = 0; i < rec::kCapacity + 1; ++i) full.append(frame);
    const auto incomplete = overflow_session.finish(full, rec::kCapacity + 1);
    require(!incomplete.complete && fs::exists(overflow_dir / "INCOMPLETE") &&
            !fs::exists(overflow_dir / "COMPLETE"), "overflow artifact was marked complete");
}

void export_fixtures(const fs::path& directory) {
    if (!fs::create_directory(directory)) throw std::runtime_error("fixture directory must not already exist");
    for (const auto channel : {rec::Channel::Hand, rec::Channel::Gaze}) {
        const auto config = channel == rec::Channel::Hand ? hand_config : gaze_config;
        const auto steps = channel == rec::Channel::Hand ? hand_steps() : gaze_steps();
        Harness h(config);
        rec::Recorder recorder(channel);
        recorder.set_initial_parameters(config);
        rec::OutputSession output(directory / (channel == rec::Channel::Hand ? "hand" : "gaze"));
        for (std::size_t i = 0; i < steps.size(); ++i) recorder.append(h.step(i, steps[i]));
        const auto result = output.finish(recorder, steps.size());
        if (!result.complete) throw std::runtime_error("fixture export failed: " + result.error);
        std::cout << "SYNTHETIC_FIXTURE " << result.trace_path.string() << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::array<std::pair<const char*, void(*)()>, 13> tests{{
        {"adapter_enabled_disabled_equivalence", adapter_equivalence},
        {"gate_enabled_disabled_equivalence", gate_equivalence},
        {"no_stale_diagnostics", no_stale_diagnostics},
        {"numerical_error_and_recovery", numerical_error_observation},
        {"independent_hand_and_gaze", independent_channels},
        {"precise_and_duplicate_timestamps", precise_timestamps},
        {"bounded_capacity_and_completeness", capacity_and_completeness},
        {"no_per_update_allocations", no_update_allocations},
        {"serialization_full_precision_and_locale", serialization_precision},
        {"configuration_change_detection", settings_change_invalidates},
        {"stream_write_failure", stream_failure},
        {"file_success_and_no_overwrite", file_session_success_and_no_overwrite},
        {"file_failure_and_overflow", file_session_failure_and_overflow},
    }};
    std::size_t passed = 0;
    for (const auto& test : tests) {
        try { test.second(); ++passed; std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) {
            allocation_probe::active = false;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << "RECORDER_TESTS " << passed << '/' << tests.size() << " tests; " << checks << " checks\n";
    if (passed != tests.size()) return 1;
    try {
        if (argc == 3 && std::string(argv[1]) == "--export") export_fixtures(argv[2]);
        else if (argc != 1) throw std::runtime_error("usage: dsp_trace_recorder_tests [--export NEW_DIRECTORY]");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
    return 0;
}
