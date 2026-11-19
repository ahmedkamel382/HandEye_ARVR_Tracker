#pragma once

// Optional evaluation instrumentation, separate from the filter mathematics.
// One main-loop thread owns a Recorder. Only append() is used in the processing
// path: it copies one numerical record into storage allocated at construction.
// Formatting and all filesystem operations belong to startup/shutdown only.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

#include "one_euro_filter.hpp"

namespace dsp::recording {

inline constexpr std::size_t kCapacity = 4096;

enum class Channel { Hand, Gaze };

inline const char* channel_name(Channel channel) noexcept {
    return channel == Channel::Hand ? "hand" : "gaze";
}

// A new value-initialized slot is used for EVERY processed observation. In
// particular, a Held/Lost/calibrating observation must not inherit a prior call.
// status is the actual dsp::FilterStatus when update_called, otherwise -1.
// outcome is the adapter's Ok=0 / Rejected=1 / Invalid=2, otherwise -1.
struct Observation {
    bool adapter_called = false;
    bool update_called = false;
    double timestamp_seconds = 0.0;
    double input_x = 0.0;
    double input_y = 0.0;
    int status = -1;
    bool has_point = false;
    double filtered_x = 0.0;
    double filtered_y = 0.0;
    int outcome = -1;
    unsigned resets = 0;
};

struct FrameRecord {
    std::uint64_t frame_id = 0;
    std::int64_t mediapipe_ms = 0;
    std::int64_t capture_ms = 0;
    double dsp_seconds = 0.0;
    int detector_count = -1;
    bool ok = false;
    double tracker_x = -1.0;
    double tracker_y = -1.0;
    int state = 0;
    int tracking = -1;
    int missed = 0;
    Observation dsp;
    bool published = false;
    double published_x = 0.0;
    double published_y = 0.0;
    int published_tracking = 2;
    std::uint64_t generation = 1;
};

static_assert(std::is_trivially_copyable<FrameRecord>::value,
              "Recorder append must remain a nonallocating numerical copy");

class Recorder {
public:
    explicit Recorder(Channel channel)
        : channel_(channel), records_(std::make_unique<FrameRecord[]>(kCapacity)),
          config_(channel == Channel::Hand ? kInitialHandConfig : kInitialGazeConfig) {}

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;
    Recorder(Recorder&&) = delete;
    Recorder& operator=(Recorder&&) = delete;

    // The integration supplies the actual selected parameters after resolving
    // command-line/saved settings. Once observations exist, metadata is immutable.
    void set_initial_parameters(FilterConfig config, TimingConfig timing = {}) {
        if (attempted_ != 0) {
            throw std::logic_error("Recorder parameters cannot change after recording starts");
        }
        config_ = config;
        timing_ = timing;
    }

    // No allocation, clock query, I/O, formatting, lock, callback or exception.
    // At capacity, processing continues and the trace is explicitly incomplete.
    bool append(const FrameRecord& record) noexcept {
        increment_saturated(attempted_);
        if (record.generation != previous_generation_) {
            increment_saturated(config_changes_);
            previous_generation_ = record.generation;
        }
        if (size_ == kCapacity) {
            increment_saturated(overflow_);
            return false;
        }
        records_[size_++] = record;
        return true;
    }

    Channel channel() const noexcept { return channel_; }
    std::uint64_t attempted() const noexcept { return attempted_; }
    std::size_t size() const noexcept { return size_; }
    std::uint64_t overflow() const noexcept { return overflow_; }
    std::uint64_t config_changes() const noexcept { return config_changes_; }

    bool complete(std::uint64_t submitted) const noexcept {
        return submitted > 0 && submitted == attempted_ && submitted == size_ &&
               overflow_ == 0 && config_changes_ == 0;
    }

    // Shutdown-only. Return value reports successful stream output, NOT that the
    // dataset is complete; the END record and complete(submitted) report that.
    // Values including NaN/Inf are preserved for invalid-input diagnostics.
    bool write(std::ostream& stream, std::uint64_t submitted) const {
        stream.imbue(std::locale::classic());
        stream << std::dec << std::defaultfloat << std::noshowpos << std::noshowbase << std::nouppercase
               << std::setprecision(std::numeric_limits<double>::max_digits10)
               << std::noboolalpha << std::setw(0);
        stream << "E4REC BEGIN version=2 channel=" << channel_name(channel_)
               << " capacity=" << kCapacity
               << " min_cutoff=" << config_.min_cutoff_hz
               << " beta=" << config_.beta
               << " derivative_cutoff=" << config_.derivative_cutoff_hz
               << " min_dt=" << timing_.min_dt_seconds
               << " reset_gap=" << timing_.reset_gap_seconds << '\n';
        for (std::size_t i = 0; i < size_; ++i) {
            const FrameRecord& r = records_[i];
            const Observation& d = r.dsp;
            stream << "E4REC F frame=" << r.frame_id
                   << " ms=" << r.mediapipe_ms << " cap=" << r.capture_ms
                   << " t=" << r.dsp_seconds << " n=" << r.detector_count
                   << " ok=" << r.ok << " x=" << r.tracker_x << " y=" << r.tracker_y
                   << " state=" << r.state << " trk=" << r.tracking << " missed=" << r.missed
                   << " adapter=" << d.adapter_called << " called=" << d.update_called
                   << " input_t=" << d.timestamp_seconds << " ix=" << d.input_x << " iy=" << d.input_y
                   << " status=" << d.status << " has=" << d.has_point
                   << " dx=" << d.filtered_x << " dy=" << d.filtered_y
                   << " outcome=" << d.outcome << " resets=" << d.resets
                   << " pub=" << r.published << " px=" << r.published_x << " py=" << r.published_y
                   << " out_trk=" << r.published_tracking << " generation=" << r.generation << '\n';
        }
        stream << "E4REC END submitted=" << submitted << " attempted=" << attempted_
               << " stored=" << size_ << " overflow=" << overflow_
               << " config_changes=" << config_changes_ << " complete=" << complete(submitted) << '\n';
        return stream.good();
    }

private:
    static void increment_saturated(std::uint64_t& value) noexcept {
        if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
    }

    Channel channel_;
    std::unique_ptr<FrameRecord[]> records_;
    FilterConfig config_;
    TimingConfig timing_;
    std::size_t size_ = 0;
    std::uint64_t attempted_ = 0;
    std::uint64_t overflow_ = 0;
    std::uint64_t config_changes_ = 0;
    std::uint64_t previous_generation_ = 1;
};

struct FinishResult {
    bool complete = false;
    std::string error;
    std::filesystem::path trace_path;
};

class OutputSession {
public:
    // Startup-only. Reserving a NEW directory is atomic; existing experiments,
    // including empty directories, are never reused. The parent must exist.
    explicit OutputSession(const std::filesystem::path& directory) {
        if (directory.empty()) throw std::invalid_argument("Recorder output directory is empty");
        directory_ = std::filesystem::absolute(directory);
        if (!std::filesystem::create_directory(directory_)) {
            throw std::runtime_error("Recorder output directory already exists: " + directory_.string());
        }
        const auto marker = directory_ / "INCOMPLETE";
        std::ofstream out(marker, std::ios::out | std::ios::binary);
        out << "Engineer 4 numerical recording, format 2.\n"
               "Only COMPLETE without INCOMPLETE, plus a valid trace footer, certifies a completed save.\n";
        out.flush();
        const bool wrote = out.good();
        out.close();
        if (!wrote || out.fail()) {
            throw std::runtime_error("Cannot create recorder INCOMPLETE marker: " + directory_.string());
        }
    }

    OutputSession(const OutputSession&) = delete;
    OutputSession& operator=(const OutputSession&) = delete;
    OutputSession(OutputSession&&) = delete;
    OutputSession& operator=(OutputSession&&) = delete;

    const std::filesystem::path& directory() const noexcept { return directory_; }

    // Shutdown-only, one attempt per session. Abrupt termination leaves the
    // INCOMPLETE marker. No destructor performs I/O or silently retries writes.
    FinishResult finish(const Recorder& recorder, std::uint64_t submitted) {
        FinishResult result;
        result.trace_path = directory_ / "trace.txt";
        if (finish_attempted_) {
            result.error = "Recorder finish was already attempted";
            return result;
        }
        finish_attempted_ = true;
        try {
            const auto partial = directory_ / "trace.partial";
            const auto incomplete = directory_ / "INCOMPLETE";
            const auto completed = directory_ / "COMPLETE";
            if (!std::filesystem::is_regular_file(incomplete)) {
                throw std::runtime_error("Recorder INCOMPLETE marker is missing or invalid");
            }
            if (path_present(partial) || path_present(result.trace_path) || path_present(completed)) {
                throw std::runtime_error("Recorder output files already exist; refusing to overwrite");
            }
            std::ofstream out(partial, std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("Cannot open recorder trace.partial");
            const bool wrote = recorder.write(out, submitted);
            out.flush();
            const bool flushed = out.good();
            out.close();
            if (!wrote || !flushed || out.fail()) {
                throw std::runtime_error("Writing or closing recorder trace.partial failed");
            }

            // copy_options::none refuses an existing destination on Windows and
            // POSIX. Unlike rename(), it cannot replace another experiment file.
            std::filesystem::copy_file(partial, result.trace_path, std::filesystem::copy_options::none);
            if (!std::filesystem::remove(partial)) {
                throw std::runtime_error("Cannot retire recorder trace.partial; save remains incomplete");
            }
            if (!recorder.complete(submitted)) {
                result.error = "Recording incomplete: count mismatch, buffer overflow, empty run, or filter configuration change";
                return result;
            }

            // Publish the success marker last. If its creation or retirement of
            // INCOMPLETE fails, importers must reject this session even if trace.txt exists.
            std::filesystem::copy_file(incomplete, completed, std::filesystem::copy_options::none);
            if (!std::filesystem::remove(incomplete)) {
                throw std::runtime_error("Cannot retire recorder INCOMPLETE marker");
            }
            result.complete = true;
        } catch (const std::exception& error) {
            result.error = error.what();
        }
        return result;
    }

private:
    // A dangling symlink is also an existing path and must never be followed.
    static bool path_present(const std::filesystem::path& path) {
        const auto status = std::filesystem::symlink_status(path);
        return status.type() != std::filesystem::file_type::not_found;
    }

    std::filesystem::path directory_;
    bool finish_attempted_ = false;
};

} // namespace dsp::recording
