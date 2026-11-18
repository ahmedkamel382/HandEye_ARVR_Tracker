"""Exercise the real recorder/OsBridge on existing synthetic-video test fixtures.

This is an integration check, not a live accuracy, jitter, or latency experiment.
It never opens a webcam and always requests --filter dsp --os log. Generated
video, stand-in models, console logs, and recordings stay in a new external
directory. The real tray also appends its normal LocalAppData log. Existing
readable tray settings are required and checked without changing or restoring them.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import evaluate_holds
import import_vs_trace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def save(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def require_saved_settings(settings):
    try:
        value = digest(settings)
    except OSError as error:
        raise RuntimeError(f"existing readable saved tray settings are required: {settings}") from error
    if value is None:
        raise RuntimeError(f"existing readable saved tray settings are required: {settings}")
    return value


def verify_saved_settings(settings, expected_digest):
    try:
        current = digest(settings)
    except OSError as error:
        raise RuntimeError("saved tray settings became unreadable; they were not restored") from error
    if current != expected_digest:
        raise RuntimeError("saved tray settings changed; they were not restored")


def run_engine(command, directory, name, settings, expected_settings_digest):
    # Refuse accidental expansion of this numerical validation into a device test.
    def option_value(option):
        if command.count(option) != 1:
            return None
        index = command.index(option) + 1
        return command[index] if index < len(command) else None

    video = option_value("--video")
    if (not video or video.startswith("--") or "--camera" in command or "--control" in command
            or option_value("--os") != "log" or option_value("--filter") != "dsp"):
        raise ValueError("integration check permits only synthetic video, DSP, and dry-run OS")
    require_saved_settings(settings)
    verify_saved_settings(settings, expected_settings_digest)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        save(directory / f"{name}.stdout.txt", result.stdout)
        save(directory / f"{name}.stderr.txt", result.stderr)
        save(directory / f"{name}.command.json", json.dumps(command, indent=2))
        return result
    finally:
        # Includes nonzero exits, subprocess errors and failures while saving logs.
        # Stop this check immediately; never overwrite a changed user settings file.
        verify_saved_settings(settings, expected_settings_digest)


def check_shutdown_summary(stdout, channel, generated_frames):
    summary = re.search(r"captured (\d+) \| processed (\d+) \| evicted (\d+) \| skipped (\d+)"
                        r" \| stale (\d+) \| python errors (\d+)", stdout)
    submitted = re.search(r"os stage: (\d+) frames submitted", stdout)
    if not summary or not submitted:
        raise RuntimeError("missing real engine shutdown summary")
    captured, processed, evicted, skipped, stale, errors = map(int, summary.groups())
    count = int(submitted[1])
    # The source consumes exactly one frame as its startup size/type probe.
    if captured != generated_frames - 1:
        raise RuntimeError(f"{channel} did not capture the complete synthetic video: "
                           f"{captured} of {generated_frames - 1} expected frames")
    if captured != processed + evicted + skipped + stale or count != processed or errors:
        raise RuntimeError(f"{channel} frame accounting or bridge errors failed")
    if "pool slots returned: 8/8 (no leaks)" not in stdout:
        raise RuntimeError(f"{channel} frame-pool return failed")
    return captured, processed, count, stale, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    executable = args.exe.resolve(strict=True)
    settings = Path(os.environ["APPDATA"]) / "HandEyeTracker/os_integration.ini"
    before_settings = require_saved_settings(settings)
    repo = Path(__file__).resolve().parents[2]
    output = Path(tempfile.mkdtemp(prefix="handeye-recorder-pipeline-", dir=args.output_parent))
    print(f"SYNTHETIC integration artifacts: {output}", flush=True)

    # Reuse teammate test fixtures without modifying their tests or production AI.
    fixture_path = repo / "core_engine/tests/e2e/run_e2e.py"
    spec = importlib.util.spec_from_file_location("existing_core_e2e", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    video = output / "synthetic_scene.avi"
    writer = fixture.cv2.VideoWriter(str(video), fixture.cv2.VideoWriter_fourcc(*"MJPG"),
                                     fixture.FPS, (fixture.W, fixture.H))
    if not writer.isOpened():
        raise RuntimeError("synthetic-video writer could not open")
    generated_frames = 0
    try:
        for _, count, face, hand, pinch in fixture.PHASES:
            image = fixture.draw(face, hand, pinch)
            for _ in range(count):
                writer.write(image)
                generated_frames += 1
    finally:
        writer.release()

    fake_ai = repo / "core_engine/tests/fake_ai"
    sys.path.insert(0, str(fake_ai / "bridge"))
    import hand_features_ref
    model = output / "synthetic_hand_model.pkl"
    hand_features_ref.train_standin_model(str(model))

    results = []
    for channel in ("hand", "gaze"):
        target = output / channel
        command = [str(executable), "--video", str(video), "--ai-root", str(fake_ai),
                   "--gaze-models", str(repo / "ai_pipeline/gaze_intent/models"),
                   "--hand-model", str(model), "--filter", "dsp", "--os", "log",
                   "--no-gaze" if channel == "hand" else "--no-hand",
                   "--hand-filter" if channel == "hand" else "--gaze-filter",
                   "1.2,4,1" if channel == "hand" else "0.8,6,1.5",
                   "--dsp-record-dir", str(target)]
        completed = run_engine(command, output, channel, settings, before_settings)
        if completed.returncode != 0:
            raise RuntimeError(f"{channel} engine exit {completed.returncode}; inspect saved logs")
        captured, processed, count, stale, errors = check_shutdown_summary(
            completed.stdout, channel, generated_frames)
        trace = target / "trace.txt"
        imported = subprocess.run([sys.executable, "-B", str(Path(import_vs_trace.__file__)),
                                   str(trace), "--channel", channel, "--submitted", str(count)],
                                  capture_output=True, text=True, timeout=30)
        if imported.returncode != 0:
            raise RuntimeError(f"{channel} CLI import failed: {imported.stderr}")
        save(output / f"{channel}.paired.csv", imported.stdout)
        coverage = evaluate_holds.evaluate(imported.stdout.encode("utf-8"))
        if coverage["frames"] != count or not coverage["status_counts"].get("Filtered"):
            raise RuntimeError(f"{channel} recorder did not retain filtered observations")
        save(output / f"{channel}.coverage.json", json.dumps(coverage, indent=2))

        # A repeated output path must fail before processing and preserve its data.
        original_hash = digest(trace)
        repeated = run_engine(command, output, channel + "_existing_directory", settings, before_settings)
        if repeated.returncode != 4 or digest(trace) != original_hash:
            raise RuntimeError("existing recording was not safely rejected")
        results.append({"channel": channel, "captured": captured, "processed": processed,
                        "submitted": count, "stale": stale, "python_errors": errors,
                        "status_counts": coverage["status_counts"], "importer_exit": imported.returncode,
                        "existing_directory_rejected": True, "trace_sha256": original_hash})
        print(f"PASS {channel}: {count} complete recorded/imported submissions", flush=True)

    verify_saved_settings(settings, before_settings)
    report = {"kind": "synthetic integration validation, not live performance",
              "executable": str(executable), "executable_sha256": digest(executable),
              "video_sha256": digest(video), "settings_unchanged": True, "runs": results}
    save(output / "validation.json", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Recorder pipeline validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
