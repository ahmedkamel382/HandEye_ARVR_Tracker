"""Deterministic cue-launcher tests: no webcam, tracker, speech, or audio playback.

Engine launch and OS discovery are mocked. Temporary files contain only test
fixtures; fixed cue deadlines and timing expectations are independent constants.
"""

import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_cued_responsiveness as cue


VERIFIED_HASH = "3A3F05472A441E225006A276D03297A38135F13827E83F1B13CA278E46167200"


def status(*, capture=20, ai=20, frames=20, x="0.35", y="0.35", state=0,
           errors=0):
    return (f"cap {capture:3d} | ai {ai:3d} fps | gaze 0.0 ms hand 40.0 ms | "
            f"latency avg 40.0 max 60.0 ms | evict 0 stale 0 pyerr {errors} | "
            f"slots 8/8 | gaze(-1.00,-1.00 s=0) hand({x},{y} s={state}) | "
            f"os frames {frames}")


class CuePlanTests(unittest.TestCase):
    def test_vertical_slow_uses_original_deadlines_and_vertical_prompts(self):
        plan = cue.cue_plan("vertical-slow")
        self.assertEqual([c.at_seconds for c in plan], [0,5,8,9,10,11,14,22,25,26,27,28,31,39])
        self.assertEqual([plan[i].message for i in (0,1,5,6,7,11,12,13)], [
            "Hold top. Keep still.", "Get ready to move down.", "MOVE! Go down.",
            "STOP! Hold bottom.", "Get ready to move up.", "MOVE! Go up.",
            "STOP! Hold top.", "Finished. Keep holding top until the camera stops."])

    def test_slow_plan_has_fixed_independent_deadlines(self):
        self.assertEqual([c.at_seconds for c in cue.cue_plan("slow")],
                         [0, 5, 8, 9, 10, 11, 14, 22, 25, 26, 27, 28, 31, 39])

    def test_fast_plan_has_fixed_independent_deadlines(self):
        self.assertEqual([c.at_seconds for c in cue.cue_plan("fast")],
                         [0, 5, 8, 9, 10, 11, 11.5, 19.5, 22.5, 23.5,
                          24.5, 25.5, 26, 34])

    def test_both_plans_have_finite_nonoverlapping_instruction_order(self):
        for mode in ("slow", "fast"):
            with self.subTest(mode=mode):
                plan = cue.cue_plan(mode)
                self.assertTrue(all(b.at_seconds > a.at_seconds
                                    for a, b in zip(plan, plan[1:])))
                self.assertTrue(all(c.clip and c.message for c in plan))
                self.assertLess(plan[-1].at_seconds, 45)

    def test_unknown_mode_is_rejected(self):
        for mode in ("", "both", "FAST", "gaze"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                cue.cue_plan(mode)


class ReadinessTests(unittest.TestCase):
    def test_startup_and_camera_messages_cannot_arm_instructions(self):
        gate = cue.ReadinessGate()
        for line in ("Camera delivers 20.0 FPS", "Filter: dsp_filter (Eng 4)",
                     "Source: 640x480", "OS: dry run"):
            self.assertFalse(gate.feed(line, 5.0))
        self.assertIsNone(gate.ready_time)

    def test_requires_two_live_reports_and_increasing_os_count(self):
        gate = cue.ReadinessGate()
        self.assertFalse(gate.feed(status(frames=20), 10.0))
        self.assertTrue(gate.feed(status(frames=40), 11.0))
        self.assertEqual(gate.ready_time, 11.0)
        self.assertEqual(gate.first_status_time, 10.0)

    def test_intervening_diagnostic_line_does_not_reset_live_reports(self):
        gate = cue.ReadinessGate()
        self.assertFalse(gate.feed(status(frames=20), 10.0))
        gate.feed("unrelated model diagnostic", 10.2)
        self.assertTrue(gate.feed(status(frames=40), 11.0))

    def test_duplicate_os_count_does_not_confirm_processing(self):
        gate = cue.ReadinessGate()
        self.assertFalse(gate.feed(status(frames=20), 10.0))
        self.assertFalse(gate.feed(status(frames=20), 11.0))
        self.assertIsNone(gate.ready_time)

    def test_invalid_hand_observation_resets_consecutive_confirmation(self):
        gate = cue.ReadinessGate()
        self.assertFalse(gate.feed(status(frames=20), 10.0))
        self.assertFalse(gate.feed(status(frames=40, x="-1", y="-1"), 11.0))
        self.assertFalse(gate.feed(status(frames=60), 12.0))
        self.assertTrue(gate.feed(status(frames=80), 13.0))

    def test_nonlive_and_invalid_statuses_do_not_arm(self):
        cases = ({"capture": 0}, {"ai": 0}, {"state": 1}, {"state": 2},
                 {"x": "-0.1"}, {"x": "1.1"}, {"y": "-1"},
                 {"y": "1.1"}, {"x": "nan"}, {"x": "inf"})
        for changes in cases:
            with self.subTest(changes=changes):
                gate = cue.ReadinessGate()
                self.assertFalse(gate.feed(status(frames=20, **changes), 10.0))
                self.assertFalse(gate.feed(status(frames=40, **changes), 11.0))
                self.assertIsNone(gate.ready_time)

    def test_zero_coordinate_is_valid_for_readiness(self):
        gate = cue.ReadinessGate()
        self.assertFalse(gate.feed(status(frames=20, x="0", y="0"), 10.0))
        self.assertTrue(gate.feed(status(frames=40, x="0", y="0"), 11.0))


class SchedulerTests(unittest.TestCase):
    def test_cues_follow_exact_virtual_clock_without_sleep_or_audio(self):
        scheduler = cue.CueScheduler("slow", 100.0)
        actual = []
        for expected in cue.cue_plan("slow"):
            actual.append(scheduler.poll(100.0 + expected.at_seconds))
        self.assertEqual(tuple(actual), cue.cue_plan("slow"))
        self.assertTrue(scheduler.complete)
        self.assertIsNone(scheduler.poll(200.0))

    def test_no_cue_before_its_deadline_or_twice_at_same_time(self):
        scheduler = cue.CueScheduler("fast", 100.0)
        self.assertEqual(scheduler.poll(100.0), cue.cue_plan("fast")[0])
        self.assertIsNone(scheduler.poll(100.0))
        self.assertIsNone(scheduler.poll(104.999))

    def test_nonfinite_or_reversing_clock_is_refused(self):
        for instant in (99.9, float("nan"), float("inf"), -float("inf")):
            with self.subTest(instant=instant), self.assertRaises(RuntimeError):
                cue.CueScheduler("fast", 100.0).poll(instant)

    def test_nonfinite_readiness_anchor_is_refused(self):
        for instant in (float("nan"), float("inf")):
            with self.subTest(instant=instant), self.assertRaises(ValueError):
                cue.CueScheduler("fast", instant)

    def test_small_delay_is_accepted_without_moving_future_deadlines(self):
        scheduler = cue.CueScheduler("slow", 100.0)
        self.assertIsNotNone(scheduler.poll(100.05))
        self.assertIsNone(scheduler.poll(104.999))
        self.assertEqual(scheduler.poll(105.0).at_seconds, 5)

    def test_late_cue_aborts_instead_of_bursting_catchup_instructions(self):
        scheduler = cue.CueScheduler("slow", 100.0)
        scheduler.poll(100.0)
        with self.assertRaises(RuntimeError):
            scheduler.poll(108.0)

    def test_dead_process_cancels_pending_instructions(self):
        scheduler = cue.CueScheduler("fast", 100.0)
        with self.assertRaises(RuntimeError):
            scheduler.poll(100.0, process_alive=False)


class CommandSafetyTests(unittest.TestCase):
    def setUp(self):
        self.exe = Path("C:/verified/core_engine.exe")
        self.recording = Path("D:/new-session/recording")

    def test_exact_camera_command_keeps_dry_run_and_fixed_hand_configuration(self):
        command = cue.engine_command(self.exe, self.recording)
        self.assertEqual(command, [str(self.exe), "--camera", "0", "--filter", "dsp",
                                   "--os", "log", "--no-gaze", "--hand-filter",
                                   "1.2,4,1", "--seconds", "55", "--dsp-record-dir",
                                   str(self.recording)])
        cue.validate_command(command, self.exe, self.recording)

    def test_argument_additions_and_changes_are_refused(self):
        base = cue.engine_command(self.exe, self.recording)
        altered = [base + ["--control"], base + ["--os", "win32"],
                   base + ["--start-active"], base + ["--no-filter"], base[:-1],
                   ["other.exe", *base[1:]]]
        for old, new in (("log", "win32"), ("dsp", "builtin"),
                         ("1.2,4,1", "9,9,9"), ("55", "500")):
            changed = base.copy()
            changed[changed.index(old)] = new
            altered.append(changed)
        for command in altered:
            with self.subTest(command=command), self.assertRaises(ValueError):
                cue.validate_command(command, self.exe, self.recording)


class ConsoleBufferTests(unittest.TestCase):
    def test_split_chunks_crlf_and_utf8_are_reassembled(self):
        buffer = cue.BoundedConsole(max_lines=4, max_line_bytes=64)
        self.assertEqual(buffer.feed(b"fir"), [])
        self.assertEqual(buffer.feed(b"st\r\nsecond "), ["first"])
        self.assertEqual(buffer.feed(b"\xc3"), [])
        self.assertEqual(buffer.feed(b"\xa9\n"), ["second \u00e9"])
        self.assertEqual(buffer.lines, ["first", "second \u00e9"])
        self.assertFalse(buffer.overflow)

    def test_excess_lines_mark_overflow_without_unbounded_growth(self):
        buffer = cue.BoundedConsole(max_lines=2, max_line_bytes=64)
        buffer.feed(b"one\ntwo\nthree\nfour\n")
        self.assertTrue(buffer.overflow)
        self.assertEqual(len(buffer.lines), 2)

    def test_long_unterminated_line_is_bounded_and_marks_overflow(self):
        buffer = cue.BoundedConsole(max_lines=4, max_line_bytes=8)
        buffer.feed(b"X" * 10000)
        buffer.feed(b"\nend\n")
        self.assertTrue(buffer.overflow)
        self.assertLessEqual(len(buffer.lines), 4)
        self.assertTrue(all(len(line.encode("utf-8")) <= 8 for line in buffer.lines))

    def test_finish_preserves_partial_line(self):
        buffer = cue.BoundedConsole(max_lines=4, max_line_bytes=32)
        buffer.feed(b"last line")
        buffer.finish()
        self.assertIn("last line", buffer.lines)


class ExclusiveOutputTests(unittest.TestCase):
    def test_existing_experiment_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="handeye-cue-test-") as directory:
            target = Path(directory) / "report.json"
            cue.save_exclusive(target, "original")
            with self.assertRaises(FileExistsError):
                cue.save_exclusive(target, "replacement")
            self.assertEqual(target.read_text(), "original")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handeye-cue-preflight-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / "repository"
        self.exe = self.repo / "out/build/windows-debug/core_engine.exe"
        self.exe.parent.mkdir(parents=True)
        self.exe.write_bytes(b"inert test fixture; never executed")
        self.output = self.directory / "experiments"
        self.output.mkdir()
        self.settings = self.directory / "os_integration.ini"
        self.settings.write_text("saved settings fixture", encoding="utf-8")
        self.branch = "codex/engineer4-dsp-recorder"
        self.executable_hash = VERIFIED_HASH

    def git_value(self, repo, arguments):
        self.assertEqual(Path(repo).resolve(), self.repo.resolve())
        if "--show-toplevel" in arguments:
            return str(self.repo.resolve())
        if "--show-current" in arguments or "--abbrev-ref" in arguments:
            return self.branch
        if "HEAD" in arguments:
            return "6354a143964d7f4aa8f1957bed0a2eed59e37341"
        return ""

    def file_hash(self, path):
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        return self.executable_hash if target.resolve() == self.exe.resolve() else "settings-hash"

    @contextlib.contextmanager
    def isolated(self, processes=()):
        with (mock.patch.object(cue, "IS_WINDOWS", True),
              mock.patch.object(cue, "git_value", side_effect=self.git_value),
              mock.patch.object(cue, "running_engines", return_value=list(processes)),
              mock.patch.object(cue, "sha256", side_effect=self.file_hash),
              mock.patch.object(cue.subprocess, "Popen") as launch):
            yield launch
            launch.assert_not_called()

    def check(self, output=None):
        return cue.preflight(self.repo, self.exe, output or self.output, self.settings)

    def test_verified_configuration_passes_without_engine_launch(self):
        with self.isolated():
            self.check()
        self.assertEqual(self.settings.read_text(), "saved settings fixture")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_shared_or_other_branch_is_refused(self):
        with self.isolated():
            for branch in ("first-full-working-demo", "master", "bassel-development", ""):
                self.branch = branch
                with self.subTest(branch=branch), self.assertRaises((RuntimeError, ValueError)):
                    self.check()

    def test_changed_executable_is_refused(self):
        self.executable_hash = "0" * 64
        with self.isolated(), self.assertRaises((RuntimeError, ValueError)):
            self.check()

    def test_missing_executable_is_refused(self):
        self.exe.unlink()
        with self.isolated(), self.assertRaises((OSError, RuntimeError, ValueError)):
            self.check()

    def test_existing_tracker_is_refused(self):
        with self.isolated(processes=(1234,)), self.assertRaises((RuntimeError, ValueError)):
            self.check()

    def test_missing_settings_are_not_created(self):
        self.settings.unlink()
        with self.isolated(), self.assertRaises((OSError, RuntimeError, ValueError)):
            self.check()
        self.assertFalse(self.settings.exists())

    def test_unreadable_settings_are_refused(self):
        original_hash = self.file_hash

        def unreadable(path):
            if Path(path) == self.settings:
                raise PermissionError("settings unreadable")
            return original_hash(path)

        with self.isolated(), mock.patch.object(cue, "sha256", side_effect=unreadable):
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                self.check()

    def test_output_inside_repository_is_refused(self):
        inside = self.repo / "new-results"
        inside.mkdir()
        with self.isolated(), self.assertRaises((RuntimeError, ValueError)):
            self.check(inside)

    def test_missing_output_parent_is_not_implicitly_created(self):
        absent = self.directory / "missing-output"
        with self.isolated(), self.assertRaises((OSError, RuntimeError, ValueError)):
            self.check(absent)
        self.assertFalse(absent.exists())

    def test_unsupported_platform_is_refused(self):
        with self.isolated(), mock.patch.object(cue, "IS_WINDOWS", False):
            with self.assertRaises((RuntimeError, ValueError)):
                self.check()


class PreparedSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handeye-cue-session-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol = self.directory / "current-protocol.md"
        self.protocol.write_text("fixed test protocol", encoding="utf-8")
        self.protocol_patch = mock.patch.object(cue, "PROTOCOL", self.protocol)
        self.protocol_patch.start()
        self.addCleanup(self.protocol_patch.stop)

    def session(self, mode="fast"):
        session = self.directory / "prepared"
        session.mkdir()
        audio = session / "audio"
        audio.mkdir()
        for key in {c.clip for c in cue.cue_plan(mode)} - {"stop", "move_stop"}:
            cue.write_wave(audio / (key + ".wav"), b"\0\0" * 2205)
        cue.write_wave(audio / "move.wav", bytes(12) + b"\x38\x00" * 10 + bytes(20))
        cue.write_wave(audio / "stop.wav", bytes(30) + b"\x55\x00" * 20 + bytes(16))
        cue.write_wave(audio / "move_stop.wav", cue.movement_wave(audio, mode))
        (session / "PREPARED").write_text("fixture", encoding="utf-8")
        protocol = cue.protocol_file(mode)
        (session / "protocol.md").write_bytes(protocol.read_bytes())
        manifest = {
            "version": cue.SESSION_VERSION, "mode": mode,
            "launcher_sha256": cue.sha256(Path(cue.__file__)),
            "protocol_sha256": cue.sha256(protocol),
            "cues": [cue.asdict(c) for c in cue.cue_plan(mode)],
            "clips": cue.check_audio(audio, mode),
            "spoken_pair": {"rate_hz": 22050, "move_onset_sample": 0,
                            "stop_onset_sample": 11025 if mode == "fast" else 66150},
            "preflight": {"exe": "never-executed.exe"},
            "engine_command": cue.engine_command("never-executed.exe", session / "recording"),
        }
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return session, manifest

    def test_synthetic_wav_manifest_roundtrip_uses_no_audio_or_process(self):
        with (mock.patch.object(cue, "play_clip") as play,
              mock.patch.object(cue.subprocess, "Popen") as process):
            session, manifest = self.session()
            actual_path, actual_manifest = cue.load_session(session)
            self.assertEqual(actual_path, session)
            self.assertEqual(actual_manifest, manifest)
            self.assertFalse((session / "recording").exists())
            play.assert_not_called()
            process.assert_not_called()

    def test_vertical_session_binds_its_own_protocol_and_rehearses_without_tracker(self):
        session, manifest = self.session("vertical-slow")
        self.assertEqual((session / "protocol.md").read_bytes(), cue.VERTICAL_PROTOCOL.read_bytes())
        self.assertNotEqual(manifest["protocol_sha256"], cue.sha256(cue.PROTOCOL))
        clock = FakeClock()
        with (mock.patch.object(cue.time, "monotonic", clock.monotonic),
              mock.patch.object(cue.time, "sleep", clock.sleep),
              mock.patch.object(cue, "play_clip") as play,
              mock.patch.object(cue, "stop_audio"),
              mock.patch.object(cue.subprocess, "Popen") as launch,
              contextlib.redirect_stdout(io.StringIO())):
            cue.rehearse(session)
        launch.assert_not_called()
        self.assertEqual(play.call_args_list[0].args[0].name, "hold_top.wav")
        log = json.loads(next(session.glob("rehearsal-*/cues.json")).read_text())
        self.assertIsNone(log["error"])
        for i in (5,11):
            move, stop = log["events"][i:i+2]
            self.assertEqual(stop["estimated_onset_monotonic"] - move["estimated_onset_monotonic"], 3.0)
        (session / "protocol.md").write_bytes(cue.PROTOCOL.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "protocol changed"):
            cue.load_session(session)

    def test_audio_tampering_is_refused(self):
        session, _ = self.session()
        target = session / "audio/one.wav"
        with target.open("r+b") as stream:
            stream.seek(44)
            stream.write(b"\x01\x00")
        with self.assertRaisesRegex(RuntimeError, "audio changed"):
            cue.load_session(session)

    def test_cue_schedule_tampering_is_refused(self):
        session, manifest = self.session()
        manifest["cues"][5]["at_seconds"] = 8
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cue manifest"):
            cue.load_session(session)

    def test_command_tampering_cannot_enable_mouse_control(self):
        session, manifest = self.session()
        command = manifest["engine_command"]
        command[command.index("log")] = "win32"
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            cue.load_session(session)

    def test_protocol_copy_tampering_is_refused(self):
        session, _ = self.session()
        (session / "protocol.md").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "protocol changed"):
            cue.load_session(session)

    def test_changed_launcher_requires_new_session(self):
        session, manifest = self.session()
        manifest["launcher_sha256"] = "0" * 64
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "launcher changed"):
            cue.load_session(session)

    def test_incomplete_preparation_is_refused(self):
        session, _ = self.session()
        (session / "PREPARED").unlink()
        with self.assertRaisesRegex(RuntimeError, "preparation is incomplete"):
            cue.load_session(session)

    def test_audio_longer_than_fast_deadline_is_refused(self):
        session, _ = self.session()
        target = session / "audio/move.wav"
        target.unlink()
        cue.write_wave(target, b"\x38\0" * cue.RATE)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            cue.check_audio(session / "audio", "fast")

    def test_waveform_word_onsets_preserve_full_words_and_fixed_sample_distance(self):
        session, _ = self.session()
        for mode, distance in (("slow", 66150), ("fast", 11025)):
            samples = memoryview(cue.movement_wave(session / "audio", mode)).cast("h")
            self.assertEqual(list(samples[:10]), [56] * 10)
            self.assertFalse(any(samples[10:distance]))
            self.assertEqual(list(samples[distance:distance + 20]), [85] * 20)
            self.assertFalse(any(samples[distance + 20:]))

    def test_slow_rehearsal_logs_both_spoken_pairs_without_starting_tracker(self):
        session, _ = self.session("slow")
        clock = FakeClock()
        with (mock.patch.object(cue.time, "monotonic", clock.monotonic),
              mock.patch.object(cue.time, "sleep", clock.sleep),
              mock.patch.object(cue, "play_clip") as play,
              mock.patch.object(cue, "stop_audio"),
              mock.patch.object(cue.subprocess, "Popen") as launch,
              contextlib.redirect_stdout(io.StringIO())):
            cue.rehearse(session)
        launch.assert_not_called()
        self.assertEqual(play.call_count, 12)
        log = json.loads(next(session.glob("rehearsal-*/cues.json")).read_text())
        self.assertIsNone(log["error"])
        events = log["events"]
        self.assertEqual(len(events), 14)
        for move, stop in ((events[5], events[6]), (events[11], events[12])):
            self.assertEqual(stop["scheduled_monotonic"] - move["scheduled_monotonic"], 3)
            self.assertEqual(stop["estimated_onset_monotonic"], move["requested_monotonic"] + 3)
            self.assertEqual(stop["kind"], "embedded_word")

    def test_silent_spoken_word_is_refused(self):
        session, _ = self.session()
        word = session / "audio/stop.wav"
        word.unlink()
        cue.write_wave(word, bytes(1000))
        with self.assertRaisesRegex(ValueError, "silent"):
            cue.check_audio(session / "audio", "fast")

    def test_one_sample_shift_in_rendered_pair_is_refused(self):
        session, _ = self.session()
        pair = session / "audio/move_stop.wav"
        pcm, _ = cue.read_wave(pair)
        pair.unlink()
        cue.write_wave(pair, bytes(2) + pcm)
        with self.assertRaisesRegex(RuntimeError, "timed spoken commands"):
            cue.load_session(session)

    def test_audio_preparation_failure_never_launches_engine_or_marks_prepared(self):
        with (mock.patch.object(cue, "preflight", return_value={}),
              mock.patch.object(cue, "render_audio", side_effect=RuntimeError("voice unavailable")),
              mock.patch.object(cue.subprocess, "Popen") as process,
              mock.patch.object(cue, "play_clip") as play):
            with self.assertRaisesRegex(RuntimeError, "voice unavailable"):
                cue.prepare_session("slow", self.directory, "repo", "exe", "settings")
            process.assert_not_called()
            play.assert_not_called()
        sessions = list(self.directory.glob("Engineer4-HandAudio-slow-*"))
        self.assertEqual(len(sessions), 1)
        self.assertTrue((sessions[0] / "PREPARING").is_file())
        self.assertFalse((sessions[0] / "PREPARED").exists())
        self.assertFalse((sessions[0] / "recording").exists())


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ImmediateReader:
    """Runs a scripted, finite stdout source without a nondeterministic thread."""
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class ScriptedOutput:
    def __init__(self, clock, events):
        self.clock = clock
        self.events = list(events)
        self.closed = False

    def read(self, count):
        if not self.events:
            return b""
        at, text = self.events.pop(0)
        self.clock.now = max(self.clock.now, at)
        return text.encode("utf-8")

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, clock, events, end=50.0, exit_code=0):
        self.clock = clock
        self.stdout = ScriptedOutput(clock, events)
        self.end = end
        self.exit_code = exit_code
        self.returncode = None
        self.signals = []
        self.killed = False

    def poll(self):
        if self.returncode is None and self.clock.now >= self.end:
            self.returncode = self.exit_code
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -1


class RecordLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handeye-cue-lifecycle-")
        self.addCleanup(self.temporary.cleanup)
        self.session = Path(self.temporary.name)
        self.settings = self.session / "existing-settings.ini"
        self.settings.write_text("unchanged settings fixture", encoding="utf-8")
        self.before = {"repo": "inert-repository", "exe": "inert-engine.exe",
                       "settings": str(self.settings), "settings_sha256": cue.sha256(self.settings),
                       "commit": "fixed-commit"}
        self.manifest = {"mode": "fast", "preflight": self.before,
                         "spoken_pair": {"rate_hz": 22050, "move_onset_sample": 0, "stop_onset_sample": 11025},
                         "clips": {c.clip: {"seconds": 0.12} for c in cue.cue_plan("fast")}}
        self.clock = FakeClock()

    @contextlib.contextmanager
    def isolated(self, *, events=None, end=50.0, answer="RECORD", stop_error=None,
                 play_error=None, verify_error=None):
        if events is None:
            events = [(1.0, status(frames=20) + "\n"),
                      (2.0, status(frames=40) + "\n")]
        process = FakeProcess(self.clock, events, end=end)
        with (mock.patch.object(cue, "load_session", return_value=(self.session, self.manifest)),
              mock.patch.object(cue, "preflight", return_value=self.before),
              mock.patch("builtins.input", return_value=answer),
              mock.patch.object(sys.stdin, "isatty", return_value=True),
              mock.patch.object(cue.subprocess, "Popen", return_value=process) as launch,
              mock.patch.object(cue.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, create=True),
              mock.patch.object(cue.threading, "Thread", ImmediateReader),
              mock.patch.object(cue.time, "monotonic", self.clock.monotonic),
              mock.patch.object(cue.time, "sleep", self.clock.sleep),
              mock.patch.object(cue, "play_clip", side_effect=play_error) as play,
              mock.patch.object(cue, "stop_audio", side_effect=stop_error),
              mock.patch.object(cue, "verify_recording", side_effect=verify_error,
                                return_value={"fixture": "offline verifier stand-in"}) as verify,
              contextlib.redirect_stdout(io.StringIO())):
            yield process, launch, play, verify

    def test_confirmation_refusal_cannot_launch_or_create_attempt(self):
        with self.isolated(answer="cancel") as (_, launch, play, verify):
            cue.record(self.session)
            launch.assert_not_called()
            play.assert_not_called()
            verify.assert_not_called()
        self.assertFalse((self.session / "LAUNCH_ATTEMPT.json").exists())

    def test_used_session_cannot_launch_or_overwrite_original(self):
        original = self.session / "recording"
        original.mkdir()
        trace = original / "trace.txt"
        trace.write_text("preserved experiment", encoding="utf-8")
        with self.isolated() as (_, launch, play, _):
            with self.assertRaisesRegex(RuntimeError, "already used"):
                cue.record(self.session)
            launch.assert_not_called()
            play.assert_not_called()
        self.assertEqual(trace.read_text(), "preserved experiment")

    def test_successful_fake_run_records_every_cue_without_devices(self):
        with self.isolated() as (process, launch, play, verify):
            cue.record(self.session)
            launch.assert_called_once()
            self.assertEqual(play.call_count, 12)
            paths = [call.args[0].name for call in play.call_args_list]
            self.assertEqual(paths.count("move_stop.wav"), 2)
            self.assertNotIn("stop.wav", paths)
            verify.assert_called_once()
            self.assertFalse(process.killed)
        events = json.loads((self.session / "cues.json").read_text())["events"]
        self.assertEqual([e["event"] for e in events[:2]], ["first_status", "ready"])
        self.assertEqual(events[0]["received_monotonic"], 1.0)
        self.assertEqual(events[1]["received_monotonic"], 2.0)
        self.assertEqual(events[-1]["event"], "process_end")
        cues = [event for event in events if event["event"] == "cue"]
        self.assertEqual([event["cue"]["at_seconds"] for event in cues],
                         [0, 5, 8, 9, 10, 11, 11.5, 19.5, 22.5, 23.5,
                          24.5, 25.5, 26, 34])
        self.assertTrue(all(0 <= event["lateness_seconds"] <= .01 for event in cues))
        for move, stop in ((cues[5], cues[6]), (cues[11], cues[12])):
            self.assertEqual(stop["kind"], "embedded_word")
            self.assertNotIn("requested_monotonic", stop)
            self.assertEqual(stop["estimated_onset_monotonic"], move["requested_monotonic"] + .5)
        result = json.loads((self.session / "session-result.json").read_text())
        self.assertTrue(result["cue_and_recording_integrity_pass"])
        self.assertEqual(result["responsiveness_acceptance"], "NOT EVALUATED")
        self.assertTrue((self.session / "SESSION_READY_FOR_ANALYSIS").is_file())

    def test_early_engine_exit_stops_future_cues_and_invalidates_session(self):
        with self.isolated(end=3.0) as (_, _, play, verify):
            with self.assertRaisesRegex(RuntimeError, "before cue sequence"):
                cue.record(self.session)
            self.assertEqual(play.call_count, 1)
            verify.assert_not_called()
        self.assertTrue((self.session / "SESSION_INVALID").is_file())

    def test_piped_confirmation_cannot_launch(self):
        with self.isolated() as (_, launch, play, _), mock.patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "interactive"):
                cue.record(self.session)
            launch.assert_not_called()
            play.assert_not_called()

    def test_first_status_after_deadline_cannot_start_cues(self):
        with self.isolated(events=[(31, status(frames=20) + "\n"), (32, status(frames=40) + "\n")]) as (_, _, play, _):
            with self.assertRaisesRegex(RuntimeError, "readiness"):
                cue.record(self.session)
            play.assert_not_called()

    def test_console_save_error_retains_invalid_marker_and_cues(self):
        (self.session / "console.txt").write_text("original", encoding="utf-8")
        failure = cue.persist_session(self.session, "new output", {}, None, None)
        self.assertIn("console.txt", failure)
        self.assertEqual((self.session / "console.txt").read_text(), "original")
        self.assertTrue((self.session / "SESSION_INVALID").is_file())
        self.assertTrue((self.session / "cues.json").is_file())

    def test_missing_readiness_withholds_all_cues_and_stops_owned_process(self):
        with self.isolated(events=[]) as (process, _, play, verify):
            with self.assertRaisesRegex(RuntimeError, "readiness"):
                cue.record(self.session)
            play.assert_not_called()
            verify.assert_not_called()
            self.assertTrue(process.signals)
        self.assertTrue((self.session / "SESSION_INVALID").is_file())

    def test_audio_start_failure_stops_owned_process(self):
        with self.isolated(play_error=RuntimeError("simulated audio failure")) as (process, _, _, verify):
            with self.assertRaisesRegex(RuntimeError, "simulated audio failure"):
                cue.record(self.session)
            self.assertTrue(process.signals)
            verify.assert_not_called()
        self.assertTrue((self.session / "SESSION_INVALID").is_file())

    def test_audio_stop_failure_cannot_skip_owned_process_cleanup(self):
        with self.isolated(play_error=RuntimeError("audio start failed"),
                           stop_error=RuntimeError("audio stop failed")) as (process, _, _, verify):
            with self.assertRaises(RuntimeError):
                cue.record(self.session)
            self.assertTrue(process.signals, "audio stop error must not strand the camera process")
            verify.assert_not_called()
        self.assertTrue((self.session / "SESSION_INVALID").is_file())

    def test_failed_trace_verification_cannot_mark_session_ready(self):
        with self.isolated(verify_error=RuntimeError("trace incomplete")):
            with self.assertRaisesRegex(RuntimeError, "trace incomplete"):
                cue.record(self.session)
        self.assertTrue((self.session / "SESSION_INVALID").is_file())
        self.assertFalse((self.session / "SESSION_READY_FOR_ANALYSIS").exists())


class SafeCliTests(unittest.TestCase):
    def test_plan_never_launches_a_process(self):
        with (mock.patch.object(cue.subprocess, "Popen") as launch,
              mock.patch.object(cue.subprocess, "run") as external,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(cue.main(["plan", "--mode", "slow"]), 0)
            launch.assert_not_called()
            external.assert_not_called()

    def test_help_never_launches_a_process(self):
        with (mock.patch.object(cue.subprocess, "Popen") as launch,
              mock.patch.object(cue.subprocess, "run") as external,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(SystemExit) as result:
                cue.main(["--help"])
            self.assertEqual(result.exception.code, 0)
            launch.assert_not_called()
            external.assert_not_called()


if __name__ == "__main__":
    unittest.main()
