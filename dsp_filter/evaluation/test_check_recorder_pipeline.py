"""Verifier guard tests. Every engine invocation is mocked; no device or tray runs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import check_recorder_pipeline as pipeline


class EngineGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handeye-verifier-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.settings = self.directory / "settings.ini"
        self.original = b"hand_filter.min_cutoff_hz = 1.2\n"
        self.settings.write_bytes(self.original)
        self.expected = pipeline.digest(self.settings)
        self.command = ["core_engine.exe", "--video", "synthetic.avi",
                        "--filter", "dsp", "--os", "log"]

    def run_engine(self, name="run", command=None):
        return pipeline.run_engine(self.command if command is None else command,
                                   self.directory, name, self.settings, self.expected)

    def test_missing_settings_refuses_engine_launch(self):
        self.settings.unlink()
        with mock.patch.object(pipeline.subprocess, "run") as launch:
            with self.assertRaisesRegex(RuntimeError, "existing readable saved tray settings"):
                self.run_engine()
            launch.assert_not_called()
        self.assertFalse(self.settings.exists())

    def test_main_checks_settings_before_creating_fixtures(self):
        executable = self.directory / "core_engine.exe"
        executable.write_bytes(b"not executed")
        with (mock.patch.object(sys, "argv", ["check", "--exe", str(executable)]),
              mock.patch.dict(pipeline.os.environ, {"APPDATA": str(self.directory)}),
              mock.patch.object(pipeline.tempfile, "mkdtemp") as create_output,
              mock.patch.object(pipeline.subprocess, "run") as launch):
            with self.assertRaisesRegex(RuntimeError, "existing readable saved tray settings"):
                pipeline.main()
            create_output.assert_not_called()
            launch.assert_not_called()

    def test_unreadable_settings_refuses_engine_launch(self):
        with (mock.patch.object(pipeline, "digest", side_effect=PermissionError("denied")),
              mock.patch.object(pipeline.subprocess, "run") as launch):
            with self.assertRaisesRegex(RuntimeError, "existing readable saved tray settings"):
                self.run_engine()
            launch.assert_not_called()

    def test_safe_command_preserves_settings_and_saves_console_logs(self):
        completed = subprocess.CompletedProcess(self.command, 0, "engine output\n", "diagnostic\n")
        with mock.patch.object(pipeline.subprocess, "run", return_value=completed) as launch:
            self.assertIs(self.run_engine(), completed)
            launch.assert_called_once_with(self.command, capture_output=True, text=True, timeout=120)
        self.assertEqual(self.settings.read_bytes(), self.original)
        self.assertEqual((self.directory / "run.stdout.txt").read_text(), completed.stdout)
        self.assertEqual((self.directory / "run.stderr.txt").read_text(), completed.stderr)
        self.assertEqual(json.loads((self.directory / "run.command.json").read_text()), self.command)

    def test_camera_real_os_and_ambiguous_options_are_refused(self):
        unsafe = [
            self.command + ["--camera", "0"],
            self.command + ["--control"],
            [*self.command[:-1], "win32"],
            self.command[0:1] + self.command[3:],
            [*self.command[:2], "", *self.command[3:]],
            self.command + ["--os", "win32"],
            self.command + ["--video", "other.avi"],
            self.command[:-1],
            [*self.command[:4], "builtin", *self.command[5:]],
        ]
        with mock.patch.object(pipeline.subprocess, "run") as launch:
            for command in unsafe:
                with self.subTest(command=command):
                    with self.assertRaisesRegex(ValueError, "only synthetic video, DSP, and dry-run OS"):
                        self.run_engine(command=command)
            launch.assert_not_called()

    def test_settings_change_is_caught_on_success_and_nonzero_exit(self):
        for returncode in (0, 4):
            with self.subTest(returncode=returncode):
                self.settings.write_bytes(self.original)

                def changed(*args, **kwargs):
                    self.settings.write_bytes(b"changed by the simulated tray\n")
                    return subprocess.CompletedProcess(self.command, returncode, "output", "error")

                with mock.patch.object(pipeline.subprocess, "run", side_effect=changed) as launch:
                    with self.assertRaisesRegex(RuntimeError, "saved tray settings changed"):
                        self.run_engine(name=f"changed-{returncode}")
                    launch.assert_called_once()
                self.assertEqual(self.settings.read_bytes(), b"changed by the simulated tray\n")
                with mock.patch.object(pipeline.subprocess, "run") as subsequent_launch:
                    with self.assertRaisesRegex(RuntimeError, "saved tray settings changed"):
                        self.run_engine(name="must-not-launch")
                    subsequent_launch.assert_not_called()

    def test_settings_change_is_caught_when_subprocess_raises(self):
        def changed_then_timeout(*args, **kwargs):
            self.settings.write_bytes(b"changed before timeout\n")
            raise subprocess.TimeoutExpired(self.command, 120)

        with mock.patch.object(pipeline.subprocess, "run", side_effect=changed_then_timeout):
            with self.assertRaisesRegex(RuntimeError, "saved tray settings changed"):
                self.run_engine()
        self.assertEqual(self.settings.read_bytes(), b"changed before timeout\n")

    def test_unchanged_settings_preserve_subprocess_failure(self):
        with mock.patch.object(pipeline.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired(self.command, 120)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.run_engine()
        self.assertEqual(self.settings.read_bytes(), self.original)

    def test_deleted_settings_are_detected_and_not_recreated(self):
        def deleted(*args, **kwargs):
            self.settings.unlink()
            return subprocess.CompletedProcess(self.command, 0, "output", "")

        with mock.patch.object(pipeline.subprocess, "run", side_effect=deleted):
            with self.assertRaisesRegex(RuntimeError, "saved tray settings changed"):
                self.run_engine()
        self.assertFalse(self.settings.exists())


def shutdown_text(captured=329, processed=320, evicted=3, skipped=1, stale=5,
                  errors=0, submitted=320, pool="8/8 (no leaks)"):
    return (f"captured {captured} | processed {processed} | evicted {evicted} | skipped {skipped}"
            f" | stale {stale} | python errors {errors}\n"
            f"os stage: {submitted} frames submitted\npool slots returned: {pool}\n")


class ShutdownAccountingTests(unittest.TestCase):
    def test_full_video_allows_accounted_queue_drops(self):
        self.assertEqual(pipeline.check_shutdown_summary(shutdown_text(), "hand", 330),
                         (329, 320, 320, 5, 0))

    def test_early_capture_fails_even_with_consistent_accounting(self):
        early = shutdown_text(captured=100, processed=100, evicted=0, skipped=0, stale=0,
                              submitted=100)
        with self.assertRaisesRegex(RuntimeError, "complete synthetic video"):
            pipeline.check_shutdown_summary(early, "hand", 330)

    def test_bad_accounting_or_bridge_errors_fail(self):
        for changes in ({"processed": 319}, {"submitted": 319}, {"errors": 1}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(RuntimeError, "frame accounting or bridge errors"):
                    pipeline.check_shutdown_summary(shutdown_text(**changes), "gaze", 330)

    def test_missing_summary_or_pool_leak_fails(self):
        with self.assertRaisesRegex(RuntimeError, "missing real engine shutdown summary"):
            pipeline.check_shutdown_summary("", "hand", 330)
        with self.assertRaisesRegex(RuntimeError, "frame-pool return failed"):
            pipeline.check_shutdown_summary(shutdown_text(pool="7/8 (LEAK!)"), "hand", 330)


if __name__ == "__main__":
    unittest.main()
