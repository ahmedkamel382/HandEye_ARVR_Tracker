"""Deterministic importer fixtures. No camera or debugger required."""

import csv
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from import_vs_trace import COLUMNS, RECORDER_COLUMNS, columns_for, convert


def detector(frame=5, ms=132, cap=132, n=42, channel="hand"):
    return f"E4DSP N {channel} frame={frame} ms={ms} cap={cap} n={n} other=false"


def published(frame=5, ms=132, channel="hand", x=0.5, y=0.25, fx=0.5, fy=0.25,
              state=0, present=True):
    return (f"E4DSP P {channel} frame={frame} ms={ms} ok=true x={x} y={y} s={state} "
            f"pub={int(present)} fx={fx} fy={fy}")


def adapter(t="0.13200000000000001", x=0.5, y=0.25, status=0, dx=None, dy=None, has=None):
    if has is None:
        has = status in (0, 1, 2)
    if dx is None:
        dx = x if has else 0
    if dy is None:
        dy = y if has else 0
    return f"E4DSP A t={t} rx={x} ry={y} status={status} has={int(has)} dx={dx} dy={dy}"


class TraceImportTests(unittest.TestCase):
    def test_single_channel_interleaved_with_dropped_camera_frames(self):
        lines = [
            "background debugger chatter",
            detector(), detector(7, 199, 199, 0), adapter(),
            published(), adapter(t="0.19900000000000001", status=1),
            detector(13, 333, 333, 0), published(7, 199),
            published(13, 333, x=-1, y=-1, present=False),
            "os stage: 3 frames submitted",
        ]
        rows = convert(lines, "hand")
        self.assertEqual(rows, [
            (5, "hand", 132, 132, 42, 0, 1, .5, .25, .5, .25, "Initialized"),
            (7, "hand", 199, 199, 0, 0, 1, .5, .25, .5, .25, "Filtered"),
            (13, "hand", 333, 333, 0, 0, 0, "", "", "", "", "Missing"),
        ])

    def test_corrected_time_and_calibrating_gaze_are_preserved(self):
        rows = convert([
            detector(n=484, channel="gaze"), adapter(),
            published(channel="gaze", state=-1), detector(6, 133, 132, 484, "gaze"),
            published(6, 133, "gaze", x=.75, fx=.625, state=0),
            adapter(t=.133, x=.75, status=1, dx=.625),
        ], "gaze", 2)
        self.assertEqual(rows[0][4:7], (484, -1, 1))
        self.assertEqual(rows[1], (6, "gaze", 133, 132, 484, 0, 1, .75, .25, .625, .25, "Filtered"))

    def test_input_crosscheck_clamps_like_adapter(self):
        rows = convert([detector(), adapter(x=1), published(x=1.25, fx=1)], "hand", 1)
        self.assertEqual(rows[0][7], 1)
        with self.assertRaisesRegex(ValueError, "upstream coordinates disagree"):
            convert([detector(), adapter(x=.75), published()], "hand", 1)

    def test_float32_upstream_and_full_precision_adapter(self):
        rows = convert([detector(), adapter(x="0.30000001192092896"),
                        published(x="0.300000012", fx="0.30000001192092896")], "hand", 1)
        self.assertEqual(rows[0][7], 0.30000001192092896)

    def test_keeps_fallback_status_and_distinct_published_point(self):
        for code, status in [(5, "RejectedTimestamp"), (6, "NumericalError")]:
            with self.subTest(status=status):
                rows = convert([detector(), adapter(status=code), published(fx=.75)], "hand", 1)
                self.assertEqual(rows[0][-6:], (1, .5, .25, .75, .25, status))

    def test_filtered_publication_must_equal_actual_dsp_output(self):
        with self.assertRaisesRegex(ValueError, "accepted DSP output differs"):
            convert([detector(), adapter(status=1, dx=.375), published(fx=.5)], "hand", 1)

    def test_optional_point_presence_must_agree_with_status(self):
        for status, has in [(0, False), (1, False), (2, False), (5, True), (6, True)]:
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "optional-point presence"):
                convert([detector(), adapter(status=status, has=has), published()], "hand", 1)

    def test_preserves_nonfinite_bypass_without_claiming_dsp_input(self):
        rows = convert([detector(), published(x="inf", fx=1)], "hand", 1)
        self.assertEqual(rows[0][-6:], (1, "", "", 1, .25, "Bypassed"))

    def test_detects_mismatched_or_duplicate_records(self):
        specs = [
            ([detector(), adapter(), published().replace("ms=132", "ms=131")], "timestamps"),
            ([detector(), adapter(t=.131), published()], "matching adapter"),
            ([detector(), adapter(), published(), adapter(t=.134)], "unpaired adapter"),
            ([detector(), adapter(), published(), published()], "duplicate P"),
            ([detector(), adapter(), adapter(), published()], "duplicate A"),
            ([detector(), detector(), adapter(), published()], "duplicate N"),
            ([detector(), published(x=-1, y=-1, present=False), adapter()], "for missing point"),
            ([detector(), adapter(dx=.75), published(fx=.75)], "initialized coordinates"),
            ([detector(n=-1), adapter(), published()], "detector count"),
            ([detector(), adapter(), published(x=-1, y=-1)], "payload presence"),
            ([detector(), adapter(), published(fx="nan")], "unit square"),
        ]
        for lines, message in specs:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                convert(lines, "hand", 1)

    def test_rejects_other_channel_or_malformed_event(self):
        for line, message in [
            (detector(channel="gaze"), "expected hand"),
            (detector().replace("other=false", "other=true"), "other tracker was enabled"),
            (detector() + " n=42", "duplicate fields"),
            (detector().replace(" n=42", ""), "incorrect"),
            (detector().replace("n=42", "n={n}"), "invalid literal"),
            (detector().replace("other=false", "other=unknown"), "boolean"),
            ("E4DSP unexpected", "incomplete"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                convert([line, adapter(), published()], "hand", 1)

    def test_requires_complete_capture(self):
        records = [detector(), adapter(), published()]
        for lines, count, message in [
            (records, None, "positive --submitted"),
            (records, 2, "incomplete capture"),
            (records + [detector(99, 700, 700)], 1, "incomplete capture"),
            (records + ["os stage: 2 frames submitted"], 1, "disagrees"),
            (records + ["os stage: 1 frames submitted"] * 2, None, "more than one run"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                convert(lines, "hand", count)

    def test_rejects_unrepresentable_or_ambiguous_time(self):
        first = 2**54
        rows = [detector(ms=first, cap=first), published(ms=first, x=-1, present=False),
                detector(6, first + 1, first + 1, 0), published(6, first + 1, x=-1, present=False)]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            convert(rows, "hand", 2)
        too_large = 2**63
        with self.assertRaisesRegex(ValueError, "timestamps"):
            convert([detector(ms=too_large, cap=too_large), published(ms=too_large)], "hand", 1)

    def test_cli_utf16_and_atomic_validation(self):
        tool = Path(__file__).with_name("import_vs_trace.py")
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "debug.txt"
            trace.write_text("\n".join([detector(), published(), adapter()]), encoding="utf-16")
            cmd = [sys.executable, "-B", str(tool), str(trace), "--channel", "hand", "--submitted"]
            good = subprocess.run(cmd + ["1"], capture_output=True, text=True, check=True)
            rows = list(csv.reader(io.StringIO(good.stdout)))
            self.assertEqual(rows[0], list(COLUMNS))
            self.assertEqual(len(rows), 2)
            bad = subprocess.run(cmd + ["2"], capture_output=True, text=True, check=False)
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(bad.stdout, "")
            self.assertIn("incomplete capture", bad.stderr)


def hand_publication(trk, missed, out_trk, *args, **kwargs):
    return f"E4HAND trk={trk} missed={missed} out_trk={out_trk} " + published(*args, **kwargs)


class HandFreshnessImportTests(unittest.TestCase):
    def setUp(self):
        self.fresh = [
            detector(), adapter(), hand_publication(0, 0, 0),
            detector(6, 199, 199), adapter(t=.199, x=.75, status=1, dx=.625),
            hand_publication(0, 0, 0, 6, 199, x=.75, fx=.625),
        ]
        self.miss = detector(7, 266, 266, 0)
        self.replay = hand_publication(1, 1, 1, 7, 266, x=.75, fx=.625)

    def test_six_cached_holds_then_lost_and_reacquired(self):
        events = self.fresh.copy()
        for missed in range(1, 7):
            frame, ms = 6 + missed, 199 + 67 * missed
            events.extend([detector(frame, ms, ms, 0),
                           hand_publication(1, missed, 1, frame, ms, x=.75, fx=.625)])
        events.extend([detector(13, 668, 668, 0),
                       hand_publication(2, 7, 2, 13, 668, x=-1, y=-1, present=False),
                       detector(14, 735, 735), hand_publication(0, 0, 0, 14, 735, x=0, fx=0),
                       adapter(t=.735, x=0)])
        rows = convert(events, "hand", 10)
        self.assertEqual(rows[1][6:], (1, .75, .25, .625, .25, "Filtered", "Tracked", 0, "Tracked"))
        for missed, result in enumerate(rows[2:8], 1):
            self.assertEqual(result[6:], (1, "", "", .625, .25, "Held", "Held", missed, "Held"))
        self.assertEqual(rows[-2][6:], (0, "", "", "", "", "Lost", "Lost", 7, "Lost"))
        self.assertEqual(rows[-1][6:], (1, 0., .25, 0., .25, "Initialized", "Tracked", 0, "Tracked"))

    def test_held_without_cache_after_error(self):
        # The bridge error clears the gate. HandNative can still remember its previous track.
        error = hand_publication(2, 0, 2, 6, 199, x=-1, y=-1, present=False).replace("ok=true", "ok=false")
        events = self.fresh[:3] + [detector(6, 199, 199, -1), error, self.miss,
                                   hand_publication(1, 1, 2, 7, 266, x=.5, present=False)]
        rows = convert(events, "hand", 3)
        self.assertEqual(rows[1][-4:], ("Lost", "Lost", 0, "Lost"))
        self.assertEqual(rows[2][6:], (0, "", "", "", "", "HeldNoCache", "Held", 1, "Lost"))
        with self.assertRaises(ValueError):
            convert(events[:-1] + [hand_publication(1, 1, 1, 7, 266)], "hand", 3)

    def test_no_cache_after_settings_replacement_cannot_resurrect_old_point(self):
        # The gate's settings replacement is not a tracker observation or an A record.
        empty = hand_publication(1, 1, 2, 7, 266, x=.75, present=False)
        events = self.fresh + [self.miss, empty]
        self.assertEqual(convert(events, "hand", 3)[-1][-4:], ("HeldNoCache", "Held", 1, "Lost"))
        stale = hand_publication(1, 2, 1, 8, 333, x=.75, fx=.625)
        with self.assertRaises(ValueError):
            convert(events + [detector(8, 333, 333, 0), stale], "hand", 4)

    def test_gate_caches_actual_tracked_publication_even_after_dsp_error(self):
        events = self.fresh[:3] + [
            detector(6, 199, 199), adapter(t=.199, x=.75, status=6),
            hand_publication(0, 0, 0, 6, 199, x=.75, fx=.75),
            self.miss, hand_publication(1, 1, 1, 7, 266, x=.75, fx=.75),
        ]
        rows = convert(events, "hand", 3)
        self.assertEqual(rows[-2][-4:], ("NumericalError", "Tracked", 0, "Tracked"))
        self.assertEqual(rows[-1][6:], (1, "", "", .75, .25, "Held", "Held", 1, "Held"))

    def test_rejects_changed_held_coordinates_or_extra_adapter(self):
        for bad in (self.replay.replace(" x=0.75", " x=0.25"),
                    self.replay.replace("fx=0.625", "fx=0.75"),
                    self.replay.replace("fy=0.25", "fy=0.25000000000000006"),
                    self.replay.replace("fx=0.625", "fx=nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert(self.fresh + [self.miss, bad], "hand", 3)
        with self.assertRaises(ValueError):
            convert(self.fresh + [self.miss, adapter(t=.266, x=.75, status=1, dx=.625),
                                  self.replay], "hand", 3)

    def test_fresh_repeated_coordinates_require_an_adapter_call(self):
        same = hand_publication(0, 0, 0, 7, 266, x=.75, fx=.6875)
        events = self.fresh + [detector(7, 266, 266), same]
        with self.assertRaises(ValueError):
            convert(events, "hand", 3)
        rows = convert(events + [adapter(t=.266, x=.75, status=1, dx=.6875)], "hand", 3)
        self.assertEqual(rows[-1][-4:], ("Filtered", "Tracked", 0, "Tracked"))

    def test_detector_and_gate_must_agree_with_tagged_state(self):
        for n, publication in [
            (42, self.replay), (0, self.replay.replace("missed=1", "missed=0")),
            (0, self.replay.replace("out_trk=1", "out_trk=0")),
            (0, self.replay.replace("pub=1", "pub=0")),
            (0, hand_publication(1, 1, 2, 7, 266)),
            (0, hand_publication(0, 0, 0, 7, 266)),
            (0, hand_publication(2, 7, 2, 7, 266, x=-1, y=-1, present=True)),
        ]:
            with self.subTest(n=n, p=publication), self.assertRaises(ValueError):
                convert(self.fresh + [detector(7, 266, 266, n), publication], "hand", 3)

    def test_mixed_legacy_and_prefixed_publications_are_rejected(self):
        for legacy_index in (2, 5):
            events = self.fresh.copy()
            marker = events[legacy_index].index("E4DSP P")
            events[legacy_index] = events[legacy_index][marker:]
            with self.subTest(legacy_index=legacy_index), self.assertRaises(ValueError):
                convert(events, "hand", 2)

    def test_rejects_bad_freshness_prefix(self):
        for prefix in ("E4HAND trk=0 out_trk=0", "E4HAND trk=0 missed=0 out_trk=0 trk=0",
                       "E4HAND trk=9 missed=0 out_trk=0", "E4HAND trk=0 missed=-1 out_trk=0",
                       "E4HAND trk=0 missed=0 out_trk=9", "E4HAND trk={expr} missed=0 out_trk=0"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                convert([detector(), adapter(), prefix + " " + published()], "hand", 1)
        with self.assertRaises(ValueError):
            convert([detector(channel="gaze", n=484), adapter(),
                     hand_publication(0, 0, 0, channel="gaze")], "gaze", 1)

    def test_tagged_pairing_still_checks_frame_timestamp_and_uniqueness(self):
        for records in ([self.miss, self.replay.replace("frame=7", "frame=8")],
                        [self.miss, self.replay.replace("ms=266", "ms=267")],
                        [self.miss, self.replay, self.replay]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                convert(self.fresh + records, "hand", 3)

    def test_cli_writes_extended_header_and_empty_held_inputs(self):
        tool = Path(__file__).with_name("import_vs_trace.py")
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "hand.txt"
            trace.write_text("\n".join(self.fresh + [self.miss, self.replay]), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(tool), str(trace),
                                     "--channel", "hand", "--submitted", "3"],
                                    capture_output=True, text=True, check=True)
        reader = csv.DictReader(io.StringIO(result.stdout))
        self.assertEqual(reader.fieldnames, [*COLUMNS, "tracking_state", "missed_frames", "published_tracking"])
        held = list(reader)[-1]
        self.assertEqual((held["raw_x"], held["raw_y"]), ("", ""))
        self.assertEqual((held["status"], held["published_tracking"], held["filtered_x"]), ("Held", "Held", "0.625"))


def recorder_frame(frame=0, t=.100123456789, **changes):
    cap = int(t * 1000) if math.isfinite(t) else 0
    fields = dict(frame=frame, ms=cap, cap=cap, t=t, n=42, ok=1, x=.5, y=.25,
                  state=0, trk=0, missed=0, adapter=1, called=1, input_t=t,
                  ix=.5, iy=.25, status=0, has=1, dx=.5, dy=.25, outcome=0,
                  resets=0, pub=1, px=.5, py=.25, out_trk=0, generation=1)
    fields.update(changes)
    return "E4REC F " + " ".join(f"{k}={v}" for k, v in fields.items())


def recorder_absent(frame, t, **changes):
    fields = dict(n=0, x=-1, y=-1, trk=2, adapter=0, called=0, input_t=0,
                  ix=0, iy=0, status=-1, has=0, dx=0, dy=0, outcome=-1,
                  pub=0, px=0, py=0, out_trk=2)
    fields.update(changes)
    return recorder_frame(frame, t, **fields)


def recorder_file(frames, channel="hand", **end_changes):
    end = dict(submitted=len(frames), attempted=len(frames), stored=len(frames),
               overflow=0, config_changes=0, complete=1)
    end.update(end_changes)
    return [f"E4REC BEGIN version=2 channel={channel} capacity=4096 min_cutoff=1.2 "
            "beta=4 derivative_cutoff=1 min_dt=1e-6 reset_gap=0.25", *frames,
            "E4REC END " + " ".join(f"{k}={v}" for k, v in end.items())]


class RecorderImportTests(unittest.TestCase):
    def test_precise_capture_clock_and_actual_dsp_result_are_preserved(self):
        t = .10012345678912345
        rows = convert(recorder_file([recorder_frame(t=t)]), "hand")
        self.assertEqual(columns_for(rows), RECORDER_COLUMNS)
        r = dict(zip(RECORDER_COLUMNS, rows[0]))
        self.assertEqual((r["observation_ms"], r["capture_ms"]), (100, 100))
        self.assertEqual(r["dsp_timestamp_seconds"], t)
        self.assertNotEqual(r["dsp_timestamp_seconds"], r["observation_ms"] / 1000)
        self.assertEqual((r["dsp_x"], r["dsp_y"]), (.5, .25))
        self.assertEqual(r["adapter_outcome"], "Ok")

    def test_submillisecond_and_duplicate_timestamps_pair_by_frame(self):
        t = .1320004
        records = [recorder_frame(t=t),
                   recorder_frame(1, t, ms=133, status=5, has=0, dx=0, dy=0,
                                  outcome=1, out_trk=1),
                   recorder_frame(2, .1320009, ms=134, status=5, has=0, dx=0, dy=0,
                                  outcome=1, out_trk=1),
                   recorder_frame(3, .132002, ms=135, status=1)]
        rows = convert(recorder_file(records), "hand")
        self.assertEqual([r[11] for r in rows],
                         ["Initialized", "RejectedTimestamp", "RejectedTimestamp", "Filtered"])
        self.assertEqual(rows[1][15], rows[0][15])
        self.assertEqual(rows[1][-2:], ("", ""))
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            convert(recorder_file([records[0], recorder_frame(1, t, ms=133, status=1)]), "hand")

    def test_gap_reinitialization_uses_precise_timing(self):
        rows = convert(recorder_file([recorder_frame(), recorder_frame(1, .5, status=2)]), "hand")
        self.assertEqual(rows[-1][11], "ReinitializedAfterGap")
        with self.assertRaisesRegex(ValueError, "gap reinitialization"):
            convert(recorder_file([recorder_frame(), recorder_frame(1, .2, status=2)]), "hand")

    def test_held_publication_uses_cached_filtered_output(self):
        records = [recorder_frame(),
                   recorder_frame(1, .2, x=.75, ix=.75, status=1, dx=.625, px=.625),
                   recorder_absent(2, .3, trk=1, missed=1, x=.75, y=.25,
                                   pub=1, px=.625, py=.25, out_trk=1)]
        rows = convert(recorder_file(records), "hand")
        self.assertEqual(rows[-1][6:15], (1, "", "", .625, .25, "Held", "Held", 1, "Held"))
        self.assertEqual(rows[-1][16:19], (0, 0, "NotCalled"))
        for bad in (records[-1].replace("px=0.625", "px=0.75"),
                    records[-1].replace("x=0.75", "x=0.5")):
            with self.assertRaises(ValueError):
                convert(recorder_file(records[:-1] + [bad]), "hand")

    def test_bridge_error_clears_cache_and_retained_hand_has_no_publication(self):
        rows = convert(recorder_file([
            recorder_frame(), recorder_absent(1, .2, n=-1, ok=0, resets=1),
            recorder_absent(2, .3, trk=1, missed=1, x=.5, y=.25),
            recorder_frame(3, .4, x=0, ix=0, dx=0, px=0),
        ]), "hand")
        self.assertEqual([r[11] for r in rows], ["Initialized", "Lost", "HeldNoCache", "Initialized"])
        self.assertEqual(rows[-1][7], 0)

    def test_confirmed_loss_then_different_position_reinitializes_directly(self):
        rows = convert(recorder_file([
            recorder_frame(), recorder_absent(1, .2, missed=7, resets=1),
            recorder_frame(2, .3, x=1, y=0, ix=1, iy=0, dx=1, dy=0, px=1, py=0),
        ]), "hand")
        self.assertEqual(rows[-1][7:11], (1, 0, 1, 0))

    def test_loss_reset_is_observed_and_held_never_resets(self):
        with self.assertRaisesRegex(ValueError, "gate's reset"):
            convert(recorder_file([recorder_frame(), recorder_absent(1, .2)]), "hand")
        held = recorder_absent(1, .2, trk=1, missed=1, x=.5, y=.25,
                               pub=1, px=.5, py=.25, out_trk=1, resets=1)
        with self.assertRaisesRegex(ValueError, "retained observation unexpectedly reset"):
            convert(recorder_file([recorder_frame(), held]), "hand")

    def test_missing_fresh_call_and_stale_absent_diagnostics_are_rejected(self):
        for bad in (recorder_absent(0, .1, n=42, trk=0, x=.5, y=.25),
                    recorder_absent(0, .1, ix=.5), recorder_absent(0, .1, dx=.5),
                    recorder_absent(0, .1, input_t=.1), recorder_absent(0, .1, has=1),
                    recorder_absent(0, .1, outcome=0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert(recorder_file([bad]), "hand")

    def test_rejection_cannot_publish_new_raw_substitute(self):
        t = .1
        good = recorder_frame(1, t, ms=101, x=.75, ix=.75, status=5, has=0, dx=0, dy=0,
                              outcome=1, out_trk=1)
        self.assertEqual(convert(recorder_file([recorder_frame(t=t), good]), "hand")[-1][9], .5)
        for bad in (good.replace("px=0.5", "px=0.75"), good.replace("out_trk=1", "out_trk=0"),
                    good.replace("outcome=1", "outcome=0")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert(recorder_file([recorder_frame(t=t), bad]), "hand")

    def test_invalid_result_drops_cache_instead_of_publishing_raw(self):
        invalid = recorder_frame(1, .2, x=.75, ix=.75, status=6, has=0, dx=0, dy=0,
                                 outcome=2, pub=0, px=0, py=0, out_trk=2)
        records = [recorder_frame(), invalid,
                   recorder_absent(2, .3, trk=1, missed=1, x=.75, y=.25)]
        rows = convert(recorder_file(records), "hand")
        self.assertEqual(rows[1][11], "NumericalError")
        self.assertEqual(rows[-1][11], "HeldNoCache")
        with self.assertRaises(ValueError):
            convert(recorder_file([recorder_frame(), invalid.replace("pub=0", "pub=1")]), "hand")

    def test_nonfinite_bypass_records_no_actual_dsp_update(self):
        bypass = recorder_frame(1, .2, x="inf", ix="inf", called=0, status=-1, has=0,
                                dx=0, dy=0, outcome=2, resets=1, pub=0, px=0, py=0, out_trk=2)
        rows = convert(recorder_file([recorder_frame(), bypass]), "hand")
        self.assertEqual(rows[-1][7:12], ("", "", "", "", "Bypassed"))
        self.assertEqual(rows[-1][16:19], (1, 0, "Invalid"))

    def test_nan_upstream_is_not_a_fresh_adapter_call(self):
        invalid = recorder_absent(1, .2, n=42, trk=0, x="nan", y=.25, resets=1)
        rows = convert(recorder_file([recorder_frame(), invalid]), "hand")
        self.assertEqual(rows[-1][11], "Lost")

    def test_gaze_calibration_loss_and_reacquisition_keep_own_semantics(self):
        records = [recorder_absent(0, .1, n=484, trk=-1, state=-1),
                   recorder_frame(1, .2, n=484, trk=-1),
                   recorder_absent(2, .3, trk=-1, resets=1),
                   recorder_frame(3, .4, n=484, trk=-1, x=.75, ix=.75, dx=.75, px=.75)]
        rows = convert(recorder_file(records, "gaze"), "gaze")
        self.assertEqual([r[11] for r in rows], ["Lost", "Initialized", "Lost", "Initialized"])
        self.assertEqual([r[12] for r in rows], ["Unavailable"] * 4)
        self.assertEqual(rows[0][5], -1)

    def test_zero_coordinates_and_neutral_state_are_valid(self):
        rows = convert(recorder_file([recorder_frame(x=0, y=0, ix=0, iy=0, dx=0, dy=0, px=0, py=0)]), "hand")
        self.assertEqual(rows[0][5:12], (0, 1, 0, 0, 0, 0, "Initialized"))

    def test_bad_timestamps_and_duplicate_frames_fail(self):
        first = recorder_frame()
        for bad in (recorder_frame(t=.1, cap=98, ms=98), recorder_frame(input_t=.1002),
                    recorder_frame(t=float("nan"), cap=100, ms=100),
                    recorder_frame(t=.100123456789, ms=101),
                    recorder_frame(generation=2), recorder_frame(n=0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert(recorder_file([bad]), "hand")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            convert(recorder_file([first, first]), "hand")
        with self.assertRaisesRegex(ValueError, "timestamps"):
            convert(recorder_file([first, recorder_frame(1, .09, status=1)]), "hand")

    def test_missing_footer_overflow_and_changed_configuration_fail(self):
        records = recorder_file([recorder_frame()])
        for lines in (records[:-1], records[1:], records + [records[-1]],
                      recorder_file([recorder_frame()], overflow=1),
                      recorder_file([recorder_frame()], complete=0),
                      recorder_file([recorder_frame()], config_changes=1),
                      recorder_file([recorder_frame()], attempted=2),
                      recorder_file([recorder_frame()], submitted=2)):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                convert(lines, "hand")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            convert(records, "hand", 2)

    def test_mixed_formats_duplicate_fields_and_bad_parameters_fail(self):
        records = recorder_file([recorder_frame()])
        cases = [records + [detector()], [records[0], records[1] + " frame=0", records[-1]],
                 [records[0].replace("beta=4", "beta=nan"), *records[1:]],
                 [records[0].replace("version=2", "version=3"), *records[1:]],
                 [records[0].replace("channel=hand", "channel=gaze"), *records[1:]]]
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                convert(lines, "hand")

    def test_cli_requires_successful_output_session_marker(self):
        tool = Path(__file__).with_name("import_vs_trace.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            trace.write_text("\n".join(recorder_file([recorder_frame()])), encoding="utf-8")
            command = [sys.executable, "-B", str(tool), str(trace), "--channel", "hand"]
            absent = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(absent.returncode, 0)
            self.assertEqual(absent.stdout, "")
            (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
            good = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(next(csv.reader(io.StringIO(good.stdout))), list(RECORDER_COLUMNS))
            (root / "INCOMPLETE").write_text("write failed\n", encoding="utf-8")
            incomplete = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertEqual(incomplete.stdout, "")


if __name__ == "__main__":
    unittest.main()
