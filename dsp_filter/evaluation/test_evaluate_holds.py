import csv
import io
import math
import unittest

from evaluate_holds import HOLD_FIELDS, TRACE_FIELDS, evaluate
from import_vs_trace import HAND_COLUMNS, RECORDER_COLUMNS, convert
from test_import_vs_trace import recorder_absent, recorder_file, recorder_frame


def table(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def row(i, t, channel="hand", **overrides):
    fields = dict(frame_id=i, channel=channel, observation_ms=t, capture_ms=t,
                  landmarks=42 if channel == "hand" else 484, state=0, published=1,
                  raw_x=0.25, raw_y=0.5, filtered_x=0.375, filtered_y=0.5, status="Filtered")
    fields.update(overrides)
    return [fields[k] for k in TRACE_FIELDS]


def report(rows, labels=(("hold", "hand", 0, 2000, "", ""),), wash_in=0):
    return evaluate(table(TRACE_FIELDS, rows), table(HOLD_FIELDS, labels), wash_in)


class HoldMetricsTests(unittest.TestCase):
    def test_hand_gate_replays_and_losses_do_not_reduce_measured_jitter(self):
        # Numbers exercise the measurement math; raw and gated output deliberately differ.
        # The bridge error drops the cache but leaves the native tracker's last point intact.
        records = [
            # n, trk, missed, output tracking, upstream x, published x, adapter status
            (0,   2, 1, 2, -1,   None,  None),
            (42,  0, 0, 0, 0.5,  0.5,   0),
            (42,  0, 0, 0, 0.25, 0.375, 1),
            (0,   1, 1, 1, 0.25, 0.375, None),
            (42,  0, 0, 0, 0.75, 0.625, 1),
            (-1,  2, 0, 2, -1,   None,  None),
            (0,   1, 1, 2, 0.75, None,  None),
            (42,  0, 0, 0, 0.5,  0.5,   0),
        ]
        events = []
        for i, (n, trk, missed, out_trk, x, fx, status) in enumerate(records):
            ms = 100 * i
            events.append(f"E4DSP N hand frame={i} ms={ms} cap={ms} n={n} other=false")
            if status is not None:
                events.append(f"E4DSP A t={ms/1000} rx={x} ry=0.5 status={status} has=1 dx={fx} dy=0.5")
            events.append(f"E4HAND trk={trk} missed={missed} out_trk={out_trk} "
                          f"E4DSP P hand frame={i} ms={ms} ok={int(n>=0)} x={x} y={0.5 if x>=0 else -1} "
                          f"s=0 pub={int(fx is not None)} fx={fx if fx is not None else 0} "
                          f"fy={0.5 if fx is not None else 0}")
        trace = table(HAND_COLUMNS, convert(events, "hand", len(records)))
        labels = table(HOLD_FIELDS, [("steady", "hand", 0, 800, "", "")])
        result = evaluate(trace, labels, wash_in_ms=0)
        self.assertEqual(result["tracking_counts"], {"Lost": 2, "Tracked": 4, "Held": 2})
        h = result["holds"][0]
        self.assertEqual((h["observed_frames"], h["used"]), (8, 2))
        self.assertEqual(h["excluded"], {"Lost": 1, "detector_error": 1, "Initialized": 2,
                                        "Held": 1, "HeldNoCache": 1})
        self.assertEqual((h["raw"]["rms_2d"], h["filtered"]["rms_2d"]), (0.25, 0.125))
        self.assertEqual(h["reduction_percent"], 50)
        self.assertEqual((h["clean_runs_used"], h["within_run_covered_ms"]), (2, 0))
        self.assertEqual(h["observation_cadence"]["intervals"], len(records) - 1)
        # Neither one-frame clean run reaches the declared 100 ms wash-in.
        warm = evaluate(trace, labels, wash_in_ms=100)["holds"][0]
        self.assertEqual((warm["used"], warm["excluded"]["wash_in"]), (0, 2))

    def test_unpublished_gaze_calibration_has_its_own_exclusion_reason(self):
        rows = [row(1, 100, "gaze", state=-1, published=0, status="Missing",
                    raw_x="", raw_y="", filtered_x="", filtered_y=""),
                row(2, 167, "gaze", state=0, published=0, landmarks=0, status="Missing",
                    raw_x="", raw_y="", filtered_x="", filtered_y="")]
        h = report(rows, [("fixation", "gaze", 0, 200, "", "")])["holds"][0]
        self.assertEqual(h["excluded"], {"calibration": 1, "unpublished": 1})
        self.assertEqual(h["used"], 0)

    def test_hand_freshness_csv_cannot_mislabel_replay_as_measurement(self):
        fresh = row(0, 0)
        held = row(0, 0, landmarks=0, status="Held", raw_x="", raw_y="")
        lost = row(0, 0, landmarks=0, status="Lost", published=0,
                   raw_x="", raw_y="", filtered_x="", filtered_y="")
        for values, metadata in [(fresh, ["Held", 1, "Held"]), (fresh, ["Tracked", 1, "Tracked"]),
                                 (held, ["Held", 1, "Lost"]), (lost, ["Held", 1, "Lost"]),
                                 (lost, ["Lost", 1, "Tracked"]), (fresh, ["Unknown", 0, "Tracked"]),
                                 (row(0, 0, "gaze"), ["Tracked", 0, "Tracked"])]:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                evaluate(table(HAND_COLUMNS, [values + metadata]))
        # A partial prefix export is neither the legacy nor the enriched CSV schema.
        with self.assertRaisesRegex(ValueError, "columns"):
            evaluate(table(HAND_COLUMNS[:-1], [fresh + ["Tracked", 0]]))

    def test_detector_to_dsp_to_publication_to_hold(self):
        events = []
        # Analytical evaluation fixture: two paired displacements of different amplitudes.
        for i, (x, fx, code) in enumerate(((0.5, 0.5, 0), (0.25, 0.375, 1), (0.75, 0.625, 1))):
            ms = i * 100
            events.extend([
                f"E4DSP N hand frame={i} ms={ms} cap={ms} n=42 other=false",
                f"E4DSP A t={ms/1000} rx={x} ry=0.5 status={code} has=true dx={fx} dy=0.5",
                f"E4DSP P hand frame={i} ms={ms} ok=true x={x} y=0.5 s=0 pub=true fx={fx} fy=0.5",
            ])
        h = report(convert(events, "hand", 3))["holds"][0]
        self.assertEqual((h["raw"]["rms_2d"], h["filtered"]["rms_2d"]), (0.25, 0.125))
        self.assertEqual((h["used"], h["reduction_percent"]), (2, 50))
        self.assertEqual(h["excluded"], {"Initialized": 1})

    def test_two_axis_rms_and_bias_have_independent_centers(self):
        rows = [row(0, 0, raw_x=0, raw_y=0, filtered_x=0.375, filtered_y=0.25),
                row(1, 66, raw_x=0.5, raw_y=1, filtered_x=0.625, filtered_y=0.75)]
        h = report(rows, [("known", "hand", 0, 67, 0.25, 0.5)])["holds"][0]
        self.assertEqual(h["used"], 2)
        self.assertEqual(h["raw"]["population_sd_xy"], [0.25, 0.5])
        self.assertAlmostEqual(h["raw"]["rms_2d"], math.sqrt(5) / 4)
        self.assertAlmostEqual(h["filtered"]["rms_2d"], math.sqrt(5) / 8)
        self.assertAlmostEqual(h["reduction_percent"], 50)
        self.assertEqual(h["raw"]["bias_xy"], [0, 0])
        self.assertEqual(h["filtered"]["bias_xy"], [0.25, 0])
        self.assertEqual(h["filtered"]["bias_distance"], 0.25)

    def test_no_labels_no_performance_claim(self):
        result = evaluate(table(TRACE_FIELDS, [row(1, 500), row(3, 600)]))
        self.assertEqual(result["unlabeled_frames"], 2)
        self.assertIsNone(result["tracking_counts"])
        self.assertEqual(result["holds"], [])
        self.assertEqual(result["observation_cadence"]["effective_hz"], 10)

    def test_zero_dispersion_and_singleton(self):
        rows = [row(i, 100*i, raw_x=0.1) for i in range(3)]
        h = report(rows)["holds"][0]
        self.assertEqual(h["raw"]["rms_2d"], 0)
        self.assertIsNone(h["reduction_percent"])
        self.assertNotIn("bias_xy", h["raw"])
        h = report(rows[:1])["holds"][0]
        self.assertEqual(h["used"], 1)
        self.assertIsNone(h["raw"])
        self.assertIsNone(h["reduction_percent"])

    def test_more_output_jitter_is_negative_reduction(self):
        rows = [row(0, 0, raw_x=0.25, filtered_x=0), row(1, 66, raw_x=0.75, filtered_x=1)]
        self.assertEqual(report(rows)["holds"][0]["reduction_percent"], -100)

    def test_exclusions_are_disjoint_and_restart_wash_in(self):
        rows = [row(0, 0, status="Initialized"), row(1, 100), row(2, 200),
                row(3, 300, landmarks=0), row(4, 400), row(5, 500),
                row(6, 600, published=0, landmarks=0, status="Missing", raw_x="", filtered_y=""),
                row(7, 700, landmarks=-1), row(8, 800, state=1),
                row(9, 900, status="RejectedTimestamp"), row(10, 1000, raw_x="-inf"),
                row(11, 1100, filtered_y="nan"), row(12, 1200, status="ReinitializedAfterGap"),
                row(13, 1300), row(14, 1400), row(15, 1500)]
        h = report(rows, wash_in=100)["holds"][0]
        self.assertEqual(h["excluded"], {"Initialized": 1, "wash_in": 3, "not_fresh": 1,
                         "unpublished": 1, "detector_error": 1, "nonneutral": 1,
                         "RejectedTimestamp": 1, "invalid_coordinates": 2, "ReinitializedAfterGap": 1})
        self.assertEqual(h["used"], 4)
        self.assertEqual(h["used"] + sum(h["excluded"].values()), len(rows))
        self.assertEqual(h["clean_runs_used"], 3)
        self.assertEqual(h["within_run_covered_ms"], 100)

    def test_fresh_identical_observations_are_allowed(self):
        h = report([row(i, t) for i, t in enumerate((0, 66, 133, 200))])["holds"][0]
        self.assertEqual(h["used"], 4)
        self.assertEqual(h["observation_cadence"]["median_ms"], 67)
        self.assertEqual(h["observation_cadence"]["effective_hz"], 15)

    def test_gaze_calibration_and_wink_excluded(self):
        rows = [row(0, 0, "gaze", state=-1), row(1, 70, "gaze", state=2),
                row(2, 140, "gaze"), row(3, 210, "gaze", filtered_x=1.001)]
        h = report(rows, [("fixation", "gaze", 0, 220, "", "")])["holds"][0]
        self.assertEqual(h["used"], 1)
        self.assertEqual(h["excluded"], {"calibration": 1, "nonneutral": 1, "invalid_coordinates": 1})

    def test_gap_and_adjacent_hold_boundaries_restart_wash_in(self):
        rows = [row(i, t) for i, t in enumerate((0, 100, 200, 500, 600, 700, 800))]
        labels = [("first", "hand", 0, 700, "", ""), ("second", "hand", 700, 900, "", "")]
        result = report(rows, labels, wash_in=100)
        a, b = result["holds"]
        self.assertEqual((a["used"], a["within_run_covered_ms"], a["used_span_ms"]), (3, 100, 500))
        self.assertEqual((b["used"], b["excluded"]), (1, {"wash_in": 1}))
        self.assertEqual(result["unlabeled_frames"], 0)

    def test_no_manufactured_time_or_silent_pair_repair(self):
        for bad in (row(0, 200), row(2, 100), row(2, 200, capture_ms=90),
                    row(2, 200, capture_ms=201), row(2, 200, "gaze"),
                    row(2, 200, status="Maybe"), row(2, 200, published="yes")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report([row(0, 100), bad])

    def test_bad_labels(self):
        for labels in ([("a", "hand", 0, 20, 0.5, "")],
                       [("a", "hand", 0, 20, "", ""), ("b", "hand", 19, 99, "", "")],
                       [("a", "gaze", 0, 20, "", "")]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                report([row(1, 50)], labels)

    def test_trace_encoding_and_unadjusted_capture_clock(self):
        data = table(TRACE_FIELDS, [row(0, 70, capture_ms=69), row(4, 71, capture_ms=69)])
        result = evaluate(data.decode().encode("utf-16"))
        self.assertEqual(result["capture_clock_corrections"], 2)
        self.assertEqual(result["observation_cadence"]["median_ms"], 1)


class RecorderHoldMetricsTests(unittest.TestCase):
    def test_precise_clock_controls_cadence_and_hold_boundaries(self):
        frames = [recorder_frame(t=.0001),
                  recorder_frame(1, .0002, ms=1, x=.25, ix=.25, status=1, dx=.375, px=.375),
                  recorder_frame(2, .0011, ms=2, x=.75, ix=.75, status=1, dx=.625, px=.625)]
        rows = convert(recorder_file(frames), "hand")
        trace = table(RECORDER_COLUMNS, rows)
        labels = table(HOLD_FIELDS, [("early", "hand", 0, 1, "", "")])
        result = evaluate(trace, labels, wash_in_ms=0)
        self.assertEqual(result["measurement_clock"], "precise_capture_seconds")
        self.assertEqual(result["capture_clock_corrections"], 2)
        self.assertAlmostEqual(result["observation_cadence"]["median_ms"], .5)
        self.assertAlmostEqual(result["observation_cadence"]["effective_hz"], 2000)
        hold = result["holds"][0]
        self.assertEqual((hold["observed_frames"], hold["used"]), (2, 1))
        self.assertEqual(hold["used_interval_ms"], [.2, .2])

    def test_duplicate_timestamp_rejection_is_retained_but_never_scored(self):
        frames = [recorder_frame(t=.1),
                  recorder_frame(1, .1, ms=101, status=5, has=0, dx=0, dy=0,
                                 outcome=1, out_trk=1)]
        trace = table(RECORDER_COLUMNS, convert(recorder_file(frames), "hand"))
        result = evaluate(trace, table(HOLD_FIELDS, [("hold", "hand", 0, 200, "", "")]), wash_in_ms=0)
        self.assertIsNone(result["observation_cadence"]["effective_hz"])
        self.assertEqual(result["holds"][0]["used"], 0)
        self.assertEqual(result["holds"][0]["excluded"], {"Initialized": 1, "RejectedTimestamp": 1})

    def test_recorder_round_trip_uses_only_same_fresh_eligible_pairs(self):
        frames = [recorder_frame(t=.1),
                  recorder_frame(1, .2, x=.25, ix=.25, status=1, dx=.375, px=.375),
                  recorder_absent(2, .3, trk=1, missed=1, x=.25, y=.25,
                                  pub=1, px=.375, py=.25, out_trk=1),
                  recorder_frame(3, .4, x=.75, ix=.75, status=1, dx=.625, px=.625)]
        result = evaluate(table(RECORDER_COLUMNS, convert(recorder_file(frames), "hand")),
                          table(HOLD_FIELDS, [("hold", "hand", 0, 500, "", "")]), wash_in_ms=0)
        hold = result["holds"][0]
        self.assertEqual(hold["used"], 2)
        self.assertEqual(hold["excluded"], {"Initialized": 1, "Held": 1})
        self.assertEqual(hold["raw"]["rms_2d"], .25)
        self.assertEqual(hold["filtered"]["rms_2d"], .125)
        self.assertEqual(hold["reduction_percent"], 50)

    def test_recorder_gaze_calibration_is_distinguished_from_loss(self):
        frames = [recorder_absent(0, .1, n=484, trk=-1, state=-1),
                  recorder_absent(1, .2, trk=-1),
                  recorder_frame(2, .3, n=484, trk=-1)]
        result = evaluate(table(RECORDER_COLUMNS, convert(recorder_file(frames, "gaze"), "gaze")),
                          table(HOLD_FIELDS, [("fixation", "gaze", 0, 500, "", "")]), wash_in_ms=0)
        self.assertEqual(result["holds"][0]["excluded"], {"calibration": 1, "Lost": 1, "Initialized": 1})

    def test_edited_recorder_csv_cannot_fake_freshness_or_actual_output(self):
        base = convert(recorder_file([recorder_frame()]), "hand")[0]
        bad_values = {"dsp_called": 0, "adapter_outcome": "Rejected", "dsp_x": .75,
                      "tracker_x": .75, "dsp_timestamp_seconds": .5, "status": "Held",
                      "filter_generation": 2, "published_tracking": "Held", "tracker_ok": 0}
        for field, bad in bad_values.items():
            values = list(base)
            values[RECORDER_COLUMNS.index(field)] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(table(RECORDER_COLUMNS, [values]))


if __name__ == "__main__":
    unittest.main()
