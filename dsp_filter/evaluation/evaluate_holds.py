"""Measure sample-weighted stationary dispersion in labeled, paired DSP observations."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from import_vs_trace import (COLUMNS as TRACE_FIELDS, HAND_COLUMNS, RECORDER_COLUMNS,
                             STATUSES as DSP_STATUSES, RecorderPublications, recorder_fields_from_csv)

HOLD_FIELDS = "trial channel start_ms end_ms target_x target_y".split()
LANDMARKS = {"hand": 42, "gaze": 484}
STATUSES = {*DSP_STATUSES, "Missing", "Bypassed"}
HAND_EVENT_STATUSES = {"Held", "HeldNoCache", "Lost"}


@dataclass(frozen=True)
class Sample:
    frame: int
    ms: float
    capture_ms: int
    landmarks: int
    state: int
    published: bool
    raw: tuple
    filtered: tuple
    status: str
    tracking_state: str | None = None
    mediapipe_ms: int | None = None


@dataclass(frozen=True)
class Hold:
    name: str
    start: int
    end: int
    target: tuple | None


def read_rows(data, *schemas):
    # Windows PowerShell redirection may produce UTF-16 with a BOM.
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    reader = csv.DictReader(io.StringIO(data.decode(encoding), newline=""), strict=True)
    columns = sorted(reader.fieldnames or [])
    if not any(columns == sorted(schema) for schema in schemas):
        raise ValueError("expected CSV columns: " + " or ".join(",".join(s) for s in schemas))
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(f"wrong field count at CSV line {reader.line_num}")
        yield row


def point(row, prefix):
    return tuple(float(v) if v.strip() else None for v in
                 (row[prefix + "_x"], row[prefix + "_y"]))


def normalized(pair):
    return all(v is not None and 0.0 <= v <= 1.0 for v in pair)


def validate_hand_row(s, missed, outgoing):
    """Check the enriched CSV's gate metadata before any row can be scored."""
    empty = (None, None)
    if s.tracking_state == "Tracked":
        valid = (missed == 0 and s.landmarks == LANDMARKS["hand"] and s.published
                 and outgoing == "Tracked" and s.status in (*DSP_STATUSES, "Bypassed"))
    elif s.tracking_state == "Held":
        if s.published:
            payload_ok = s.status == "Held" and outgoing == "Held" and normalized(s.filtered)
        else:
            payload_ok = s.status == "HeldNoCache" and outgoing == "Lost" and s.filtered == empty
        valid = missed > 0 and s.landmarks == 0 and s.raw == empty and payload_ok
    else:
        valid = (s.tracking_state == "Lost" and missed >= 0 and s.status == "Lost"
                 and outgoing == "Lost" and not s.published and s.raw == empty and s.filtered == empty)
    if not valid:
        raise ValueError(f"inconsistent hand freshness, status or publication on frame {s.frame}")


def read_trace(data):
    samples = []
    channel = None
    recorder_gate = None
    for row in read_rows(data, TRACE_FIELDS, HAND_COLUMNS, RECORDER_COLUMNS):
        source = row["channel"]
        if source not in LANDMARKS or (channel is not None and source != channel):
            raise ValueError("one trace must contain exactly one known channel and run")
        channel = source
        recorded = "dsp_timestamp_seconds" in row
        tracking = row.get("tracking_state")
        if tracking is not None and source != "hand" and not recorded:
            raise ValueError("hand freshness columns apply only to hand captures")
        allowed_statuses = STATUSES if tracking is None else STATUSES | HAND_EVENT_STATUSES
        if row["published"] not in ("0", "1") or row["status"] not in allowed_statuses:
            raise ValueError(f"unknown publication flag or DSP status on frame {row['frame_id']}")
        mediapipe_ms = int(row["observation_ms"])
        if recorded:
            if recorder_gate is None:
                recorder_gate = RecorderPublications(source)
            verified = recorder_gate.pair(recorder_fields_from_csv(row))
            if verified[11] != row["status"]:
                raise ValueError("recorder CSV status disagrees with its actual update/publication")
            measurement_ms = float(row["dsp_timestamp_seconds"]) * 1000.0
        else:
            measurement_ms = mediapipe_ms
        s = Sample(int(row["frame_id"]), measurement_ms, int(row["capture_ms"]),
                   int(row["landmarks"]), int(row["state"]), row["published"] == "1",
                   point(row, "raw"), point(row, "filtered"), row["status"], tracking,
                   mediapipe_ms if recorded else None)
        if tracking is not None and not recorded:
            validate_hand_row(s, int(row["missed_frames"]), row["published_tracking"])
        if min(s.frame, s.capture_ms) < 0 or not s.capture_ms <= mediapipe_ms <= 2**63 - 1:
            raise ValueError(f"invalid frame or capture/observation time on frame {s.frame}")
        if not recorded and samples and (s.frame <= samples[-1].frame or s.ms <= samples[-1].ms
                        or s.capture_ms < samples[-1].capture_ms):
            raise ValueError(f"duplicate or out-of-order frame/time at frame {s.frame}")
        samples.append(s)
    if not samples:
        raise ValueError("empty trace")
    return channel, samples


def read_holds(data, channel):
    holds = []
    names = set()
    for row in read_rows(data, HOLD_FIELDS):
        name = row["trial"].strip()
        start, end = int(row["start_ms"]), int(row["end_ms"])
        if row["channel"] != channel or not name or name in names or not 0 <= start < end:
            raise ValueError(f"invalid channel, duplicate name or interval for trial {name!r}")
        target = point(row, "target")
        if target == (None, None):
            target = None
        elif not normalized(target):
            raise ValueError(f"target for {name!r} needs two normalized coordinates")
        holds.append(Hold(name, start, end, target))
        names.add(name)
    holds.sort(key=lambda h: h.start)
    if not holds or any(b.start < a.end for a, b in zip(holds, holds[1:])):
        raise ValueError("supply at least one stationary hold, with no overlapping labels")
    return holds


def exclude(s, channel):
    if s.landmarks < 0:
        return "detector_error"
    if channel == "gaze" and s.state == -1:
        return "calibration"
    if s.status in HAND_EVENT_STATUSES:
        return s.status
    if not s.published:
        return "unpublished"
    if s.landmarks != LANDMARKS[channel]:
        return "not_fresh"
    if s.state != 0:
        return "nonneutral"
    if s.status != "Filtered":
        return s.status
    if not (normalized(s.raw) and normalized(s.filtered)):
        return "invalid_coordinates"
    return None


def cadence(times):
    dt = [b - a for a, b in zip(times, times[1:])]
    if not dt:
        return None
    return {"intervals": len(dt), "min_ms": min(dt), "median_ms": statistics.median(dt),
            "max_ms": max(dt), "effective_hz": 1000 * len(dt) / sum(dt) if sum(dt) > 0 else None}


def dispersion(points, target):
    axes = list(zip(*points))
    center = tuple(axis[0] + statistics.fmean(x - axis[0] for x in axis) for axis in axes)
    sd = [statistics.pstdev(axis) for axis in axes]
    result = {"mean_xy": center, "population_sd_xy": sd, "rms_2d": math.hypot(*sd)}
    if target is not None:
        result.update(bias_xy=[a - b for a, b in zip(center, target)],
                      bias_distance=math.dist(center, target))
    return result


def summarize_hold(samples, channel, hold, wash_in_ms, max_gap_ms):
    rows = [s for s in samples if hold.start <= s.ms < hold.end]
    used = []
    reasons = Counter()
    clean_since = previous_ms = None
    previous_used = False
    runs = covered_ms = 0
    for s in rows:
        reason = exclude(s, channel)
        if reason is not None:
            reasons[reason] += 1
            clean_since = None
            previous_used = False
        else:
            if clean_since is None or s.ms - previous_ms > max_gap_ms:
                clean_since = s.ms
                previous_used = False
            if s.ms - clean_since < wash_in_ms:
                reasons["wash_in"] += 1
            else:
                used.append(s)
                if previous_used:
                    covered_ms += s.ms - previous_ms
                else:
                    runs += 1
                previous_used = True
        previous_ms = s.ms

    result = {"trial": hold.name, "interval_ms": [hold.start, hold.end],
              "label_duration_ms": hold.end - hold.start, "target_xy": hold.target,
              "observed_frames": len(rows), "used": len(used), "excluded": dict(reasons),
              "observation_cadence": cadence([s.ms for s in rows]),
              "used_observation_cadence": cadence([s.ms for s in used]),
              "used_interval_ms": [used[0].ms, used[-1].ms] if used else None,
              "used_span_ms": used[-1].ms - used[0].ms if used else None,
              "clean_runs_used": runs, "within_run_covered_ms": covered_ms,
              "raw": None, "filtered": None, "reduction_percent": None}
    if len(used) >= 2:
        raw = dispersion([s.raw for s in used], hold.target)
        filtered = dispersion([s.filtered for s in used], hold.target)
        result.update(raw=raw, filtered=filtered)
        if raw["rms_2d"] > 0.0:
            result["reduction_percent"] = 100 * (1 - filtered["rms_2d"] / raw["rms_2d"])
    return result


def evaluate(trace, labels=None, wash_in_ms=500, max_gap_ms=250):
    if wash_in_ms < 0 or max_gap_ms <= 0:
        raise ValueError("wash-in must be non-negative and maximum gap positive")
    channel, samples = read_trace(trace)
    holds = [] if labels is None else read_holds(labels, channel)
    summaries = [summarize_hold(samples, channel, h, wash_in_ms, max_gap_ms) for h in holds]
    return {"channel": channel, "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "holds_sha256": hashlib.sha256(labels).hexdigest() if labels is not None else None,
            "wash_in_ms": wash_in_ms, "max_gap_ms": max_gap_ms, "frames": len(samples),
            "observed_interval_ms": [samples[0].ms, samples[-1].ms],
            "capture_clock_corrections": sum(s.capture_ms != (s.mediapipe_ms if s.mediapipe_ms is not None else s.ms)
                                             for s in samples),
            "measurement_clock": "precise_capture_seconds" if samples[0].mediapipe_ms is not None else "legacy_observation_ms",
            "observation_cadence": cadence([s.ms for s in samples]),
            "status_counts": dict(Counter(s.status for s in samples)),
            "tracking_counts": (dict(Counter(s.tracking_state for s in samples))
                                if samples[0].tracking_state is not None else None),
            "landmark_counts": dict(Counter(s.landmarks for s in samples)),
            "unlabeled_frames": len(samples) - sum(h["observed_frames"] for h in summaries),
            "holds": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="one run of import_vs_trace.py output")
    parser.add_argument("--holds", type=Path, help="stationary trial labels on the observation clock")
    parser.add_argument("--wash-in-ms", type=int, default=500)
    parser.add_argument("--max-gap-ms", type=int, default=250)
    args = parser.parse_args()
    try:
        result = evaluate(args.trace.read_bytes(), args.holds.read_bytes() if args.holds else None,
                          args.wash_in_ms, args.max_gap_ms)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
