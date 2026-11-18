"""Validate one complete single-channel Visual Studio trace or E4REC v2 recording."""

import argparse
import codecs
import csv
import math
import re
import struct
import sys
from pathlib import Path


COLUMNS = (
    "frame_id", "channel", "observation_ms", "capture_ms", "landmarks", "state",
    "published", "raw_x", "raw_y", "filtered_x", "filtered_y", "status",
)
HAND_COLUMNS = COLUMNS + ("tracking_state", "missed_frames", "published_tracking")
RECORDER_COLUMNS = HAND_COLUMNS + (
    "dsp_timestamp_seconds", "adapter_called", "dsp_called", "adapter_outcome",
    "reset_count", "filter_generation", "tracker_ok", "tracker_x", "tracker_y", "dsp_x", "dsp_y",
)
TRACKING_STATES = {0: "Tracked", 1: "Held", 2: "Lost"}
OUTCOMES = {-1: "NotCalled", 0: "Ok", 1: "Rejected", 2: "Invalid"}
STATUSES = (
    "Initialized", "Filtered", "ReinitializedAfterGap", "TrackingLost",
    "InvalidInput", "RejectedTimestamp", "NumericalError",
)
FIELDS = {
    "N": {"frame", "ms", "cap", "n", "other"},
    "A": {"t", "rx", "ry", "status", "has", "dx", "dy"},
    "P": {"frame", "ms", "ok", "x", "y", "s", "pub", "fx", "fy"},
}
RECORDER_FIELDS = {
    "BEGIN": set("version channel capacity min_cutoff beta derivative_cutoff min_dt reset_gap".split()),
    "F": set("frame ms cap t n ok x y state trk missed adapter called input_t ix iy status has dx dy "
             "outcome resets pub px py out_trk generation".split()),
    "END": set("submitted attempted stored overflow config_changes complete".split()),
}


def precise_capture_time(seconds, capture_ms):
    """Compare the two representations without replacing the real DSP clock.

    The native integer duration cast truncates the original time_point, whereas
    conversion to binary64 seconds and multiplication can round at a millisecond
    boundary. Allow four ULPs only for this cross-representation check.
    """
    if not math.isfinite(seconds) or seconds < 0:
        return False
    milliseconds = seconds * 1000.0
    if not math.isfinite(milliseconds):
        return False
    tolerance = 4 * math.ulp(milliseconds)
    return capture_ms - tolerance <= milliseconds <= capture_ms + 1 + tolerance


def same_numbers(left, right):
    return all(a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(left, right))


class RecorderPublications:
    """Check the latest core contract without recomputing the production filter."""

    def __init__(self, channel, min_dt=None, reset_gap=None):
        self.channel = channel
        self.previous_frame = self.previous_ms = self.previous_cap = -1
        self.previous_time = -1.0
        self.cached_output = None
        self.last_tracker_point = None
        self.accepted_time = None
        self.min_dt = min_dt
        self.reset_gap = reset_gap

    def pair(self, v):
        frame, ms, cap = (int(v[k]) for k in ("frame", "ms", "cap"))
        t = float(v["t"])
        if (not 0 <= frame < 2**64 or frame <= self.previous_frame
                or not 0 <= cap <= ms < 2**63 or cap < self.previous_cap
                or ms != max(cap, self.previous_ms + 1)
                or not precise_capture_time(t, cap) or t < self.previous_time):
            raise ValueError("duplicate/out-of-order frame or inconsistent precise timestamps")
        n, state, trk, missed, outgoing, generation, resets = (
            int(v[k]) for k in ("n", "state", "trk", "missed", "out_trk", "generation", "resets"))
        ok, adapter, called, has, pub = (flag(v[k]) for k in ("ok", "adapter", "called", "has", "pub"))
        code, outcome = int(v["status"]), int(v["outcome"])
        x, y, input_t, ix, iy, dx, dy, px, py = (
            float(v[k]) for k in ("x", "y", "input_t", "ix", "iy", "dx", "dy", "px", "py"))
        if (outgoing not in TRACKING_STATES or outcome not in OUTCOMES or missed < 0 or resets < 0
                or generation != 1 or n < -1):
            raise ValueError("invalid freshness, configuration generation or diagnostic metadata")
        if self.channel == "hand":
            if trk not in TRACKING_STATES:
                raise ValueError("hand freshness is missing")
            if trk == 0 and not (n == 42 and ok and missed == 0):
                raise ValueError("inconsistent fresh hand detector result")
            if trk == 1:
                if not (n == 0 and ok and missed > 0 and 0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("inconsistent retained hand detector result")
                if self.last_tracker_point is not None and (x, y) != self.last_tracker_point:
                    raise ValueError("retained hand coordinates changed without a detection")
            if trk == 2 and not (n in (-1, 0) and ok == (n == 0) and (x, y) == (-1, -1)):
                raise ValueError("inconsistent lost hand detector result")
        elif trk != -1 or missed != 0 or n not in (-1, 0, 484) or ok != (n >= 0):
            raise ValueError("inconsistent gaze detector metadata")
        if self.channel == "gaze" and (n < 484 or state == -1) and (x, y) != (-1, -1):
            raise ValueError("missing/calibrating gaze must not contain usable coordinates")

        detected = ok and x >= 0 and y >= 0
        should_call = detected and trk in (-1, 0)
        if adapter != should_call:
            raise ValueError("fresh observation and adapter-call presence disagree")
        if pub and not (0 <= px <= 1 and 0 <= py <= 1):
            raise ValueError("published coordinates are outside the finite unit square")
        if not pub and (px, py) != (0, 0):
            raise ValueError("unpublished frame contains stale publication coordinates")
        raw = ("", "")
        dsp_point = ("", "")
        if resets:
            self.accepted_time = None
        if not adapter:
            if (called or has or code != -1 or outcome != -1
                    or (input_t, ix, iy, dx, dy) != (0, 0, 0, 0, 0)):
                raise ValueError("uncalled adapter contains stale diagnostic values")
            if trk == 1:
                if resets:
                    raise ValueError("retained observation unexpectedly reset the DSP")
                if pub:
                    if outgoing != 1 or (px, py) != self.cached_output:
                        raise ValueError("Held output does not match the previously cached publication")
                    status = "Held"
                else:
                    if outgoing != 2 or self.cached_output is not None:
                        raise ValueError("Held without a publication disagrees with the cache")
                    status = "HeldNoCache"
            else:
                if pub or outgoing != 2:
                    raise ValueError("lost/calibrating observation unexpectedly published a point")
                if self.cached_output is not None and resets == 0:
                    raise ValueError("tracking loss did not record the gate's reset")
                self.cached_output = None
                self.accepted_time = None
                if ok:
                    self.last_tracker_point = None
                status = "Lost"
        else:
            if input_t != t:
                raise ValueError("adapter timestamp differs from the observation timestamp")
            finite = math.isfinite(x) and math.isfinite(y)
            if called != finite:
                raise ValueError("DSP-call presence disagrees with the adapter numerical guard")
            if not called:
                if (code != -1 or has or outcome != 2 or (dx, dy) != (0, 0)
                        or not same_numbers((ix, iy), (x, y)) or pub or outgoing != 2):
                    raise ValueError("nonfinite bypass contains a DSP result or publication")
                self.cached_output = None
                self.accepted_time = None
                status = "Bypassed"
            else:
                if (ix, iy) != (clamp_input(x), clamp_input(y)):
                    raise ValueError("actual DSP input disagrees with upstream coordinates")
                if code not in (0, 1, 2, 4, 5, 6):
                    raise ValueError("unknown or impossible adapter DSP status")
                status = STATUSES[code]
                raw = (ix, iy)
                accepted = code in (0, 1, 2)
                if has != accepted or (not has and (dx, dy) != (0, 0)):
                    raise ValueError("DSP optional result or absent-result storage is inconsistent")
                if accepted:
                    if outcome != 0 or not pub or outgoing != 0 or (dx, dy) != (px, py):
                        raise ValueError("accepted DSP output differs from the publication")
                    if code in (0, 2) and (dx, dy) != (ix, iy):
                        raise ValueError("initialized output differs from its actual input")
                    elapsed = None if self.accepted_time is None else t - self.accepted_time
                    if code == 0 and elapsed is not None:
                        raise ValueError("initialization reused an existing track without a reset")
                    if code in (1, 2) and (elapsed is None or elapsed <= 0):
                        raise ValueError("accepted update has no history or a non-increasing timestamp")
                    if code == 1 and self.min_dt is not None and not self.min_dt <= elapsed <= self.reset_gap:
                        raise ValueError("Filtered status disagrees with configured timing limits")
                    if code == 2 and self.reset_gap is not None and elapsed <= self.reset_gap:
                        raise ValueError("gap reinitialization has no sufficiently long observation gap")
                    dsp_point = (dx, dy)
                    self.cached_output = (px, py)
                    self.accepted_time = t
                elif code == 5:
                    if outcome != 1:
                        raise ValueError("rejected DSP observation has the wrong adapter outcome")
                    if self.cached_output is None:
                        if pub or outgoing != 2:
                            raise ValueError("rejection without a cache published a substitute")
                    elif not pub or outgoing != 1 or (px, py) != self.cached_output:
                        raise ValueError("rejected observation did not retain the cached publication")
                else:
                    if outcome != 2 or pub or outgoing != 2:
                        raise ValueError("invalid DSP observation published a substitute")
                    self.cached_output = None
                    self.accepted_time = None
            self.last_tracker_point = (x, y)

        self.previous_frame, self.previous_ms, self.previous_cap, self.previous_time = frame, ms, cap, t
        tracking = TRACKING_STATES.get(trk, "Unavailable")
        return (frame, self.channel, ms, cap, n, state, int(pub), *raw,
                *((px, py) if pub else ("", "")), status, tracking, missed, TRACKING_STATES[outgoing],
                t, int(adapter), int(called), OUTCOMES[outcome], resets, generation, int(ok), x, y, *dsp_point)


def convert_recorder(lines, channel, submitted=None):
    """Import one completed version-2 recorder file; pair by frame, never rounded time."""
    header = footer = None
    frames = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] != "E4REC" or parts[1] not in RECORDER_FIELDS:
            raise ValueError(f"line {number}: mixed or malformed recorder format")
        kind = parts[1]
        values = trace_fields(parts[2:], RECORDER_FIELDS[kind], kind)
        if kind == "BEGIN":
            if header is not None or frames or footer is not None:
                raise ValueError("duplicate or misplaced recorder header")
            header = values
        elif kind == "F":
            if header is None or footer is not None:
                raise ValueError("frame outside recorder header/footer")
            frames.append(values)
        else:
            if header is None or footer is not None:
                raise ValueError("duplicate or misplaced recorder footer")
            footer = values
    if header is None or footer is None:
        raise ValueError("incomplete recording: header/footer missing")
    if header["version"] != "2" or header["channel"] != channel or int(header["capacity"]) != 4096:
        raise ValueError("unsupported recorder version, channel or capacity")
    minimum, beta, derivative, min_dt, gap = (
        float(header[k]) for k in ("min_cutoff", "beta", "derivative_cutoff", "min_dt", "reset_gap"))
    if (not all(math.isfinite(v) for v in (minimum, beta, derivative, min_dt, gap))
            or min(minimum, derivative, min_dt) <= 0 or beta < 0 or gap <= min_dt):
        raise ValueError("invalid recorded filter/timing configuration")
    count, attempted, stored, overflow, changes = (
        int(footer[k]) for k in ("submitted", "attempted", "stored", "overflow", "config_changes"))
    if (not flag(footer["complete"]) or overflow != 0 or changes != 0
            or not 0 < count == attempted == stored == len(frames) <= 4096):
        raise ValueError("incomplete recording: counts, overflow or configuration changes")
    if submitted is not None and submitted != count:
        raise ValueError("--submitted disagrees with the recorder shutdown summary")
    gate = RecorderPublications(channel, min_dt, gap)
    rows = []
    for frame in frames:
        try:
            rows.append(gate.pair(frame))
        except ValueError as exc:
            raise ValueError(f"frame {frame['frame']}: {exc}") from exc
    return rows


def flag(text):
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"expected a boolean, got {text!r}")


def trace_fields(parts, required, kind):
    values = dict(part.split("=", 1) for part in parts)
    if len(values) != len(parts) or values.keys() != required:
        raise ValueError(f"incorrect or duplicate fields in {kind} record")
    return values


def read_events(lines, channel):
    events = {kind: {} for kind in FIELDS}
    summary = None
    for number, line in enumerate(lines, 1):
        end = re.search(r"\bos stage: (\d+) frames submitted\s*$", line)
        if end:
            if summary is not None:
                raise ValueError(f"line {number}: more than one run summary")
            summary = int(end[1])
        marker = re.search(r"\bE4DSP\b", line)
        hand = re.search(r"\bE4HAND\b", line)
        if not marker:
            if hand:
                raise ValueError(f"line {number}: E4HAND prefix without a publication")
            continue
        try:
            parts = line[marker.end():].split()
            if not parts or parts[0] not in FIELDS:
                raise ValueError("unknown or incomplete E4DSP record")
            kind = parts.pop(0)
            if kind != "A" and (not parts or parts.pop(0) != channel):
                raise ValueError(f"expected {channel} detector and publication records only")
            values = trace_fields(parts, FIELDS[kind], kind)
            if hand:
                if channel != "hand" or kind != "P" or hand.start() > marker.start():
                    raise ValueError("E4HAND must prefix a hand publication")
                prefix = line[hand.end():marker.start()].split()
                values.update(trace_fields(prefix, {"trk", "missed", "out_trk"}, "E4HAND"))
            if kind == "A":
                key = float(values["t"])
                if not math.isfinite(key):
                    raise ValueError("nonfinite adapter timestamp")
            else:
                key = int(values["frame"])
                if not 0 <= key < 2**64:
                    raise ValueError("frame ID outside the uint64 range")
            if key in events[kind]:
                raise ValueError(f"duplicate {kind} record for {key}")
            events[kind][key] = values
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc
    return events, summary


def clamp_input(value):
    # ModuleOut.x/y are binary32; adapter rx/ry are clamped binary64.
    bounded = min(1.0, max(0.0, value))
    return struct.unpack("=f", struct.pack("=f", bounded))[0]


def pair_point(publication, adapter):
    x, y = float(publication["x"]), float(publication["y"])
    detected = flag(publication["ok"]) and x >= 0 and y >= 0
    published = flag(publication["pub"])
    if detected != published:
        raise ValueError("upstream detection disagrees with OS payload presence")
    if not detected:
        if adapter is not None:
            raise ValueError("unexpected adapter call for missing point")
        return 0, "", "", "", "", "Missing"

    fx, fy = float(publication["fx"]), float(publication["fy"])
    if not (0 <= fx <= 1 and 0 <= fy <= 1):
        raise ValueError("OS payload is outside the finite unit square")
    if not (math.isfinite(x) and math.isfinite(y)):
        if adapter is not None:
            raise ValueError("nonfinite coordinate unexpectedly entered the adapter update")
        return 1, "", "", fx, fy, "Bypassed"
    if adapter is None:
        raise ValueError("published a finite point without a matching adapter update")
    rx, ry = float(adapter["rx"]), float(adapter["ry"])
    if (rx, ry) != (clamp_input(x), clamp_input(y)):
        raise ValueError("DSP input and corresponding upstream coordinates disagree")
    code = int(adapter["status"])
    if not 0 <= code < len(STATUSES):
        raise ValueError(f"unknown FilterStatus {code}")
    status = STATUSES[code]
    accepted = status in ("Initialized", "Filtered", "ReinitializedAfterGap")
    if flag(adapter["has"]) != accepted:
        raise ValueError(f"DSP optional-point presence disagrees with {status}")
    if accepted and (float(adapter["dx"]), float(adapter["dy"])) != (fx, fy):
        raise ValueError("accepted DSP output differs from the published payload")
    if status in ("Initialized", "ReinitializedAfterGap") and (fx, fy) != (rx, ry):
        raise ValueError("initialized coordinates differ from the published payload")
    # On rejection or numerical error these are published fallback values.
    # The status travels with them, so they cannot count as Filtered rows.
    return 1, rx, ry, fx, fy, status


class HandPublications:
    """Verify the current hand gate from complete publications in frame order."""

    def __init__(self):
        self.last_tracker_point = None
        self.cached_output = None

    def pair(self, publication, adapter, landmarks):
        tracking = TRACKING_STATES.get(int(publication["trk"]))
        outgoing = TRACKING_STATES.get(int(publication["out_trk"]))
        missed = int(publication["missed"])
        if tracking is None or outgoing is None or missed < 0:
            raise ValueError("invalid or unset hand freshness or miss count")
        ok, published = flag(publication["ok"]), flag(publication["pub"])
        upstream = float(publication["x"]), float(publication["y"])

        if tracking == "Tracked":
            if not (landmarks == 42 and missed == 0 and ok and published and outgoing == "Tracked"):
                raise ValueError("inconsistent Tracked detector or publication")
            result = pair_point(publication, adapter)
            self.last_tracker_point = upstream
            # FreshnessGate caches the actual publication, including an adapter fallback.
            self.cached_output = result[3:5]
        elif tracking == "Held":
            if landmarks != 0 or missed == 0 or not ok or adapter is not None:
                raise ValueError("Held needs a detector miss and no adapter update")
            if not all(0 <= value <= 1 for value in upstream):
                raise ValueError("invalid retained hand coordinates")
            if self.last_tracker_point is not None and self.last_tracker_point != upstream:
                raise ValueError("retained hand coordinates differ from the last observation")
            self.last_tracker_point = upstream
            if published:
                output = float(publication["fx"]), float(publication["fy"])
                if outgoing != "Held" or output != self.cached_output:
                    raise ValueError("Held publication does not match a cached hand output")
                result = (1, "", "", *output, "Held")
            else:
                # A filter/settings replacement can clear the gate between publications.
                if outgoing != "Lost":
                    raise ValueError("Held without a cache must publish Lost without a point")
                self.cached_output = None
                result = (0, "", "", "", "", "HeldNoCache")
        else:
            if (outgoing != "Lost" or published or adapter is not None
                    or upstream != (-1.0, -1.0) or ok != (landmarks == 0)):
                raise ValueError("inconsistent Lost observation or publication")
            self.cached_output = None
            if ok:  # A real tracking loss clears HandNative's history; a bridge error does not.
                self.last_tracker_point = None
            result = (0, "", "", "", "", "Lost")
        return (*result, tracking, missed, outgoing)


def convert(lines, channel, submitted=None):
    lines = list(lines)
    if channel not in ("hand", "gaze"):
        raise ValueError("choose one channel")
    if any(re.search(r"\bE4REC\b", line) for line in lines):
        return convert_recorder(lines, channel, submitted)
    return convert_legacy(lines, channel, submitted)


def columns_for(rows):
    if len(rows[0]) == len(RECORDER_COLUMNS):
        return RECORDER_COLUMNS
    return HAND_COLUMNS if len(rows[0]) == len(HAND_COLUMNS) else COLUMNS


def recorder_fields_from_csv(row):
    """Reconstruct diagnostic fields so an edited CSV cannot bypass gate checks."""
    names = {value: key for key, value in TRACKING_STATES.items()}
    names["Unavailable"] = -1
    outcomes = {value: key for key, value in OUTCOMES.items()}
    try:
        adapter, called = flag(row["adapter_called"]), flag(row["dsp_called"])
        if called and row["status"] not in STATUSES:
            raise ValueError("a DSP call needs an actual DSP status")
        code = STATUSES.index(row["status"]) if called else -1
        has_x, has_y = bool(row["dsp_x"].strip()), bool(row["dsp_y"].strip())
        if has_x != has_y or (not called and (has_x or row["raw_x"].strip() or row["raw_y"].strip())):
            raise ValueError("absent DSP update has stale or partial coordinates")
        if called and not (row["raw_x"].strip() and row["raw_y"].strip()):
            raise ValueError("actual DSP call is missing its input")
        published = flag(row["published"])
        if not published and (row["filtered_x"].strip() or row["filtered_y"].strip()):
            raise ValueError("unpublished CSV row has stale output")
        return dict(
            frame=row["frame_id"], ms=row["observation_ms"], cap=row["capture_ms"],
            t=row["dsp_timestamp_seconds"], n=row["landmarks"], ok=row["tracker_ok"],
            x=row["tracker_x"], y=row["tracker_y"], state=row["state"],
            trk=str(names[row["tracking_state"]]), missed=row["missed_frames"],
            adapter=str(int(adapter)), called=str(int(called)),
            input_t=row["dsp_timestamp_seconds"] if adapter else "0",
            ix=row["raw_x"] if called else row["tracker_x"] if adapter else "0",
            iy=row["raw_y"] if called else row["tracker_y"] if adapter else "0",
            status=str(code), has=str(int(has_x)), dx=row["dsp_x"] if has_x else "0",
            dy=row["dsp_y"] if has_y else "0", outcome=str(outcomes[row["adapter_outcome"]]),
            resets=row["reset_count"], pub=str(int(published)),
            px=row["filtered_x"] if published else "0", py=row["filtered_y"] if published else "0",
            out_trk=str(names[row["published_tracking"]]), generation=row["filter_generation"],
        )
    except KeyError as exc:
        raise ValueError(f"invalid recorder CSV metadata: {exc}") from exc


def convert_legacy(lines, channel, submitted=None):
    if channel not in ("hand", "gaze"):
        raise ValueError("choose one channel")
    events, summary = read_events(lines, channel)
    if submitted is not None and summary is not None and submitted != summary:
        raise ValueError("--submitted disagrees with the engine shutdown summary")
    count = submitted if submitted is not None else summary
    if count is None or count <= 0:
        raise ValueError("supply the completed run's positive --submitted count or OS summary")
    detectors, adapters, publications = events["N"], events["A"], events["P"]
    if len(publications) != count or detectors.keys() != publications.keys():
        raise ValueError("detector/publication frames or submitted count disagree; incomplete capture")
    tagged = sum("trk" in publication for publication in publications.values())
    if 0 < tagged < count:
        raise ValueError("mixed legacy and E4HAND publications or a missing freshness prefix")
    hand_gate = HandPublications() if tagged else None

    rows = []
    previous_ms = previous_cap = -1
    for frame in sorted(publications):
        detector, publication = detectors[frame], publications[frame]
        try:
            if flag(detector["other"]):
                raise ValueError("the other tracker was enabled")
            ms, cap, n = int(publication["ms"]), int(detector["cap"]), int(detector["n"])
            if (ms != int(detector["ms"]) or not 0 <= cap <= ms < 2**63 or cap < previous_cap
                    or ms != max(cap, previous_ms + 1)):
                raise ValueError("inconsistent capture or observation timestamps")
            if previous_ms >= 0 and ms / 1000.0 == previous_ms / 1000.0:
                raise ValueError("ambiguous adapter timestamp in binary64 seconds")
            if n < -1:
                raise ValueError("invalid detector row count")
            adapter = adapters.pop(ms / 1000.0, None)
            if hand_gate is not None:
                point = hand_gate.pair(publication, adapter, n)
            else:
                point = pair_point(publication, adapter)
                if point[0] and (n == -1 or (channel == "gaze" and n < 484)
                                 or (channel == "hand" and n not in (0, 42))):
                    raise ValueError("detector count cannot produce this published point")
            rows.append((frame, channel, ms, cap, n, int(publication["s"]), *point))
            previous_ms, previous_cap = ms, cap
        except ValueError as exc:
            raise ValueError(f"frame {frame}: {exc}") from exc
    if adapters:
        raise ValueError("unpaired adapter update(s); check the run and timestamp precision")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="complete Debug Output or finalized E4REC trace.txt (UTF-8/UTF-16)")
    parser.add_argument("--channel", required=True, choices=("hand", "gaze"))
    parser.add_argument("--submitted", type=int, help="OS frame count from that run's shutdown summary")
    args = parser.parse_args()
    try:
        with open(args.trace, "rb") as stream:
            bom = stream.read(2)
        encoding = "utf-16" if bom in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE) else "utf-8-sig"
        with open(args.trace, encoding=encoding) as stream:
            lines = stream.readlines()
        if any(re.search(r"\bE4REC\b", line) for line in lines):
            directory = Path(args.trace).parent
            if not (directory / "COMPLETE").is_file() or (directory / "INCOMPLETE").exists():
                raise ValueError("recorder output has no successful COMPLETE marker or remains INCOMPLETE")
        rows = convert(lines, args.channel, args.submitted)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"trace import failed: {exc}\n")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(columns_for(rows))
    writer.writerows(rows)


if __name__ == "__main__":
    main()
