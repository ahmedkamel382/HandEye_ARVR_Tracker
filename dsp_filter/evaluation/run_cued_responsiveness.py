"""Optional, manually operated hand experiment cues; never imported by production.

plan: no devices/files. prepare: render WAV files silently in a new external folder.
rehearse: explicitly requested audio only. record: explicit typed confirmation,
verified recorder executable, fixed DSP/dry-run arguments, then bounded diagnostics.
Cue clock timestamps are NOT camera timestamps or measured acoustic onset times.

Audio-only practice: rehearse --session SESSION_DIRECTORY. Hold through
"Three, two, one". Start on the first sound of "Move!" and reach the other
mark on the first sound of "Stop!". Both words share one prerecorded waveform.
The STOP event logs its sample offset from MOVE's playback request.
"""

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

IS_WINDOWS = os.name == "nt"
EXPECTED_BRANCH = "codex/engineer4-dsp-recorder"
EXPECTED_EXE_SHA256 = "3A3F05472A441E225006A276D03297A38135F13827E83F1B13CA278E46167200"
ENGINE_SECONDS = 55
MAX_CUE_LATENESS = 0.10
READINESS_TIMEOUT = 30.0
READY_AFTER_FIRST_STATUS_LIMIT = 8.0
SESSION_VERSION = 2
RATE = 22050
MOVEMENT_SECONDS = {"slow": 3.0, "fast": 0.5, "vertical-slow": 3.0}
PROTOCOL = Path(__file__).with_name("RESPONSIVENESS_PROTOCOL.md")
VERTICAL_PROTOCOL = Path(__file__).with_name("VERTICAL_RESPONSIVENESS_PROTOCOL.md")


def protocol_file(mode):
    return VERTICAL_PROTOCOL if mode == "vertical-slow" else PROTOCOL


@dataclass(frozen=True)
class Cue:
    at_seconds: float
    clip: str
    message: str


def cue_plan(mode):
    if mode not in MOVEMENT_SECONDS:
        raise ValueError("unknown responsiveness mode")
    start, end = ("top", "bottom") if mode == "vertical-slow" else ("left", "right")
    outward, inward = ("down", "up") if mode == "vertical-slow" else ("right", "left")
    duration = MOVEMENT_SECONDS[mode]
    arrival = 11.0 + duration
    back = arrival + 14.0
    finish = back + duration
    return (
        Cue(0, "hold_" + start, f"Hold {start}. Keep still."),
        Cue(5, "prepare_" + end, f"Get ready to move {outward}."),
        Cue(8, "three", "Three"), Cue(9, "two", "Two"), Cue(10, "one", "One"),
        Cue(11, "move_stop", f"MOVE! Go {outward}."),
        Cue(arrival, "stop", f"STOP! Hold {end}."),
        Cue(arrival + 8, "prepare_" + start, f"Get ready to move {inward}."),
        Cue(arrival + 11, "three", "Three"), Cue(arrival + 12, "two", "Two"),
        Cue(arrival + 13, "one", "One"),
        Cue(back, "move_stop", f"MOVE! Go {inward}."),
        Cue(finish, "stop", f"STOP! Hold {start}."),
        Cue(finish + 8, "finished", f"Finished. Keep holding {start} until the camera stops."),
    )


class CueScheduler:
    """Monotonic deadlines; never catch up by issuing a burst of late cues."""
    def __init__(self, mode, ready_time):
        if not math.isfinite(ready_time):
            raise ValueError("ready time must be finite")
        self.cues = cue_plan(mode)
        self.ready_time = ready_time
        self.index = 0
        self.previous_time = ready_time

    @property
    def complete(self):
        return self.index == len(self.cues)

    def poll(self, now, process_alive=True):
        if not math.isfinite(now) or now < self.previous_time:
            raise RuntimeError("cue clock must be finite and monotonic")
        self.previous_time = now
        if not process_alive and not self.complete:
            raise RuntimeError("process exited before cue sequence completed")
        if self.complete:
            return None
        cue = self.cues[self.index]
        delay = now - (self.ready_time + cue.at_seconds)
        if delay < 0:
            return None
        if delay > MAX_CUE_LATENESS:
            raise RuntimeError("cue deadline missed; no catch-up cues will be issued")
        self.index += 1
        return cue


STATUS = re.compile(
    r"^cap\s+(\d+)\s*\|\s*ai\s+(\d+)\s+fps\s*\|.*?"
    r"pyerr\s+(\d+).*?hand\(([^,]+),([^ ]+)\s+s=(-?\d+)\)\s*\|\s*os frames\s+(\d+)\s*$")


class ReadinessGate:
    """Two live, neutral status reports; full freshness is checked from trace later."""
    def __init__(self):
        self.first_status_time = None
        self.ready_time = None
        self.last_os_frames = None
        self.previous_valid = False

    def feed(self, line, now):
        if not math.isfinite(now):
            raise ValueError("status time must be finite")
        match = STATUS.fullmatch(line.strip())
        if not match:
            return False
        if self.first_status_time is None:
            self.first_status_time = now
        cap, ai, errors, x, y, state, count = match.groups()
        x, y = float(x), float(y)
        count = int(count)
        valid = (int(cap) > 0 and int(ai) > 0 and int(errors) == 0 and int(state) == 0
                 and math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1
                 and count > 0)
        advanced = self.last_os_frames is not None and count > self.last_os_frames
        became_ready = self.ready_time is None and valid and self.previous_valid and advanced
        self.previous_valid = valid
        self.last_os_frames = count
        if became_ready:
            self.ready_time = now
        return became_ready


class BoundedConsole:
    """Bound each retained line and total line count; always keep draining input."""
    def __init__(self, max_lines=4096, max_line_bytes=4096):
        if max_lines <= 0 or max_line_bytes <= 0:
            raise ValueError("console bounds must be positive")
        self.max_lines, self.max_line_bytes = max_lines, max_line_bytes
        self.lines = []
        self.pending = bytearray()
        self.discard_line = False
        self.overflow = False

    def _emit(self):
        line = bytes(self.pending).decode("utf-8", errors="replace").rstrip("\r")
        self.pending.clear()
        if self.discard_line:
            self.discard_line = False
            return []
        if len(self.lines) >= self.max_lines:
            self.overflow = True
            return []
        self.lines.append(line)
        return [line]

    def feed(self, chunk):
        output = []
        for part_index, part in enumerate(chunk.split(b"\n")):
            if part_index:
                output.extend(self._emit())
            if not self.discard_line:
                if len(self.pending) + len(part) > self.max_line_bytes:
                    self.pending.clear()
                    self.discard_line = self.overflow = True
                else:
                    self.pending.extend(part)
        return output

    def finish(self):
        return self._emit() if self.pending or self.discard_line else []


def save_exclusive(path, text):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def save_json(path, value):
    save_exclusive(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_value(repo, args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, timeout=10).stdout.strip()


def powershell():
    return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"


def run_powershell(script, payload=None):
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return subprocess.run([str(powershell()), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                          input="" if payload is None else json.dumps(payload), text=True,
                          encoding="utf-8", capture_output=True, check=True, timeout=60,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def running_engines():
    script = "$ErrorActionPreference='Stop'; @(Get-Process -Name core_engine -ErrorAction SilentlyContinue).Count"
    return int(run_powershell(script).stdout.strip())


def preflight(repo, exe, output_parent, settings):
    if not IS_WINDOWS:
        raise RuntimeError("Windows is required for the verified executable and audio")
    repo, exe, output_parent, settings = map(lambda p: Path(p).resolve(), (repo, exe, output_parent, settings))
    if not output_parent.is_dir() or output_parent == repo or repo in output_parent.parents:
        raise ValueError("output parent must be an existing directory outside the repository")
    if git_value(repo, ["branch", "--show-current"]) != EXPECTED_BRANCH:
        raise RuntimeError("unexpected branch; no branch change was attempted")
    if not exe.is_file() or sha256(exe).upper() != EXPECTED_EXE_SHA256:
        raise RuntimeError("executable hash differs from the verified recorder build")
    if not settings.is_file():
        raise RuntimeError("existing readable saved tray settings are required")
    try:
        settings_hash = sha256(settings)
    except OSError as error:
        raise RuntimeError("existing readable saved tray settings are required") from error
    if running_engines():
        raise RuntimeError("another core_engine process is already running")
    return {"repo": str(repo), "exe": str(exe), "output_parent": str(output_parent),
            "settings": str(settings), "settings_sha256": settings_hash,
            "commit": git_value(repo, ["rev-parse", "HEAD"]), "branch": EXPECTED_BRANCH,
            "exe_sha256": EXPECTED_EXE_SHA256, "git_status": git_value(repo, ["status", "--short"])}


def engine_command(exe, record_dir):
    return [str(exe), "--camera", "0", "--filter", "dsp", "--os", "log", "--no-gaze",
            "--hand-filter", "1.2,4,1", "--seconds", str(ENGINE_SECONDS), "--dsp-record-dir", str(record_dir)]


def validate_command(command, exe, record_dir):
    if command != engine_command(exe, record_dir):
        raise ValueError("only the fixed camera/DSP/dry-run hand command is allowed")


def write_wave(path, pcm):
    with Path(path).open("xb") as stream, wave.open(stream, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(pcm)


def read_wave(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError("unexpected audio WAV format")
        frames = wav.getnframes()
        if not 0 < frames <= RATE * 15:
            raise ValueError("audio clip is empty or too long")
        pcm = wav.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("truncated audio clip")
        return pcm, frames / RATE


def speech_bounds(pcm):
    """First and last signed PCM samples above the synthesizer's silence floor."""
    samples = memoryview(pcm).cast("h")
    first = next((i for i in range(len(samples)) if abs(samples[i]) > 5), None)
    if first is None:
        raise ValueError("silent speech clip")
    last = next(i for i in range(len(samples) - 1, first - 1, -1) if abs(samples[i]) > 5)
    return first, last


def movement_wave(directory, mode):
    # One playback stream gives both words the same device startup latency.
    words = []
    for name in ("move", "stop"):
        pcm, _ = read_wave(directory / (name + ".wav"))
        first, last = speech_bounds(pcm)
        words.append(pcm[2 * first:2 * (last + 1)])
    spacing = round(RATE * MOVEMENT_SECONDS[mode])
    if len(words[0]) // 2 + round(RATE * .05) > spacing:
        raise ValueError("spoken MOVE overlaps the STOP safety margin")
    return words[0].ljust(2 * spacing, b"\0") + words[1] + bytes(2 * round(RATE * .05))


def render_audio(directory, mode):
    """Silent synthesis using installed Windows voices; no speaker playback."""
    directory.mkdir(exist_ok=False)
    countdown = ("three", "two", "one")
    speech = {c.clip: c.message for c in cue_plan(mode) if c.clip not in ("move_stop", "stop")}
    speech.update(three="Three.", two="Two.", one="One.", move="Move!", stop="Stop!")
    jobs = [{"text": text, "rate": 2 if key in ("move", "stop") else 0,
             "path": str(directory / (key + (".source.wav" if key in countdown else ".wav")))}
            for key, text in speech.items()]
    script = r"""
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$jobs = [Console]::In.ReadToEnd() | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  foreach ($job in $jobs) {
    if (Test-Path -LiteralPath $job.path) { throw 'Refusing existing audio output' }
    $voice.Rate = [int]$job.rate
    $voice.SetOutputToWaveFile([string]$job.path, $format)
    $voice.Speak([string]$job.text)
    $voice.SetOutputToNull()
  }
  [Console]::Out.WriteLine($voice.Voice.Name)
} finally { $voice.Dispose() }
"""
    voice = run_powershell(script, jobs).stdout.strip()
    # Windows voices append sentence-ending silence even to a single number.
    # Preserve all voiced samples plus a 50 ms tail without moving the cue deadlines.
    for key in countdown:
        pcm, _ = read_wave(directory / (key + ".source.wav"))
        _, last = speech_bounds(pcm)
        write_wave(directory / (key + ".wav"), pcm[:2 * (last + 1 + RATE // 20)])
    write_wave(directory / "move_stop.wav", movement_wave(directory, mode))
    return voice


def check_audio(directory, mode):
    clips = {}
    cues = cue_plan(mode)
    for i, cue in enumerate(cues):
        if cue.clip == "stop":  # embedded in the preceding move_stop.wav
            continue
        path = directory / (cue.clip + ".wav")
        pcm, duration = read_wave(path)
        following = i + 1
        if cue.clip == "move_stop":
            if pcm != movement_wave(directory, mode):
                raise RuntimeError("recorded movement audio differs from the timed spoken commands")
            following += 1
        available = cues[following].at_seconds - cue.at_seconds - .05 if following < len(cues) else 5
        if duration > available:
            raise RuntimeError(f"audio clip {cue.clip} overlaps the next cue")
        clips[cue.clip] = {"seconds": duration, "sha256": sha256(path)}
    return clips


def prepare_session(mode, output_parent, repo, exe, settings):
    cue_plan(mode)
    before = preflight(repo, exe, output_parent, settings)
    session = Path(tempfile.mkdtemp(prefix=f"Engineer4-HandAudio-{mode}-", dir=output_parent))
    save_exclusive(session / "PREPARING", "No recording has been launched. Retain incomplete preparation if an error occurs.\n")
    with (session / "protocol.md").open("xb") as copy:
        copy.write(protocol_file(mode).read_bytes())
    voice = render_audio(session / "audio", mode)
    clips = check_audio(session / "audio", mode)
    manifest = {"version": SESSION_VERSION, "mode": mode, "preflight": before,
                "launcher_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(protocol_file(mode)),
                "voice": voice, "clips": clips, "cues": [asdict(c) for c in cue_plan(mode)],
                "engine_command": engine_command(exe, session / "recording"),
                "spoken_pair": {"rate_hz": RATE, "move_onset_sample": 0,
                                "stop_onset_sample": round(RATE * MOVEMENT_SECONDS[mode])},
                "clock_limit": "Waveform offsets are sample-verified. Device latency and alignment to the camera clock are unmeasured."}
    save_json(session / "manifest.json", manifest)
    save_exclusive(session / "PREPARED", "Audio files synthesized silently. No camera or speaker was activated.\n")
    return session


def load_session(session):
    session = Path(session).resolve(strict=True)
    if not (session / "PREPARED").is_file():
        raise RuntimeError("session preparation is incomplete")
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != SESSION_VERSION or manifest["launcher_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("launcher changed since preparation; prepare a new session")
    expected_protocol = sha256(protocol_file(manifest["mode"]))
    if manifest["protocol_sha256"] != expected_protocol or sha256(session / "protocol.md") != expected_protocol:
        raise RuntimeError("protocol changed since preparation")
    if manifest["cues"] != [asdict(c) for c in cue_plan(manifest["mode"])]:
        raise RuntimeError("cue manifest differs from the fixed schedule")
    if manifest["clips"] != check_audio(session / "audio", manifest["mode"]):
        raise RuntimeError("audio changed since preparation")
    if manifest["spoken_pair"] != {"rate_hz": RATE, "move_onset_sample": 0,
                                   "stop_onset_sample": round(RATE * MOVEMENT_SECONDS[manifest["mode"]])}:
        raise RuntimeError("speech timing metadata changed")
    validate_command(manifest["engine_command"], manifest["preflight"]["exe"], session / "recording")
    return session, manifest


def play_clip(path):
    import winsound
    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


def stop_audio():
    import winsound
    winsound.PlaySound(None, 0)


class CuePlayback:
    """Keep software issue times separate from positions on the audio stream."""
    def __init__(self, session, manifest):
        self.session = session
        self.manifest = manifest
        self.pair_request = None
        self.last_end = None

    def issue(self, cue, scheduled, events):
        observed = time.monotonic()
        if observed - scheduled > MAX_CUE_LATENESS:
            raise RuntimeError("cue playback deadline missed")
        event = {"event": "cue", "cue": asdict(cue), "scheduled_monotonic": scheduled,
                 "observed_monotonic": observed}
        if cue.clip == "stop":
            if self.pair_request is None:
                raise RuntimeError("STOP has no active movement waveform")
            timing = self.manifest["spoken_pair"]
            offset = timing["stop_onset_sample"] / timing["rate_hz"]
            estimated = self.pair_request + offset
            event.update(kind="embedded_word", stream_request_monotonic=self.pair_request,
                         wave_onset_seconds=offset, estimated_onset_monotonic=estimated,
                         lateness_seconds=estimated - scheduled)
            self.pair_request = None
        else:
            play_clip(self.session / "audio" / (cue.clip + ".wav"))
            returned = time.monotonic()
            self.last_end = returned + self.manifest["clips"][cue.clip]["seconds"]
            event.update(kind="playback_request", requested_monotonic=observed,
                         returned_monotonic=returned, lateness_seconds=observed - scheduled)
            if cue.clip == "move_stop":
                self.pair_request = observed
                event.update(wave_onset_seconds=0.0, estimated_onset_monotonic=observed)
            if returned - observed > MAX_CUE_LATENESS:
                events.append(event)
                raise RuntimeError("audio playback request stalled; trial cues invalid")
        events.append(event)
        print(cue.message, flush=True)


def rehearse(session):
    session, manifest = load_session(session)
    print("AUDIO ONLY. Hold through the countdown. Start on MOVE! Arrive and hold on STOP! Ctrl+C cancels.")
    clock = time.monotonic
    start = clock(); scheduler = CueScheduler(manifest["mode"], start)
    events = []
    playback = CuePlayback(session, manifest)
    error = None
    try:
        while not scheduler.complete:
            now = clock(); cue = scheduler.poll(now)
            if cue:
                playback.issue(cue, start + cue.at_seconds, events)
            time.sleep(.005)
        time.sleep(manifest["clips"][scheduler.cues[-1].clip]["seconds"])
    except (KeyboardInterrupt, RuntimeError, OSError) as exc:
        error = str(exc) or "cancelled"
    finally:
        try:
            stop_audio()
        except Exception as exc:
            error = error or "audio cleanup failed: " + str(exc)
        output = Path(tempfile.mkdtemp(prefix="rehearsal-", dir=session))
        save_json(output / "cues.json", {"kind": "audio-only rehearsal, no camera", "events": events, "error": error})
    if error:
        raise RuntimeError(error)


def stop_owned_process(process):
    if process.poll() is not None:
        return False
    forced = False
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        forced = True
        process.kill()
        process.wait(timeout=5)
    return forced


def persist_session(session, log, cues, recording, failure):
    """Save independently; one failed output must not suppress the invalid marker."""
    errors = []
    for name, value in (("console.txt", log), ("cues.json", cues)):
        try:
            if isinstance(value, str):
                save_exclusive(session / name, value)
            else:
                save_json(session / name, value)
        except (OSError, ValueError) as exc:
            errors.append(f"could not save {name}: {exc}")
    if errors:
        failure = "; ".join(([failure] if failure else []) + errors)
    try:
        save_json(session / "session-result.json", {"cue_and_recording_integrity_pass": failure is None,
                                                   "failure": failure, "recording": recording,
                                                   "responsiveness_acceptance": "NOT EVALUATED"})
    except (OSError, ValueError) as exc:
        failure = (failure + "; " if failure else "") + "could not save session result: " + str(exc)
    try:
        save_exclusive(session / ("SESSION_READY_FOR_ANALYSIS" if failure is None else "SESSION_INVALID"),
                       "Preserve all files. " + (failure or "Integrity only; responsiveness still requires analysis.") + "\n")
        if failure is None:
            (session / "SESSION_INCOMPLETE").unlink(missing_ok=True)
    except OSError as exc:
        failure = (failure + "; " if failure else "") + "could not save session marker: " + str(exc)
        # A ready marker may have been partially created; never claim its existence proves success.
        try:
            save_exclusive(session / "SESSION_INVALID", failure + "\n")
        except OSError:
            pass  # CLI reports failure even when the volume cannot accept any error marker.
    return failure


def verify_recording(session, stdout):
    """Offline completeness check, not responsiveness acceptance or numerical replay."""
    import import_vs_trace
    import evaluate_holds
    summary = re.search(r"captured (\d+) \| processed (\d+) \| evicted (\d+) \| skipped (\d+) \| stale (\d+) \| python errors (\d+)", stdout)
    submitted = re.search(r"os stage: (\d+) frames submitted", stdout)
    if not summary or not submitted:
        raise RuntimeError("missing shutdown frame accounting")
    captured, processed, evicted, skipped, stale, errors = map(int, summary.groups())
    if captured != processed + evicted + skipped + stale or int(submitted[1]) != processed or errors:
        raise RuntimeError("shutdown accounting or Python errors failed")
    if "pool slots returned: 8/8 (no leaks)" not in stdout:
        raise RuntimeError("frame pool was not fully returned")
    result = subprocess.run([sys.executable, "-B", str(Path(import_vs_trace.__file__)),
                             str(session / "recording/trace.txt"), "--channel", "hand", "--submitted", str(processed)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("trace importer failed: " + result.stderr)
    save_exclusive(session / "paired.csv", result.stdout)
    coverage = evaluate_holds.evaluate(result.stdout.encode("utf-8"))
    save_json(session / "coverage.json", coverage)
    return {"captured": captured, "submitted": processed, "stale": stale,
            "trace_sha256": sha256(session / "recording/trace.txt"), "statuses": coverage["status_counts"]}


def record(session):
    session, manifest = load_session(session)
    before = manifest["preflight"]
    current = preflight(before["repo"], before["exe"], session.parent, before["settings"])
    if current["settings_sha256"] != before["settings_sha256"] or current["commit"] != before["commit"]:
        raise RuntimeError("settings or commit changed since preparation; prepare a new session")
    if (session / "recording").exists() or (session / "LAUNCH_ATTEMPT.json").exists():
        raise RuntimeError("session already used; existing experiment data will not be overwritten")
    print("This command WILL open camera 0, in --os log mode, for 55 seconds of processing.")
    initial_hold = cue_plan(manifest["mode"])[0].message
    print(f"{initial_hold} Start on MOVE! Arrive and hold on STOP! F9 stops the engine.")
    if not sys.stdin.isatty():
        raise RuntimeError("record requires manual confirmation in an interactive terminal")
    if input("Type RECORD to launch this one experiment, or anything else to cancel: ").strip() != "RECORD":
        print("Cancelled; camera was not opened.")
        return
    # Check again after the human may have spent time reading the confirmation.
    latest = preflight(before["repo"], before["exe"], session.parent, before["settings"])
    if latest["commit"] != before["commit"] or latest["settings_sha256"] != before["settings_sha256"]:
        raise RuntimeError("settings or commit changed before launch")
    command = engine_command(before["exe"], session / "recording")
    validate_command(command, before["exe"], session / "recording")
    save_json(session / "LAUNCH_ATTEMPT.json", {"command": command, "preflight": latest})
    save_exclusive(session / "SESSION_INCOMPLETE", "Cue/console integrity not yet verified. Preserve this session.\n")
    console = BoundedConsole(); inbox = queue.Queue(maxsize=256); reader_error = []
    events = []; failure = None; process = None; reader = None; forced = False
    gate = ReadinessGate(); scheduler = None; clock = time.monotonic; started = clock()
    playback = CuePlayback(session, manifest)
    try:
        process = subprocess.Popen(command, cwd=before["repo"], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        def drain():
            try:
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    for line in console.feed(chunk):
                        if STATUS.fullmatch(line.strip()):
                            try:
                                inbox.put_nowait((line, clock()))
                            except queue.Full:
                                if not reader_error:
                                    reader_error.append("status queue overflow")
                console.finish()
            except Exception as exc:
                if not reader_error:
                    reader_error.append("console reader failed: " + str(exc))
        reader = threading.Thread(target=drain, name="e4-console-drain", daemon=True)
        reader.start()
        print(f"Waiting for two live Neutral hand status reports. {initial_hold}", flush=True)
        while process.poll() is None:
            now = clock()
            if now - started > 100:
                raise RuntimeError("engine watchdog timeout")
            if console.overflow or reader_error:
                raise RuntimeError("bounded console diagnostics failed: " + "; ".join(reader_error))
            while True:
                try:
                    line, received = inbox.get_nowait()
                except queue.Empty:
                    break
                first_status = gate.first_status_time is None
                became_ready = gate.feed(line, received)
                if first_status and gate.first_status_time is not None:
                    if received - started > READINESS_TIMEOUT:
                        raise RuntimeError("first live status arrived after the readiness deadline")
                    events.append({"event": "first_status", "received_monotonic": received, "status": line})
                if became_ready:
                    if received - gate.first_status_time > READY_AFTER_FIRST_STATUS_LIMIT:
                        raise RuntimeError("hand readiness was too late for the fixed recording duration")
                    if now - received > MAX_CUE_LATENESS:
                        raise RuntimeError("readiness report was delayed in the launcher")
                    scheduler = CueScheduler(manifest["mode"], received)
                    events.append({"event": "ready", "received_monotonic": received, "status": line})
            if scheduler is None:
                if ((gate.first_status_time is None and now - started > READINESS_TIMEOUT)
                        or (gate.first_status_time is not None and now - gate.first_status_time > READY_AFTER_FIRST_STATUS_LIMIT)):
                    raise RuntimeError("no timely live Neutral hand readiness; movement cues withheld")
            else:
                cue = scheduler.poll(clock(), process_alive=process.poll() is None)
                if cue:
                    playback.issue(cue, scheduler.ready_time + cue.at_seconds, events)
            time.sleep(.005)
        if scheduler is None or not scheduler.complete:
            raise RuntimeError("engine exited before cue sequence completed")
        if playback.last_end is not None and clock() < playback.last_end:
            raise RuntimeError("engine exited before the final prompt could finish")
        if process.returncode != 0:
            raise RuntimeError(f"engine exited with code {process.returncode}")
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or "cancelled by user"
    finally:
        try:
            stop_audio()
        except Exception as exc:
            failure = failure or "audio cleanup failed: " + str(exc)
        try:
            if process is not None and process.poll() is None:
                forced = stop_owned_process(process)
        except Exception as exc:
            failure = (failure + "; " if failure else "") + "owned process cleanup failed: " + str(exc)
        events.append({"event": "process_end", "observed_monotonic": clock(),
                       "exit_code": process.poll() if process is not None else None, "failure": failure})
        if reader:
            reader.join(timeout=5)
            if reader.is_alive():
                failure = failure or "console reader did not finish"
        if process and process.stdout and (not reader or not reader.is_alive()):
            process.stdout.close()
        if console.overflow or reader_error:
            failure = failure or "console log or reader overflow/error"
        try:
            if sha256(before["settings"]) != before["settings_sha256"]:
                failure = failure or "saved settings changed; original settings were NOT restored"
        except OSError:
            failure = failure or "saved settings became unreadable; nothing was restored"
        log = "\n".join(console.lines) + "\n"
    recording = None
    if failure is None:
        try:
            recording = verify_recording(session, log)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            failure = str(exc)
    failure = persist_session(session, log,
                              {"clock": "Python monotonic seconds, separate from recorder clock",
                               "launch_monotonic": started, "events": events, "failure": failure,
                               "forced_process_termination": forced}, recording, failure)
    print("Session: " + str(session))
    if failure:
        raise RuntimeError(failure)
    print("Recording saved and imported. Responsiveness has not yet been scored.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="print cue plan; no devices or files")
    plan.add_argument("--mode", choices=MOVEMENT_SECONDS, required=True)
    prepare = sub.add_parser("prepare", help="silently prepare audio and protocol; no camera/playback")
    prepare.add_argument("--mode", choices=MOVEMENT_SECONDS, required=True)
    prepare.add_argument("--output-parent", type=Path, required=True)
    for name in ("rehearse", "record"):
        p = sub.add_parser(name, help="audio only" if name == "rehearse" else "manual confirmation before camera")
        p.add_argument("--session", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"mode": args.mode, "engine_seconds": ENGINE_SECONDS,
                          "cues": [asdict(c) for c in cue_plan(args.mode)]}, indent=2))
    elif args.action == "prepare":
        repo = Path(__file__).resolve().parents[2]
        session = prepare_session(args.mode, args.output_parent.resolve(), repo,
                                  repo / "out/build/windows-debug/core_engine.exe",
                                  Path(os.environ["APPDATA"]) / "HandEyeTracker/os_integration.ini")
        print("Prepared silently; no camera or audio playback: " + str(session))
    elif args.action == "rehearse":
        rehearse(args.session)
    else:
        record(args.session)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("Cued experiment stopped: " + str(error), file=sys.stderr)
        raise SystemExit(1)
