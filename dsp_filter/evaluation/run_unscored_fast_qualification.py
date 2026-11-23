# -*- coding: utf-8 -*-
"""UNSCORED E2-D FAST readiness capture — audio cues + native DSP trace.

This is NOT Engineer 4's scored launcher and NOT a scored trial. It reuses the
launcher's fixed cue schedule, audio rendering and playback helpers WITHOUT
modifying that launcher, and runs the native engine directly with
--dsp-record-dir so the trace contains frame_id, capture_s, detection state,
landmark count and raw/filtered x/y required by H1.

It performs no preflight branch/SHA gate: the scored identity decision is
separate and remains BLOCKED. All attempts are preserved; existing files are
never overwritten.

Usage:
  python run_unscored_fast_qualification.py --run-id E2D_UNSCORED_YYYYMMDD_01 \
      --out-dir C:\\Users\\Doctor Laptop\\Downloads --mode fast \
      --exe ..\\..\\out\\build\\windows-release\\core_engine.exe --prepare-only
  python run_unscored_fast_qualification.py ... --dry-run        # print plan only
  python run_unscored_fast_qualification.py ...                  # real capture
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cued_responsiveness as L  # exact cue schedule/audio helpers, unmodified

STATUS = re.compile(
    r"cap\s+(\d+)\s*\|\s*ai\s+(\d+)\s*\|\s*pyerr\s+(\d+)\s*\|\s*"
    r"x\s+(-?[\d.]+)\s*\|\s*y\s+(-?[\d.]+)\s*\|\s*state\s+(\d+)\s*\|\s*n\s+(\d+)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def build_manifest(session, run_id, mode, exe, repo, prepare):
    cues = [asdict(c) for c in L.cue_plan(mode)]
    return {
        "label": "UNSCORED E2-D FAST READINESS CAPTURE",
        "scored": False,
        "trial4_replaced": False,
        "run_id": run_id,
        "mode": mode,
        "engine_seconds": L.ENGINE_SECONDS,
        "engine": {"path": str(exe), "sha256": sha256(exe)},
        "source": {"branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()},
        "voice": prepare["voice"],
        "clips": prepare["clips"],
        "cues": cues,
        "spoken_pair": {"rate_hz": L.RATE, "move_onset_sample": 0,
                        "stop_onset_sample": round(L.RATE * L.MOVEMENT_SECONDS[mode])},
        "clock_limit": "Cue times are software issue times, not camera timestamps or measured acoustic onsets.",
        "physical_setup": {
            "marker_layout": "LEFT - 10 cm - CENTER - 10 cm - RIGHT (to be measured)",
            "hand_to_camera_cm": "", "camera_model": "", "camera_mode": "",
            "camera_connection": "", "lighting": "", "exposure": "",
            "same_height": "", "same_depth": "", "measured_marker_distances_cm": "",
            "note": "Fill from the physical setup record before evaluating gates."},
        "command": L.engine_command(exe, session / "recording"),
        "command_note": "Direct native engine, NOT the scored launcher. No identity gate is applied.",
    }


def prepare_session(args, repo):
    session = Path(args.out_dir) / args.run_id
    if session.exists():
        raise RuntimeError("run directory already exists: %s" % session)
    session.mkdir(parents=True)   # render_audio() creates the audio directory itself
    cues = L.cue_plan(args.mode)                       # validates mode
    voice = L.render_audio(session / "audio", args.mode)  # also writes move_stop.wav
    clips = L.check_audio(session / "audio", args.mode)
    manifest = build_manifest(session, args.run_id, args.mode, Path(args.exe).resolve(),
                              repo, {"voice": voice, "clips": clips})
    L.save_json(session / "manifest.json", manifest)
    L.save_exclusive(session / "PREPARED",
                     "UNSCORED audio prepared; no camera or speaker was activated.\n")
    return session, manifest


def run_capture(session, manifest):
    command = manifest["command"]
    events = []
    clock = time.monotonic
    started = clock()
    process = subprocess.Popen(command, cwd=manifest["_repo"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=0, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    linebuf = []

    def drain():
        for chunk in iter(lambda: process.stdout.read(1), b""):
            if not chunk:
                break
            if chunk in (b"\n", b"\r"):
                line = "".join(linebuf).strip()
                linebuf.clear()
                if line:
                    lineq.append((line, clock()))
            else:
                linebuf.append(chunk.decode("utf-8", "replace"))

    lineq = []
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    gate = L.ReadinessGate()
    scheduler = None
    playback = L.CuePlayback(session, manifest)
    failure = None
    console = []
    try:
        while process.poll() is None:
            now = clock()
            if now - started > 120:
                raise RuntimeError("engine watchdog timeout")
            while lineq:
                line, received = lineq.pop(0)
                console.append(line)
                became = gate.feed(line, received)
                if became and scheduler is None:
                    scheduler = L.CueScheduler(manifest["mode"], received)
                    events.append({"event": "ready", "received_monotonic": received})
            if scheduler is not None:
                cue = scheduler.poll(clock(), process_alive=process.poll() is None)
                if cue:
                    playback.issue(cue, scheduler.ready_time + cue.at_seconds, events)
            time.sleep(0.005)
        if scheduler is None or not scheduler.complete:
            raise RuntimeError("engine exited before cue sequence completed")
        if process.returncode != 0:
            raise RuntimeError("engine exit code %d" % process.returncode)
    except Exception as exc:
        failure = str(exc) or "cancelled"
    finally:
        try:
            L.stop_audio()
        except Exception as exc:
            failure = failure or ("audio cleanup failed: %s" % exc)
        (session / "console.txt").write_text("\n".join(console) + "\n", encoding="utf-8")
        L.save_json(session / "cues.json", {
            "kind": "UNSCORED cue issue times; not camera timestamps",
            "mode": manifest["mode"],
            "cues": manifest["cues"],   # at_seconds values used by the gate calculator
            "events": events, "error": failure})
        L.save_json(session / "MANIFEST_STATUS.json", {
            "scored": False, "completed": failure is None,
            "trace": str(session / "recording"), "error": failure})
    if failure:
        raise RuntimeError(failure)
    return session


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=sorted(L.MOVEMENT_SECONDS), default="fast")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--preview", action="store_true",
                    help="add --preview so the operator can see the hand (UNSCORED only; drawing only)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="run the capture from an already prepared session")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    if a.preview:
        # Patching the command here keeps the scored launcher untouched. --preview changes
        # only preview drawing, not tracking, DSP, recorder or timing.
        pass
    if a.run:
        # Reuse an existing prepared session; never re-render or overwrite audio.
        session = Path(a.out_dir) / a.run_id
        if not (session / "manifest.json").is_file():
            raise RuntimeError("no prepared manifest in %s; run --prepare-only first" % session)
        if (session / "recording").exists():
            raise RuntimeError("recording already exists; use a new --run-id")
        with open(session / "manifest.json", "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        print("Reusing prepared UNSCORED session:", session)
    else:
        session, manifest = prepare_session(a, repo)
    print("UNSCORED run directory:", session)
    print("Command:", " ".join(manifest["command"]))
    if a.prepare_only or a.dry_run:
        print("Prepared only (or dry run). No camera or speaker was activated.")
        return 0
    if a.preview and not manifest.get("preview_enabled"):
        cmd = list(manifest["command"])
        if "--preview" not in cmd:
            cmd.append("--preview")
        manifest["command"] = cmd
        manifest["preview_enabled"] = True
        manifest["command_note"] = ("UNSCORED only: --preview added for operator visibility; "
                                    "preview drawing does not change tracking/DSP/recorder.")
        with open(session / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print("--preview added (UNSCORED): operator can see the hand")
    manifest["_repo"] = str(repo)
    run_capture(session, manifest)
    print("Capture finished. Run check_unscored_fast_readiness.py next.")
    print("THIS IS UNSCORED QUALIFICATION EVIDENCE, NOT A SCORED FAST TRIAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
