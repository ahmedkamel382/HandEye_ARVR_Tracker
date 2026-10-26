#!/usr/bin/env python3
"""Live-camera check for hand_index_tracker.dll through its public C ABI.

This script deliberately does not import index_tracker.py. It loads the
compiled DLL with ctypes, calls only the exported hand_index_* functions, and
never calls OS pointer/click APIs. A camera is opened only when this script is
run explicitly.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception as exc:
    print(f"FAIL camera dependencies: {exc}")
    raise SystemExit(2)


HAND_INDEX_OK = 0
HAND_INDEX_STATUS_TRACKED = 1
HAND_INDEX_STATUS_LOST = 0
HAND_INDEX_LANDMARK_INDEX = 8


class HandIndexConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("python_dir", ctypes.c_char_p),
        ("site_packages_dir", ctypes.c_char_p),
        ("model_path", ctypes.c_char_p),
        ("tracking_region", ctypes.c_float * 4),
        ("smoothing_alpha", ctypes.c_float),
        ("min_hand_detection_confidence", ctypes.c_float),
        ("min_hand_presence_confidence", ctypes.c_float),
        ("min_tracking_confidence", ctypes.c_float),
    ]


class HandIndexResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("valid", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("landmark_index", ctypes.c_int32),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("camera_x", ctypes.c_float),
        ("camera_y", ctypes.c_float),
    ]


class ProbeFailure(Exception):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def load_dll(path: Path):
    if not path.is_file():
        raise ProbeFailure(f"DLL not found: {path}", 2)
    try:
        dll = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ProbeFailure(f"cannot load DLL {path}: {exc}", 2) from exc

    dll.hand_index_version.argtypes = []
    dll.hand_index_version.restype = ctypes.c_char_p
    dll.hand_index_abi_version.argtypes = []
    dll.hand_index_abi_version.restype = ctypes.c_uint32
    dll.hand_index_error_string.argtypes = [ctypes.c_int]
    dll.hand_index_error_string.restype = ctypes.c_char_p
    dll.hand_index_config_default.argtypes = [ctypes.POINTER(HandIndexConfig)]
    dll.hand_index_config_default.restype = None
    dll.hand_index_result_init.argtypes = [ctypes.POINTER(HandIndexResult)]
    dll.hand_index_result_init.restype = None
    dll.hand_index_create.argtypes = [
        ctypes.POINTER(HandIndexConfig),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    dll.hand_index_create.restype = ctypes.c_int
    dll.hand_index_process_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(HandIndexResult),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    dll.hand_index_process_bgr.restype = ctypes.c_int
    dll.hand_index_destroy.argtypes = [ctypes.c_void_p]
    dll.hand_index_destroy.restype = None
    return dll


def error_text(dll, code: int) -> str:
    value = dll.hand_index_error_string(code)
    return value.decode("utf-8", errors="replace") if value else str(code)


def open_camera(index: int):
    for backend, label in ((cv2.CAP_DSHOW, "DirectShow"), (0, "default")):
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, label
        cap.release()
    return None, ""


def validate_tracked(result: HandIndexResult) -> None:
    if result.valid != 1:
        raise ProbeFailure("valid result does not set valid=1", 5)
    if result.status != HAND_INDEX_STATUS_TRACKED:
        raise ProbeFailure("valid result does not report TRACKED", 5)
    if result.landmark_index != HAND_INDEX_LANDMARK_INDEX:
        raise ProbeFailure("result does not report landmark index 8", 5)
    for name in ("x", "y", "camera_x", "camera_y"):
        value = float(getattr(result, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProbeFailure(f"{name} is not a finite normalized coordinate: {value}", 5)


def validate_lost(result: HandIndexResult) -> None:
    if result.valid != 0:
        raise ProbeFailure("lost result sets valid=1", 5)
    if result.status != HAND_INDEX_STATUS_LOST:
        raise ProbeFailure("lost result does not report LOST", 5)
    for name in ("x", "y", "camera_x", "camera_y"):
        value = float(getattr(result, name))
        if value != -1.0:
            raise ProbeFailure(f"lost result has stale {name}={value}", 5)


def run(args: argparse.Namespace) -> int:
    dll = load_dll(args.dll)
    version = dll.hand_index_version().decode("utf-8", errors="replace")
    abi_version = dll.hand_index_abi_version()
    print(f"DLL: {args.dll}")
    print(f"Version: {version}, ABI: {abi_version}")
    if abi_version != 1:
        raise ProbeFailure(f"unsupported DLL ABI version: {abi_version}", 2)

    config = HandIndexConfig()
    dll.hand_index_config_default(ctypes.byref(config))
    if config.struct_size != ctypes.sizeof(HandIndexConfig):
        raise ProbeFailure("camera probe ABI does not match the DLL config struct", 2)

    probe_result = HandIndexResult()
    dll.hand_index_result_init(ctypes.byref(probe_result))
    if probe_result.struct_size != ctypes.sizeof(HandIndexResult):
        raise ProbeFailure("camera probe ABI does not match the DLL result struct", 2)

    python_dir = str(args.python_dir).encode("utf-8")
    model_path = str(args.model).encode("utf-8")
    site_packages = (
        str(args.site_packages).encode("utf-8") if args.site_packages else None
    )
    config.python_dir = python_dir
    config.model_path = model_path
    config.site_packages_dir = site_packages

    error_buffer = ctypes.create_string_buffer(1024)
    tracker = ctypes.c_void_p()
    code = dll.hand_index_create(
        ctypes.byref(config),
        ctypes.byref(tracker),
        error_buffer,
        len(error_buffer),
    )
    if code != HAND_INDEX_OK or not tracker.value:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise ProbeFailure(
            f"hand_index_create failed: {error_text(dll, code)}: {message or '(no message)'}",
            4,
        )
    print("PASS MediaPipe tracker initialized for camera test")

    cap = None
    exit_code = 0
    try:
        cap, backend = open_camera(args.camera_index)
        if cap is None:
            raise ProbeFailure(
                f"cannot open camera index {args.camera_index}; close other camera apps "
                "or choose another index with -CameraIndex",
                3,
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Camera opened: index={args.camera_index}, backend={backend}, "
            f"size={width}x{height}, duration={args.seconds:.1f}s"
        )
        print("Show one hand to the camera. This probe never moves the mouse or clicks.")

        deadline = time.monotonic() + args.seconds
        frames = 0
        tracked = 0
        lost = 0
        read_failures = 0
        seen_tracked = False
        reacquired = False
        previous_valid = False
        last_report = time.monotonic()

        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= 10:
                    raise ProbeFailure(
                        "camera stopped returning frames; it may be busy or disconnected",
                        3,
                    )
                time.sleep(0.05)
                continue
            read_failures = 0
            frame = np.ascontiguousarray(frame)

            result = HandIndexResult()
            dll.hand_index_result_init(ctypes.byref(result))
            error_buffer.value = b"\x00"
            code = dll.hand_index_process_bgr(
                tracker,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                frame.shape[1],
                frame.shape[0],
                int(frame.strides[0]),
                ctypes.byref(result),
                error_buffer,
                len(error_buffer),
            )
            if code != HAND_INDEX_OK:
                message = error_buffer.value.decode("utf-8", errors="replace")
                raise ProbeFailure(
                    f"hand_index_process_bgr failed: {error_text(dll, code)}: "
                    f"{message or '(no message)'}",
                    5,
                )

            frames += 1
            if result.valid:
                validate_tracked(result)
                tracked += 1
                if seen_tracked and not previous_valid:
                    reacquired = True
                seen_tracked = True
                previous_valid = True
            else:
                validate_lost(result)
                lost += 1
                previous_valid = False

            now = time.monotonic()
            if now - last_report >= 2.0:
                print(f"  frames={frames} tracked={tracked} lost={lost}")
                last_report = now

        if frames == 0:
            raise ProbeFailure("camera opened but supplied no frames", 3)
        if tracked == 0:
            raise ProbeFailure(
                "camera opened and frames were processed, but no hand was tracked; "
                "check lighting and keep one hand visible",
                6,
            )
        print(
            f"PASS live camera frames={frames} tracked={tracked} lost={lost} "
            f"reacquired={str(reacquired).lower()}"
        )
    except ProbeFailure as exc:
        print(f"FAIL {exc}")
        exit_code = exc.exit_code
    finally:
        try:
            if cap is not None:
                cap.release()
                print("PASS camera released")
        finally:
            dll.hand_index_destroy(tracker)
            print("PASS tracker destroyed")
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise hand_index_tracker.dll with a live camera."
    )
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--python-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ProbeFailure as exc:
        print(f"FAIL {exc}")
        raise SystemExit(exc.exit_code)
    except KeyboardInterrupt:
        print("FAIL camera test interrupted")
        raise SystemExit(130)
