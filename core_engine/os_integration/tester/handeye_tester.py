"""HandEye Tester - entry point.

Double-click HandEyeTester.exe (built next to os_integration_capi.dll), or run:
    .venv\\Scripts\\python.exe core_engine\\os_integration\\tester\\handeye_tester.py

Uses the teammates' trackers unchanged and Engineer 5's OsController through
os_integration_capi.dll. Log: %LOCALAPPDATA%\\HandEyeTracker\\tester.log
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_PATH = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "HandEyeTracker" / "tester.log"


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(LOG_PATH), level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("tester")


def enable_dpi_awareness() -> None:
    # Physical pixels everywhere, so the screen map matches what SendInput uses.
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))   # per-monitor v2
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def show_fatal(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "HandEye Tester", 0x10)
    except Exception:
        print(message, file=sys.stderr)


def main() -> int:
    log = setup_logging()
    log.info("---- tester starting (python %s, %s)", sys.version.split()[0], sys.executable)
    sys.path.insert(0, str(HERE))
    enable_dpi_awareness()
    try:
        import tkinter as tk
        import osi_capi
        from tester_app import TesterApp
        library = osi_capi.OsiLibrary()
        log.info("library %s v%d", library.path, library.version)
    except Exception as error:
        log.error("startup failed:\n%s", traceback.format_exc())
        show_fatal(f"The tester could not start:\n\n{error}\n\nLog: {LOG_PATH}\n\n"
                   "If a package is missing, run setup.bat in the project folder. If the library is missing, "
                   "build core_engine/os_integration.")
        return 1

    root = tk.Tk()

    def report_callback_exception(exc, value, tb):
        log.error("UI error:\n%s", "".join(traceback.format_exception(exc, value, tb)))

    root.report_callback_exception = report_callback_exception
    TesterApp(root, library, lambda text: log.info("%s", text))
    root.mainloop()
    log.info("---- tester closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
