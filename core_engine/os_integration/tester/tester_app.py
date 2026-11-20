"""HandEye Tester: the window.

Pick what to test (index finger only, hand gestures, gaze, winks, the full
system), watch the camera with overlays, see where the cursor would go, tune
the settings live, run an accuracy test and record sessions to CSV.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

import osi_capi
from tester_engine import (EYE_STATES, HAND_STATES, MODES, REPO, TRACKING_NAMES, IniDoc, TesterEngine,
                           prepare_ai_imports, user_settings_path)

# ----------------------------------------------------------------------------- theme

BG = "#0f1419"
PANEL = "#161c24"
PANEL_2 = "#1d2530"
BORDER = "#2a3441"
TEXT = "#e6edf3"
MUTED = "#8b98a7"
ACCENT = "#4cc2ff"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 10)
FONT_TITLE = ("Segoe UI Semibold", 17)
FONT_HEAD = ("Segoe UI Semibold", 12)
FONT_MONO = ("Consolas", 9)


def apply_theme(root: tk.Tk) -> None:
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL_2, bordercolor=BORDER,
                    lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL_2, focuscolor=ACCENT, font=FONT,
                    selectbackground=ACCENT, selectforeground=BG, insertcolor=TEXT)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Card.TFrame", background=PANEL_2)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Card.TLabel", background=PANEL_2)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONT_SMALL)
    style.configure("CardMuted.TLabel", background=PANEL_2, foreground=MUTED, font=FONT_SMALL)
    style.configure("Head.TLabel", background=PANEL, foreground=TEXT, font=FONT_HEAD)
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
    style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=FONT_SMALL)
    style.configure("Value.TLabel", background=PANEL, foreground=TEXT, font=FONT_MONO)
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 7), borderwidth=0,
                    font=FONT_BOLD)
    style.map("TNotebook.Tab", background=[("selected", PANEL_2)], foreground=[("selected", ACCENT)])
    style.configure("TButton", background=PANEL_2, foreground=TEXT, borderwidth=1, padding=(12, 6),
                    font=FONT_BOLD)
    style.map("TButton", background=[("active", BORDER), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
    style.configure("Accent.TButton", background=ACCENT, foreground=BG)
    style.map("Accent.TButton", background=[("active", "#7fd3ff"), ("disabled", PANEL)])
    style.configure("Danger.TButton", background="#3a1f24", foreground=BAD)
    style.map("Danger.TButton", background=[("active", "#55262d")])
    for name in ("TCheckbutton", "TRadiobutton"):
        style.configure(name, background=PANEL, foreground=TEXT, indicatorbackground=PANEL_2,
                        indicatorforeground=ACCENT, font=FONT)
        style.map(name, background=[("active", PANEL)], indicatorbackground=[("selected", ACCENT)])
    style.configure("Card.TRadiobutton", background=PANEL_2, font=FONT_BOLD)
    style.map("Card.TRadiobutton", background=[("active", PANEL_2)])
    style.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL_2, sliderthickness=14)
    style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2, foreground=TEXT, arrowcolor=TEXT)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL_2)], foreground=[("readonly", TEXT)])
    style.configure("TSpinbox", fieldbackground=PANEL_2, foreground=TEXT, arrowcolor=TEXT)
    style.configure("Vertical.TScrollbar", background=PANEL_2, troughcolor=PANEL, arrowcolor=MUTED)
    root.option_add("*TCombobox*Listbox.background", PANEL_2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)


class Scrollable(ttk.Frame):
    """A vertically scrolling panel (for the long tabs)."""

    def __init__(self, parent):
        super().__init__(parent, style="Panel.TFrame")
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, style="Panel.TFrame", padding=(14, 12))
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


def section(parent, title: str, hint: str = "") -> ttk.Frame:
    ttk.Label(parent, text=title, style="Head.TLabel").pack(anchor="w", pady=(10, 2))
    if hint:
        ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=420, justify="left").pack(anchor="w")
    frame = ttk.Frame(parent, style="Panel.TFrame")
    frame.pack(fill="x", pady=(6, 4))
    return frame


class Pill(tk.Label):
    """Small coloured status badge."""

    def __init__(self, parent, text: str):
        super().__init__(parent, text=text, bg=PANEL_2, fg=MUTED, font=FONT_SMALL, padx=10, pady=3)

    def set(self, text: str, colour: str):
        self.configure(text=f"● {text}", fg=colour)


# ----------------------------------------------------------------------------- tuning model

class Slider:
    def __init__(self, parent, label: str, lo: float, hi: float, fmt: str, on_change: Callable[[], None],
                 hint: str = ""):
        self.fmt = fmt
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=3)
        top = ttk.Frame(row, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="Panel.TLabel").pack(side="left")
        self.value_label = ttk.Label(top, text="", style="Value.TLabel")
        self.value_label.pack(side="right")
        self.var = tk.DoubleVar(value=lo)
        ttk.Scale(row, from_=lo, to=hi, variable=self.var,
                  command=lambda _=None: (self._show(), on_change())).pack(fill="x")
        if hint:
            ttk.Label(row, text=hint, style="Muted.TLabel", wraplength=420, justify="left").pack(anchor="w")

    def _show(self):
        self.value_label.configure(text=self.fmt.format(self.var.get()))

    def get(self) -> float:
        return float(self.var.get())

    def set(self, value: float):
        self.var.set(value)
        self._show()


# ----------------------------------------------------------------------------- app

class TesterApp:
    TICK_MS = 33

    def __init__(self, root: tk.Tk, library: osi_capi.OsiLibrary, log: Callable[[str], None]):
        self.root = root
        self.lib = library
        self.log_file = log
        self.engine = TesterEngine(library, self._log_from_engine)
        self._pending_log: List[str] = []
        self._log_lock = threading.Lock()
        self._photo = None
        self._tuning_after = None
        self._loading_tuning = False
        self._prev_click_total = 0
        self._screen_trail: List[Tuple[float, float]] = []
        self.test = AccuracyTest()

        root.title("HandEye Tester")
        # Fit the usable screen area (above the taskbar), centred.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(1360, sw - 60), min(860, sh - 110)
        root.geometry(f"{width}x{height}+{max(0, (sw - width) // 2)}+{max(0, (sh - height) // 2 - 20)}")
        root.minsize(min(1100, width), min(680, height))
        apply_theme(root)
        self._set_icon()
        self._build()
        self._load_tuning_from_settings()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<F5>", lambda e: self.toggle_run())
        root.bind("<space>", lambda e: self.toggle_active() if self.engine.running else None)
        self.root.after(self.TICK_MS, self.tick)
        self.log("Ready. Choose what to test, then press Start (F5).")

    # ---------------------------------------------------------------- layout
    def _set_icon(self):
        icon = tk.PhotoImage(width=32, height=32)
        for y in range(32):
            for x in range(32):
                d = math.hypot(x - 15.5, y - 15.5)
                if d < 6:
                    icon.put("#ffffff", (x, y))
                elif d < 14:
                    icon.put(ACCENT, (x, y))
        self.root.iconphoto(True, icon)
        self._icon = icon

    def _build(self):
        header = ttk.Frame(self.root, padding=(18, 12, 18, 6))
        header.pack(fill="x")
        left = ttk.Frame(header)
        left.pack(side="left")
        ttk.Label(left, text="HandEye Tester", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="Test each part of the hand & eye virtual mouse on its own, or the whole system.",
                  style="Sub.TLabel").pack(anchor="w")
        pills = ttk.Frame(header)
        pills.pack(side="right")
        self.pill_camera = Pill(pills, "Camera")
        self.pill_track = Pill(pills, "Tracking")
        self.pill_output = Pill(pills, "Output")
        for p in (self.pill_camera, self.pill_track, self.pill_output):
            p.pack(side="left", padx=4)
        self.pill_camera.set("Camera off", MUTED)
        self.pill_track.set("Not tracking", MUTED)
        self.pill_output.set("Simulation", GOOD)

        body = ttk.Frame(self.root, padding=(18, 4, 18, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left: camera + screen map
        view = ttk.Frame(body, style="Panel.TFrame", padding=10)
        view.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=4)
        view.rowconfigure(2, weight=1)
        # A fixed container, so the picture follows the layout instead of driving it.
        holder = tk.Frame(view, bg="#05080b", height=420)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.pack_propagate(False)
        self.video = tk.Label(holder, bg="#05080b", fg=MUTED, font=FONT_HEAD,
                              text="Camera preview\n\nChoose a test on the right and press Start")
        self.video.pack(fill="both", expand=True)
        screen_head = ttk.Frame(view, style="Panel.TFrame")
        screen_head.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        ttk.Label(screen_head, text="Where the cursor goes", style="Head.TLabel").pack(side="left")
        self.test_button = ttk.Button(screen_head, text="Start accuracy test", command=self.toggle_test)
        self.test_button.pack(side="right")
        self.test_label = ttk.Label(screen_head, text="", style="Muted.TLabel")
        self.test_label.pack(side="right", padx=10)
        self.screen = tk.Canvas(view, bg="#05080b", highlightthickness=0, height=170)
        self.screen.grid(row=2, column=0, sticky="nsew")

        # Right: controls
        tabs = ttk.Notebook(body)
        tabs.grid(row=0, column=1, sticky="nsew")
        self.tabs = tabs
        self._build_test_tab(tabs)
        self._build_tuning_tab(tabs)
        self._build_live_tab(tabs)
        self._build_log_tab(tabs)
        self._build_help_tab(tabs)

        footer = ttk.Frame(self.root, padding=(18, 0, 18, 10))
        footer.pack(fill="x")
        self.footer = ttk.Label(footer, text="F5 start/stop  ·  Space activate/pause output  ·  "
                                "Esc pauses the real mouse instantly  ·  moving the real mouse takes over",
                                style="Sub.TLabel")
        self.footer.pack(side="left")
        self.status = ttk.Label(footer, text="Stopped", style="Sub.TLabel")
        self.status.pack(side="right")

    def _build_test_tab(self, tabs):
        page = Scrollable(tabs)
        tabs.add(page, text="  Test  ")
        b = page.body

        modes = section(b, "1  What do you want to test?")
        self.mode_var = tk.StringVar(value="index")
        for mode in MODES.values():
            card = ttk.Frame(modes, style="Card.TFrame", padding=(10, 7))
            card.pack(fill="x", pady=3)
            rb = ttk.Radiobutton(card, text=f"{mode.title}", value=mode.key, variable=self.mode_var,
                                 style="Card.TRadiobutton", command=self._mode_changed)
            rb.pack(anchor="w")
            desc = ttk.Label(card, text=mode.description, style="CardMuted.TLabel", wraplength=410, justify="left")
            desc.pack(anchor="w", padx=(22, 0))
            for widget in (card, desc):
                widget.bind("<Button-1>", lambda e, k=mode.key: (self.mode_var.set(k), self._mode_changed()))

        output = section(b, "2  Output", "Simulation runs Engineer 5's real mouse logic but never touches the "
                                         "mouse. Real mouse moves and clicks for real (starts paused).")
        self.real_var = tk.BooleanVar(value=False)
        ttk.Radiobutton(output, text="Simulation (safe: nothing is clicked)", value=False, variable=self.real_var,
                        command=self._output_changed).pack(anchor="w")
        ttk.Radiobutton(output, text="Real mouse (moves and clicks for real)", value=True, variable=self.real_var,
                        command=self._output_changed).pack(anchor="w")
        self.monitors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(output, text="Use all monitors (default: primary monitor only)",
                        variable=self.monitors_var, command=self._options_changed).pack(anchor="w", pady=(4, 0))

        options = section(b, "3  Options")
        self.clicks_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=True)
        self.landmarks_var = tk.BooleanVar(value=True)
        self.zone_var = tk.BooleanVar(value=True)
        self.trail_var = tk.BooleanVar(value=True)
        self.lowlight_var = tk.BooleanVar(value=True)
        self.roi_var = tk.BooleanVar(value=True)
        for text, var in [("Allow clicks (pinch, fist, winks)", self.clicks_var),
                          ("Smoothing (Engineer 4's 1-Euro filter, like the engine)", self.smooth_var),
                          ("Draw landmarks", self.landmarks_var), ("Draw the hand's active zone", self.zone_var),
                          ("Draw the pointer trail", self.trail_var),
                          ("Hand: low-light enhancement", self.lowlight_var),
                          ("Hand: recover a lost hand near its last position", self.roi_var)]:
            ttk.Checkbutton(options, text=text, variable=var, command=self._options_changed).pack(anchor="w")
        self.index_alpha = Slider(options, "Index finger smoothing (Engineer 2's tracker)", 0.05, 1.0, "{:.2f}",
                                  self._options_changed, "Lower = steadier but slower. 1.00 = no smoothing.")
        self.index_alpha.set(0.35)
        cam = ttk.Frame(options, style="Panel.TFrame")
        cam.pack(fill="x", pady=(6, 0))
        ttk.Label(cam, text="Camera", style="Panel.TLabel").pack(side="left")
        self.camera_var = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=9, width=4, textvariable=self.camera_var,
                    command=self._options_changed).pack(side="left", padx=6)
        self.res_var = tk.StringVar(value="640 x 480")
        res = ttk.Combobox(cam, values=["640 x 480", "960 x 540", "1280 x 720"], textvariable=self.res_var,
                           state="readonly", width=11)
        res.pack(side="left", padx=6)
        res.bind("<<ComboboxSelected>>", lambda e: self._options_changed())

        actions = section(b, "4  Run")
        row1 = ttk.Frame(actions, style="Panel.TFrame")
        row1.pack(fill="x")
        self.run_button = ttk.Button(row1, text="▶  Start", style="Accent.TButton", command=self.toggle_run)
        self.run_button.pack(side="left", fill="x", expand=True)
        self.active_button = ttk.Button(row1, text="Pause output", command=self.toggle_active, state="disabled")
        self.active_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        row2 = ttk.Frame(actions, style="Panel.TFrame")
        row2.pack(fill="x", pady=(8, 0))
        self.record_button = ttk.Button(row2, text="●  Record CSV", command=self.toggle_record)
        self.record_button.pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Run checks", command=self.run_checks).pack(side="left", fill="x", expand=True,
                                                                         padx=(8, 0))

    def _build_tuning_tab(self, tabs):
        page = Scrollable(tabs)
        tabs.add(page, text="  Tuning  ")
        b = page.body
        ttk.Label(b, text="Changes apply instantly to the running test. They are NOT saved unless you press "
                          "“Save as my settings”.", style="Muted.TLabel", wraplength=420,
                  justify="left").pack(anchor="w")
        ch = self._tuning_changed
        s = {}
        g = section(b, "Cursor")
        s["zone"] = Slider(g, "Hand active zone margin", 0.0, 0.35, "{:.2f}", ch,
                           "How much of the camera edge is ignored. Bigger = smaller hand movements reach the "
                           "screen edges.")
        s["gaze"] = Slider(g, "Gaze range (degrees each way)", 4, 30, "±{:.0f}°", ch,
                           "How far you look to reach the screen edge.")
        self.source_var = tk.StringVar(value="hand_preferred")
        row = ttk.Frame(g, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Cursor follows", style="Panel.TLabel").pack(side="left")
        combo = ttk.Combobox(row, values=["hand_preferred", "hand_only", "gaze_only"], textvariable=self.source_var,
                             state="readonly", width=16)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", lambda e: ch())
        s["snap"] = Slider(g, "Click snap-back", 0.0, 0.3, "{:.2f} s", ch,
                           "How far back in time the cursor jumps when a click starts.")
        s["drag"] = Slider(g, "Drag threshold", 0, 120, "{:.0f} px", ch,
                           "How far a held pinch must move before it becomes a drag.")
        g = section(b, "Gestures")
        s["pinch"] = Slider(g, "Pinch must last", 0.0, 0.3, "{:.2f} s", ch)
        s["release"] = Slider(g, "Release must last", 0.0, 0.3, "{:.2f} s", ch)
        s["fist"] = Slider(g, "Fist must last", 0.1, 1.0, "{:.2f} s", ch)
        s["wink"] = Slider(g, "Wink must last", 0.05, 0.8, "{:.2f} s", ch,
                           "Longer = fewer accidental clicks from normal blinks.")
        s["closure"] = Slider(g, "Eyes closed to pause", 0.3, 3.0, "{:.2f} s", ch)
        self.fist_var = tk.StringVar(value="right_click")
        row = ttk.Frame(g, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Fist does", style="Panel.TLabel").pack(side="left")
        combo = ttk.Combobox(row, values=["right_click", "middle_click", "toggle_pause", "none"],
                             textvariable=self.fist_var, state="readonly", width=16)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", lambda e: ch())
        g = section(b, "Smoothing (Engineer 4's filter)", "Lower minimum cutoff = steadier but laggier. "
                                                          "Higher beta = faster on quick moves.")
        s["h_cut"] = Slider(g, "Hand minimum cutoff", 0.1, 5.0, "{:.2f} Hz", ch)
        s["h_beta"] = Slider(g, "Hand beta", 0.0, 20.0, "{:.1f}", ch)
        s["g_cut"] = Slider(g, "Gaze minimum cutoff", 0.1, 5.0, "{:.2f} Hz", ch)
        s["g_beta"] = Slider(g, "Gaze beta", 0.0, 20.0, "{:.1f}", ch)
        self.sliders: Dict[str, Slider] = s
        buttons = section(b, "Settings")
        for text, command in [("Reset to defaults", self.reset_tuning),
                              ("Load my saved settings", self.load_user_settings),
                              ("Save as my settings", self.save_user_settings)]:
            ttk.Button(buttons, text=text, command=command).pack(fill="x", pady=2)
        self.tuning_status = ttk.Label(b, text="", style="Muted.TLabel", wraplength=420)
        self.tuning_status.pack(anchor="w", pady=6)

    def _build_live_tab(self, tabs):
        page = Scrollable(tabs)
        tabs.add(page, text="  Live data  ")
        b = page.body
        self.live: Dict[str, ttk.Label] = {}
        groups = [
            ("Hand (Engineer 2)", ["Tracker status", "Freshness", "Pointer", "Gesture", "Confidence",
                                   "Time per frame"]),
            ("Eyes (Engineer 3)", ["Face", "Eye state", "Gaze angle", "Head angle", "Gaze pointer",
                                   "Time per frame "]),
            ("OS stage (Engineer 5)", ["Cursor follows", "Cursor pixel", "Output", "Confirmed gesture",
                                       "Confirmed eye", "Stuck button"]),
            ("Counters", ["Left down / up", "Left clicks", "Right clicks", "Middle clicks", "Pauses",
                          "Refused by Windows"]),
            ("Quality", ["Frame rate", "Processing", "Brightness", "Camera size"]),
        ]
        for title, names in groups:
            g = section(b, title)
            g.columnconfigure(1, weight=1)
            for i, name in enumerate(names):
                ttk.Label(g, text=name.strip(), style="Muted.TLabel").grid(row=i, column=0, sticky="w", pady=1)
                value = ttk.Label(g, text="-", style="Value.TLabel")
                value.grid(row=i, column=1, sticky="e", pady=1)
                self.live[name] = value
            if title == "Quality":
                self.quality_bar = tk.Canvas(g, height=14, bg=PANEL_2, highlightthickness=0)
                self.quality_bar.grid(row=len(names), column=0, columnspan=2, sticky="ew", pady=(6, 0))
                ttk.Label(g, text="tracked / held / lost over the last few seconds", style="Muted.TLabel").grid(
                    row=len(names) + 1, column=0, columnspan=2, sticky="w")

    def _build_log_tab(self, tabs):
        page = ttk.Frame(tabs, style="Panel.TFrame", padding=10)
        tabs.add(page, text="  Log  ")
        self.log_text = tk.Text(page, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat", font=FONT_MONO,
                                wrap="word", state="disabled", padx=8, pady=6)
        self.log_text.pack(fill="both", expand=True)
        row = ttk.Frame(page, style="Panel.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Clear", command=self._clear_log).pack(side="left")
        ttk.Button(row, text="Save log...", command=self._save_log).pack(side="left", padx=8)

    def _build_help_tab(self, tabs):
        page = Scrollable(tabs)
        tabs.add(page, text="  Help  ")
        b = page.body
        text = [
            ("What is this?", "A test bench for the HandEye tracker. It runs your teammates' trackers exactly as "
             "they are and feeds them into Engineer 5's real mouse logic (OsController), so you can check one part "
             "at a time."),
            ("How to test", "1. Pick a test.  2. Keep Simulation on.  3. Press Start (F5).  Watch the camera view "
             "and the 'Where the cursor goes' map. Switch tests while running; the camera stays on."),
            ("Index finger only", "Point with your index finger. The cursor should follow smoothly with no clicks. "
             "Good first test for lighting and camera position."),
            ("Hand + gestures", "Pinch thumb and index = left button (keep pinching and move to drag). Make a fist "
             "and hold 0.3 s = right click. The Live data tab counts every action."),
            ("Gaze and winks", "Look at the screen with both eyes open for the first second (the eye model "
             "calibrates). Wink deliberately (about half a second) to click; close both eyes for a second to pause."),
            ("Accuracy test", "Press 'Start accuracy test'. Move the cursor onto each target: click it (pinch, wink "
             "or fist), or rest on it for 1 s in Index finger mode. You get hits, time and error at the end."),
            ("Real mouse", "Switch Output to Real mouse, press Activate. Esc pauses instantly; moving the real mouse "
             "takes over for 1.5 s. Do not run the tester and the engine at the same time (one camera)."),
            ("Recording", "Record CSV saves every frame (positions, states, actions, frame rate) to "
             "Documents\\HandEyeTests for later analysis."),
            ("Files", f"Tester log: %LOCALAPPDATA%\\HandEyeTracker\\tester.log\nYour settings: {user_settings_path()}"),
        ]
        for title, body in text:
            ttk.Label(b, text=title, style="Head.TLabel").pack(anchor="w", pady=(10, 2))
            ttk.Label(b, text=body, style="Panel.TLabel", wraplength=420, justify="left").pack(anchor="w")

    # ---------------------------------------------------------------- logging
    def _log_from_engine(self, text: str):
        self.log_file(text)
        with self._log_lock:
            self._pending_log.append(text)

    def log(self, text: str):
        self.log_file(text)
        self._append_log(text)

    def _append_log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{stamp}  {text}\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 2000:
            self.log_text.delete("1.0", f"{lines - 2000}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")],
                                            initialfile="handeye_tester_log.txt")
        if path:
            Path(path).write_text(self.log_text.get("1.0", "end"), encoding="utf-8")

    # ---------------------------------------------------------------- config changes
    def _config(self) -> dict:
        w, h = (int(v) for v in self.res_var.get().split("x"))
        try:
            camera = int(self.camera_var.get())
        except (tk.TclError, ValueError):
            camera = 0
        return dict(mode=self.mode_var.get(), camera_index=camera, width=w, height=h, real_mouse=self.real_var.get(),
                    all_monitors=self.monitors_var.get(), smoothing=self.smooth_var.get(), clicks=self.clicks_var.get(),
                    lowlight=self.lowlight_var.get(), roi_recovery=self.roi_var.get(),
                    index_alpha=self.index_alpha.get(), show_landmarks=self.landmarks_var.get(),
                    show_zone=self.zone_var.get(), show_trail=self.trail_var.get())

    def _mode_changed(self):
        self.engine.configure(**self._config())
        self._screen_trail.clear()

    def _options_changed(self):
        self.engine.configure(**self._config())

    def _output_changed(self):
        if self.real_var.get():
            if not messagebox.askokcancel(
                    "Real mouse",
                    "The tester will move and click your real mouse.\n\n"
                    "It starts PAUSED: press Activate when ready.\n"
                    "Esc pauses instantly. Moving the real mouse takes over for 1.5 s.",
                    icon="warning", parent=self.root):
                self.real_var.set(False)
                return
        self.engine.configure(**self._config())

    # ---------------------------------------------------------------- run control
    def toggle_run(self):
        if self.engine.running:
            self.engine.stop()
            self.run_button.configure(text="▶  Start")
            self.active_button.configure(state="disabled")
            self.log("Stopped")
            self.video.configure(image="", text="Stopped\n\nPress Start (F5) to continue")
            self._photo = None
        else:
            self.engine.configure(**self._config())
            self.engine.set_settings_text(self._tuning_ini())
            self.engine.start()
            self.run_button.configure(text="■  Stop")
            self.active_button.configure(state="normal")

    def toggle_active(self):
        active = self.engine.snapshot().report.active
        self.engine.set_active(not active)

    def toggle_record(self):
        if self.engine.snapshot().recording:
            self.engine.stop_recording()
            self.record_button.configure(text="●  Record CSV")
            return
        if not self.engine.running:
            messagebox.showinfo("Record", "Start a test first.", parent=self.root)
            return
        folder = Path.home() / "Documents" / "HandEyeTests"
        path = folder / f"{self.mode_var.get()}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.engine.start_recording(path)
        self.record_button.configure(text="■  Stop recording")

    # ---------------------------------------------------------------- tuning
    def _load_tuning_from_settings(self):
        doc = IniDoc(self.engine.settings_text())
        self._loading_tuning = True
        try:
            hand = _region(doc.get("mapping", "hand_region"), (0.15, 0.15, 0.85, 0.85))
            gaze = _region(doc.get("mapping", "gaze_region"), (0.3, 0.3, 0.7, 0.7))
            s = self.sliders
            s["zone"].set(max(0.0, min(0.35, min(hand[0], hand[1], 1 - hand[2], 1 - hand[3]))))
            s["gaze"].set(max(4.0, min(30.0, abs(gaze[2] - gaze[0]) / 2 * 60)))
            s["snap"].set(doc.get_float("stabilization", "anchor_lookback_seconds", 0.1))
            s["drag"].set(doc.get_float("stabilization", "drag_threshold_px", 30))
            s["pinch"].set(doc.get_float("gestures", "pinch_onset_seconds", 0.06))
            s["release"].set(doc.get_float("gestures", "pinch_release_seconds", 0.08))
            s["fist"].set(doc.get_float("gestures", "fist_onset_seconds", 0.3))
            s["wink"].set(doc.get_float("gestures", "wink_onset_seconds", 0.25))
            s["closure"].set(doc.get_float("gestures", "closure_onset_seconds", 1.0))
            s["h_cut"].set(doc.get_float("hand_filter", "min_cutoff_hz", 1.2))
            s["h_beta"].set(doc.get_float("hand_filter", "beta", 4.0))
            s["g_cut"].set(doc.get_float("gaze_filter", "min_cutoff_hz", 0.8))
            s["g_beta"].set(doc.get_float("gaze_filter", "beta", 6.0))
            self.source_var.set(doc.get("mapping", "cursor_source") or "hand_preferred")
            self.fist_var.set(doc.get("gestures", "fist_action") or "right_click")
            self._gaze_inverted = gaze[2] < gaze[0], gaze[3] < gaze[1]
            self._gaze_centre = ((gaze[0] + gaze[2]) / 2, (gaze[1] + gaze[3]) / 2)
        finally:
            self._loading_tuning = False

    def _tuning_ini(self) -> str:
        doc = IniDoc(self.engine.settings_text())
        s = self.sliders
        m = round(s["zone"].get(), 3)
        doc.set("mapping", "hand_region", (m, m, 1 - m, 1 - m))
        half = s["gaze"].get() / 60.0
        cx, cy = getattr(self, "_gaze_centre", (0.5, 0.5))
        inv_x, inv_y = getattr(self, "_gaze_inverted", (False, False))
        x0, x1 = (cx + half, cx - half) if inv_x else (cx - half, cx + half)
        y0, y1 = (cy + half, cy - half) if inv_y else (cy - half, cy + half)
        doc.set("mapping", "gaze_region", tuple(round(v, 4) for v in (x0, y0, x1, y1)))
        doc.set("mapping", "cursor_source", self.source_var.get())
        doc.set("stabilization", "anchor_lookback_seconds", round(s["snap"].get(), 3))
        doc.set("stabilization", "drag_threshold_px", round(s["drag"].get(), 1))
        doc.set("gestures", "pinch_onset_seconds", round(s["pinch"].get(), 3))
        doc.set("gestures", "pinch_release_seconds", round(s["release"].get(), 3))
        doc.set("gestures", "fist_onset_seconds", round(s["fist"].get(), 3))
        doc.set("gestures", "wink_onset_seconds", round(s["wink"].get(), 3))
        doc.set("gestures", "closure_onset_seconds", round(s["closure"].get(), 3))
        doc.set("gestures", "fist_action", self.fist_var.get())
        doc.set("hand_filter", "min_cutoff_hz", round(s["h_cut"].get(), 3))
        doc.set("hand_filter", "beta", round(s["h_beta"].get(), 2))
        doc.set("gaze_filter", "min_cutoff_hz", round(s["g_cut"].get(), 3))
        doc.set("gaze_filter", "beta", round(s["g_beta"].get(), 2))
        return doc.text()

    def _tuning_changed(self):
        if self._loading_tuning:
            return
        if self._tuning_after:
            self.root.after_cancel(self._tuning_after)
        self._tuning_after = self.root.after(250, self._push_tuning)

    def _push_tuning(self):
        self._tuning_after = None
        self.engine.set_settings_text(self._tuning_ini())
        self.tuning_status.configure(text="Applied to the running test (not saved).", foreground=MUTED)

    def reset_tuning(self):
        self.engine.reset_settings()
        self._load_tuning_from_settings()
        self.tuning_status.configure(text="Defaults restored (not saved).", foreground=MUTED)

    def load_user_settings(self):
        path = user_settings_path()
        if not path.is_file():
            messagebox.showinfo("Settings", f"No saved settings yet:\n{path}", parent=self.root)
            return
        self.engine.set_settings_text(path.read_text(encoding="utf-8"))
        self._load_tuning_from_settings()
        self.tuning_status.configure(text=f"Loaded {path}", foreground=MUTED)

    def save_user_settings(self):
        path = user_settings_path()
        if not messagebox.askyesno("Save settings",
                                   f"Save these values as your settings?\n\n{path}\n\n"
                                   "The engine and the tray app will use them next time they start.",
                                   parent=self.root):
            return
        text = self._tuning_ini()
        probe_error = self._validate(text)
        if probe_error:
            messagebox.showerror("Settings", f"Not saved: {probe_error}", parent=self.root)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".ini.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        self.tuning_status.configure(text=f"Saved to {path}", foreground=GOOD)
        self.log(f"Settings saved to {path}")

    def _validate(self, text: str) -> Optional[str]:
        try:
            probe = self.lib.controller(text, False, False)
            probe.close()
            return None
        except ValueError as error:
            return str(error)

    # ---------------------------------------------------------------- accuracy test
    def toggle_test(self):
        if self.test.running:
            self.test.stop()
            self.test_button.configure(text="Start accuracy test")
            self.test_label.configure(text=self.test.summary())
            self.log("Accuracy test: " + self.test.summary())
            return
        if not self.engine.running:
            messagebox.showinfo("Accuracy test", "Start a test first.", parent=self.root)
            return
        self.test.start(dwell=self.mode_var.get() == "index")
        self.test_button.configure(text="Stop test")
        self.log("Accuracy test started: reach each target"
                 + (" and rest on it for 1 s" if self.test.dwell else " and click it"))

    # ---------------------------------------------------------------- checks
    def run_checks(self):
        window = tk.Toplevel(self.root)
        window.title("Checks")
        window.configure(bg=PANEL)
        window.geometry("560x420")
        window.transient(self.root)
        text = tk.Text(window, bg=PANEL_2, fg=TEXT, font=FONT_MONO, relief="flat", padx=10, pady=8)
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("end", "Running checks...\n")

        def worker():
            results = system_checks(self.lib, camera=not self.engine.running,
                                    camera_index=self._config()["camera_index"])
            def show():
                text.delete("1.0", "end")
                for ok, name, detail in results:
                    text.insert("end", f"{'PASS' if ok else 'FAIL'}  {name}\n", "ok" if ok else "bad")
                    if detail:
                        text.insert("end", f"      {detail}\n", "detail")
                failed = sum(1 for r in results if not r[0])
                text.insert("end", "\nALL CHECKS PASSED\n" if not failed else f"\n{failed} check(s) failed\n",
                            "ok" if not failed else "bad")
                text.tag_configure("ok", foreground=GOOD)
                text.tag_configure("bad", foreground=BAD)
                text.tag_configure("detail", foreground=MUTED)
                text.configure(state="disabled")
            self.root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------------- refresh
    def tick(self):
        try:
            snap = self.engine.snapshot()
            with self._log_lock:
                pending, self._pending_log = self._pending_log, []
            for line in pending:
                self._append_log(line)
            self._update_video(snap)
            self._update_pills(snap)
            self._update_live(snap)
            self._update_screen(snap)
            self._update_buttons(snap)
        except Exception as error:   # never let a display glitch kill the window
            self.log_file(f"tick error: {error!r}")
        self.root.after(self.TICK_MS, self.tick)

    def _update_video(self, snap):
        if snap.frame is None:
            if snap.error:
                self.video.configure(image="", text=f"Problem:\n\n{snap.error}", fg=BAD)
                self._photo = None
            elif snap.running:
                self.video.configure(image="", text=snap.status, fg=MUTED)
            return
        width = max(self.video.winfo_width(), 100)
        height = max(self.video.winfo_height(), 100)
        h, w = snap.frame.shape[:2]
        scale = min(width / w, height / h)
        image = Image.fromarray(snap.frame)
        if abs(scale - 1.0) > 0.02:
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self._photo, text="")

    def _update_pills(self, snap):
        if not snap.running:
            self.pill_camera.set("Camera off", MUTED)
            self.pill_track.set("Not tracking", MUTED)
        else:
            colour = GOOD if snap.fps >= 24 else WARN if snap.fps >= 12 else BAD
            self.pill_camera.set(f"Camera {snap.fps:.0f} fps", colour)
            mode = MODES[snap.mode]
            if mode.hand:
                t = snap.hand.tracking
                self.pill_track.set(f"Hand {TRACKING_NAMES.get(t, '?')}",
                                    {osi_capi.TRACKED: GOOD, osi_capi.HELD: WARN}.get(t, BAD))
            elif mode.gaze:
                if snap.gaze.state == -1:
                    self.pill_track.set("Eyes calibrating", WARN)
                else:
                    self.pill_track.set("Face found" if snap.gaze.face else "No face", GOOD if snap.gaze.face else BAD)
            else:
                self.pill_track.set("Camera only", MUTED)
        real = self.real_var.get()
        active = snap.report.active and snap.running
        if not snap.running:
            self.pill_output.set("Real mouse (off)" if real else "Simulation (off)", MUTED)
        elif real:
            self.pill_output.set("REAL MOUSE " + ("ACTIVE" if active else "paused") +
                                 (" (yielding)" if snap.yielding else ""), BAD if active else WARN)
        else:
            self.pill_output.set("Simulation " + ("active" if active else "paused"), GOOD if active else WARN)
        self.status.configure(text=snap.status + (f"  ·  recording {snap.recorded_rows} frames"
                                                  if snap.recording else ""))

    def _update_live(self, snap):
        v = self.live

        def put(name, text, colour=TEXT):
            v[name].configure(text=text, foreground=colour)

        h, g, r, c = snap.hand, snap.gaze, snap.report, snap.counters
        put("Tracker status", h.status)
        put("Freshness", TRACKING_NAMES.get(h.tracking, "-"),
            {osi_capi.TRACKED: GOOD, osi_capi.HELD: WARN}.get(h.tracking, BAD))
        put("Pointer", f"{h.point[0]:.3f}, {h.point[1]:.3f}" if h.point else "-")
        put("Gesture", HAND_STATES.get(h.state, str(h.state)))
        put("Confidence", f"{h.confidence:.2f}")
        put("Time per frame", f"{h.ms:.1f} ms")
        put("Face", "yes" if g.face else "no", GOOD if g.face else MUTED)
        put("Eye state", EYE_STATES.get(g.state, str(g.state)), WARN if g.state == -1 else TEXT)
        put("Gaze angle", _angles(g.yaw, g.pitch))
        put("Head angle", _angles(g.head_yaw, g.head_pitch))
        put("Gaze pointer", f"{g.point[0]:.3f}, {g.point[1]:.3f}" if g.point else "-")
        put("Time per frame ", f"{g.ms:.1f} ms")
        put("Cursor follows", r.source)
        put("Cursor pixel", f"{r.cursor[0]}, {r.cursor[1]}" if r.cursor else "-")
        put("Output", ("active" if r.active else "paused") + (" (inhibited)" if r.inhibited else ""),
            GOOD if r.active else WARN)
        put("Confirmed gesture", HAND_STATES.get(r.confirmed_hand, str(r.confirmed_hand)))
        put("Confirmed eye", EYE_STATES.get(r.confirmed_eye, str(r.confirmed_eye)))
        put("Stuck button", "YES - retrying" if r.release_pending else "no", BAD if r.release_pending else GOOD)
        put("Left down / up", f"{c.left_down} / {c.left_up}")
        put("Left clicks", str(c.left_click))
        put("Right clicks", str(c.right_click))
        put("Middle clicks", str(c.middle_click))
        put("Pauses", str(c.pauses))
        put("Refused by Windows", str(c.refused), BAD if c.refused else TEXT)
        put("Frame rate", f"{snap.fps:.1f} fps")
        put("Processing", f"{snap.loop_ms:.1f} ms")
        put("Brightness", f"{snap.brightness:.0f} / 255", WARN if snap.running and snap.brightness < 60 else TEXT)
        put("Camera size", f"{snap.camera_size[0]} x {snap.camera_size[1]}" if snap.camera_size[0] else "-")
        bar = self.quality_bar
        bar.delete("all")
        width = max(bar.winfo_width(), 10)
        x = 0
        for pct, colour in ((snap.tracked_pct, GOOD), (snap.held_pct, WARN), (snap.lost_pct, BAD)):
            w = width * pct / 100.0
            if w > 0:
                bar.create_rectangle(x, 0, x + w, 14, fill=colour, width=0)
            x += w

    def _update_screen(self, snap):
        canvas = self.screen
        canvas.delete("all")
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        left, top, sw, sh = snap.target_rect
        if sw <= 0 or sh <= 0:
            return
        scale = min((cw - 20) / sw, (ch - 20) / sh)
        ox, oy = (cw - sw * scale) / 2, (ch - sh * scale) / 2
        canvas.create_rectangle(ox, oy, ox + sw * scale, oy + sh * scale, outline=BORDER, fill="#0b1016")
        canvas.create_text(ox + 8, oy + 8, anchor="nw", fill=MUTED, font=FONT_SMALL,
                           text=f"screen {sw} x {sh}" + ("  (simulation)" if not self.real_var.get() else ""))

        def to_canvas(px, py):
            return ox + (px - left) * scale, oy + (py - top) * scale

        # Accuracy test
        clicked = False
        total = snap.counters.left_down + snap.counters.left_click + snap.counters.right_click
        if total > self._prev_click_total:
            clicked = True
        self._prev_click_total = total
        cursor = snap.report.cursor if snap.running else None
        if self.test.running:
            target = self.test.current(sw, sh)
            tx, ty = to_canvas(left + target[0], top + target[1])
            radius = self.test.radius * scale
            canvas.create_oval(tx - radius, ty - radius, tx + radius, ty + radius, outline=ACCENT, width=2)
            canvas.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill=ACCENT, width=0)
            if cursor:
                self.test.update((cursor[0] - left, cursor[1] - top), clicked)
            self.test_label.configure(text=self.test.progress())
            if not self.test.running:
                self.test_button.configure(text="Start accuracy test")
                self.test_label.configure(text=self.test.summary())
                self.log("Accuracy test: " + self.test.summary())
        if cursor:
            cx, cy = to_canvas(*cursor)
            self._screen_trail.append((cx, cy))
            self._screen_trail = self._screen_trail[-30:]
            if len(self._screen_trail) > 1:
                canvas.create_line(*[v for p in self._screen_trail for v in p], fill="#24506b", width=2,
                                   smooth=True)
            colour = BAD if snap.report.release_pending else (GOOD if snap.report.active else WARN)
            canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, outline=colour, width=2)
            canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=colour, width=0)
            if clicked:
                canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, outline=TEXT, width=1)

    def _update_buttons(self, snap):
        if snap.running:
            self.active_button.configure(text="Pause output" if snap.report.active else
                                         ("Activate REAL mouse" if self.real_var.get() else "Activate output"))
        if not snap.running and self.engine.running is False and self.run_button.cget("text").endswith("Stop"):
            self.run_button.configure(text="▶  Start")
            self.active_button.configure(state="disabled")
        if not snap.recording and self.record_button.cget("text").endswith("recording"):
            self.record_button.configure(text="●  Record CSV")

    def close(self):
        self.engine.stop()
        self.root.destroy()


# ----------------------------------------------------------------------------- accuracy test

class AccuracyTest:
    """Ten targets on the (virtual) screen. Hit = click inside, or 1 s dwell."""

    TARGETS = 10
    DWELL_SECONDS = 1.0

    def __init__(self):
        self.running = False
        self.dwell = False
        self.radius = 60
        self.results: List[Tuple[float, float]] = []   # (seconds, error px)
        self._target: Optional[Tuple[float, float]] = None
        self._shown_at = 0.0
        self._inside_since: Optional[float] = None
        self._misses = 0
        self._rng = random.Random()

    def start(self, dwell: bool):
        self.running, self.dwell = True, dwell
        self.results, self._misses, self._target = [], 0, None

    def stop(self):
        self.running = False

    def current(self, width: int, height: int) -> Tuple[float, float]:
        if self._target is None:
            self.radius = max(30, min(width, height) // 18)
            m = self.radius * 2
            self._target = (self._rng.uniform(m, width - m), self._rng.uniform(m, height - m))
            self._shown_at = time.perf_counter()
            self._inside_since = None
        return self._target

    def update(self, cursor: Tuple[float, float], clicked: bool):
        if not self.running or self._target is None:
            return
        error = math.hypot(cursor[0] - self._target[0], cursor[1] - self._target[1])
        inside = error <= self.radius
        now = time.perf_counter()
        hit = False
        if self.dwell:
            if inside:
                self._inside_since = self._inside_since or now
                hit = now - self._inside_since >= self.DWELL_SECONDS
            else:
                self._inside_since = None
        elif clicked:
            if inside:
                hit = True
            else:
                self._misses += 1
        if hit:
            self.results.append((now - self._shown_at, error))
            self._target = None
            if len(self.results) >= self.TARGETS:
                self.running = False

    def progress(self) -> str:
        return f"target {len(self.results) + 1} of {self.TARGETS}" + (f"  ·  misses {self._misses}"
                                                                      if self._misses else "")

    def summary(self) -> str:
        if not self.results:
            return "no targets hit"
        times = [r[0] for r in self.results]
        errors = [r[1] for r in self.results]
        return (f"{len(self.results)} hits, {self._misses} misses, average {sum(times) / len(times):.2f} s, "
                f"average error {sum(errors) / len(errors):.0f} px")


# ----------------------------------------------------------------------------- helpers

def _angles(yaw: float, pitch: float) -> str:
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return "-"
    return f"yaw {yaw:+.1f}°  pitch {pitch:+.1f}°"


def _region(text: Optional[str], default):
    try:
        values = [float(v) for v in (text or "").replace(",", " ").split()]
        return tuple(values) if len(values) == 4 else default
    except ValueError:
        return default


def system_checks(lib: osi_capi.OsiLibrary, camera: bool, camera_index: int) -> List[Tuple[bool, str, str]]:
    """Everything the tester needs, each with a hint when it fails."""
    results: List[Tuple[bool, str, str]] = []
    results.append((True, f"Engineer 5 library v{lib.version}", str(lib.path)))
    probe = lib.smoothing_filter(1.0, 0.0, 1.0)
    results.append((probe is not None, "Engineer 4 smoothing filter",
                    "" if probe else "library built without dsp_filter: smoothing option unavailable"))
    if probe:
        probe.close()
    assets = [REPO / "assets" / n for n in ("hand_landmarker.task", "face_landmarker.task", "hand_gesture_rf.pkl")]
    assets += [REPO / "ai_pipeline" / "gaze_intent" / "models" / n
               for n in ("gesture_model.dll", "gaze_pitch.dll", "gaze_yaw.dll")]
    for path in assets:
        results.append((path.is_file(), f"Model file {path.name}", "" if path.is_file() else f"missing: {path}"))
    prepare_ai_imports()
    for module, hint in [("cv2", "opencv-python"), ("mediapipe", "mediapipe"), ("sklearn", "scikit-learn"),
                         ("tl2cgen", "tl2cgen (gaze models)")]:
        try:
            __import__(module)
            results.append((True, f"Python package {module}", ""))
        except Exception as error:
            results.append((False, f"Python package {module}", f"{error} - run setup.bat ({hint})"))
    for module in ("index_tracker", "hand_tracker", "gaze_intent"):
        try:
            __import__(module)
            results.append((True, f"Teammate module {module}.py", ""))
        except Exception as error:
            results.append((False, f"Teammate module {module}.py", str(error)))
    if camera:
        try:
            import cv2
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            results.append((bool(ok), f"Camera {camera_index}",
                            f"{frame.shape[1]} x {frame.shape[0]}" if ok else
                            "cannot read frames: close other apps using the camera or pick another index"))
        except Exception as error:
            results.append((False, f"Camera {camera_index}", str(error)))
    else:
        results.append((True, "Camera", "in use by the running test (skipped)"))
    return results
