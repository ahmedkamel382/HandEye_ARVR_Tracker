"""HandEye Tester: the window.

Pick what to test (index finger only, hand gestures, gaze, winks, the full
system), watch the camera with overlays, see where the cursor would go, tune
the settings live, run an accuracy test and record sessions to CSV.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

import osi_capi
from tester_engine import (ENGINE_MATCH, EYE_STATES, HAND_STATES, MODES, REPO, TRACKING_NAMES, IniDoc, TesterConfig,
                           TesterEngine, engine_differences, format_value, gaze_is_uncalibrated, prepare_ai_imports,
                           read_settings, saved_all_monitors, user_settings_path)

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

# slider: (section, key, default, decimals)
VALUE_CONTROLS = {
    "snap": ("stabilization", "anchor_lookback_seconds", 0.1, 3),
    "drag": ("stabilization", "drag_threshold_px", 30.0, 1),
    "pinch": ("gestures", "pinch_onset_seconds", 0.06, 3),
    "release": ("gestures", "pinch_release_seconds", 0.08, 3),
    "fist": ("gestures", "fist_onset_seconds", 0.3, 3),
    "wink": ("gestures", "wink_onset_seconds", 0.25, 3),
    "closure": ("gestures", "closure_onset_seconds", 1.0, 3),
    "h_cut": ("hand_filter", "min_cutoff_hz", 1.2, 3),
    "h_beta": ("hand_filter", "beta", 4.0, 2),
    "g_cut": ("gaze_filter", "min_cutoff_hz", 0.8, 3),
    "g_beta": ("gaze_filter", "beta", 6.0, 2),
}
DEFAULT_REGIONS = {"hand_region": (0.15, 0.15, 0.85, 0.85), "gaze_region": (0.25, 0.58, 0.75, 0.82)}
# slider: (region key, axis 0 = x / 1 = y, slider units per unit of region span)
REGION_CONTROLS = {
    "zone_x": ("hand_region", 0, 1.0),
    "zone_y": ("hand_region", 1, 1.0),
    "gaze_x": ("gaze_region", 0, 30.0),     # span / 2 * 60 = degrees each way
    "gaze_y": ("gaze_region", 1, 30.0),
}


def resize_axis(doc: IniDoc, key: str, axis: int, span: float) -> str:
    """The region with one axis resized about its own centre, direction kept. The other axis is copied as written."""
    default = DEFAULT_REGIONS[key]
    text = doc.get("mapping", key)
    values = _region(text, default)
    written = (text or "").replace(",", " ").split()
    if values is default or len(written) != 4:
        written = [format_value(v) for v in values]
    low, high = values[axis], values[axis + 2]
    centre, half = (low + high) / 2, (span / 2 if high >= low else -span / 2)
    written[axis], written[axis + 2] = format_value(centre - half), format_value(centre + half)
    return " ".join(written)


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
        self._pending_log: List[str] = []
        self._log_lock = threading.Lock()
        self.engine = TesterEngine(library, self._log_from_engine)      # may log about the saved settings
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
        self._use_saved_screen()
        self._update_notes()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<F5>", lambda e: self.toggle_run())
        root.bind("<space>", self._space_pressed)
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
        ttk.Checkbutton(output, text="Use all monitors (starts as your settings say)",
                        variable=self.monitors_var, command=self._options_changed).pack(anchor="w", pady=(4, 0))

        options = section(b, "3  Options")
        self.clicks_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=True)
        self.landmarks_var = tk.BooleanVar(value=True)
        self.zone_var = tk.BooleanVar(value=True)
        self.trail_var = tk.BooleanVar(value=True)
        self.lowlight_var = tk.BooleanVar(value=TesterConfig.lowlight)
        self.roi_var = tk.BooleanVar(value=TesterConfig.roi_recovery)
        for text, var in [("Allow clicks (pinch, fist, winks)", self.clicks_var),
                          ("Smoothing (Engineer 4's 1-Euro filter, like the engine)", self.smooth_var),
                          ("Draw landmarks", self.landmarks_var), ("Draw the hand's active zone", self.zone_var),
                          ("Draw the pointer trail", self.trail_var),
                          ("Hand: low-light enhancement (not in core_engine)", self.lowlight_var),
                          ("Hand: recover a lost hand near its last position (not in core_engine)", self.roi_var)]:
            ttk.Checkbutton(options, text=text, variable=var, command=self._options_changed).pack(anchor="w")
        self.index_alpha = Slider(options, "Index finger smoothing (Engineer 2's tracker)", 0.05, 1.0, "{:.2f}",
                                  self._options_changed, "Lower = steadier but slower. 1.00 = off, so only Engineer 4's filter smooths.")
        self.index_alpha.set(TesterConfig.index_alpha)
        self.parity_label = ttk.Label(options, text="", style="Muted.TLabel", wraplength=420, justify="left")
        self.parity_label.pack(anchor="w", pady=(8, 2))
        ttk.Button(options, text="Match core_engine options", command=self.match_engine).pack(anchor="w")
        self.gaze_note = ttk.Label(options, text="", style="Muted.TLabel", wraplength=420, justify="left")
        self.gaze_note.pack(anchor="w", pady=(4, 0))
        cam = ttk.Frame(options, style="Panel.TFrame")
        cam.pack(fill="x", pady=(6, 0))
        ttk.Label(cam, text="Camera", style="Panel.TLabel").pack(side="left")
        self.camera_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(cam, from_=0, to=9, width=4, textvariable=self.camera_var, command=self._options_changed)
        spin.pack(side="left", padx=6)
        for event in ("<Return>", "<FocusOut>"):        # a typed number, not only the arrows
            spin.bind(event, lambda e: self._options_changed())
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
        s["zone_x"] = Slider(g, "Hand zone width", 0.1, 1.0, "{:.2f}", ch)
        s["zone_y"] = Slider(g, "Hand zone height", 0.1, 1.0, "{:.2f}", ch,
                             "Part of the camera picture that covers the whole screen. Smaller = smaller hand "
                             "movements reach the edges. A calibrated centre and direction are kept.")
        s["gaze_x"] = Slider(g, "Gaze range left-right", 1, 45, "±{:.1f}°", ch)
        s["gaze_y"] = Slider(g, "Gaze range up-down", 1, 45, "±{:.1f}°", ch,
                             "How far you look to reach the screen edge. A calibrated centre and direction are kept.")
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
                                   "Python tracker time"]),
            ("Eyes (Engineer 3)", ["Face", "Eye state", "Gaze angle", "Head angle", "Gaze pointer",
                                   "Python tracker time "]),
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
            ("Comparing with core_engine", "The defaults switch off what core_engine does not have: low-light "
             "enhancement, lost-hand recovery and a second smoothing stage. The trackers here are still the Python "
             "ones, so timings in this window are not core_engine's latency."),
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
        self._update_notes()
        self.root.focus_set()

    def _options_changed(self):
        self.engine.configure(**self._config())
        self._update_notes()

    def match_engine(self):
        self.lowlight_var.set(ENGINE_MATCH["lowlight"])
        self.roi_var.set(ENGINE_MATCH["roi_recovery"])
        self.smooth_var.set(ENGINE_MATCH["smoothing"])
        self.clicks_var.set(ENGINE_MATCH["clicks"])
        self.index_alpha.set(ENGINE_MATCH["index_alpha"])
        self._use_saved_screen()
        self._options_changed()

    def _use_saved_screen(self):
        all_monitors = saved_all_monitors(self.engine.settings_text())
        self.monitors_var.set(all_monitors)
        if self.engine.config.all_monitors != all_monitors:
            self.engine.configure(all_monitors=all_monitors)

    def _update_notes(self):
        config = TesterConfig(**self._config())
        mode = MODES[config.mode]
        if mode.key == "camera":
            self.parity_label.configure(text="", foreground=MUTED)
        else:
            differs = engine_differences(config, self.engine.settings_text())
            if self._tuning_replaced or self._tuning_ini() != self._tuning_base:
                differs.append("tuning changes are not saved")
            if self.engine.saved_settings_error:
                differs.append("your settings file has an error")
            self.parity_label.configure(
                text=("Differs from core_engine: " + "; ".join(differs) + "." if differs else
                      "Options match core_engine. (Python trackers: timings are not the engine's latency.)"),
                foreground=WARN if differs else GOOD)
        uncalibrated = mode.gaze == "pointer" and gaze_is_uncalibrated(self.engine.settings_text(),
                                                                       self.lib.default_settings_ini())
        self.gaze_note.configure(
            text=("Gaze is not calibrated: the cursor can stick to an edge or move the wrong way. In core_engine, "
                  "use the tray icon: Calibrate > Gaze range.") if uncalibrated else "",
            foreground=WARN)

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
        self.root.focus_set()

    # ---------------------------------------------------------------- run control
    def toggle_run(self):
        if self.engine.running:
            self.engine.stop()
            if self.engine.running:
                self.log("Still stopping: waiting for the camera. Try again in a moment.")
                return
            self.run_button.configure(text="▶  Start")
            self.active_button.configure(state="disabled")
            self.log("Stopped")
            self.video.configure(image="", text="Stopped\n\nPress Start (F5) to continue")
            self._photo = None
        else:
            self.engine.configure(**self._config())
            # Start never rewrites settings; only an edit still waiting for its 250 ms delay is applied now.
            if self._tuning_after:
                self.root.after_cancel(self._tuning_after)
                self._push_tuning()
            self.engine.start()
            self.root.focus_set()       # so Space is the activate / pause shortcut, not a second click on Start
            self.run_button.configure(text="■  Stop")
            self.active_button.configure(state="normal")

    # Widgets that use Space themselves.
    SPACE_WIDGETS = {"TButton", "Button", "TCheckbutton", "Checkbutton", "TRadiobutton", "Radiobutton", "TSpinbox",
                     "Spinbox", "TEntry", "Entry", "TCombobox", "Text"}

    def _space_pressed(self, event):
        """Space may always pause. It activates only when the focused widget has no use for Space."""
        if not self.engine.running or event.widget is self.active_button:
            return                      # that button toggles by itself
        widget = event.widget
        uses_space = isinstance(widget, str) or widget.winfo_class() in self.SPACE_WIDGETS
        if self.engine.snapshot().report.active or not uses_space:
            self.toggle_active()

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
    def _control_values(self) -> dict:
        values = {name: slider.get() for name, slider in self.sliders.items()}
        values.update(source=self.source_var.get(), fist_action=self.fist_var.get())
        return values

    def _cancel_tuning_push(self):
        if self._tuning_after:
            self.root.after_cancel(self._tuning_after)
            self._tuning_after = None

    def _load_tuning_from_settings(self, replaced: bool = False):
        """Show the current settings. The text stays the source of truth: sliders only describe edits to it."""
        self._cancel_tuning_push()
        self._tuning_base = self.engine.settings_text()
        self._tuning_replaced = replaced        # Reset: Save then writes the whole text
        doc = IniDoc(self._tuning_base)
        self._loading_tuning = True
        try:
            for name, (key, axis, scale) in REGION_CONTROLS.items():
                values = _region(doc.get("mapping", key), DEFAULT_REGIONS[key])
                self.sliders[name].set(abs(values[axis + 2] - values[axis]) * scale)
            for name, (section_name, key, default, _) in VALUE_CONTROLS.items():
                self.sliders[name].set(doc.get_float(section_name, key, default))
            self.source_var.set(doc.get("mapping", "cursor_source") or "hand_preferred")
            self.fist_var.set(doc.get("gestures", "fist_action") or "right_click")
            self._loaded_values = self._control_values()
        finally:
            self._loading_tuning = False

    def _apply_edits(self, text: str) -> str:
        """`text` with only the controls the user changed; exactly `text` when none were."""
        now = self._control_values()
        edited = {name for name, value in now.items() if value != self._loaded_values[name]}
        if not edited:
            return text
        doc = IniDoc(text)
        for name, (key, axis, scale) in REGION_CONTROLS.items():
            if name in edited:
                doc.set("mapping", key, resize_axis(doc, key, axis, now[name] / scale))
        for name, (section_name, key, _, decimals) in VALUE_CONTROLS.items():
            if name in edited:
                doc.set(section_name, key, round(now[name], decimals))
        if "source" in edited:
            doc.set("mapping", "cursor_source", now["source"])
        if "fist_action" in edited:
            doc.set("gestures", "fist_action", now["fist_action"])
        return doc.text()

    def _tuning_ini(self) -> str:
        return self._apply_edits(self._tuning_base)

    def _tuning_changed(self):
        if self._loading_tuning:
            return
        self._cancel_tuning_push()
        self._tuning_after = self.root.after(250, self._push_tuning)

    def _push_tuning(self):
        self._tuning_after = None
        text = self._tuning_ini()
        if text == self.engine.settings_text():
            return
        error = self.engine.check_settings(text)
        if error:
            self.tuning_status.configure(text=f"Not applied: {error}", foreground=BAD)
            return
        self.engine.set_settings_text(text)
        self.tuning_status.configure(text="Applied to the running test (not saved).", foreground=MUTED)
        self._update_notes()

    def reset_tuning(self):
        self.engine.reset_settings()
        self._load_tuning_from_settings(replaced=True)
        self._use_saved_screen()
        self._update_notes()
        self.tuning_status.configure(text="Defaults restored (not saved).", foreground=MUTED)

    def load_user_settings(self):
        path = user_settings_path()
        if not path.is_file():
            messagebox.showinfo("Settings", f"No saved settings yet:\n{path}", parent=self.root)
            return
        try:
            text = read_settings(path)
            error = self.engine.check_settings(text)
        except (OSError, UnicodeDecodeError) as problem:
            error = str(problem)
        if error:
            messagebox.showerror("Settings", f"Not loaded: {error}", parent=self.root)
            return
        self.engine.saved_settings_error = None
        self.engine.set_settings_text(text)
        self._load_tuning_from_settings()
        self._use_saved_screen()
        self._update_notes()
        self.tuning_status.configure(text=f"Loaded {path}", foreground=MUTED)

    def save_user_settings(self):
        path = user_settings_path()
        what = ("Replace your settings with the defaults and your edits? This also erases your calibration."
                if self._tuning_replaced else "Save the values you changed to your settings?")
        if not messagebox.askyesno("Save settings",
                                   f"{what}\n\n{path}\n\n"
                                   "The engine and the tray app will use them next time they start.",
                                   parent=self.root):
            return
        # The tray may have calibrated since this window opened: edit the file as it is now. A file that cannot
        # be loaded is never written over, except by an explicit Reset and Save.
        base = self._tuning_base
        if not self._tuning_replaced and path.exists():
            try:
                saved = read_settings(path)
                problem = self.engine.check_settings(saved)
            except (OSError, UnicodeDecodeError) as error:
                problem = str(error)
            if problem:
                messagebox.showerror(
                    "Settings",
                    f"Not saved, and your file was left as it is, because it has an error:\n\n{problem}\n\n"
                    "Correct the file, or press Reset to defaults and then Save (that keeps a .bak copy).",
                    parent=self.root)
                return
            base = saved
        text = self._apply_edits(base)
        error = self.engine.check_settings(text)
        if error:
            messagebox.showerror("Settings", f"Not saved: {error}", parent=self.root)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.copyfile(path, path.with_name(path.name + ".bak"))
            temporary = path.with_name(path.name + ".tester.tmp")       # the tray has its own .tmp
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        except OSError as error:
            messagebox.showerror("Settings", f"Not saved: {error}", parent=self.root)
            return
        self.engine.saved_settings_error = None
        self.engine.set_settings_text(text)
        self._load_tuning_from_settings()
        self._update_notes()
        self.tuning_status.configure(text=f"Saved to {path}", foreground=GOOD)
        self.log(f"Settings saved to {path}")

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

        # Tk belongs to the main thread: read it here, and let the main thread poll for the results.
        camera, camera_index = not self.engine.running, self._config()["camera_index"]
        finished: List[List[Tuple[bool, str, str]]] = []

        def worker():
            try:
                finished.append(system_checks(self.lib, camera=camera, camera_index=camera_index))
            except Exception as error:
                finished.append([(False, "Checks", repr(error))])

        def show():
            if not window.winfo_exists():
                return          # closed before the checks finished
            if not finished:
                self.root.after(100, show)
                return
            results = finished[0]
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

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, show)

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
        put("Python tracker time", f"{h.ms:.1f} ms")
        put("Face", "yes" if g.face else "no", GOOD if g.face else MUTED)
        put("Eye state", EYE_STATES.get(g.state, str(g.state)), WARN if g.state == -1 else TEXT)
        put("Gaze angle", _angles(g.yaw, g.pitch))
        put("Head angle", _angles(g.head_yaw, g.head_pitch))
        put("Gaze pointer", f"{g.point[0]:.3f}, {g.point[1]:.3f}" if g.point else "-")
        put("Python tracker time ", f"{g.ms:.1f} ms")
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
        total = (snap.counters.left_down + snap.counters.left_click + snap.counters.right_click
                 + snap.counters.middle_click)
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
        self.engine.stop()                  # the watchdog releases at once, even if a camera read is blocked
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
