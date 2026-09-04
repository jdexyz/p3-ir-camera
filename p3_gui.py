#!/usr/bin/env python3
"""Tk control panel for the P3 thermal camera.

A windowed front-end over P3Viewer for sessions where recording has to start
and stop on cue. All frame processing is P3Viewer's; this module only drives
it and shows the result, so the GUI and the OpenCV viewer stay identical
image-for-image.

Threading: one capture thread owns the USB device and runs
P3Viewer.process_frame(). Tk polls the newest rendered frame from the main
thread. Anything that talks to the camera is posted to the capture thread as
a command rather than called directly, since libusb transfers must not
overlap.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from tkinter import filedialog, messagebox, ttk

import os
import queue
import threading
import time
import tkinter as tk

from numpy.typing import NDArray
from PIL import Image, ImageTk

import cv2
import numpy as np
import usb.util

from p3_camera import GainMode
from p3_viewer import AGCMode, ColormapID, HotspotMode, P3Viewer

import audio
import uvc


# Dark palette, chosen so the thermal image stays the brightest thing on screen.
BG = "#1e1f22"
BG_PANEL = "#2b2d31"
BG_INPUT = "#383a40"
FG = "#e3e5e8"
FG_DIM = "#9a9ea6"
ACCENT = "#4a9eff"
REC_RED = "#f04747"
OK_GREEN = "#43b581"


def default_output_dir() -> str:
    """Where recordings should land by default.

    Not the working directory: launched from a shortcut that is the source
    tree, and a few minutes of raw frames is a few hundred MB.
    """
    videos = os.path.join(os.path.expanduser("~"), "Videos")
    return videos if os.path.isdir(videos) else os.getcwd()


class State(str, Enum):
    """Connection state of the capture thread."""

    CONNECTING = "Connecting"
    STREAMING = "Streaming"
    RECONNECTING = "Reconnecting"
    STOPPED = "Stopped"


# A frame read blocks for up to 10 s inside libusb, so the stall watchdog has
# to allow longer than that before deciding the link is dead.
STALL_TIMEOUT = 14.0
RETRY_DELAY = 1.5


class CaptureThread(threading.Thread):
    """Owns the camera, produces rendered frames, and survives disconnects.

    The device can vanish underneath us in two ways that matter: the cable is
    pulled, or the machine sleeps and the handle is invalidated on resume. Both
    are recoverable, so a failure re-enters the connect loop rather than ending
    the session. A stall watchdog covers the case where reads neither fail nor
    deliver -- what a resume from sleep looks like -- since a silently frozen
    image is worse than a visible error.
    """

    def __init__(self, viewer: P3Viewer) -> None:
        super().__init__(daemon=True)
        self.viewer = viewer
        self.commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self.frame: NDArray[np.uint8] | None = None
        self.error: BaseException | None = None
        self.state: State = State.CONNECTING
        self.last_error: str | None = None
        self.reconnects = 0
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._lock = threading.Lock()

    def post(self, fn: Callable[[], None]) -> None:
        """Run fn on the capture thread, between frames."""
        self.commands.put(fn)

    def latest(self) -> NDArray[np.uint8] | None:
        with self._lock:
            return self.frame

    def stop(self) -> None:
        self._stop.set()

    def request_reconnect(self) -> None:
        """Drop the device and reconnect at the next opportunity."""
        self._reconnect.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.state = State.CONNECTING
                self.viewer.open_camera()
                self.ready.set()
                self.state = State.STREAMING
                self._stream()
            except BaseException as e:  # noqa: BLE001 - must not kill the thread
                self.last_error = f"{type(e).__name__}: {e}"
                self.error = e
            finally:
                self._teardown()

            if self._stop.is_set():
                break
            self.state = State.RECONNECTING
            self.reconnects += 1
            self._reconnect.clear()
            # The device often needs a moment to re-enumerate, especially on
            # resume from sleep, so retry indefinitely rather than giving up.
            self._stop.wait(RETRY_DELAY)

        self.state = State.STOPPED

    # -- internals --------------------------------------------------------

    def _stream(self) -> None:
        last_frame = time.time()
        while not self._stop.is_set() and not self._reconnect.is_set():
            self._drain_commands()

            frame = self.viewer.process_frame()
            now = time.time()
            if frame is not None:
                last_frame = now
                with self._lock:
                    self.frame = frame
            elif now - last_frame > STALL_TIMEOUT:
                raise TimeoutError(
                    f"no frame for {now - last_frame:.0f}s; link presumed dead"
                )
            else:
                # A read that returns nothing without blocking would otherwise
                # spin this loop at full speed.
                time.sleep(0.005)

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                cmd()
            except Exception as e:  # a bad command must not kill capture
                print(f"Command failed: {e}")

    def _teardown(self) -> None:
        """Close the device, keeping any in-progress recording valid.

        A recording is finalized rather than abandoned: the frames captured
        before the disconnect are still good data, and a half-written mp4 with
        no sidecar is not.
        """
        try:
            self.viewer.close_camera()
        except Exception:
            pass
        # Release the stale handle so the next usb.core.find() gets a fresh
        # one; without this a post-sleep reconnect keeps failing.
        try:
            if self.viewer.camera.dev is not None:
                usb.util.dispose_resources(self.viewer.camera.dev)
        except Exception:
            pass
        self.viewer.camera.dev = None
        self.viewer.camera.streaming = False


class P3GUI:
    """Tk control panel."""

    def __init__(self, root: tk.Tk, viewer: P3Viewer,
                 uvc_index: int | None = None) -> None:
        self.root = root
        self.viewer = viewer
        self.capture = CaptureThread(viewer)
        self._photo: ImageTk.PhotoImage | None = None
        self.uvc_cam: uvc.UVCCamera | None = None
        # True between pressing record and pressing stop. The thermal writer
        # opens a frame later than the click, so "not recording yet" must not
        # be mistaken for "recording finished".
        self._record_intent = False
        self._view_missing_note = ""
        self.audio_rec: audio.AudioRecorder | None = None
        self._session_dir: str | None = None
        self._mux_thread: threading.Thread | None = None
        self._closing = False
        self.uvc_devices: list[tuple[int, int, int]] = []
        self._uvc_preferred = uvc_index

        root.title("P3 Thermal Camera")
        root.geometry("1280x800")
        root.minsize(900, 600)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._init_style()
        self._build()

        self.capture.start()
        self.root.after(100, self._tick)

    # -- setup ------------------------------------------------------------

    def _init_style(self) -> None:
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=BG_INPUT)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=BG_PANEL)
        st.configure("TLabel", background=BG_PANEL, foreground=FG)
        st.configure("Dim.TLabel", background=BG_PANEL, foreground=FG_DIM)
        st.configure("Status.TLabel", background=BG, foreground=FG_DIM)
        st.configure(
            "Value.TLabel", background=BG_PANEL, foreground=ACCENT,
            font=("Segoe UI", 10, "bold"),
        )
        st.configure(
            "TLabelframe", background=BG_PANEL, foreground=FG_DIM,
            bordercolor="#41434a",
        )
        st.configure("TLabelframe.Label", background=BG_PANEL, foreground=FG_DIM)
        st.configure("TButton", background=BG_INPUT, foreground=FG, borderwidth=0,
                     focuscolor=BG_INPUT, padding=6)
        st.map("TButton", background=[("active", "#4a4d55")])
        st.configure("TCheckbutton", background=BG_PANEL, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG_PANEL)])
        st.configure("TRadiobutton", background=BG_PANEL, foreground=FG)
        st.map("TRadiobutton", background=[("active", BG_PANEL)])
        st.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG,
                     borderwidth=0, insertcolor=FG)
        st.configure("TSpinbox", fieldbackground=BG_INPUT, foreground=FG,
                     arrowcolor=FG, borderwidth=0)
        st.configure("TCombobox", fieldbackground=BG_INPUT, foreground=FG,
                     arrowcolor=FG, borderwidth=0)
        # A readonly combobox draws its text using the selection colours, which
        # default to dark-on-dark against this palette.
        st.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            foreground=[("readonly", FG)],
            selectbackground=[("readonly", BG_INPUT)],
            selectforeground=[("readonly", FG)],
        )
        self.root.option_add("*TCombobox*Listbox.background", BG_INPUT)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _build(self) -> None:
        # Status bar is packed first so it always reserves its height; packed
        # after an expanding frame it gets pushed off the bottom of the window.
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x")
        self.status = ttk.Label(bar, text="Starting...", style="Status.TLabel",
                                anchor="w", padding=(8, 4))
        self.status.pack(fill="x")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # Control column ----------------------------------------------------
        # Scrollable, so the lower sections stay reachable on short windows.
        col = ttk.Frame(main, style="Panel.TFrame", width=330)
        col.pack(side="right", fill="y")
        col.pack_propagate(False)

        canvas = tk.Canvas(col, bg=BG_PANEL, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(col, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(canvas, style="Panel.TFrame")
        window = canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>",
                  lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"),
        )

        # Preview -----------------------------------------------------------
        # pack_propagate(False) stops the Label from resizing this frame to fit
        # its image; without it the image size and the widget size feed each
        # other and the preview grows without bound.
        self.stage = tk.Frame(main, bg="#000000")
        self.stage.pack(side="left", fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.preview = tk.Label(self.stage, bg="#000000",
                                text="Connecting to camera...",
                                fg=FG_DIM, font=("Segoe UI", 11))
        self.preview.place(relx=0.5, rely=0.5, anchor="center")

        self._build_connection(side)
        self._build_view(side)
        self._build_record(side)
        self._build_visible(side)
        self._build_scale(side)
        self._build_camera(side)
        self._build_display(side)
        self._sync_strength_enabled()

    def _build_connection(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Camera link ", padding=10)
        box.pack(fill="x", padx=10, pady=(12, 0))
        self.lbl_conn = ttk.Label(box, text="○ Connecting...", style="Dim.TLabel")
        self.lbl_conn.pack(anchor="w")
        self.btn_reconnect = ttk.Button(box, text="Reconnect",
                                        command=self._reconnect)
        self.btn_reconnect.pack(fill="x", pady=(8, 0))

    def _reconnect(self) -> None:
        """Force a reconnect; also the manual recovery if auto-retry is stuck."""
        self.capture.request_reconnect()

    def _build_view(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" View ", padding=10)
        box.pack(fill="x", padx=10, pady=6)
        self.var_view = tk.StringVar(value="both")
        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x")
        for i, (value, label) in enumerate(
            (("both", "Both"), ("thermal", "Thermal"), ("visible", "Visible"))
        ):
            ttk.Radiobutton(row, text=label, value=value, variable=self.var_view,
                            command=self._apply_view).grid(row=0, column=i,
                                                           sticky="w", padx=(0, 10))

    def _apply_view(self) -> None:
        """Open or release the webcam to match what the view needs."""
        self._ensure_uvc()

    def _wants_visible(self) -> bool:
        """The webcam is needed to show it, or to record it."""
        return self.var_view.get() in ("both", "visible") or self.var_uvc_on.get()

    def _build_record(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Recording ", padding=10)
        box.pack(fill="x", padx=10, pady=(12, 6))

        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Folder", style="Dim.TLabel").pack(side="left")
        ttk.Button(row, text="Browse", width=8, command=self._pick_dir).pack(side="right")
        self.var_dir = tk.StringVar(value=default_output_dir())
        ttk.Entry(box, textvariable=self.var_dir).pack(fill="x", pady=(2, 8))

        ttk.Label(box, text="Name", style="Dim.TLabel").pack(anchor="w")
        self.var_name = tk.StringVar(value=self._default_name())
        ttk.Entry(box, textvariable=self.var_name).pack(fill="x", pady=(2, 10))

        self.btn_record = tk.Button(
            box, text="●  Start Recording", command=self._toggle_record,
            bg=REC_RED, fg="white", activebackground="#d63c3c",
            activeforeground="white", relief="flat", bd=0,
            font=("Segoe UI", 11, "bold"), pady=10, cursor="hand2",
        )
        self.btn_record.pack(fill="x")

        self.lbl_rec = ttk.Label(box, text="Not recording", style="Dim.TLabel")
        self.lbl_rec.pack(anchor="w", pady=(8, 0))

        ttk.Button(box, text="Snapshot (PNG + raw)",
                   command=self._snapshot).pack(fill="x", pady=(8, 0))

    def _build_visible(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Visible camera ", padding=10)
        box.pack(fill="x", padx=10, pady=6)

        self.var_uvc_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Record visible alongside thermal",
                        variable=self.var_uvc_on,
                        command=self._toggle_uvc).pack(anchor="w")

        self.var_uvc_dev = tk.StringVar(value="detecting...")
        self.cb_uvc = ttk.Combobox(box, textvariable=self.var_uvc_dev,
                                   state="disabled", values=[])
        self.cb_uvc.pack(fill="x", pady=(8, 0))
        self.cb_uvc.bind("<<ComboboxSelected>>", lambda _e: self._select_uvc())

        self.lbl_uvc = ttk.Label(box, text="Detecting cameras...",
                                 style="Dim.TLabel")
        self.lbl_uvc.pack(anchor="w", pady=(8, 0))

        self.var_audio_on = tk.BooleanVar(value=audio.available())
        ttk.Checkbutton(box, text="Record sound", variable=self.var_audio_on,
                        command=self._sync_audio_enabled).pack(anchor="w",
                                                               pady=(10, 0))
        self.audio_inputs = audio.list_inputs()
        self.var_mic = tk.StringVar()
        self.cb_mic = ttk.Combobox(box, textvariable=self.var_mic,
                                   state="readonly",
                                   values=[n for _, n, _, _ in self.audio_inputs])
        self.cb_mic.pack(fill="x", pady=(4, 0))
        preferred = audio.preferred_input(self.audio_inputs)
        if preferred is not None:
            self.var_mic.set(preferred[1])
        else:
            self.var_mic.set("sounddevice not installed"
                             if not audio.available() else "no microphone found")
        self._sync_audio_enabled()

        # Probing opens each device in turn and costs a second or two, so it
        # must not block the window from appearing.
        threading.Thread(target=self._probe_uvc, daemon=True).start()

    def _sync_audio_enabled(self) -> None:
        usable = audio.available() and bool(self.audio_inputs)
        if not usable:
            self.var_audio_on.set(False)
        self.cb_mic.config(
            state="readonly" if usable and self.var_audio_on.get() else "disabled"
        )

    def _selected_mic(self) -> tuple[int, str, int, float] | None:
        name = self.var_mic.get()
        return next((d for d in self.audio_inputs if d[1] == name), None)

    def _probe_uvc(self) -> None:
        devices = uvc.list_cameras()
        self.root.after(0, lambda: self._uvc_probed(devices))

    def _uvc_probed(self, devices: list[tuple[int, int, int]]) -> None:
        self.uvc_devices = devices
        if not devices:
            self.lbl_uvc.config(text="No visible camera found")
            return
        self.cb_uvc.config(values=[uvc.describe(*d) for d in devices],
                           state="readonly")
        chosen = None
        if self._uvc_preferred is not None:
            chosen = next((d for d in devices if d[0] == self._uvc_preferred), None)
        # Default to the highest-resolution device: a dedicated capture camera
        # outresolves a built-in webcam, which always takes index 0.
        chosen = chosen or uvc.best_camera(devices)
        assert chosen is not None
        self.var_uvc_dev.set(uvc.describe(*chosen))
        self.lbl_uvc.config(text=f"{len(devices)} found")
        self._ensure_uvc()

    def _selected_uvc(self) -> tuple[int, int, int] | None:
        label = self.var_uvc_dev.get()
        return next((d for d in self.uvc_devices if uvc.describe(*d) == label), None)

    def _toggle_uvc(self) -> None:
        self._ensure_uvc()

    def _select_uvc(self) -> None:
        """Reopen after the device choice changes."""
        self._close_uvc()
        self._ensure_uvc()

    def _ensure_uvc(self) -> None:
        """Hold the webcam open exactly while it is being shown or recorded.

        Previewing and recording are separate needs; the device is released
        when neither applies so its light goes out and other apps can use it.
        """
        if not self._wants_visible():
            if self.uvc_cam is not None and not self.uvc_cam.recording:
                self._close_uvc()
                self.lbl_uvc.config(text="Idle", foreground=FG_DIM)
            return
        if self.uvc_cam is not None:
            return
        device = self._selected_uvc()
        if device is None:
            return
        index, w, h = device
        self.uvc_cam = uvc.UVCCamera(index=index, width=w, height=h,
                                     show_timestamp=self.var_stamp.get())
        self.uvc_cam.start()
        self.lbl_uvc.config(text=f"Opening camera {index}...", foreground=FG_DIM)

    def _close_uvc(self) -> None:
        if self.uvc_cam is not None:
            self.uvc_cam.stop()
            self.uvc_cam.join(timeout=3.0)
            self.uvc_cam = None

    def _build_scale(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Temperature scale ", padding=10)
        box.pack(fill="x", padx=10, pady=6)

        self.var_agc = tk.IntVar(value=int(self.viewer.agc_mode))
        for mode, label in (
            (AGCMode.FACTORY, "Factory AGC (auto)"),
            (AGCMode.TEMPORAL_1, "Percentile (auto)"),
            (AGCMode.FIXED_RANGE, "Fixed range (absolute)"),
            (AGCMode.LOG_RANGE, "Log range (absolute)"),
        ):
            ttk.Radiobutton(box, text=label, value=int(mode), variable=self.var_agc,
                            command=self._apply_agc).pack(anchor="w")

        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Min °C", style="Dim.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Max °C", style="Dim.TLabel").grid(row=0, column=1, sticky="w",
                                                                   padx=(8, 0))
        self.var_min = tk.StringVar(value=f"{self.viewer.fixed_range[0]:g}")
        self.var_max = tk.StringVar(value=f"{self.viewer.fixed_range[1]:g}")
        ttk.Entry(row, textvariable=self.var_min, width=9).grid(row=1, column=0, sticky="ew")
        ttk.Entry(row, textvariable=self.var_max, width=9).grid(row=1, column=1, sticky="ew",
                                                               padx=(8, 0))
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        srow = ttk.Frame(box, style="Panel.TFrame")
        srow.pack(fill="x", pady=(10, 0))
        self.lbl_title_strength = ttk.Label(srow, text="Log strength",
                                            style="Dim.TLabel")
        self.lbl_title_strength.pack(side="left")
        self.lbl_strength = ttk.Label(srow, text=f"{self.viewer.log_strength:.0f}",
                                      style="Value.TLabel")
        self.lbl_strength.pack(side="right")
        self.var_strength = tk.DoubleVar(value=self.viewer.log_strength)
        self.scale_strength = ttk.Scale(box, from_=1.0, to=400.0,
                                        variable=self.var_strength,
                                        command=self._apply_strength)
        self.scale_strength.pack(fill="x")

        ttk.Button(box, text="Apply range",
                   command=self._apply_range).pack(fill="x", pady=(8, 0))

        presets = ttk.Frame(box, style="Panel.TFrame")
        presets.pack(fill="x", pady=(6, 0))
        for i, (lo, hi) in enumerate(((20, 350), (0, 550), (15, 40))):
            ttk.Button(presets, text=f"{lo}–{hi}", width=7,
                       command=lambda a=lo, b=hi: self._set_range(a, b)).grid(
                           row=0, column=i, padx=(0 if i == 0 else 4, 0), sticky="ew")
            presets.columnconfigure(i, weight=1)

    def _build_camera(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Camera ", padding=10)
        box.pack(fill="x", padx=10, pady=6)

        ttk.Label(box, text="Gain", style="Dim.TLabel").pack(anchor="w")
        initial_gain = self.viewer.startup_gain_mode or self.viewer.camera.gain_mode
        self.var_gain = tk.IntVar(value=int(initial_gain))
        grow = ttk.Frame(box, style="Panel.TFrame")
        grow.pack(fill="x", pady=(2, 8))
        ttk.Radiobutton(grow, text="Low (0–550 °C)", value=int(GainMode.LOW),
                        variable=self.var_gain, command=self._apply_gain).pack(anchor="w")
        ttk.Radiobutton(grow, text="High (−20–150 °C)", value=int(GainMode.HIGH),
                        variable=self.var_gain, command=self._apply_gain).pack(anchor="w")

        erow = ttk.Frame(box, style="Panel.TFrame")
        erow.pack(fill="x", pady=(0, 8))
        ttk.Label(erow, text="Emissivity", style="Dim.TLabel").pack(side="left")
        self.var_emis = tk.DoubleVar(value=self.viewer.camera.env_params.emissivity)
        ttk.Spinbox(erow, from_=0.1, to=1.0, increment=0.05, width=6,
                    textvariable=self.var_emis, command=self._apply_emis).pack(side="right")

        ttk.Button(box, text="Trigger shutter / NUC",
                   command=self._shutter).pack(fill="x")

    def _build_display(self, parent: tk.Widget) -> None:
        box = ttk.Labelframe(parent, text=" Display ", padding=10)
        box.pack(fill="x", padx=10, pady=6)

        crow = ttk.Frame(box, style="Panel.TFrame")
        crow.pack(fill="x", pady=(0, 8))
        ttk.Label(crow, text="Colormap", style="Dim.TLabel").pack(side="left")
        self.var_cmap = tk.StringVar(value=ColormapID(self.viewer.colormap_idx).name)
        cb = ttk.Combobox(crow, textvariable=self.var_cmap, width=13, state="readonly",
                          values=[c.name for c in ColormapID])
        cb.pack(side="right")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_cmap())

        self.var_enh = tk.BooleanVar(value=self.viewer.enhanced)
        self.var_ret = tk.BooleanVar(value=self.viewer.show_reticule)
        self.var_cbar = tk.BooleanVar(value=self.viewer.show_colorbar)
        self.var_mirror = tk.BooleanVar(value=self.viewer.mirror)
        self.var_stamp = tk.BooleanVar(value=self.viewer.show_timestamp)
        self.var_hot = tk.BooleanVar(value=self.viewer.hotspot_mode != HotspotMode.OFF)
        for text, var in (
            ("Enhanced (CLAHE + DDE)", self.var_enh),
            ("Reticule", self.var_ret),
            ("Colorbar", self.var_cbar),
            ("Min/max markers", self.var_hot),
            ("Date/time stamp", self.var_stamp),
            ("Mirror", self.var_mirror),
        ):
            ttk.Checkbutton(box, text=text, variable=var,
                            command=self._apply_display).pack(anchor="w")

        ttk.Button(box, text="Rotate 90°",
                   command=self._rotate).pack(fill="x", pady=(8, 0))

    # -- helpers ----------------------------------------------------------

    def _default_name(self) -> str:
        return time.strftime("p3_%Y%m%d_%H%M%S")

    @property
    def _recording(self) -> bool:
        return self.viewer._rec_raw is not None

    # -- actions ----------------------------------------------------------

    def _pick_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.var_dir.get())
        if d:
            self.var_dir.set(d)

    def _toggle_record(self) -> None:
        if self._recording:
            # Finalize the companion streams first so their metadata makes it
            # into the sidecar, which _stop_recording writes.
            self._record_intent = False
            had_video = self.uvc_cam is not None and self.uvc_cam.recording
            self._stop_uvc_recording()
            sound = self._stop_audio()
            if sound:
                self.viewer.extra_metadata.setdefault("audio", sound)
            self.capture.post(self.viewer._stop_recording)
            if self._session_dir:
                self._finish_session(self._session_dir, had_video)
            self.btn_record.config(text="●  Start Recording", bg=REC_RED,
                                   activebackground="#d63c3c")
            self.var_name.set(self._default_name())
            return

        folder = self.var_dir.get().strip()
        name = self.var_name.get().strip()
        if not name:
            messagebox.showerror("Recording", "Enter a name for the recording.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Recording", f"Folder does not exist:\n{folder}")
            return

        # One folder per session: a capture is several files that only make
        # sense together, and they were easy to lose among earlier recordings.
        session = os.path.join(folder, name)
        occupied = os.path.isdir(session) and os.listdir(session)
        if occupied and not messagebox.askyesno(
            "Overwrite?", f"{name} already exists and is not empty. Overwrite?"
        ):
            return
        try:
            os.makedirs(session, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Recording", f"Could not create folder:\n{e}")
            return

        self._session_dir = session
        base = os.path.join(session, "thermal")

        self._record_intent = True
        if self.uvc_cam is not None and self.var_uvc_on.get():
            self.uvc_cam.start_recording(os.path.join(session, "visible.mp4"))
        self._start_audio(session)

        # Let the capture thread open the files; it owns the writer.
        self.viewer._pending_record = base
        self.btn_record.config(text="■  Stop Recording", bg="#5c6370",
                               activebackground="#6b7280")

    def _start_audio(self, session: str) -> None:
        """Begin capturing sound for this session, if enabled."""
        if not self.var_audio_on.get():
            return
        mic = self._selected_mic()
        if mic is None:
            return
        index, _name, channels, samplerate = mic
        self.audio_rec = audio.AudioRecorder(
            os.path.join(session, "audio.wav"),
            device=index, channels=channels, samplerate=samplerate,
        )
        self.audio_rec.start()

    def _stop_audio(self) -> dict[str, object]:
        rec, self.audio_rec = self.audio_rec, None
        if rec is None:
            return {}
        meta = rec.stop()
        if rec.error:
            return {"error": rec.error}
        if meta:
            meta["file"] = os.path.basename(str(meta["file"]))
        return meta

    def _finish_session(self, session: str, had_video: bool) -> None:
        """Mux the sound into the visible mp4 once both have been written.

        Runs off the Tk thread: ffmpeg copies the video stream but still has to
        encode the audio, which should not freeze the window.
        """
        video = os.path.join(session, "visible.mp4")
        sound = os.path.join(session, "audio.wav")
        if not (had_video and os.path.exists(video) and os.path.exists(sound)):
            return

        def work() -> None:
            merged = os.path.join(session, "visible_av.mp4")
            error = audio.mux(video, sound, merged)
            if error is None and os.path.exists(merged):
                try:
                    os.replace(merged, video)
                    os.remove(sound)
                    note = "sound muxed into visible.mp4"
                except OSError as e:
                    note = f"mux kept separate: {e}"
            else:
                # Keeping the WAV is the safe outcome: nothing is lost, the
                # two files just have to be combined by hand.
                note = f"sound kept as audio.wav ({error})"
            if self._closing:
                return
            try:
                self.root.after(
                    0, lambda: self.lbl_uvc.config(text=note, foreground=FG_DIM)
                )
            except tk.TclError:
                # The window went away while ffmpeg was running; the files are
                # already on disk, so there is nothing left to report.
                pass

        self._mux_thread = threading.Thread(target=work, daemon=True)
        self._mux_thread.start()

    def _stop_uvc_recording(self) -> None:
        """Finalize the visible recording and stage it for the sidecar."""
        if self.uvc_cam is None or not self.uvc_cam.recording:
            return
        meta = self.uvc_cam.stop_recording()
        if meta:
            meta["file"] = "visible.mp4"
            meta["note"] = (
                "Independent device; align with the thermal stream using "
                "started_at and measured_fps rather than frame index."
            )
            self.viewer.extra_metadata = {"visible": meta}

    def _snapshot(self) -> None:
        def do() -> None:
            self.viewer._screenshot()
            if self.viewer._last_thermal is not None:
                self.viewer._dump(self.viewer._last_thermal)
        self.capture.post(do)

    def _apply_agc(self) -> None:
        self.viewer.agc_mode = AGCMode(self.var_agc.get())
        self._sync_strength_enabled()

    def _sync_strength_enabled(self) -> None:
        """Grey the strength slider out unless it actually affects the image.

        It only applies in LOG_RANGE. Left enabled in the linear modes it reads
        as a live control that is being ignored, which is exactly how it looks
        at startup when --log was not passed.
        """
        active = self.viewer.agc_mode == AGCMode.LOG_RANGE
        self.scale_strength.state(["!disabled"] if active else ["disabled"])
        self.lbl_strength.config(style="Value.TLabel" if active else "Dim.TLabel")
        self.lbl_title_strength.config(
            text="Log strength" if active else "Log strength (log mode only)"
        )

    def _set_range(self, lo: float, hi: float) -> None:
        self.var_min.set(f"{lo:g}")
        self.var_max.set(f"{hi:g}")
        self._apply_range()

    def _apply_range(self) -> None:
        try:
            lo = float(self.var_min.get())
            hi = float(self.var_max.get())
        except ValueError:
            messagebox.showerror("Range", "Min and max must be numbers.")
            return
        if lo >= hi:
            messagebox.showerror("Range", "Min must be less than max.")
            return
        self.viewer.fixed_range = (lo, hi)
        # Applying a range must not knock the user out of log mode.
        if self.viewer.agc_mode != AGCMode.LOG_RANGE:
            self.var_agc.set(int(AGCMode.FIXED_RANGE))
            self.viewer.agc_mode = AGCMode.FIXED_RANGE
        self._sync_strength_enabled()
        if hi > 150 and self.viewer.camera.gain_mode != GainMode.LOW:
            self.var_gain.set(int(GainMode.LOW))
            self._apply_gain()

    def _apply_strength(self, _value: str | None = None) -> None:
        strength = float(self.var_strength.get())
        self.viewer.log_strength = strength
        self.lbl_strength.config(text=f"{strength:.0f}")

    def _apply_gain(self) -> None:
        mode = GainMode(self.var_gain.get())
        self.capture.post(lambda: self.viewer.camera.set_gain_mode(mode))

    def _apply_emis(self) -> None:
        self.viewer.camera.env_params.emissivity = float(self.var_emis.get())

    def _shutter(self) -> None:
        self.capture.post(self.viewer.camera.trigger_shutter)

    def _apply_cmap(self) -> None:
        self.viewer.colormap_idx = int(ColormapID[self.var_cmap.get()])

    def _apply_display(self) -> None:
        v = self.viewer
        v.enhanced = self.var_enh.get()
        v.use_clahe = self.var_enh.get()
        v.show_reticule = self.var_ret.get()
        v.show_colorbar = self.var_cbar.get()
        v.mirror = self.var_mirror.get()
        v.hotspot_mode = HotspotMode.MINMAX if self.var_hot.get() else HotspotMode.OFF
        v.show_timestamp = self.var_stamp.get()
        if self.uvc_cam is not None:
            self.uvc_cam.show_timestamp = self.var_stamp.get()

    def _rotate(self) -> None:
        self.viewer.rotation = (self.viewer.rotation + 90) % 360

    # -- loop -------------------------------------------------------------

    def _tick(self) -> None:
        # A camera error is no longer fatal: the capture thread reconnects on
        # its own, so the window stays up and reports what is happening.
        frame = self._compose()
        if frame is not None:
            self._show(frame)
        self._update_connection()
        self._update_status()
        self.root.after(33, self._tick)

    def _update_connection(self) -> None:
        state = self.capture.state
        if state is State.STREAMING:
            self.lbl_conn.config(text="● Connected", foreground=OK_GREEN)
            self.btn_reconnect.config(text="Reconnect")
        elif state is State.CONNECTING:
            self.lbl_conn.config(text="○ Connecting...", foreground=FG_DIM)
        else:
            detail = f" ({self.capture.reconnects})" if self.capture.reconnects else ""
            self.lbl_conn.config(text=f"● Reconnecting{detail}...",
                                 foreground=REC_RED)
            # Say why, so an unplugged cable is distinguishable from a hang.
            if self.capture.last_error:
                self.preview.config(
                    image="",
                    text=f"Camera disconnected\n\n{self.capture.last_error}\n\n"
                         "Reconnecting automatically...",
                )
                self._photo = None

    def _compose(self) -> NDArray[np.uint8] | None:
        """Build the preview image for the selected view."""
        view = self.var_view.get()
        thermal = (self.capture.latest()
                   if self.capture.state is State.STREAMING else None)
        visible = self.uvc_cam.latest() if self.uvc_cam is not None else None

        if view == "thermal":
            return thermal
        if view == "visible":
            return visible
        # Both: fall back to whichever is available rather than showing nothing.
        if thermal is None:
            return visible
        if visible is None:
            return thermal
        return self._stacked(thermal, visible)

    @staticmethod
    def _stacked(
        top: NDArray[np.uint8], bottom: NDArray[np.uint8], gap: int = 6
    ) -> NDArray[np.uint8]:
        """Stack two frames at a common width.

        Both feeds are landscape, so placing them side by side produces a very
        wide, short image that wastes most of a normal window. Stacking keeps
        each one as large as the window's width allows.

        The webcam frame is scaled to the thermal frame's width rather than the
        other way round; upscaling the thermal image would cost time and add
        no detail.
        """
        w = top.shape[1]
        bh = max(1, int(bottom.shape[0] * w / bottom.shape[1]))
        bottom = cv2.resize(bottom, (w, bh), interpolation=cv2.INTER_AREA)
        divider = np.zeros((gap, w, 3), dtype=np.uint8)
        return np.vstack([top, divider, bottom])

    def _show(self, frame: NDArray[np.uint8]) -> None:
        # Measure the container, never the Label: the Label's own size is a
        # function of the image we are about to give it.
        avail_w = self.stage.winfo_width()
        avail_h = self.stage.winfo_height()
        h, w = frame.shape[:2]
        if avail_w > 10 and avail_h > 10:
            scale = min(avail_w / w, avail_h / h, 2.0)
            new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
            if (new_w, new_h) != (w, h):
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.config(image=self._photo, text="")

    def _update_status(self) -> None:
        v = self.viewer
        stats = v._last_stats
        parts = [f"{v.fps:4.1f} fps"]
        if stats:
            parts.append(f"Spot {stats['tspot']:.1f}°C")
            parts.append(f"Scene {stats['tmin']:.1f}–{stats['tmax']:.1f}°C")
        parts.append(f"Gain {v.camera.gain_mode.name}")
        if v.agc_mode == AGCMode.FIXED_RANGE:
            parts.append(f"Scale {v.fixed_range[0]:g}–{v.fixed_range[1]:g}°C linear")
        elif v.agc_mode == AGCMode.LOG_RANGE:
            parts.append(
                f"Scale {v.fixed_range[0]:g}–{v.fixed_range[1]:g}°C log "
                f"(k={v.log_strength:.0f})"
            )
        else:
            parts.append("Scale auto")
        self.status.config(text="   |   ".join(parts))

        if self.var_gain.get() != int(v.camera.gain_mode):
            self.var_gain.set(int(v.camera.gain_mode))

        if self._recording:
            elapsed = time.time() - v._rec_start
            mb = v._rec_count * (v._rec_shape[0] * v._rec_shape[1] * 2 / 1e6) \
                if v._rec_shape else 0.0
            dot = "●" if int(elapsed * 2) % 2 == 0 else "○"
            self.lbl_rec.config(
                text=f"{dot} REC  {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}   "
                     f"{v._rec_count} frames   {mb:.0f} MB raw",
                foreground=REC_RED,
            )
            # The capture thread may have opened the files after the click.
            self.btn_record.config(text="■  Stop Recording", bg="#5c6370")
        else:
            self.lbl_rec.config(text="Not recording", foreground=FG_DIM)
            self.btn_record.config(text="●  Start Recording", bg=REC_RED,
                                   activebackground="#d63c3c")
            # A disconnect finalizes the thermal recording on the capture
            # thread; the visible stream has to follow it down. Waiting for
            # the pending start to clear avoids tearing down a recording that
            # simply has not opened yet.
            if self._record_intent and self.viewer._pending_record is None:
                self._record_intent = False
                had_video = self.uvc_cam is not None and self.uvc_cam.recording
                self._stop_uvc_recording()
                self._stop_audio()
                if self._session_dir:
                    self._finish_session(self._session_dir, had_video)

        self._update_uvc_status()

    def _update_uvc_status(self) -> None:
        cam = self.uvc_cam
        if cam is None:
            return
        if cam.error:
            self.lbl_uvc.config(text=cam.error, foreground=REC_RED)
        elif cam.recording:
            self.lbl_uvc.config(text=f"● REC  {cam.frames_written} frames",
                                foreground=REC_RED)
        elif cam.opened.is_set():
            frame = cam.latest()
            size = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "-"
            self.lbl_uvc.config(text=f"Ready  {size} @ {cam.fps:g} fps",
                                foreground=OK_GREEN)

    def _on_close(self) -> None:
        if self._recording and not messagebox.askyesno(
            "Quit", "A recording is in progress. Stop it and quit?"
        ):
            return
        self._closing = True
        self._stop_uvc_recording()
        self._stop_audio()
        self._close_uvc()
        # Let a mux that is already running finish writing before the process
        # goes away, so the session is not left with a stray WAV.
        if self._mux_thread is not None and self._mux_thread.is_alive():
            self._mux_thread.join(timeout=30.0)
        self.capture.stop()
        self.capture.join(timeout=5.0)
        self.root.destroy()


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="P3/P1 thermal camera GUI")
    parser.add_argument("--model", choices=["p1", "p3"], default="p3",
                        help="Camera model (default: p3)")
    parser.add_argument("--range", nargs=2, type=float, metavar=("MIN_C", "MAX_C"),
                        default=None, help="Start with this fixed absolute scale")
    parser.add_argument("--log", action="store_true",
                        help="Start with the logarithmic mapping of --range")
    parser.add_argument("--log-strength", type=float, default=50.0,
                        help="Log curve strength; higher lifts the cool end "
                             "more (default: 50)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Do not burn a date/time stamp into the video")
    parser.add_argument("--uvc", type=int, default=None, metavar="INDEX",
                        help="Preselect this visible-camera index; the default "
                             "is the highest-resolution device found")
    parser.add_argument("--gain", choices=["low", "high"], default=None,
                        help="Sensor gain mode at startup")
    parser.add_argument("--record-fps", type=float, default=25.0,
                        help="Frame rate written into the mp4 (default: 25.0)")
    args = parser.parse_args()

    if args.range is not None and args.range[0] >= args.range[1]:
        parser.error("--range MIN_C must be less than MAX_C")
    if args.log and args.range is None:
        parser.error("--log requires --range")
    if args.log_strength <= 0:
        parser.error("--log-strength must be positive")

    viewer = P3Viewer(
        model=args.model,
        fixed_range=tuple(args.range) if args.range else None,
        log_scale=args.log,
        log_strength=args.log_strength,
        show_timestamp=not args.no_timestamp,
        gain_mode=GainMode[args.gain.upper()] if args.gain else None,
        record_fps=args.record_fps,
    )
    # GUI previews at native scale and fits to the window instead.
    viewer.zoom = 2

    root = tk.Tk()
    P3GUI(root, viewer, uvc_index=args.uvc)
    root.mainloop()


if __name__ == "__main__":
    main()
