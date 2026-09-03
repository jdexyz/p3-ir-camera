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

from p3_camera import GainMode
from p3_viewer import AGCMode, ColormapID, HotspotMode, P3Viewer


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


class CaptureThread(threading.Thread):
    """Owns the camera and produces rendered frames."""

    def __init__(self, viewer: P3Viewer) -> None:
        super().__init__(daemon=True)
        self.viewer = viewer
        self.commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self.frame: NDArray[np.uint8] | None = None
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def post(self, fn: Callable[[], None]) -> None:
        """Run fn on the capture thread, between frames."""
        self.commands.put(fn)

    def latest(self) -> NDArray[np.uint8] | None:
        with self._lock:
            return self.frame

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self.viewer.open_camera()
            self.ready.set()
            while not self._stop.is_set():
                while True:
                    try:
                        cmd = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        cmd()
                    except Exception as e:  # a bad command must not kill capture
                        print(f"Command failed: {e}")

                frame = self.viewer.process_frame()
                if frame is not None:
                    with self._lock:
                        self.frame = frame
        except BaseException as e:
            self.error = e
            self.ready.set()
        finally:
            try:
                self.viewer.close_camera()
            except Exception:
                pass


class P3GUI:
    """Tk control panel."""

    def __init__(self, root: tk.Tk, viewer: P3Viewer) -> None:
        self.root = root
        self.viewer = viewer
        self.capture = CaptureThread(viewer)
        self._photo: ImageTk.PhotoImage | None = None

        root.title("P3 Thermal Camera")
        root.geometry("1280x800")
        root.minsize(900, 600)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._init_style()
        self._build()

        self.capture.start()
        self.root.after(100, self._wait_for_camera)

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

        self._build_record(side)
        self._build_scale(side)
        self._build_camera(side)
        self._build_display(side)

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
        ttk.Label(srow, text="Log strength", style="Dim.TLabel").pack(side="left")
        self.lbl_strength = ttk.Label(srow, text=f"{self.viewer.log_strength:.0f}",
                                      style="Value.TLabel")
        self.lbl_strength.pack(side="right")
        self.var_strength = tk.DoubleVar(value=self.viewer.log_strength)
        ttk.Scale(box, from_=1.0, to=400.0, variable=self.var_strength,
                  command=self._apply_strength).pack(fill="x")

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
        self.var_hot = tk.BooleanVar(value=self.viewer.hotspot_mode != HotspotMode.OFF)
        for text, var in (
            ("Enhanced (CLAHE + DDE)", self.var_enh),
            ("Reticule", self.var_ret),
            ("Colorbar", self.var_cbar),
            ("Min/max markers", self.var_hot),
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
            self.capture.post(self.viewer._stop_recording)
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

        base = os.path.join(folder, name)
        exists = any(os.path.exists(base + ext) for ext in (".mp4", ".raw", ".json"))
        if exists and not messagebox.askyesno(
            "Overwrite?", f"{name} already exists. Overwrite?"
        ):
            return

        # Let the capture thread open the files; it owns the writer.
        self.viewer._pending_record = base
        self.btn_record.config(text="■  Stop Recording", bg="#5c6370",
                               activebackground="#6b7280")

    def _snapshot(self) -> None:
        def do() -> None:
            self.viewer._screenshot()
            if self.viewer._last_thermal is not None:
                self.viewer._dump(self.viewer._last_thermal)
        self.capture.post(do)

    def _apply_agc(self) -> None:
        self.viewer.agc_mode = AGCMode(self.var_agc.get())

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
        if hi > 150 and self.viewer.camera.gain_mode != GainMode.LOW:
            self.var_gain.set(int(GainMode.LOW))
            self._apply_gain()

    def _apply_strength(self, _value: str | None = None) -> None:
        strength = float(self.var_strength.get())
        self.viewer.log_strength = strength
        self.lbl_strength.config(text=f"{strength:.0f}")
        # The slider only means anything in log mode, so dragging it selects it.
        if self.viewer.agc_mode != AGCMode.LOG_RANGE:
            self.var_agc.set(int(AGCMode.LOG_RANGE))
            self.viewer.agc_mode = AGCMode.LOG_RANGE

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

    def _rotate(self) -> None:
        self.viewer.rotation = (self.viewer.rotation + 90) % 360

    # -- loop -------------------------------------------------------------

    def _wait_for_camera(self) -> None:
        if self.capture.error is not None:
            messagebox.showerror("Camera error", str(self.capture.error))
            self.root.destroy()
            return
        if not self.capture.ready.is_set():
            self.root.after(100, self._wait_for_camera)
            return
        self._tick()

    def _tick(self) -> None:
        if self.capture.error is not None:
            messagebox.showerror("Camera error", str(self.capture.error))
            self.root.destroy()
            return

        frame = self.capture.latest()
        if frame is not None:
            self._show(frame)
        self._update_status()
        self.root.after(33, self._tick)

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

    def _on_close(self) -> None:
        if self._recording and not messagebox.askyesno(
            "Quit", "A recording is in progress. Stop it and quit?"
        ):
            return
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
        gain_mode=GainMode[args.gain.upper()] if args.gain else None,
        record_fps=args.record_fps,
    )
    # GUI previews at native scale and fits to the window instead.
    viewer.zoom = 2

    root = tk.Tk()
    P3GUI(root, viewer)
    root.mainloop()


if __name__ == "__main__":
    main()
