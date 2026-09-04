#!/usr/bin/env python3
"""Replay a recorded session, with click-to-read temperatures.

The thermal pane is rendered from `thermal.raw` rather than from the recorded
video, because the video is 8-bit colour-mapped pixels while the raw stream is
the sensor's own counts. Only the raw data can answer "what temperature is
that point", and only it maps cleanly back to a sensor pixel.

The visible pane and the sound come from `combined.mp4`, cropped below the
thermal pane using the split height stored in the sidecar.
"""

from __future__ import annotations

from tkinter import filedialog, messagebox, ttk

import json
import os
import subprocess
import tempfile
import time
import tkinter as tk
import wave

from numpy.typing import NDArray
from PIL import Image, ImageTk

import cv2
import numpy as np

from p3_viewer import (
    AGCMode,
    ColormapID,
    agc_fixed,
    agc_log,
    apply_colormap,
    raw_to_celsius,
)

import audio
import thermal_store


BG = "#1e1f22"
BG_PANEL = "#2b2d31"
FG = "#e3e5e8"
FG_DIM = "#9a9ea6"
ACCENT = "#4a9eff"


class Session:
    """A recorded session on disk."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        meta_path = os.path.join(folder, "thermal.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No thermal.json in {folder}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        frames, rows, cols = self.meta["raw_shape"]
        raw_path = os.path.join(folder, os.path.basename(self.meta["raw_file"]))
        if not os.path.exists(raw_path):
            # Fall back across formats so a session recorded either way opens.
            for candidate in ("thermal.tz", "thermal.raw"):
                if os.path.exists(os.path.join(folder, candidate)):
                    raw_path = os.path.join(folder, candidate)
                    break
        self.raw = thermal_store.open_frames(raw_path, (frames, rows, cols))
        self.frames = int(frames)
        self.rows = int(rows)
        self.cols = int(cols)
        self.fps = float(self.meta.get("measured_fps") or 25.0) or 25.0

        self.fixed_range = tuple(self.meta.get("fixed_range_c") or (20.0, 350.0))
        self.log_strength = float(self.meta.get("log_strength") or 50.0)
        self.agc_mode = (
            AGCMode.LOG_RANGE
            if self.meta.get("agc_mode") == "LOG_RANGE"
            else AGCMode.FIXED_RANGE
        )

        combined = self.meta.get("combined") or {}
        self.split = int(combined.get("thermal_pane_height") or 0)
        self.video_path = os.path.join(folder, "combined.mp4")
        self.has_video = os.path.exists(self.video_path)
        if self.has_video and self.split <= 0:
            # Recorded before the split was stored. The thermal pane spans the
            # full width and keeps the sensor's aspect ratio, so its height
            # follows from the video width.
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            estimated = round(width * self.rows / self.cols)
            self.split = estimated if 0 < estimated < height else 0
            self.has_video = self.split > 0

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def frame(self, index: int) -> NDArray[np.uint16]:
        """One thermal frame, whichever storage format it came from."""
        return np.asarray(self.raw[int(np.clip(index, 0, self.frames - 1))])

    def temperature(self, frame: int, row: int, col: int) -> float:
        """Temperature in Celsius at one sensor pixel of one frame."""
        row = int(np.clip(row, 0, self.rows - 1))
        col = int(np.clip(col, 0, self.cols - 1))
        return float(raw_to_celsius(self.frame(frame)[row, col]))


class AudioPlayer:
    """Plays the session's sound, extracted from the mp4."""

    def __init__(self, video: str) -> None:
        self.path: str | None = None
        self.error: str | None = None
        self._stream = None
        self._wav: wave.Wave_read | None = None
        self._extract(video)

    def _extract(self, video: str) -> None:
        exe = audio.ffmpeg_exe()
        if exe is None or not os.path.exists(video):
            self.error = "no ffmpeg" if exe is None else "no video"
            return
        out = os.path.join(tempfile.mkdtemp(prefix="p3replay_"), "audio.wav")
        done = subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", video, "-vn", out],
            capture_output=True, check=False,
        )
        if done.returncode != 0 or not os.path.exists(out):
            self.error = "no sound track"
            return
        self.path = out

    @property
    def available(self) -> bool:
        return self.path is not None and audio.available()

    def play_from(self, seconds: float) -> None:
        """(Re)start playback at a position, matching the video's clock."""
        self.stop()
        if not self.available:
            return
        import sounddevice as sd

        wav = wave.open(self.path, "rb")
        rate = wav.getframerate()
        wav.setpos(min(int(seconds * rate), wav.getnframes() - 1))
        channels = wav.getnchannels()
        self._wav = wav

        def callback(outdata, frames, _time, _status):
            data = wav.readframes(frames)
            need = frames * channels * 2
            if len(data) < need:
                data += b"\x00" * (need - len(data))
            outdata[:] = np.frombuffer(data, dtype="<i2").reshape(frames, channels)

        self._stream = sd.OutputStream(
            samplerate=rate, channels=channels, dtype="int16", callback=callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None


class ReplayWindow:
    """Playback window with a temperature probe."""

    def __init__(self, root: tk.Tk, session: Session) -> None:
        self.root = root
        self.session = session
        self.player = AudioPlayer(session.video_path) if session.has_video else None
        self.cap = cv2.VideoCapture(session.video_path) if session.has_video else None

        self.index = 0
        self.playing = False
        self._play_started = 0.0
        self._play_from_index = 0
        self._photo: ImageTk.PhotoImage | None = None
        self._probe: tuple[int, int] | None = None
        self._thermal_box = (0, 0, 0, 0)  # where the thermal pane is on screen

        root.title(f"Replay - {os.path.basename(session.folder)}")
        root.geometry("1100x900")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._show()

    # -- ui ---------------------------------------------------------------

    def _build(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x")
        self.status = ttk.Label(bar, text="", anchor="w", padding=(8, 4),
                                background=BG, foreground=FG_DIM)
        self.status.pack(fill="x")

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side="bottom", fill="x")
        self.btn_play = tk.Button(controls, text="▶  Play", width=10,
                                  command=self._toggle_play, relief="flat",
                                  bg="#383a40", fg=FG, activebackground="#4a4d55",
                                  activeforeground=FG, cursor="hand2")
        self.btn_play.pack(side="left")
        self.var_pos = tk.DoubleVar(value=0.0)
        self.scale_pos = ttk.Scale(controls, from_=0, to=max(self.session.frames - 1, 1),
                                   variable=self.var_pos, command=self._seek)
        self.scale_pos.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_time = ttk.Label(controls, text="0.0s", background=BG,
                                  foreground=FG_DIM, width=14)
        self.lbl_time.pack(side="right")

        self.stage = tk.Frame(self.root, bg="#000000")
        self.stage.pack(fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.view = tk.Label(self.stage, bg="#000000", cursor="crosshair")
        self.view.place(relx=0.5, rely=0.5, anchor="center")
        self.view.bind("<Button-1>", self._on_click)

    # -- rendering --------------------------------------------------------

    def _render_thermal(self) -> NDArray[np.uint8]:
        """Colour-map the raw frame using the session's own AGC settings."""
        s = self.session
        frame = np.asarray(s.frame(self.index))
        if s.agc_mode == AGCMode.LOG_RANGE:
            img = agc_log(frame, *s.fixed_range, strength=s.log_strength)
        else:
            img = agc_fixed(frame, *s.fixed_range)
        return apply_colormap(img, ColormapID.IRONBOW)

    def _visible_pane(self) -> NDArray[np.uint8] | None:
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame[self.session.split:]

    def _show(self) -> None:
        thermal = self._render_thermal()
        # Upscale the sensor image so a click lands on a pixel precisely.
        scale = 3
        thermal = cv2.resize(
            thermal,
            (self.session.cols * scale, self.session.rows * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        if self._probe is not None:
            row, col = self._probe
            cx, cy = int((col + 0.5) * scale), int((row + 0.5) * scale)
            temp = self.session.temperature(self.index, row, col)
            cv2.drawMarker(thermal, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(thermal, f"{temp:.1f}C", (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        composed = thermal
        visible = self._visible_pane()
        if visible is not None:
            w = thermal.shape[1]
            h = max(1, int(visible.shape[0] * w / visible.shape[1]))
            visible = cv2.resize(visible, (w, h), interpolation=cv2.INTER_AREA)
            composed = np.vstack(
                [thermal, np.zeros((6, w, 3), np.uint8), visible]
            )

        self._thermal_box = (0, 0, thermal.shape[1], thermal.shape[0])
        aw, ah = self.stage.winfo_width(), self.stage.winfo_height()
        self._display_scale = 1.0
        if aw > 10 and ah > 10:
            f = min(aw / composed.shape[1], ah / composed.shape[0])
            if f < 1.0:
                composed = cv2.resize(
                    composed,
                    (max(1, int(composed.shape[1] * f)),
                     max(1, int(composed.shape[0] * f))),
                    interpolation=cv2.INTER_AREA,
                )
                self._display_scale = f
        self._thermal_scale = scale
        rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.config(image=self._photo)

        t = self.index / self.session.fps
        self.lbl_time.config(text=f"{t:5.1f}s / {self.session.duration:.1f}s")
        frame = np.asarray(self.session.frame(self.index))
        self.status.config(
            text=f"Frame {self.index + 1}/{self.session.frames}   |   "
                 f"Scene {raw_to_celsius(frame.min()):.1f}-"
                 f"{raw_to_celsius(frame.max()):.1f}°C   |   "
                 f"{'Log' if self.session.agc_mode == AGCMode.LOG_RANGE else 'Linear'} "
                 f"{self.session.fixed_range[0]:g}-{self.session.fixed_range[1]:g}°C"
                 + ("   |   click the thermal image to read a temperature"
                    if self._probe is None else "")
        )

    # -- interaction ------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        """Map a click back to a sensor pixel and probe it."""
        x = event.x / max(self._display_scale, 1e-6)
        y = event.y / max(self._display_scale, 1e-6)
        _, _, tw, th = self._thermal_box
        if not (0 <= x < tw and 0 <= y < th):
            return  # clicked the visible pane, which carries no temperatures
        col = int(x / self._thermal_scale)
        row = int(y / self._thermal_scale)
        self._probe = (row, col)
        self._show()

    def _seek(self, _value: str | None = None) -> None:
        index = int(float(self.var_pos.get()))
        if index == self.index:
            return
        self.index = index
        if self.playing:
            self._start_clock()
        self._show()

    def _start_clock(self) -> None:
        self._play_started = time.time()
        self._play_from_index = self.index
        if self.player is not None:
            self.player.play_from(self.index / self.session.fps)

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            if self.index >= self.session.frames - 1:
                self.index = 0
            self.btn_play.config(text="❚❚  Pause")
            self._start_clock()
            self._tick()
        else:
            self.btn_play.config(text="▶  Play")
            if self.player is not None:
                self.player.stop()

    def _tick(self) -> None:
        if not self.playing:
            return
        # Drive frames from the wall clock, not by counting ticks, so playback
        # keeps pace with the sound instead of drifting behind it.
        elapsed = time.time() - self._play_started
        index = self._play_from_index + int(elapsed * self.session.fps)
        if index >= self.session.frames:
            self.index = self.session.frames - 1
            self.playing = False
            self.btn_play.config(text="▶  Play")
            if self.player is not None:
                self.player.stop()
        elif index != self.index:
            self.index = index
        self.var_pos.set(self.index)
        self._show()
        if self.playing:
            self.root.after(15, self._tick)

    def _close(self) -> None:
        self.playing = False
        if self.player is not None:
            self.player.stop()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def open_session(parent: tk.Misc | None = None, folder: str | None = None) -> None:
    """Pick a session folder and open a replay window for it."""
    if folder is None:
        folder = filedialog.askdirectory(title="Open a recording session")
        if not folder:
            return
    try:
        session = Session(folder)
    except Exception as e:
        messagebox.showerror("Replay", f"Could not open session:\n{e}")
        return
    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.configure(bg=BG)
    ReplayWindow(win, session)


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay a P3 recording session")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Session folder (containing thermal.json)")
    args = parser.parse_args()

    root = tk.Tk()
    root.withdraw()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=FG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG)
    open_session(folder=args.folder)
    root.mainloop()


if __name__ == "__main__":
    main()
