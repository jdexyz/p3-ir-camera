#!/usr/bin/env python3
"""Visible-light USB (UVC) camera capture, for recording alongside the P3.

The thermal camera and a webcam are independent devices with their own clocks,
so frames cannot be interleaved one-for-one. Instead each stream is recorded at
its own rate and the sidecar records both start times and measured rates, which
is enough to align them afterwards.
"""

from __future__ import annotations

import sys
import threading
import time

from numpy.typing import NDArray

import cv2
import numpy as np

from overlay import draw_timestamp


# DirectShow rather than the Windows default of Media Foundation: MSMF accepts
# focus writes, returns success, and silently ignores them, while DirectShow
# actually drives the lens. Both handle this camera's full resolution.
BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

# Focus values this class will emit. UVC cameras expose a coarse, quantised
# range and clamp anything outside it, so the exact bounds are the device's
# business, not ours.
FOCUS_MIN = 0
FOCUS_MAX = 260
FOCUS_STEP = 10


def _open(index: int) -> tuple[cv2.VideoCapture | None, bool]:
    """Open a camera, preferring the backend that supports focus.

    Returns (capture, using_preferred). Falling back matters because the
    preferred backend can fail to open a device that the default one still
    handles; capturing without focus control beats not capturing.
    """
    cap = cv2.VideoCapture(index, BACKEND)
    if cap.isOpened():
        return cap, True
    cap.release()
    if BACKEND == cv2.CAP_ANY:
        return None, False
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if cap.isOpened():
        return cap, False
    cap.release()
    return None, False

# Opening a camera costs ~1-2 s per index, so probing is deliberately shallow.
MAX_PROBE_INDEX = 4

# Used when the device does not report a usable rate.
DEFAULT_FPS = 30.0

# A camera can open successfully and then never deliver a frame -- Media
# Foundation reports MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED when another
# application holds the device. Waiting silently forever hides that, so give up
# and report it.
NO_FRAME_TIMEOUT = 5.0


# Asking for more than any webcam provides makes each one settle on its own
# maximum, which is the only thing distinguishing them through OpenCV.
_PROBE_REQUEST = (4096, 2160)


def list_cameras(max_index: int = MAX_PROBE_INDEX) -> list[tuple[int, int, int]]:
    """Probe for openable UVC cameras and their maximum resolution.

    OpenCV exposes indices but no device names, and indices are reassigned when
    devices are plugged or unplugged. Resolution is therefore the practical way
    to tell a 2K capture device from a built-in webcam.

    Returns:
        List of (index, width, height), lowest index first.
    """
    found: list[tuple[int, int, int]] = []
    for index in range(max_index):
        cap, _preferred = _open(index)
        if cap is None:
            continue
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _PROBE_REQUEST[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _PROBE_REQUEST[1])
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            found.append((index, int(frame.shape[1]), int(frame.shape[0])))
        finally:
            cap.release()
    return found


def describe(index: int, width: int, height: int) -> str:
    """Dropdown label for a probed camera."""
    return f"Camera {index} - {width}x{height}"


def best_camera(
    cameras: list[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    """Pick the highest-resolution camera.

    A dedicated capture device almost always outresolves a built-in webcam, so
    this defaults to the external one rather than to index 0.
    """
    return max(cameras, key=lambda c: c[1] * c[2], default=None)


class UVCCamera(threading.Thread):
    """Grabs from a UVC camera continuously, recording when asked.

    Grabbing runs whether or not a recording is active so the preview stays
    live and so a recording starts on an already-warm device instead of
    swallowing the camera's start-up delay.
    """

    def __init__(
        self,
        index: int = 0,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        show_timestamp: bool = True,
        autofocus: bool = True,
        focus: int | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.index = index
        self.show_timestamp = show_timestamp
        # None means take the rate from the device. Writing a webcam's frames
        # into a container labelled with the thermal camera's rate plays the
        # result back at the wrong speed.
        self.fps = fps or DEFAULT_FPS
        self._fps_explicit = fps is not None
        self.width = width
        self.height = height

        self.error: str | None = None
        self.opened = threading.Event()

        self._cap: cv2.VideoCapture | None = None
        self._frame: NDArray[np.uint8] | None = None
        self._writer: cv2.VideoWriter | None = None
        self._pending_path: str | None = None
        self._size: tuple[int, int] | None = None
        self._count = 0
        self._start_time = 0.0
        self._last_meta: dict[str, object] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Focus is applied on the grab thread, which owns the capture handle.
        self._pending_autofocus: bool | None = autofocus
        self._pending_focus: int | None = focus
        self.autofocus = autofocus
        self.focus: float | None = None
        self.focus_supported = False
        self.focus_backend = True

    # -- public API -------------------------------------------------------

    def latest(self) -> NDArray[np.uint8] | None:
        with self._lock:
            return self._frame

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._writer is not None or self._pending_path is not None

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._count

    def start_recording(self, path: str) -> None:
        """Request recording to path; the grab thread opens the writer."""
        with self._lock:
            if self._writer is None:
                self._pending_path = path

    def stop_recording(self) -> dict[str, object]:
        """Finalize the recording and return its metadata."""
        with self._lock:
            self._pending_path = None
            writer, self._writer = self._writer, None
            count, start = self._count, self._start_time
            size = self._size
            self._count = 0
        if writer is None:
            return {}
        writer.release()
        elapsed = time.time() - start
        meta = {
            "frames": count,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
            "duration_s": round(elapsed, 3),
            "measured_fps": round(count / elapsed, 3) if elapsed > 0 else 0.0,
            "resolution": list(size) if size else None,
            "device_index": self.index,
            "video_fps": self.fps,
            "timestamped": self.show_timestamp,
            "autofocus": self.autofocus,
            "focus": self.focus,
        }
        self._last_meta = meta
        return meta

    def set_autofocus(self, enabled: bool) -> None:
        """Queue an autofocus change for the grab thread."""
        with self._lock:
            self._pending_autofocus = enabled
            self.autofocus = enabled

    def set_focus(self, value: int) -> None:
        """Queue a manual focus position for the grab thread."""
        with self._lock:
            self._pending_focus = int(value)

    def stop(self) -> None:
        self._stop.set()

    # -- thread -----------------------------------------------------------

    def run(self) -> None:
        try:
            cap, preferred = _open(self.index)
            if cap is None:
                self.error = f"Could not open UVC camera {self.index}"
                self.opened.set()
                return
            self.focus_backend = preferred
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self._fps_explicit:
                reported = cap.get(cv2.CAP_PROP_FPS)
                if 1.0 < reported < 240.0:
                    self.fps = float(reported)
            # A camera that reports a focus position has a driveable lens.
            self.focus_supported = cap.get(cv2.CAP_PROP_FOCUS) >= 0
            self._cap = cap
            self.opened.set()

            last_ok = time.time()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # A momentary read failure is normal on some webcams; a
                    # sustained one means the device is not really ours.
                    if time.time() - last_ok > NO_FRAME_TIMEOUT:
                        self.error = (
                            f"Camera {self.index} opened but delivered no frames "
                            "- another application may be using it"
                        )
                        return
                    time.sleep(0.02)
                    continue
                last_ok = time.time()
                self._apply_focus(cap)
                captured = time.time()
                with self._lock:
                    self._frame = frame
                    self._open_pending_writer(frame)
                    if self._writer is not None:
                        # Copy first: the stamp must not land in the frame
                        # handed out by latest(), which callers may reuse.
                        out = self._fit(frame).copy()
                        if self.show_timestamp:
                            draw_timestamp(out, captured)
                        self._writer.write(out)
                        self._count += 1
        except Exception as e:
            self.error = str(e)
        finally:
            self.stop_recording()
            if self._cap is not None:
                self._cap.release()

    # -- internals --------------------------------------------------------

    def _apply_focus(self, cap: cv2.VideoCapture) -> None:
        """Push any queued focus change, then read back what the lens took.

        The device arbitrates: it quantises manual values and moves the lens
        itself while autofocus is on, so the read-back is the truth rather than
        whatever was requested.
        """
        with self._lock:
            autofocus, self._pending_autofocus = self._pending_autofocus, None
            focus, self._pending_focus = self._pending_focus, None

        if autofocus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if focus is not None and not self.autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, float(focus))
        if autofocus is not None or focus is not None or self._count % 30 == 0:
            self.focus = cap.get(cv2.CAP_PROP_FOCUS)

    def _open_pending_writer(self, frame: NDArray[np.uint8]) -> None:
        """Open the writer once a real frame has sized it. Caller holds lock."""
        if self._pending_path is None:
            return
        path, self._pending_path = self._pending_path, None
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        if not writer.isOpened():
            self.error = f"Could not open video writer for {path}"
            return
        self._writer = writer
        self._size = (w, h)
        self._count = 0
        self._start_time = time.time()

    def _fit(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Force the writer's locked frame size; webcams can change mode."""
        assert self._size is not None
        w, h = self._size
        if (frame.shape[1], frame.shape[0]) != (w, h):
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame
