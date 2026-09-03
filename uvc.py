#!/usr/bin/env python3
"""Visible-light USB (UVC) camera capture, for recording alongside the P3.

The thermal camera and a webcam are independent devices with their own clocks,
so frames cannot be interleaved one-for-one. Instead each stream is recorded at
its own rate and the sidecar records both start times and measured rates, which
is enough to align them afterwards.
"""

from __future__ import annotations

import threading
import time

from numpy.typing import NDArray

import cv2
import numpy as np


# Opening a camera costs ~1-2 s per index, so probing is deliberately shallow.
MAX_PROBE_INDEX = 4

# Used when the device does not report a usable rate.
DEFAULT_FPS = 30.0


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
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
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
    ) -> None:
        super().__init__(daemon=True)
        self.index = index
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
        }
        self._last_meta = meta
        return meta

    def stop(self) -> None:
        self._stop.set()

    # -- thread -----------------------------------------------------------

    def run(self) -> None:
        try:
            cap = cv2.VideoCapture(self.index)
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not cap.isOpened():
                self.error = f"Could not open UVC camera {self.index}"
                self.opened.set()
                return
            if not self._fps_explicit:
                reported = cap.get(cv2.CAP_PROP_FPS)
                if 1.0 < reported < 240.0:
                    self.fps = float(reported)
            self._cap = cap
            self.opened.set()

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # A momentary read failure is normal on some webcams; a
                    # persistent one is handled by the caller noticing the
                    # frame stops updating.
                    time.sleep(0.02)
                    continue
                with self._lock:
                    self._frame = frame
                    self._open_pending_writer(frame)
                    if self._writer is not None:
                        self._writer.write(self._fit(frame))
                        self._count += 1
        except Exception as e:
            self.error = str(e)
        finally:
            self.stop_recording()
            if self._cap is not None:
                self._cap.release()

    # -- internals --------------------------------------------------------

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
