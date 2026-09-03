"""Tests for the GUI capture thread's disconnect/stall recovery."""

from __future__ import annotations

import time

import numpy as np
import pytest

from p3_gui import CaptureThread, State

import p3_gui


class FakeCamera:
    def __init__(self) -> None:
        self.dev = object()
        self.streaming = True
        self.gain_mode = 0


class FakeViewer:
    """Stands in for P3Viewer, with scriptable failures."""

    def __init__(self, script=None) -> None:
        self.camera = FakeCamera()
        self.script = list(script or [])
        self.stall_forever = False
        self.opens = 0
        self.closes = 0
        self.frames = 0
        self.open_fails = 0

    def open_camera(self):
        self.opens += 1
        if self.open_fails > 0:
            self.open_fails -= 1
            raise OSError("device not ready")
        self.camera.dev = object()
        self.camera.streaming = True
        return ("P3", "0.0")

    def close_camera(self):
        self.closes += 1

    def process_frame(self):
        if self.stall_forever:
            time.sleep(0.01)
            return None
        if self.script:
            action = self.script.pop(0)
            if isinstance(action, Exception):
                raise action
            if action is None:
                return None
        self.frames += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(p3_gui, "RETRY_DELAY", 0.05)


def test_streams_frames(fast_retry):
    viewer = FakeViewer()
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: thread.latest() is not None)
        assert thread.state is State.STREAMING
    finally:
        thread.stop()
        thread.join(timeout=5)


def test_recovers_from_usb_error(fast_retry):
    """A read failure must reconnect, not end the session."""
    viewer = FakeViewer(script=[OSError("[Errno 19] No such device")])
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: thread.reconnects >= 1)
        assert wait_for(lambda: thread.state is State.STREAMING)
        assert viewer.opens >= 2, "should have reopened the device"
        assert thread.last_error is not None
    finally:
        thread.stop()
        thread.join(timeout=5)


def test_recovers_from_stall(monkeypatch, fast_retry):
    """Reads that neither fail nor deliver must trip the watchdog.

    This is the resume-from-sleep case: the image simply stops updating.
    """
    monkeypatch.setattr(p3_gui, "STALL_TIMEOUT", 0.3)
    viewer = FakeViewer()
    viewer.stall_forever = True
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: thread.reconnects >= 1, timeout=15)
        assert "no frame" in (thread.last_error or "")
    finally:
        thread.stop()
        thread.join(timeout=5)


def test_retries_until_device_returns(fast_retry):
    """Reconnect keeps trying while the device is still absent."""
    viewer = FakeViewer()
    viewer.open_fails = 3
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: thread.state is State.STREAMING, timeout=15)
        assert viewer.opens >= 4
    finally:
        thread.stop()
        thread.join(timeout=5)


def test_manual_reconnect(fast_retry):
    viewer = FakeViewer()
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: thread.state is State.STREAMING)
        before = viewer.opens
        thread.request_reconnect()
        assert wait_for(lambda: viewer.opens > before)
        assert wait_for(lambda: thread.state is State.STREAMING)
    finally:
        thread.stop()
        thread.join(timeout=5)


def test_teardown_releases_device(fast_retry):
    """A stale handle must be dropped so the next find() gets a fresh one."""
    viewer = FakeViewer(script=[OSError("boom")])
    thread = CaptureThread(viewer)
    thread.start()
    try:
        assert wait_for(lambda: viewer.closes >= 1)
    finally:
        thread.stop()
        thread.join(timeout=5)
    assert viewer.closes >= 1


def test_stop_ends_thread(fast_retry):
    viewer = FakeViewer()
    thread = CaptureThread(viewer)
    thread.start()
    assert wait_for(lambda: thread.state is State.STREAMING)
    thread.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert thread.state is State.STOPPED
