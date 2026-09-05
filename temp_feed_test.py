"""Tests for the Sonia temperature feed server.

The rule these mostly guard is that a missing reading produces silence, never a
held-over value: the consumer's staleness timeout is what cuts the ultrasound,
and a repeated value would defeat it.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from temp_feed import TempFeedServer

import temp_feed


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Feed:
    """A running server on a free port, torn down after the test."""

    def __init__(self, provider, hz=50.0):
        self.port = free_port()
        self.server = TempFeedServer(provider, port=self.port, hz=hz)
        self.server.start()
        assert self.server.started.wait(5), "server never started"
        assert self.server.error is None, self.server.error

    def connect(self, timeout=5.0):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        sock.settimeout(timeout)
        return sock

    def close(self):
        self.server.stop()


@pytest.fixture
def feed():
    made = []

    def make(provider, hz=50.0):
        f = Feed(provider, hz=hz)
        made.append(f)
        return f

    yield make
    for f in made:
        f.close()


def read_lines(sock, count, timeout=5.0):
    """Read at least `count` newline-terminated lines."""
    buf, deadline = b"", time.time() + timeout
    while buf.count(b"\n") < count and time.time() < deadline:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            break
        if not chunk:
            break
        buf += chunk
    return [ln for ln in buf.decode().split("\n") if ln]


# --- format ------------------------------------------------------------------


def test_streams_ndjson_with_max_temp_c(feed):
    f = feed(lambda: 97.5)
    with f.connect() as sock:
        lines = read_lines(sock, 5)
    assert len(lines) >= 5
    for line in lines:
        assert json.loads(line) == {"max_temp_c": 97.5}


def test_every_line_is_newline_terminated(feed):
    f = feed(lambda: 20.0)
    with f.connect() as sock:
        time.sleep(0.3)
        raw = sock.recv(65536).decode()
    assert raw.endswith("\n")
    assert all(json.loads(p)["max_temp_c"] == 20.0 for p in raw.split("\n") if p)


def test_rate_is_roughly_as_configured(feed):
    f = feed(lambda: 30.0, hz=20.0)
    with f.connect() as sock:
        time.sleep(1.0)
        count = sock.recv(65536).decode().count("\n")
    # Generous bounds: this only needs to prove the pacing is honoured at all.
    assert 8 <= count <= 40, count


# --- the safety rule ---------------------------------------------------------


def test_none_sends_nothing(feed):
    """No fresh reading must produce silence, not a placeholder."""
    f = feed(lambda: None)
    with f.connect() as sock:
        time.sleep(0.4)
        sock.settimeout(0.3)
        with pytest.raises((TimeoutError, socket.timeout)):
            sock.recv(4096)
    assert f.server.sent == 0
    assert f.server.skipped > 0


def test_stops_and_resumes_without_repeating(feed):
    """A gap must be a gap; the value must not be held across it."""
    state = {"value": 25.0}
    f = feed(lambda: state["value"], hz=50.0)
    with f.connect() as sock:
        first = read_lines(sock, 3)
        state["value"] = None
        time.sleep(0.4)
        # Drain whatever was already in flight. It may well be nothing, which
        # is itself the point, so a timeout here is fine.
        sock.settimeout(0.3)
        try:
            sock.recv(65536)
        except TimeoutError:
            pass
        with pytest.raises((TimeoutError, socket.timeout)):
            sock.recv(4096)
        state["value"] = 26.0
        resumed = read_lines(sock, 3)
    assert json.loads(first[0])["max_temp_c"] == 25.0
    assert json.loads(resumed[-1])["max_temp_c"] == 26.0


def test_rejects_out_of_range_readings(feed):
    """The consumer rejects these, so they must never be emitted."""
    for bad in (-60.0, 1200.0):
        f = feed(lambda bad=bad: bad)
        with f.connect() as sock:
            time.sleep(0.3)
            sock.settimeout(0.3)
            with pytest.raises((TimeoutError, socket.timeout)):
                sock.recv(4096)
        assert f.server.sent == 0


def test_rejects_nan_and_infinity(feed):
    """NaN and infinity are not legal JSON and must never be emitted."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not temp_feed._valid(bad)


def test_provider_error_is_silence_not_a_crash(feed):
    """A provider that raises means no reading, which means send nothing."""
    def boom():
        raise RuntimeError("acquisition stalled")

    f = feed(boom)
    with f.connect() as sock:
        time.sleep(0.3)
        sock.settimeout(0.3)
        with pytest.raises((TimeoutError, socket.timeout)):
            sock.recv(4096)
    assert f.server.sent == 0
    assert f.server.running


# --- robustness --------------------------------------------------------------


def test_survives_client_disconnect(feed):
    f = feed(lambda: 42.0)
    sock = f.connect()
    read_lines(sock, 2)
    sock.close()
    time.sleep(0.2)
    # A second client must still be served.
    with f.connect() as again:
        assert read_lines(again, 2)
    assert f.server.running


def test_serves_several_clients(feed):
    f = feed(lambda: 55.5)
    with f.connect() as a, f.connect() as b:
        assert json.loads(read_lines(a, 2)[0])["max_temp_c"] == 55.5
        assert json.loads(read_lines(b, 2)[0])["max_temp_c"] == 55.5


def test_ignores_anything_the_client_sends(feed):
    """The protocol is one-way; client bytes are discarded."""
    f = feed(lambda: 12.0)
    with f.connect() as sock:
        sock.sendall(b'{"please": "stop"}\n')
        assert read_lines(sock, 3)
    assert f.server.running


def test_port_in_use_is_reported_not_raised():
    port = free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        second = TempFeedServer(lambda: 1.0, port=port)
        second.start()
        assert second.started.wait(5)
        # Windows refuses the duplicate bind; POSIX with SO_REUSEADDR may not,
        # so accept either a reported error or a working second listener.
        assert second.error is not None or second.running
        second.stop()
    finally:
        blocker.close()


def test_restart_rebinds_the_same_port(feed):
    """SO_REUSEADDR, so restarting does not hit 'address already in use'."""
    f = feed(lambda: 33.0)
    port = f.port
    f.close()
    time.sleep(0.3)
    again = TempFeedServer(lambda: 34.0, port=port)
    again.start()
    try:
        assert again.started.wait(5)
        assert again.error is None, again.error
    finally:
        again.stop()


def test_status_reports_stopped_feed(feed):
    f = feed(lambda: None)
    with f.connect():
        time.sleep(0.3)
        assert "no fresh reading" in f.server.status() or "holding" in f.server.status()
