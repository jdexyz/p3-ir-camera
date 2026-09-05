#!/usr/bin/env python3
"""Temperature feed server for the Sonia ultrasonic press.

Sonia regulates ultrasonic power in a closed loop against the sample's maximum
temperature, and has no way to measure that itself. This module streams the
measurement to it as newline-delimited JSON over TCP.

The one rule that is not obvious: **when there is no fresh reading, send
nothing**. Sonia treats a gap longer than its staleness timeout as a sensor
fault and cuts the ultrasound, and that timeout is the only thing protecting
the sample -- the machine chars wood in seconds. Resending the last value, or
emitting a placeholder, looks like a healthy feed and defeats the guard
entirely, leaving the press driving power against a temperature that is no
longer real. So the provider returns None and nothing goes out.

See TEMP_FEED_PROTOCOL.md in the sonia repository for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable

import json
import socket
import threading
import time


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999
DEFAULT_HZ = 10.0

# The consumer rejects anything outside this, so never emit it either.
MIN_TEMP_C = -50.0
MAX_TEMP_C = 1000.0

# A stalled socket must not hold a client thread forever.
SEND_TIMEOUT_S = 2.0

# Readings are sampled at send time rather than queued: a backlog would deliver
# old temperatures late, which is worse than delivering none.
TempProvider = Callable[[], float | None]


def _valid(value: float | None) -> bool:
    """A reading is sendable only if it is finite and physically plausible."""
    if value is None:
        return False
    value = float(value)
    # NaN and infinity are not legal JSON and must never be emitted.
    if value != value or value in (float("inf"), float("-inf")):
        return False
    return MIN_TEMP_C <= value <= MAX_TEMP_C


class TempFeedServer:
    """Serves the maximum temperature to any client that connects."""

    def __init__(
        self,
        provider: TempProvider,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        hz: float = DEFAULT_HZ,
    ) -> None:
        self.provider = provider
        self.host = host
        self.port = port
        self.period = 1.0 / max(hz, 0.1)

        self.error: str | None = None
        self.clients = 0
        self.sent = 0
        self.skipped = 0
        self.last_sent: float | None = None
        self.last_sent_at: float = 0.0
        self.started = threading.Event()

        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        srv = self._srv
        if srv is not None:
            # Unblocks accept().
            try:
                srv.close()
            except OSError:
                pass

    @property
    def running(self) -> bool:
        return self.started.is_set() and not self._stop.is_set()

    # -- server -----------------------------------------------------------

    def _serve(self) -> None:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Lets a quick restart rebind instead of failing on TIME_WAIT.
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(4)
            srv.settimeout(0.5)
            self._srv = srv
            self.started.set()
        except OSError as e:
            self.error = f"{self.host}:{self.port}: {e}"
            self.started.set()
            return

        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break  # socket closed by stop()
            threading.Thread(
                target=self._handle, args=(conn, addr), daemon=True
            ).start()

        try:
            srv.close()
        except OSError:
            pass

    def _handle(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        with self._lock:
            self.clients += 1
        try:
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(SEND_TIMEOUT_S)
                next_send = time.monotonic()
                while not self._stop.is_set():
                    value = None
                    try:
                        value = self.provider()
                    except Exception:
                        # A provider that raises means no reading, which is
                        # exactly the case where nothing must be sent.
                        value = None

                    if _valid(value):
                        payload = json.dumps({"max_temp_c": float(value)}) + "\n"
                        conn.sendall(payload.encode("utf-8"))
                        with self._lock:
                            self.sent += 1
                            self.last_sent = float(value)
                            self.last_sent_at = time.time()
                    else:
                        with self._lock:
                            self.skipped += 1

                    next_send += self.period
                    delay = next_send - time.monotonic()
                    if delay < 0:
                        # Fell behind: resynchronise rather than trying to
                        # catch up, so readings stay live.
                        next_send = time.monotonic()
                        delay = 0.0
                    if self._stop.wait(delay):
                        break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # a client that goes away is routine, not an error
        finally:
            with self._lock:
                self.clients -= 1

    # -- reporting --------------------------------------------------------

    def status(self) -> str:
        """One line describing the feed, for a status bar."""
        if self.error:
            return f"Temp feed error: {self.error}"
        if not self.started.is_set():
            return "Temp feed starting..."
        where = f"{self.host}:{self.port}"
        if self.clients == 0:
            return f"Temp feed {where} (no client)"
        if self.last_sent is None:
            return f"Temp feed {where}, {self.clients} client(s), holding"
        age = time.time() - self.last_sent_at
        if age > 1.0:
            return (f"Temp feed {where}, {self.clients} client(s), "
                    f"STOPPED ({age:.1f}s - no fresh reading)")
        return (f"Temp feed {where}, {self.clients} client(s), "
                f"{self.last_sent:.1f}°C")
