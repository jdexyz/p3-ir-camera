#!/usr/bin/env python3
"""Microphone capture, for recording sound alongside the visible camera.

OpenCV's VideoWriter is video-only, so audio is captured separately to a WAV
and muxed into the visible mp4 when the recording stops.

sounddevice is optional: if it is unavailable the rest of the application still
works and audio is simply disabled, since a missing microphone must not stop a
thermal capture.
"""

from __future__ import annotations

import subprocess
import threading
import time
import wave


try:  # optional dependency
    import sounddevice as _sd
except Exception:  # pragma: no cover - depends on the host
    _sd = None

try:  # ships a static ffmpeg build; system ffmpeg is used if this is absent
    import imageio_ffmpeg as _imageio_ffmpeg
except Exception:  # pragma: no cover
    _imageio_ffmpeg = None

SAMPLE_WIDTH = 2  # int16
DEFAULT_SAMPLERATE = 44100

# Names that suggest a microphone built into a capture device rather than the
# machine's own array, so a webcam's mic is preferred over the laptop's.
_CAMERA_HINTS = ("camera", "webcam", "cam ")


def available() -> bool:
    """True when audio capture is possible on this machine."""
    return _sd is not None


def ffmpeg_exe() -> str | None:
    """Path to an ffmpeg binary, or None if there is none."""
    if _imageio_ffmpeg is not None:
        try:
            return _imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    from shutil import which

    return which("ffmpeg")


def list_inputs() -> list[tuple[int, str, int, float]]:
    """Input devices as (index, name, channels, default samplerate).

    Windows exposes the same microphone once per host API, so entries are
    deduplicated by name; the first occurrence is kept.
    """
    if _sd is None:
        return []
    seen: set[str] = set()
    out: list[tuple[int, str, int, float]] = []
    try:
        devices = _sd.query_devices()
    except Exception:
        return []
    for index, dev in enumerate(devices):
        channels = int(dev.get("max_input_channels", 0))
        name = str(dev.get("name", "")).strip()
        if channels < 1 or not name or name in seen:
            continue
        # Not a real capture device, just the host API's router.
        if "sound mapper" in name.lower() or "primary sound" in name.lower():
            continue
        seen.add(name)
        out.append((index, name, channels,
                    float(dev.get("default_samplerate", DEFAULT_SAMPLERATE))))
    return out


def preferred_input(
    inputs: list[tuple[int, str, int, float]],
) -> tuple[int, str, int, float] | None:
    """Pick the microphone most likely to belong to the visible camera."""
    for entry in inputs:
        if any(hint in entry[1].lower() for hint in _CAMERA_HINTS):
            return entry
    return inputs[0] if inputs else None


class AudioRecorder(threading.Thread):
    """Captures a microphone to a WAV file for the life of one recording."""

    def __init__(
        self,
        path: str,
        device: int | None = None,
        channels: int = 2,
        samplerate: float = DEFAULT_SAMPLERATE,
    ) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.device = device
        self.channels = max(1, min(2, channels))
        self.samplerate = int(samplerate)
        self.error: str | None = None
        self.frames = 0
        self.started_at = 0.0
        self._stop = threading.Event()

    def stop(self) -> dict[str, object]:
        """Stop capture and return metadata for the sidecar."""
        self._stop.set()
        self.join(timeout=5.0)
        if self.error or not self.frames:
            return {}
        duration = self.frames / float(self.samplerate)
        return {
            "file": self.path,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)
            ),
            "duration_s": round(duration, 3),
            "samplerate": self.samplerate,
            "channels": self.channels,
        }

    def run(self) -> None:
        if _sd is None:
            self.error = "sounddevice is not installed"
            return
        try:
            with wave.open(self.path, "wb") as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(SAMPLE_WIDTH)
                wav.setframerate(self.samplerate)
                self.started_at = time.time()
                with _sd.InputStream(
                    device=self.device,
                    channels=self.channels,
                    samplerate=self.samplerate,
                    dtype="int16",
                    blocksize=1024,
                ) as stream:
                    while not self._stop.is_set():
                        data, overflowed = stream.read(1024)
                        del overflowed  # a dropped block is not worth failing on
                        wav.writeframes(data.tobytes())
                        self.frames += len(data)
        except Exception as e:
            self.error = str(e)


def mux(
    video: str,
    audio: str,
    output: str,
    timeout: float = 120.0,
    timescale: float = 1.0,
    audio_skip: float = 0.0,
) -> str | None:
    """Combine a video file and a WAV into one mp4, correcting drift.

    Two corrections keep sound aligned with picture:

    * `timescale` rescales the video's timestamps. A VideoWriter has to be told
      a frame rate before the first frame exists, so the container's declared
      rate is a guess; the real rate is only known once capture ends. Left
      uncorrected the picture runs fast or slow against the sound, and the
      error accumulates over the whole recording.
    * `audio_skip` trims the head of the WAV. Audio capture starts as soon as
      recording is pressed, but the video writer cannot open until a frame has
      arrived to size it, so the sound leads the picture by that gap.

    The video stream is copied rather than re-encoded, so this costs little and
    cannot degrade the recording.

    Returns:
        An error string, or None on success.
    """
    exe = ffmpeg_exe()
    if exe is None:
        return "no ffmpeg available"
    cmd = [exe, "-y", "-loglevel", "error"]
    if abs(timescale - 1.0) > 1e-6:
        cmd += ["-itsscale", f"{timescale:.6f}"]
    cmd += ["-i", video]
    if audio_skip > 0.001:
        cmd += ["-ss", f"{audio_skip:.3f}"]
    cmd += [
        "-i", audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output,
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except Exception as e:
        return str(e)
    if done.returncode != 0:
        return (done.stderr or b"").decode(errors="replace").strip()[:300] or "ffmpeg failed"
    return None
