"""Tests for microphone selection and muxing helpers."""

from __future__ import annotations

import os
import subprocess
import wave

import cv2
import numpy as np
import pytest

import audio


# (index, name, channels, samplerate) as list_inputs returns them.
LAPTOP = (0, "Microphone Array (Realtek Audio)", 4, 44100.0)
WEBCAM = (1, "Microphone (UGREEN Camera 2K)", 2, 44100.0)
HEADSET = (2, "Headset (Jabra Elite 85h)", 1, 8000.0)


def test_prefers_a_camera_microphone():
    """The webcam's own mic belongs with the webcam's video."""
    assert audio.preferred_input([LAPTOP, WEBCAM, HEADSET]) == WEBCAM


def test_prefers_camera_regardless_of_order():
    assert audio.preferred_input([WEBCAM, LAPTOP]) == WEBCAM


def test_falls_back_to_first_input():
    assert audio.preferred_input([LAPTOP, HEADSET]) == LAPTOP


def test_no_inputs_gives_none():
    assert audio.preferred_input([]) is None


def test_ffmpeg_lookup_returns_path_or_none():
    exe = audio.ffmpeg_exe()
    assert exe is None or isinstance(exe, str)


def test_mux_reports_missing_input():
    """A failed mux must report, not raise: the WAV is kept instead."""
    err = audio.mux("does_not_exist.mp4", "also_missing.wav", "out.mp4")
    assert err, "expected an error string"
    assert isinstance(err, str)


def test_available_matches_import():
    assert audio.available() is (audio._sd is not None)


# --- mux timing corrections -------------------------------------------------


def _probe_duration(path):
    """Container duration in seconds, via ffmpeg's own report."""
    out = subprocess.run([audio.ffmpeg_exe(), "-i", path], capture_output=True)
    for line in (out.stderr or b"").decode(errors="replace").splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def _make_clip(tmp_path, frames=60, declared_fps=30.0, seconds_of_sound=6.0):
    video = str(tmp_path / "v.mp4")
    sound = str(tmp_path / "a.wav")
    writer = cv2.VideoWriter(
        video, cv2.VideoWriter_fourcc(*"mp4v"), declared_fps, (320, 240)
    )
    for i in range(frames):
        writer.write(np.full((240, 320, 3), (i * 4) % 255, np.uint8))
    writer.release()
    with wave.open(sound, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * int(8000 * seconds_of_sound))
    return video, sound


needs_ffmpeg = pytest.mark.skipif(
    audio.ffmpeg_exe() is None, reason="ffmpeg not available"
)


@needs_ffmpeg
def test_mux_produces_both_streams(tmp_path):
    video, sound = _make_clip(tmp_path)
    out = str(tmp_path / "out.mp4")
    assert audio.mux(video, sound, out) is None
    assert os.path.exists(out)
    assert _probe_duration(out) > 0


@needs_ffmpeg
def test_timescale_retimes_the_video(tmp_path):
    """A writer is told a frame rate before the real one is known.

    Declaring 30 fps for footage really captured at 15 would play the picture
    at double speed against the sound, so the timestamps are rescaled.
    """
    video, sound = _make_clip(tmp_path, frames=60, declared_fps=30.0)
    plain, fixed = str(tmp_path / "plain.mp4"), str(tmp_path / "fixed.mp4")
    assert audio.mux(video, sound, plain) is None
    assert audio.mux(video, sound, fixed, timescale=2.0) is None
    assert _probe_duration(plain) == pytest.approx(2.0, abs=0.15)
    assert _probe_duration(fixed) == pytest.approx(4.0, abs=0.15)


@needs_ffmpeg
def test_audio_skip_trims_the_lead(tmp_path):
    """Sound starts on the button press, video once a frame sized the writer."""
    video, sound = _make_clip(tmp_path, frames=150, declared_fps=30.0,
                              seconds_of_sound=6.0)
    out = str(tmp_path / "out.mp4")
    assert audio.mux(video, sound, out, audio_skip=2.0) is None
    # 6 s of sound minus a 2 s lead leaves 4 s, shorter than the 5 s video.
    assert _probe_duration(out) == pytest.approx(4.0, abs=0.2)
