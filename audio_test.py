"""Tests for microphone selection and muxing helpers."""

from __future__ import annotations

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
