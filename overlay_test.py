"""Tests for the burned-in timestamp overlay."""

from __future__ import annotations

import time

import numpy as np

from overlay import TIMESTAMP_FORMAT, draw_timestamp, format_timestamp


def test_format_uses_capture_time():
    when = time.mktime((2026, 9, 3, 14, 5, 30, 0, 0, -1))
    assert format_timestamp(when) == "2026-09-03 14:05:30"


def test_format_has_date_and_time():
    text = format_timestamp()
    assert len(text) == len("YYYY-MM-DD HH:MM:SS")
    assert text.count("-") == 2 and text.count(":") == 2


def test_draws_into_frame():
    img = np.zeros((760, 1024, 3), dtype=np.uint8)
    draw_timestamp(img, 0.0)
    assert img.any(), "nothing was drawn"


def test_draws_in_top_right():
    img = np.zeros((760, 1024, 3), dtype=np.uint8)
    draw_timestamp(img)
    top, bottom = img[:60], img[60:]
    assert top.any(), "stamp should be at the top"
    assert not bottom.any(), "stamp should not extend down the frame"
    # Right half only, so it never collides with the top-left readout.
    assert not img[:60, :400].any()


def test_scales_with_frame_height():
    """A 1440-row webcam frame and a 768-row thermal frame both stay legible."""
    heights = {}
    for h, w in ((768, 1024), (1440, 2560)):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        draw_timestamp(img)
        rows = np.where(img.any(axis=(1, 2)))[0]
        heights[h] = rows.max() - rows.min()
    assert heights[1440] > heights[768], "text should grow with the frame"


def test_survives_small_frames():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    draw_timestamp(img)
    assert img.any()


def test_readable_on_white():
    """White text alone would vanish on a saturated thermal frame.

    OpenCV 5 clamps putText thickness at 2, so legibility comes from a
    darkened plate behind the text rather than an outlined glyph.
    """
    img = np.full((760, 1024, 3), 255, dtype=np.uint8)
    draw_timestamp(img)
    darkened = (img.min(axis=2) < 200).sum()
    assert darkened > 500, f"expected a dark plate behind the text, got {darkened}"


def test_readable_on_black():
    img = np.zeros((760, 1024, 3), dtype=np.uint8)
    draw_timestamp(img)
    assert (img.max(axis=2) > 200).sum() > 200, "text should be bright on a dark frame"


def test_custom_format():
    when = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    assert format_timestamp(when, "%H:%M") == "03:04"
    assert TIMESTAMP_FORMAT.startswith("%Y")
