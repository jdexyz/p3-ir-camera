#!/usr/bin/env python3
"""Frame overlays shared by the thermal and visible recorders.

Kept separate from both so the two writers stamp frames identically, and so
uvc.py does not have to import the whole viewer to draw text.
"""

from __future__ import annotations

import time

from numpy.typing import NDArray

import cv2
import numpy as np


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Font size is derived from frame height so the stamp stays the same physical
# size whether it lands on a 192-row thermal frame or a 1440-row webcam frame.
_REFERENCE_HEIGHT = 760.0
_MIN_SCALE = 0.4
_MAX_SCALE = 1.6

_WHITE = (255, 255, 255)

# Opacity of the plate drawn behind the text.
_PLATE_ALPHA = 0.55


def format_timestamp(when: float | None = None, fmt: str = TIMESTAMP_FORMAT) -> str:
    """Local-time stamp for a capture time (defaults to now)."""
    return time.strftime(fmt, time.localtime(time.time() if when is None else when))


def draw_timestamp(
    img: NDArray[np.uint8],
    when: float | None = None,
    fmt: str = TIMESTAMP_FORMAT,
    margin: int = 10,
) -> NDArray[np.uint8]:
    """Burn a date and time into the top-right of a BGR frame, in place.

    Top-right is the only corner the viewer leaves free: the spot/range readout
    is top-left, the settings line bottom-left, and the colorbar runs down the
    right edge below it.

    The text is drawn with a dark outline because a thermal image can be white
    at the top of its scale, where plain white text would vanish.
    """
    text = format_timestamp(when, fmt)
    h, w = img.shape[:2]
    scale = float(np.clip(h / _REFERENCE_HEIGHT, _MIN_SCALE, _MAX_SCALE))
    thickness = max(1, int(round(scale * 1.5)))

    (tw, th), _baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(margin, w - tw - margin)
    y = margin + th

    # A darkened plate rather than an outlined glyph: OpenCV 5 clamps putText
    # stroke thickness at 2, so an outline drawn by over-stroking is silently
    # covered by the text pass and the stamp vanishes on a white-hot frame.
    pad = max(3, int(round(scale * 5)))
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(w, x + tw + pad), min(h, y + pad)
    plate = img[y0:y1, x0:x1]
    if plate.size:
        cv2.addWeighted(plate, 1.0 - _PLATE_ALPHA, np.zeros_like(plate),
                        _PLATE_ALPHA, 0, plate)

    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _WHITE, thickness,
        cv2.LINE_AA,
    )
    return img
