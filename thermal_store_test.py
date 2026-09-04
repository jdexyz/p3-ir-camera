"""Tests for block-compressed thermal storage."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_store import (
    DEFAULT_SHIFT,
    ThermalReader,
    ThermalWriter,
    open_frames,
)


ROWS, COLS = 192, 256


def make_frames(count=60, seed=0):
    """Frames resembling the sensor: a slow drift plus noise, low bits unused."""
    rng = np.random.default_rng(seed)
    base = 19000 + np.arange(count)[:, None, None] * 3
    noise = rng.integers(0, 64, size=(count, ROWS, COLS))
    frames = (base + noise).astype(np.uint16)
    # The P3 never sets the bottom two bits.
    return (frames >> DEFAULT_SHIFT << DEFAULT_SHIFT).astype(np.uint16)


def write(tmp_path, frames, **kw):
    path = str(tmp_path / "t.tz")
    with ThermalWriter(path, ROWS, COLS, **kw) as w:
        for f in frames:
            w.write(f)
    return path


def test_roundtrip_is_exact(tmp_path):
    frames = make_frames(60)
    with ThermalReader(write(tmp_path, frames)) as r:
        assert len(r) == 60
        assert r.shape == (60, ROWS, COLS)
        for i in range(60):
            assert np.array_equal(r[i], frames[i]), f"frame {i} differs"


def test_random_access_matches_sequential(tmp_path):
    """Scrubbing must give the same frames as playing through."""
    frames = make_frames(80)
    with ThermalReader(write(tmp_path, frames, block_frames=10)) as r:
        for i in (73, 0, 41, 12, 79, 5, 41):
            assert np.array_equal(r[i], frames[i]), f"frame {i} differs"


def test_negative_index(tmp_path):
    frames = make_frames(30)
    with ThermalReader(write(tmp_path, frames)) as r:
        assert np.array_equal(r[-1], frames[-1])


def test_out_of_range(tmp_path):
    with ThermalReader(write(tmp_path, make_frames(10))) as r,             pytest.raises(IndexError):
        _ = r[10]


def test_partial_final_block(tmp_path):
    """A recording rarely stops on a block boundary."""
    frames = make_frames(23)
    with ThermalReader(write(tmp_path, frames, block_frames=10)) as r:
        assert len(r) == 23
        assert np.array_equal(r[22], frames[22])


def test_single_frame(tmp_path):
    frames = make_frames(1)
    with ThermalReader(write(tmp_path, frames)) as r:
        assert len(r) == 1
        assert np.array_equal(r[0], frames[0])


def test_compresses_substantially(tmp_path):
    import os

    frames = make_frames(75)
    path = write(tmp_path, frames)
    ratio = frames.nbytes / os.path.getsize(path)
    assert ratio > 2.0, f"expected real compression, got {ratio:.2f}x"


def test_shift_is_lossless_for_this_sensor(tmp_path):
    """The bottom two bits are never set, so dropping them loses nothing."""
    frames = make_frames(20)
    assert not np.any(frames & 0b11), "fixture should mimic the sensor"
    with ThermalReader(write(tmp_path, frames)) as r:
        assert np.array_equal(r[7], frames[7])


def test_larger_shift_is_bounded_loss(tmp_path):
    """A bigger shift trades precision for size, by a known amount."""
    frames = make_frames(20)
    with ThermalReader(write(tmp_path, frames, shift=4)) as r:
        err = np.abs(r[5].astype(np.int32) - frames[5].astype(np.int32)).max()
        assert err <= 0b1111, "loss must stay within the discarded bits"


def test_open_frames_reads_raw(tmp_path):
    """Existing .raw recordings must keep working."""
    frames = make_frames(12)
    path = str(tmp_path / "thermal.raw")
    frames.tofile(path)
    data = open_frames(path, shape=(12, ROWS, COLS))
    assert np.array_equal(data[3], frames[3])


def test_open_frames_reads_compressed(tmp_path):
    frames = make_frames(12)
    with open_frames(write(tmp_path, frames)) as data:
        assert np.array_equal(data[3], frames[3])


def test_rejects_foreign_file(tmp_path):
    path = str(tmp_path / "bad.tz")
    with open(path, "wb") as f:
        f.write(b"not a thermal file at all" * 8)
    with pytest.raises(ValueError):
        ThermalReader(path)


def test_rejects_unfinalized_file(tmp_path):
    """A recording cut short mid-write must fail loudly, not read as garbage."""
    frames = make_frames(30)
    path = str(tmp_path / "t.tz")
    writer = ThermalWriter(path, ROWS, COLS)
    for f in frames:
        writer.write(f)
    writer._flush()
    writer._file.close()  # no index, no footer
    with pytest.raises(ValueError):
        ThermalReader(path)
