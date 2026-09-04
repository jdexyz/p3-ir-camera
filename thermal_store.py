#!/usr/bin/env python3
"""Block-compressed storage for the 16-bit thermal stream.

A raw recording is about 150 MB per minute, which is a lot to keep for a long
session. Consecutive thermal frames are nearly identical, so storing each frame
as its difference from the previous one leaves small numbers that compress hard.

Compression normally costs random access, which replay needs for scrubbing.
Frames are therefore grouped into independent blocks with an index, so seeking
decompresses one block rather than the whole recording.

The container is deliberately plain: a header, a run of compressed blocks, an
index, and a footer pointing at the index. Nothing here needs a library beyond
the standard one -- Python 3.14 ships zstd.

Layout::

    magic "P3TZ" | version | rows | cols | block_frames | shift
    block 0 ... block n-1                (zstd-compressed delta frames)
    index: (offset, length, frames) per block
    index offset | frame count | magic "P3TZEND"
"""

from __future__ import annotations

import struct

from compression import zstd
from numpy.typing import NDArray

import numpy as np


MAGIC = b"P3TZ"
FOOTER = b"P3TZEND"
VERSION = 1

# One block per second of capture: small enough that a seek decompresses
# little, large enough that the delta chain still pays for itself.
DEFAULT_BLOCK_FRAMES = 25
DEFAULT_LEVEL = 10

# The P3 leaves the bottom two bits of every sample at zero -- its real quantum
# is 4 counts (0.0625 C), not 1. Dropping them is exactly lossless; the reader
# shifts them back. Raising this trades precision for size.
DEFAULT_SHIFT = 2

_HEADER = struct.Struct("<4sHHHHB")
_INDEX_ENTRY = struct.Struct("<QIH")
_FOOTER = struct.Struct("<QI7s")


def _delta_encode(block: NDArray[np.uint16]) -> bytes:
    """First frame absolute, the rest as differences from the one before."""
    out = np.empty(block.shape, dtype=np.int16)
    out[0] = block[0].astype(np.int16)
    if len(block) > 1:
        out[1:] = (block[1:].astype(np.int32) - block[:-1].astype(np.int32)).astype(
            np.int16
        )
    return out.tobytes()


def _delta_decode(data: bytes, frames: int, rows: int, cols: int) -> NDArray[np.uint16]:
    diff = np.frombuffer(data, dtype="<i2").reshape(frames, rows, cols)
    return np.cumsum(diff.astype(np.int32), axis=0, dtype=np.int32).astype(np.uint16)


class ThermalWriter:
    """Writes frames into a block-compressed file."""

    def __init__(
        self,
        path: str,
        rows: int,
        cols: int,
        block_frames: int = DEFAULT_BLOCK_FRAMES,
        level: int = DEFAULT_LEVEL,
        shift: int = DEFAULT_SHIFT,
    ) -> None:
        self.path = path
        self.rows = rows
        self.cols = cols
        self.block_frames = max(1, block_frames)
        self.level = level
        self.shift = shift
        self.frames = 0

        self._file = open(path, "wb")
        self._file.write(
            _HEADER.pack(MAGIC, VERSION, rows, cols, self.block_frames, shift)
        )
        self._index: list[tuple[int, int, int]] = []
        self._pending: list[NDArray[np.uint16]] = []

    def write(self, frame: NDArray[np.uint16]) -> None:
        self._pending.append(np.asarray(frame, dtype=np.uint16) >> self.shift)
        self.frames += 1
        if len(self._pending) >= self.block_frames:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        block = np.stack(self._pending)
        self._pending = []
        blob = zstd.compress(_delta_encode(block), self.level)
        offset = self._file.tell()
        self._file.write(blob)
        self._index.append((offset, len(blob), len(block)))

    def close(self) -> None:
        if self._file.closed:
            return
        self._flush()
        index_offset = self._file.tell()
        for entry in self._index:
            self._file.write(_INDEX_ENTRY.pack(*entry))
        self._file.write(_FOOTER.pack(index_offset, self.frames, FOOTER))
        self._file.close()

    def __enter__(self) -> ThermalWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ThermalReader:
    """Random access to a block-compressed thermal file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "rb")
        try:
            self._read_directory()
        except Exception:
            # Rejecting a file must not leave its handle open behind us.
            self._file.close()
            raise
        self._cache_block = -1
        self._cache: NDArray[np.uint16] | None = None

    def _read_directory(self) -> None:
        header = self._file.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError(f"{self.path} is too short to be a P3TZ file")
        magic, version, self.rows, self.cols, self.block_frames, self.shift = (
            _HEADER.unpack(header)
        )
        if magic != MAGIC:
            raise ValueError(f"{self.path} is not a P3TZ file")
        if version != VERSION:
            raise ValueError(f"unsupported P3TZ version {version}")

        self._file.seek(-_FOOTER.size, 2)
        index_offset, self.frames, footer = _FOOTER.unpack(self._file.read(_FOOTER.size))
        if footer != FOOTER:
            raise ValueError(f"{self.path} is truncated or not finalized")

        self._file.seek(index_offset)
        self._index: list[tuple[int, int, int]] = []
        start = 0
        self._starts: list[int] = []
        while True:
            chunk = self._file.read(_INDEX_ENTRY.size)
            if len(chunk) < _INDEX_ENTRY.size:
                break
            offset, length, frames = _INDEX_ENTRY.unpack(chunk)
            if offset >= index_offset:
                break
            self._index.append((offset, length, frames))
            self._starts.append(start)
            start += frames

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.frames, self.rows, self.cols)

    def __len__(self) -> int:
        return self.frames

    def _block_for(self, index: int) -> int:
        # Blocks are uniform except possibly the last, so this is arithmetic
        # rather than a search.
        return min(index // self.block_frames, len(self._index) - 1)

    def __getitem__(self, index: int) -> NDArray[np.uint16]:
        if index < 0:
            index += self.frames
        if not 0 <= index < self.frames:
            raise IndexError(index)
        block = self._block_for(index)
        if block != self._cache_block:
            offset, length, frames = self._index[block]
            self._file.seek(offset)
            data = zstd.decompress(self._file.read(length))
            self._cache = _delta_decode(data, frames, self.rows, self.cols)
            self._cache_block = block
        assert self._cache is not None
        return (self._cache[index - self._starts[block]] << self.shift).astype(
            np.uint16
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> ThermalReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RawWriter:
    """Plain uncompressed writer, with the same interface as ThermalWriter.

    Keeps the two recording paths identical at the call site, so choosing a
    format is a constructor choice rather than a branch in the capture loop.
    """

    def __init__(self, path: str, rows: int, cols: int) -> None:
        self.path = path
        self.rows = rows
        self.cols = cols
        self.shift = 0
        self.frames = 0
        self._file = open(path, "wb")

    def write(self, frame: NDArray[np.uint16]) -> None:
        self._file.write(np.ascontiguousarray(frame, dtype="<u2").tobytes())
        self.frames += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> RawWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_frames(path: str, shape: tuple[int, int, int] | None = None):
    """Open either a compressed or a plain raw thermal file.

    Older recordings are plain `.raw`, so both keep working; the plain form is
    memory-mapped rather than read, which is what made it worth keeping.
    """
    if path.endswith(".raw"):
        if shape is None:
            raise ValueError("a raw file needs its shape from the sidecar")
        return np.memmap(path, dtype="<u2", mode="r").reshape(shape)
    return ThermalReader(path)
