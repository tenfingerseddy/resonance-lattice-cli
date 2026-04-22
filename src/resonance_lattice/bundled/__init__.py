# SPDX-License-Identifier: BUSL-1.1
"""Bundled source-blob packer/reader.

Packs raw source files into the `.rlat` knowledge model as individually
zstd-compressed frames, indexed by posix-normalised relative path. Used
by ``BundledStore`` (lossless mode with embedded bytes) to serve whole
files at query time without touching the filesystem.

Individual frames (not one big zstd stream) let us decompress a single
file on demand without paying for the rest of the corpus — O(1) random
access against the blob section.

Format (written into the knowledge model's source_store slot when
``store_mode='bundled'``):

    [4 bytes]  b"RLBD"                    # magic
    [4 bytes]  uint32 version             # bundled-format version
    [8 bytes]  uint64 meta_size           # embedded metadata SQLite size
    [8 bytes]  uint64 index_size          # index JSON size
    [8 bytes]  uint64 blob_size           # concatenated frames total size
    [meta_size bytes]    metadata SQLite  # encoder config, source manifest, etc.
    [index_size bytes]   index JSON       # {rel_path: {offset, length, sha256}}
    [blob_size bytes]    blob payload     # concatenated zstd frames

Offsets inside ``index`` are relative to the start of the blob payload.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Mapping
from pathlib import PurePosixPath

MAGIC = b"RLBD"
BUNDLED_VERSION = 1
HEADER_SIZE = 4 + 4 + 8 + 8 + 8  # magic + version + 3 * uint64

# zstd level 19: heavy compression, good for one-shot pack at build time.
# Markdown / source text typically compresses 40-60 % at this level.
ZSTD_LEVEL = 19


def _posix(rel_path: str) -> str:
    """Normalise a relative path to posix-style forward slashes."""
    return PurePosixPath(rel_path.replace("\\", "/")).as_posix()


def is_bundled_payload(data: bytes) -> bool:
    """Sniff: does ``data`` start with the bundled-blob magic?"""
    return len(data) >= 4 and data[:4] == MAGIC


def pack(
    files: Mapping[str, bytes],
    meta_sqlite: bytes = b"",
) -> bytes:
    """Pack ``files`` (rel_path -> raw bytes) into a bundled-blob payload.

    Args:
        files: Mapping of relative path (normalised to posix) to raw file
            bytes. Caller is responsible for collecting these — typically
            by walking ``source_root`` and reading every file referenced
            from the knowledge model manifest.
        meta_sqlite: Embedded metadata SQLite bytes (encoder config,
            source manifest, retrieval config, profile) — the same
            bytes that external mode stores in the source_store slot.

    Returns:
        Full bundled payload ready to drop into the knowledge model's
        source_store section.
    """
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)

    index: dict[str, dict] = {}
    blob_buf = io.BytesIO()
    offset = 0
    for raw_path, data in files.items():
        rel = _posix(raw_path)
        frame = cctx.compress(data)
        sha = hashlib.sha256(data).hexdigest()
        index[rel] = {
            "offset": offset,
            "length": len(frame),
            "sha256": sha,
            "raw_size": len(data),
        }
        blob_buf.write(frame)
        offset += len(frame)

    index_bytes = json.dumps(index, sort_keys=True).encode("utf-8")
    blob_bytes = blob_buf.getvalue()

    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<I", BUNDLED_VERSION))
    out.write(struct.pack("<Q", len(meta_sqlite)))
    out.write(struct.pack("<Q", len(index_bytes)))
    out.write(struct.pack("<Q", len(blob_bytes)))
    out.write(meta_sqlite)
    out.write(index_bytes)
    out.write(blob_bytes)
    return out.getvalue()


def unpack_header(data: bytes) -> tuple[int, int, int, int]:
    """Return ``(version, meta_size, index_size, blob_size)`` from ``data``.

    Raises ValueError if the magic is wrong or the payload is truncated.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"Bundled payload too short: {len(data)} < {HEADER_SIZE}"
        )
    if data[:4] != MAGIC:
        raise ValueError(
            f"Invalid bundled magic: {data[:4]!r} (expected {MAGIC!r})"
        )
    version = struct.unpack_from("<I", data, 4)[0]
    meta_size = struct.unpack_from("<Q", data, 8)[0]
    index_size = struct.unpack_from("<Q", data, 16)[0]
    blob_size = struct.unpack_from("<Q", data, 24)[0]
    return version, meta_size, index_size, blob_size


def split(data: bytes) -> tuple[bytes, dict, bytes]:
    """Split a bundled payload into ``(meta_sqlite, index, blob_bytes)``."""
    version, meta_size, index_size, blob_size = unpack_header(data)
    if version != BUNDLED_VERSION:
        raise ValueError(
            f"Unsupported bundled version: {version} (this build expects {BUNDLED_VERSION})"
        )
    cursor = HEADER_SIZE
    meta = data[cursor:cursor + meta_size]
    cursor += meta_size
    index_bytes = data[cursor:cursor + index_size]
    cursor += index_size
    blob = data[cursor:cursor + blob_size]
    if len(meta) != meta_size or len(index_bytes) != index_size or len(blob) != blob_size:
        raise ValueError(
            "Bundled payload truncated: section sizes do not match "
            "declared lengths in the header."
        )
    index = json.loads(index_bytes.decode("utf-8")) if index_bytes else {}
    return meta, index, blob


class BlobReader:
    """Random-access reader for a bundled blob payload.

    Holds the compressed bytes in memory (they are already the smallest
    representation — decompressing on demand is cheap) and decodes single
    files on request. Not thread-safe; wrap in a lock if you share one
    reader across threads.
    """

    def __init__(self, index: dict, blob_bytes: bytes) -> None:
        self._index = index
        self._blob = blob_bytes
        self._cache: dict[str, bytes] = {}

    @classmethod
    def from_payload(cls, data: bytes) -> BlobReader:
        _meta, index, blob = split(data)
        return cls(index=index, blob_bytes=blob)

    def __contains__(self, rel_path: str) -> bool:
        return _posix(rel_path) in self._index

    @property
    def paths(self) -> list[str]:
        return sorted(self._index.keys())

    def entry(self, rel_path: str) -> dict | None:
        return self._index.get(_posix(rel_path))

    def read_bytes(self, rel_path: str) -> bytes:
        """Return raw (decompressed) bytes for a single file."""
        rel = _posix(rel_path)
        if rel in self._cache:
            return self._cache[rel]
        entry = self._index.get(rel)
        if entry is None:
            raise KeyError(f"Bundled store has no file at {rel!r}")
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        frame = self._blob[entry["offset"]:entry["offset"] + entry["length"]]
        raw = dctx.decompress(frame)
        self._cache[rel] = raw
        return raw
