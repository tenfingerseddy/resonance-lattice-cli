# SPDX-License-Identifier: BUSL-1.1
"""Binary .rlat file format serialisation.

Implements the file format from Spec Section 10. The .rlat file contains:
  - 64-byte header with metadata
  - Field tensor block (dense, factored, or PQ)
  - Phase registry (LSH tables + source pointers)
  - Source store (multi-resolution content)

Supports optional zstd or lz4 compression on the field tensor and registry blocks.

Layout:
    [HEADER 64 bytes]
    [FIELD TENSOR BLOCK — variable size]
    [PHASE REGISTRY — variable size]
    [SOURCE STORE — embedded SQLite or serialised content]
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
from pathlib import Path


@contextlib.contextmanager
def _atomic_open_wb(path):
    """Open ``path`` for binary writing via a ``.tmp`` sibling + atomic rename.

    On Windows with OneDrive / Dropbox / iCloud syncing the target directory,
    the sync client can grab an exclusive lock on a file the moment it sees
    a change, racing the ``rlat save`` pipeline and crashing it with
    ``PermissionError``. Writing to ``<path>.tmp`` and then ``os.replace``-ing
    over ``<path>`` collapses the lock window to a single atomic rename, which
    never collides with a sync read. Drop-in replacement for
    ``open(path, "wb")``.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        f = open(tmp, "wb")
        try:
            yield f
        finally:
            f.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.config import (
    Compression,
    FieldType,
    LatticeConfig,
    Precision,
)

# Magic bytes
MAGIC = b"RLAT"
FORMAT_VERSION = 3
# v1: original layout (implicit embedded store). Still readable; treated as store_mode="embedded".
# v2: adds explicit store_mode byte at header offset 0x39 (embedded | external).
# v3: extends store_mode enum with bundled mode (raw source files packed into
#     the source_store slot via the bundled-blob format, see resonance_lattice.bundled).
#     Remote mode is reserved as value 2 for Phase 4 of feat/three-mode-store.
MIN_SUPPORTED_VERSION = 1

# Field type codes
FIELD_TYPE_MAP = {
    FieldType.DENSE: 0,
    FieldType.FACTORED: 1,
    FieldType.PQ: 2,
    FieldType.ASYMMETRIC_DENSE: 3,
    FieldType.MULTI_VECTOR: 4,
}
FIELD_TYPE_REVERSE = {v: k for k, v in FIELD_TYPE_MAP.items()}

# Precision codes
PRECISION_MAP = {Precision.F16: 0, Precision.BF16: 1, Precision.F32: 2}
PRECISION_REVERSE = {v: k for k, v in PRECISION_MAP.items()}
PRECISION_DTYPE = {Precision.F16: np.float16, Precision.BF16: np.float16, Precision.F32: np.float32}

# Compression codes
COMPRESSION_MAP = {Compression.NONE: 0, Compression.ZSTD: 1, Compression.LZ4: 2}
COMPRESSION_REVERSE = {v: k for k, v in COMPRESSION_MAP.items()}

# Store mode codes — introduced in format v2, extended in v3.
#
# Canonical names (three-mode-store v1.0.0-era): bundled / local / remote.
# Historical name "external" is kept as an alias for "local" so older
# cartridges and in-flight branches keep loading. "embedded" stays as
# the name for the deprecated legacy SQLite store (slated for v2.0.0
# removal) — it is distinct from "bundled" (which is lossless).
STORE_MODE_EMBEDDED = 0  # legacy: pre-chunked SQLite SourceStore (deprecated v2.0.0)
STORE_MODE_EXTERNAL = 1  # lossless + local disk (canonical name: "local")
STORE_MODE_LOCAL = STORE_MODE_EXTERNAL  # alias, same wire value
STORE_MODE_REMOTE = 2    # lossless + HTTP origin with SHA-pinned cache
STORE_MODE_BUNDLED = 3   # v3+: lossless with raw source files packed inside the cartridge
STORE_MODE_VALUES = {
    "embedded": STORE_MODE_EMBEDDED,
    "external": STORE_MODE_EXTERNAL,
    "local": STORE_MODE_LOCAL,     # canonical alias for "external"
    "remote": STORE_MODE_REMOTE,
    "bundled": STORE_MODE_BUNDLED,
}
# When turning a wire value back into a name, prefer the historical
# "external" spelling for now so existing tools / tests don't break.
# Phase 6+1 of the rename will flip this to "local" once the doc/CLI
# surfaces all move over.
STORE_MODE_NAMES = {
    STORE_MODE_EMBEDDED: "embedded",
    STORE_MODE_EXTERNAL: "external",
    STORE_MODE_REMOTE: "remote",
    STORE_MODE_BUNDLED: "bundled",
}


def _compress(data: bytes, compression: Compression) -> bytes:
    """Compress data with the specified algorithm."""
    if compression == Compression.NONE:
        return data
    elif compression == Compression.ZSTD:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data)
    elif compression == Compression.LZ4:
        import lz4.frame
        return lz4.frame.compress(data)
    return data


def _decompress(data: bytes, compression: Compression) -> bytes:
    """Decompress data with the specified algorithm."""
    if compression == Compression.NONE:
        return data
    elif compression == Compression.ZSTD:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    elif compression == Compression.LZ4:
        import lz4.frame
        return lz4.frame.decompress(data)
    return data


class RlatHeader:
    """64-byte .rlat file header."""

    STRUCT_FORMAT = "<4sHBHQBBBBHH"  # up to reserved
    # Magic(4) + Version(2) + Bands(1) + Dim(2) + SourceCount(8) +
    # FieldType(1) + Precision(1) + Compression(1) + PQSubspaces(1) +
    # PQCodebookSize(2) + SVDRank(2)
    # Then 3 uint64 offsets + 1 uint64 ANN index offset + padding

    SIZE = 64

    def __init__(
        self,
        bands: int = 5,
        dim: int = 2048,
        source_count: int = 0,
        field_type: FieldType = FieldType.DENSE,
        precision: Precision = Precision.F32,
        compression: Compression = Compression.NONE,
        pq_subspaces: int = 8,
        pq_codebook_size: int = 1024,
        svd_rank: int = 512,
        field_tensor_offset: int = 0,
        registry_offset: int = 0,
        source_store_offset: int = 0,
        ann_index_offset: int = 0,
        store_mode: str = "embedded",
        version: int = FORMAT_VERSION,
    ) -> None:
        if store_mode not in STORE_MODE_VALUES:
            raise ValueError(
                f"Invalid store_mode: {store_mode!r}. Expected one of {list(STORE_MODE_VALUES)}"
            )
        self.bands = bands
        self.dim = dim
        self.source_count = source_count
        self.field_type = field_type
        self.precision = precision
        self.compression = compression
        self.pq_subspaces = pq_subspaces
        self.pq_codebook_size = pq_codebook_size
        self.svd_rank = svd_rank
        self.field_tensor_offset = field_tensor_offset
        self.registry_offset = registry_offset
        self.source_store_offset = source_store_offset
        self.ann_index_offset = ann_index_offset
        self.store_mode = store_mode
        # `version` tracks which format version this header was loaded from.
        # Always written as FORMAT_VERSION (current) in to_bytes; back-compat
        # loading of older cartridges records their original version here.
        self.version = version

    def to_bytes(self) -> bytes:
        """Serialise header to 64 bytes."""
        buf = io.BytesIO()

        # [0x00] Magic
        buf.write(MAGIC)
        # [0x04] Version
        buf.write(struct.pack("<H", FORMAT_VERSION))
        # [0x06] Bands
        buf.write(struct.pack("<B", self.bands))
        # [0x07] Dim
        buf.write(struct.pack("<H", self.dim))
        # [0x09] Source count
        buf.write(struct.pack("<Q", self.source_count))
        # [0x11] Field type
        buf.write(struct.pack("<B", FIELD_TYPE_MAP[self.field_type]))
        # [0x12] Precision
        buf.write(struct.pack("<B", PRECISION_MAP[self.precision]))
        # [0x13] Compression
        buf.write(struct.pack("<B", COMPRESSION_MAP[self.compression]))
        # [0x14] PQ subspaces
        buf.write(struct.pack("<B", self.pq_subspaces))
        # [0x15] PQ codebook size
        buf.write(struct.pack("<H", self.pq_codebook_size))
        # [0x17] SVD rank
        buf.write(struct.pack("<H", self.svd_rank))
        # [0x19] Field tensor offset
        buf.write(struct.pack("<Q", self.field_tensor_offset))
        # [0x21] Registry offset
        buf.write(struct.pack("<Q", self.registry_offset))
        # [0x29] Source store offset
        buf.write(struct.pack("<Q", self.source_store_offset))
        # [0x31] ANN index offset
        buf.write(struct.pack("<Q", self.ann_index_offset))
        # [0x39] Store mode — added in format v2. v1 cartridges have \x00
        # (padding) here, which decodes as "embedded" via back-compat.
        buf.write(struct.pack("<B", STORE_MODE_VALUES[self.store_mode]))

        # Pad to 64 bytes
        data = buf.getvalue()
        padding = self.SIZE - len(data)
        if padding > 0:
            data += b"\x00" * padding

        return data[:self.SIZE]

    @classmethod
    def from_bytes(cls, data: bytes) -> RlatHeader:
        """Parse header from 64 bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Header too short: {len(data)} < {cls.SIZE}")

        magic = data[0:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r} (expected {MAGIC!r})")

        version = struct.unpack_from("<H", data, 4)[0]
        # Accept any format version from MIN_SUPPORTED_VERSION up to the current
        # FORMAT_VERSION. Fields introduced in later versions have safe defaults
        # when reading an older cartridge (e.g. v1 → store_mode="embedded").
        if version < MIN_SUPPORTED_VERSION or version > FORMAT_VERSION:
            raise ValueError(
                f"Unsupported version: {version} "
                f"(supported range: {MIN_SUPPORTED_VERSION}-{FORMAT_VERSION})"
            )

        bands = struct.unpack_from("<B", data, 6)[0]
        dim = struct.unpack_from("<H", data, 7)[0]
        source_count = struct.unpack_from("<Q", data, 9)[0]
        field_type_code = struct.unpack_from("<B", data, 17)[0]
        precision_code = struct.unpack_from("<B", data, 18)[0]
        compression_code = struct.unpack_from("<B", data, 19)[0]
        pq_subspaces = struct.unpack_from("<B", data, 20)[0]
        pq_codebook_size = struct.unpack_from("<H", data, 21)[0]
        svd_rank = struct.unpack_from("<H", data, 23)[0]
        field_tensor_offset = struct.unpack_from("<Q", data, 25)[0]
        registry_offset = struct.unpack_from("<Q", data, 33)[0]
        source_store_offset = struct.unpack_from("<Q", data, 41)[0]
        ann_index_offset = struct.unpack_from("<Q", data, 49)[0]

        # v2+: explicit store_mode byte at 0x39.
        # v1: that byte is padding (always \x00) which naturally decodes to
        # STORE_MODE_EMBEDDED — correct, since v1 only supported embedded stores.
        if version >= 2:
            store_mode_code = struct.unpack_from("<B", data, 57)[0]
            store_mode = STORE_MODE_NAMES.get(store_mode_code, "embedded")
        else:
            store_mode = "embedded"

        return cls(
            bands=bands,
            dim=dim,
            source_count=source_count,
            field_type=FIELD_TYPE_REVERSE[field_type_code],
            precision=PRECISION_REVERSE[precision_code],
            compression=COMPRESSION_REVERSE[compression_code],
            pq_subspaces=pq_subspaces,
            pq_codebook_size=pq_codebook_size,
            svd_rank=svd_rank,
            field_tensor_offset=field_tensor_offset,
            registry_offset=registry_offset,
            source_store_offset=source_store_offset,
            ann_index_offset=ann_index_offset,
            store_mode=store_mode,
            version=version,
        )


def save_dense_field(
    path: str | Path,
    field_tensor: NDArray,
    config: LatticeConfig,
    source_count: int = 0,
    registry_data: bytes = b"",
    store_data: bytes = b"",
    ann_index_data: bytes = b"",
    store_mode: str = "embedded",
) -> None:
    """Save a dense field to .rlat format.

    Args:
        path: Output file path.
        field_tensor: Shape (B, D, D) tensor.
        config: Lattice configuration.
        source_count: Number of sources encoded.
        registry_data: Serialised registry bytes.
        store_data: Serialised store bytes.
        ann_index_data: Serialised ANN index bytes.
        store_mode: "embedded" packs the source store inside the knowledge model;
            "external" records that content is resolved from disk at query
            time. Written into the v2 header byte 0x39.
    """
    path = Path(path)
    precision = config.precision
    compression = config.compression
    dtype = PRECISION_DTYPE.get(precision, np.float32)

    # Convert and compress field tensor
    tensor_bytes = field_tensor.astype(dtype).tobytes()
    compressed_tensor = _compress(tensor_bytes, compression)

    # Compress registry block with the same algorithm as the field tensor
    compressed_registry = _compress(registry_data, compression) if registry_data else b""

    # Compute offsets
    header_size = RlatHeader.SIZE
    field_offset = header_size
    registry_offset = field_offset + len(compressed_tensor)
    store_offset = registry_offset + len(compressed_registry)
    ann_offset = store_offset + len(store_data) if ann_index_data else 0

    header = RlatHeader(
        bands=config.bands,
        dim=config.dim,
        source_count=source_count,
        field_type=FieldType.DENSE,
        precision=precision,
        compression=compression,
        field_tensor_offset=field_offset,
        registry_offset=registry_offset,
        source_store_offset=store_offset,
        ann_index_offset=ann_offset,
        store_mode=store_mode,
    )

    with _atomic_open_wb(path) as f:
        f.write(header.to_bytes())
        f.write(compressed_tensor)
        f.write(compressed_registry)
        f.write(store_data)
        if ann_index_data:
            f.write(ann_index_data)


def load_dense_field(
    path: str | Path,
) -> tuple[RlatHeader, NDArray, bytes, bytes, bytes]:
    """Load a dense field from .rlat format.

    Args:
        path: Input file path.

    Returns:
        Tuple of (header, field_tensor, registry_data, store_data, ann_index_data).
    """
    path = Path(path)

    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)

        if header.field_type != FieldType.DENSE:
            raise ValueError(f"Expected dense field, got {header.field_type}")

        # Read field tensor
        f.seek(header.field_tensor_offset)
        if header.registry_offset > header.field_tensor_offset:
            tensor_size = header.registry_offset - header.field_tensor_offset
        else:
            tensor_size = -1

        compressed = f.read(tensor_size) if tensor_size > 0 else f.read()

        compression = header.compression
        raw = _decompress(compressed, compression)

        dtype = PRECISION_DTYPE.get(header.precision, np.float32)
        tensor = np.frombuffer(raw, dtype=dtype).reshape(
            header.bands, header.dim, header.dim
        )

        # Read and decompress registry data
        registry_data = b""
        store_end = header.ann_index_offset if header.ann_index_offset > 0 else None
        if header.registry_offset > 0 and header.source_store_offset > header.registry_offset:
            f.seek(header.registry_offset)
            raw_registry = f.read(header.source_store_offset - header.registry_offset)
            try:
                registry_data = _decompress(raw_registry, compression)
            except Exception:
                registry_data = raw_registry

        # Read store data
        store_data = b""
        if header.source_store_offset > 0:
            f.seek(header.source_store_offset)
            if store_end:
                store_data = f.read(store_end - header.source_store_offset)
            else:
                store_data = f.read()

        # Read ANN index data
        ann_index_data = b""
        if header.ann_index_offset > 0:
            f.seek(header.ann_index_offset)
            ann_index_data = f.read()

    return header, tensor.astype(np.float32), registry_data, store_data, ann_index_data


def save_factored_field(
    path: str | Path,
    U_list: list[NDArray],
    sigma_list: list[NDArray],
    V_list: list[NDArray],
    config: LatticeConfig,
    source_count: int = 0,
    store_mode: str = "embedded",
) -> None:
    """Save a factored (SVD) field to .rlat format.

    Per band: U[D,K] + sigma[K] + V[D,K] serialised contiguously.
    """
    path = Path(path)
    precision = config.precision
    compression = config.compression
    dtype = PRECISION_DTYPE.get(precision, np.float32)

    buf = io.BytesIO()
    for b in range(config.bands):
        if U_list[b] is not None:
            k = U_list[b].shape[1]
            # Write rank
            buf.write(struct.pack("<I", k))
            buf.write(U_list[b].astype(dtype).tobytes())
            buf.write(sigma_list[b].astype(dtype).tobytes())
            buf.write(V_list[b].astype(dtype).tobytes())
        else:
            buf.write(struct.pack("<I", 0))

    tensor_bytes = buf.getvalue()
    compressed = _compress(tensor_bytes, compression)

    header_size = RlatHeader.SIZE
    field_offset = header_size

    header = RlatHeader(
        bands=config.bands,
        dim=config.dim,
        source_count=source_count,
        field_type=FieldType.FACTORED,
        precision=precision,
        compression=compression,
        svd_rank=config.svd_rank,
        field_tensor_offset=field_offset,
        registry_offset=field_offset + len(compressed),
        source_store_offset=field_offset + len(compressed),
        store_mode=store_mode,
    )

    with _atomic_open_wb(path) as f:
        f.write(header.to_bytes())
        f.write(compressed)


def load_factored_field(
    path: str | Path,
) -> tuple[RlatHeader, list[NDArray | None], list[NDArray | None], list[NDArray | None]]:
    """Load a factored field from .rlat format.

    Returns:
        (header, U_list, sigma_list, V_list) per band.
    """
    path = Path(path)

    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)

        if header.field_type != FieldType.FACTORED:
            raise ValueError(f"Expected factored field, got {header.field_type}")

        f.seek(header.field_tensor_offset)
        compressed_size = header.registry_offset - header.field_tensor_offset
        compressed = f.read(compressed_size) if compressed_size > 0 else f.read()

    raw = _decompress(compressed, header.compression)
    dtype = PRECISION_DTYPE.get(header.precision, np.float32)

    buf = io.BytesIO(raw)
    U_list: list[NDArray | None] = []
    sigma_list: list[NDArray | None] = []
    V_list: list[NDArray | None] = []

    for b in range(header.bands):
        k = struct.unpack("<I", buf.read(4))[0]
        if k == 0:
            U_list.append(None)
            sigma_list.append(None)
            V_list.append(None)
        else:
            D = header.dim
            U = np.frombuffer(buf.read(D * k * np.dtype(dtype).itemsize), dtype=dtype).reshape(D, k)
            sigma = np.frombuffer(buf.read(k * np.dtype(dtype).itemsize), dtype=dtype).copy()
            V = np.frombuffer(buf.read(D * k * np.dtype(dtype).itemsize), dtype=dtype).reshape(D, k)
            U_list.append(U.astype(np.float32))
            sigma_list.append(sigma.astype(np.float32))
            V_list.append(V.astype(np.float32))

    return header, U_list, sigma_list, V_list


# ═══════════════════════════════════════════════════════════
# PQ Field serialisation
# ═══════════════════════════════════════════════════════════

def save_pq_field(
    path: str | Path,
    codebooks: list[list[NDArray]],
    qfield: list[list[NDArray]],
    config: LatticeConfig,
    source_count: int = 0,
    store_mode: str = "embedded",
) -> None:
    """Save a PQ field to .rlat format.

    Per band, per subspace: codebook[K, D/M] + quantised_field[K, K].

    Args:
        path: Output file path.
        codebooks: [B][M] arrays of shape (K, sub_dim).
        qfield: [B][M] arrays of shape (K, K).
        config: Lattice configuration.
        source_count: Number of sources encoded.
    """
    path = Path(path)
    precision = config.precision
    compression = config.compression
    dtype = PRECISION_DTYPE.get(precision, np.float32)

    buf = io.BytesIO()
    B = config.bands
    M = config.pq_subspaces

    for b in range(B):
        for m in range(M):
            cb = codebooks[b][m]
            qf = qfield[b][m]
            if cb is not None:
                buf.write(cb.astype(dtype).tobytes())
                buf.write(qf.astype(dtype).tobytes())
            else:
                # Write zeros if codebook not trained
                K = config.pq_codebook_size
                sub_dim = config.dim // M
                buf.write(np.zeros((K, sub_dim), dtype=dtype).tobytes())
                buf.write(np.zeros((K, K), dtype=dtype).tobytes())

    tensor_bytes = buf.getvalue()
    compressed = _compress(tensor_bytes, compression)

    header_size = RlatHeader.SIZE
    field_offset = header_size

    header = RlatHeader(
        bands=B,
        dim=config.dim,
        source_count=source_count,
        field_type=FieldType.PQ,
        precision=precision,
        compression=compression,
        pq_subspaces=M,
        pq_codebook_size=config.pq_codebook_size,
        field_tensor_offset=field_offset,
        registry_offset=field_offset + len(compressed),
        source_store_offset=field_offset + len(compressed),
        store_mode=store_mode,
    )

    with _atomic_open_wb(path) as f:
        f.write(header.to_bytes())
        f.write(compressed)


def load_pq_field(
    path: str | Path,
) -> tuple[RlatHeader, list[list[NDArray]], list[list[NDArray]]]:
    """Load a PQ field from .rlat format.

    Returns:
        (header, codebooks[B][M], qfield[B][M]).
    """
    path = Path(path)

    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)

        if header.field_type != FieldType.PQ:
            raise ValueError(f"Expected PQ field, got {header.field_type}")

        f.seek(header.field_tensor_offset)
        compressed_size = header.registry_offset - header.field_tensor_offset
        compressed = f.read(compressed_size) if compressed_size > 0 else f.read()

    raw = _decompress(compressed, header.compression)
    dtype = PRECISION_DTYPE.get(header.precision, np.float32)

    B = header.bands
    M = header.pq_subspaces
    K = header.pq_codebook_size
    sub_dim = header.dim // M
    elem_size = np.dtype(dtype).itemsize

    buf = io.BytesIO(raw)
    codebooks: list[list[NDArray]] = [[] for _ in range(B)]
    qfield: list[list[NDArray]] = [[] for _ in range(B)]

    for b in range(B):
        for m in range(M):
            cb_bytes = buf.read(K * sub_dim * elem_size)
            cb = np.frombuffer(cb_bytes, dtype=dtype).reshape(K, sub_dim)
            codebooks[b].append(cb.astype(np.float32).copy())

            qf_bytes = buf.read(K * K * elem_size)
            qf = np.frombuffer(qf_bytes, dtype=dtype).reshape(K, K)
            qfield[b].append(qf.astype(np.float32).copy())

    return header, codebooks, qfield


# ═══════════════════════════════════════════════════════════
# Memory-mapped field access
# ═══════════════════════════════════════════════════════════

def mmap_dense_field(path: str | Path) -> tuple[RlatHeader, NDArray]:
    """Memory-map a dense field for zero-copy read access.

    The field tensor is mapped directly from disk. Changes to the array
    are NOT written back (copy-on-write).

    Args:
        path: .rlat file path.

    Returns:
        (header, field_tensor_mmap) where field_tensor is (B, D, D).
    """

    path = Path(path)

    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
    header = RlatHeader.from_bytes(header_bytes)

    if header.field_type != FieldType.DENSE:
        raise ValueError(f"mmap only supported for dense fields, got {header.field_type}")
    if header.compression != Compression.NONE:
        raise ValueError("mmap requires uncompressed .rlat files (compression=none)")

    dtype = PRECISION_DTYPE.get(header.precision, np.float32)
    shape = (header.bands, header.dim, header.dim)
    offset = header.field_tensor_offset

    # np.memmap handles cross-platform mmap
    tensor = np.memmap(
        str(path),
        dtype=dtype,
        mode="r",  # read-only
        offset=offset,
        shape=shape,
    )

    return header, tensor


# ── Field-only export/import (PR #60) ─────────────────────────────────
#
# A field-only cartridge contains the header + field tensor + registry
# but NO source store. This enables privacy-preserving sharing:
# recipients can profile, compare, and merge without seeing source text.
#
# The source_store_offset is set to 0 to signal a field-only cartridge.


def is_field_only(path: str | Path) -> bool:
    """Check if a .rlat file is a field-only knowledge model (no store)."""
    path = Path(path)
    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
    header = RlatHeader.from_bytes(header_bytes)
    return header.source_store_offset == 0


def save_field_only(
    output_path: str | Path,
    loaded_data: tuple,
) -> None:
    """Save a field-only knowledge model (field + registry, no store).

    Args:
        output_path: Output file path.
        loaded_data: Tuple from load_dense_field/load_factored_field/load_pq_field:
            (header, field_tensor, registry_data, store_data, ...).
            The store_data (and any trailing elements) are discarded.
    """
    header, field_tensor, registry_data, *_rest = loaded_data
    output_path = Path(output_path)

    # Re-save with store omitted
    dtype = PRECISION_DTYPE.get(header.precision, np.float32)
    tensor_bytes = field_tensor.astype(dtype).tobytes()
    compressed_tensor = _compress(tensor_bytes, header.compression)

    header_size = RlatHeader.SIZE
    field_offset = header_size
    registry_offset = field_offset + len(compressed_tensor)

    new_header = RlatHeader(
        bands=header.bands,
        dim=header.dim,
        source_count=header.source_count,
        field_type=header.field_type,
        precision=header.precision,
        compression=header.compression,
        pq_subspaces=header.pq_subspaces,
        pq_codebook_size=header.pq_codebook_size,
        svd_rank=header.svd_rank,
        field_tensor_offset=field_offset,
        registry_offset=registry_offset,
        source_store_offset=0,  # Signal: field-only, no store
        ann_index_offset=0,
        # Preserve the store_mode from the source header. A field-only export
        # has source_store_offset=0 regardless, but the mode still conveys
        # author intent for downstream consumers.
        store_mode=header.store_mode,
    )

    with _atomic_open_wb(output_path) as f:
        f.write(new_header.to_bytes())
        f.write(compressed_tensor)
        f.write(registry_data)


def load_field_only(
    path: str | Path,
) -> tuple[RlatHeader, NDArray, bytes]:
    """Load a field-only knowledge model (field + registry, no store).

    Works for both field-only and full knowledge models (just ignores the store).

    Args:
        path: Input .rlat file path.

    Returns:
        Tuple of (header, field_tensor, registry_data).
    """
    path = Path(path)

    with open(path, "rb") as f:
        header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)

        # Read field tensor
        f.seek(header.field_tensor_offset)
        if header.registry_offset > header.field_tensor_offset:
            tensor_size = header.registry_offset - header.field_tensor_offset
        else:
            tensor_size = -1
        compressed = f.read(tensor_size) if tensor_size > 0 else f.read()
        raw = _decompress(compressed, header.compression)
        dtype = PRECISION_DTYPE.get(header.precision, np.float32)
        tensor = np.frombuffer(raw, dtype=dtype).reshape(
            header.bands, header.dim, header.dim
        )

        # Read registry data
        registry_data = b""
        if header.registry_offset > 0:
            f.seek(header.registry_offset)
            if header.source_store_offset > header.registry_offset:
                registry_data = f.read(header.source_store_offset - header.registry_offset)
            else:
                # Field-only: read to end of file
                registry_data = f.read()

    return header, tensor.astype(np.float32), registry_data
