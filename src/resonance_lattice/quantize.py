# SPDX-License-Identifier: BUSL-1.1
"""Phase vector quantization for registry compression.

Implements a data-oblivious quantization scheme inspired by TurboQuant
(PolarQuant + QJL, ICLR 2026). Key properties:

- No codebook: quantization is data-oblivious, so independently-built
  fields can merge without codebook alignment.
- Streaming-compatible: can quantize incrementally as sources are added.
- 4-bit default: ~87% compression on registry storage.
- Pure NumPy: no external dependencies for MVP.

The scheme per band:
1. Store L2 norm as float16 (2 bytes).
2. Store min and max of the unit vector as float16 (4 bytes).
3. Quantize the unit-normalized vector using min-max scaling to N bits.
4. Pack quantized values into bytes.

At 4 bits/value with D=2048, B=5:
- Original: 5 * 2048 * 4 = 40,960 bytes per source
- Quantized: 5 * (2 + 4 + 2048/2) = 5,150 bytes per source (~87% savings)
"""

from __future__ import annotations

import struct

import numpy as np
from numpy.typing import NDArray


def quantize_phases(
    phase_vectors: NDArray[np.float32],
    bits: int = 4,
) -> bytes:
    """Quantize a (B, D) phase vector array to compact bytes.

    Each band is stored as:
        [norm: float16 (2 bytes)]
        [vmin: float16 (2 bytes)]
        [vmax: float16 (2 bytes)]
        [quantized values: ceil(D * bits / 8) bytes]

    Args:
        phase_vectors: Shape (B, D) float32 phase vectors.
        bits: Bits per value (1-8). Default 4.

    Returns:
        Compact byte representation.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1-8, got {bits}")

    B, D = phase_vectors.shape
    buf = bytearray()

    # Header: bits setting (1 byte)
    buf.append(bits)

    max_val = (1 << bits) - 1  # e.g. 15 for 4-bit

    for b in range(B):
        vec = phase_vectors[b]
        norm = float(np.linalg.norm(vec))

        # Store norm as float16
        buf.extend(struct.pack("<e", norm))

        # Normalize
        if norm > 1e-12:
            unit = vec / norm
        else:
            unit = np.zeros_like(vec)

        # Min-max of the unit vector
        vmin = float(np.min(unit))
        vmax = float(np.max(unit))
        buf.extend(struct.pack("<e", vmin))
        buf.extend(struct.pack("<e", vmax))

        # Map [vmin, vmax] -> [0, max_val]
        span = vmax - vmin
        if span < 1e-12:
            quantized = np.zeros(D, dtype=np.uint8)
        else:
            scaled = (unit - vmin) / span  # [0, 1]
            quantized = np.round(scaled * max_val).astype(np.uint8)
            quantized = np.clip(quantized, 0, max_val)

        # Pack into bytes
        if bits == 8:
            buf.extend(quantized.tobytes())
        elif bits == 4:
            buf.extend(_pack_4bit(quantized, D))
        elif bits == 2:
            buf.extend(_pack_2bit(quantized, D))
        elif bits == 1:
            buf.extend(np.packbits(quantized, bitorder="little").tobytes())
        else:
            buf.extend(_pack_generic(quantized, D, bits))

    return bytes(buf)


def dequantize_phases(
    data: bytes,
    bands: int,
    dim: int,
) -> NDArray[np.float32]:
    """Dequantize bytes back to (B, D) float32 phase vectors.

    Args:
        data: Bytes from quantize_phases().
        bands: Number of bands.
        dim: Dimension per band.

    Returns:
        Reconstructed (B, D) float32 array (approximate).
    """
    offset = 0

    # Read bits setting
    bits = data[offset]
    offset += 1

    max_val = (1 << bits) - 1

    result = np.zeros((bands, dim), dtype=np.float32)

    for b in range(bands):
        # Read norm, vmin, vmax
        norm = struct.unpack_from("<e", data, offset)[0]
        offset += 2
        vmin = struct.unpack_from("<e", data, offset)[0]
        offset += 2
        vmax = struct.unpack_from("<e", data, offset)[0]
        offset += 2

        # Unpack quantized values
        if bits == 8:
            nbytes = dim
            quantized = np.frombuffer(data[offset:offset + nbytes], dtype=np.uint8).copy()
            offset += nbytes
        elif bits == 4:
            nbytes = (dim + 1) // 2
            quantized = _unpack_4bit(data[offset:offset + nbytes], dim)
            offset += nbytes
        elif bits == 2:
            nbytes = (dim + 3) // 4
            quantized = _unpack_2bit(data[offset:offset + nbytes], dim)
            offset += nbytes
        elif bits == 1:
            nbytes = (dim + 7) // 8
            quantized = np.unpackbits(
                np.frombuffer(data[offset:offset + nbytes], dtype=np.uint8),
                bitorder="little",
            )[:dim].copy()
            offset += nbytes
        else:
            nbytes = _generic_packed_size(dim, bits)
            quantized = _unpack_generic(data[offset:offset + nbytes], dim, bits)
            offset += nbytes

        # Dequantize: map [0, max_val] back to [vmin, vmax]
        span = vmax - vmin
        if max_val > 0 and span > 1e-12:
            unit = (quantized.astype(np.float32) / max_val) * span + vmin
        else:
            unit = np.full(dim, vmin, dtype=np.float32)

        result[b] = unit * norm

    return result


def quantized_size(bands: int, dim: int, bits: int = 4) -> int:
    """Compute the byte size of a quantized phase vector.

    Args:
        bands: Number of bands.
        dim: Dimension per band.
        bits: Bits per value.

    Returns:
        Total bytes.
    """
    per_band = 6  # float16 norm + float16 vmin + float16 vmax
    if bits == 8:
        per_band += dim
    elif bits == 4:
        per_band += (dim + 1) // 2
    elif bits == 2:
        per_band += (dim + 3) // 4
    elif bits == 1:
        per_band += (dim + 7) // 8
    else:
        per_band += _generic_packed_size(dim, bits)

    return 1 + bands * per_band  # 1 byte header for bits


# ── 4-bit packing helpers ──

def _pack_4bit(values: NDArray[np.uint8], count: int) -> bytes:
    """Pack array of 4-bit values (0-15) into bytes, two per byte."""
    nbytes = (count + 1) // 2
    packed = np.zeros(nbytes, dtype=np.uint8)
    # Low nibble: even indices, high nibble: odd indices
    even = values[0::2]
    packed[:len(even)] = even & 0x0F
    if count > 1:
        odd = values[1::2]
        packed[:len(odd)] |= (odd & 0x0F) << 4
    return packed.tobytes()


def _unpack_4bit(data: bytes, count: int) -> NDArray[np.uint8]:
    """Unpack 4-bit values from packed bytes."""
    raw = np.frombuffer(data, dtype=np.uint8)
    # Interleave low and high nibbles
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    values = np.empty(len(raw) * 2, dtype=np.uint8)
    values[0::2] = low
    values[1::2] = high
    return values[:count]


# ── 2-bit packing helpers ──

def _pack_2bit(values: NDArray[np.uint8], count: int) -> bytes:
    """Pack array of 2-bit values (0-3) into bytes, four per byte."""
    nbytes = (count + 3) // 4
    packed = np.zeros(nbytes, dtype=np.uint8)
    for i in range(count):
        byte_idx = i // 4
        shift = (i % 4) * 2
        packed[byte_idx] |= (values[i] & 0x03) << shift
    return packed.tobytes()


def _unpack_2bit(data: bytes, count: int) -> NDArray[np.uint8]:
    """Unpack 2-bit values from packed bytes."""
    raw = np.frombuffer(data, dtype=np.uint8)
    values = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        byte_idx = i // 4
        shift = (i % 4) * 2
        values[i] = (raw[byte_idx] >> shift) & 0x03
    return values


# ── Generic bit packing ──

def _generic_packed_size(count: int, bits: int) -> int:
    """Compute packed byte size for arbitrary bit width."""
    total_bits = count * bits
    return (total_bits + 7) // 8


def _pack_generic(values: NDArray[np.uint8], count: int, bits: int) -> bytes:
    """Pack values with arbitrary bit width."""
    total_bits = count * bits
    nbytes = (total_bits + 7) // 8
    packed = bytearray(nbytes)
    bit_pos = 0
    mask = (1 << bits) - 1
    for i in range(count):
        val = int(values[i]) & mask
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        packed[byte_idx] |= (val << bit_offset) & 0xFF
        if bit_offset + bits > 8 and byte_idx + 1 < nbytes:
            packed[byte_idx + 1] |= val >> (8 - bit_offset)
        bit_pos += bits
    return bytes(packed)


def _unpack_generic(data: bytes, count: int, bits: int) -> NDArray[np.uint8]:
    """Unpack values with arbitrary bit width."""
    values = np.zeros(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_pos = 0
    for i in range(count):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        val = (data[byte_idx] >> bit_offset) & mask
        if bit_offset + bits > 8 and byte_idx + 1 < len(data):
            val |= (data[byte_idx + 1] << (8 - bit_offset)) & mask
        values[i] = val
        bit_pos += bits
    return values
