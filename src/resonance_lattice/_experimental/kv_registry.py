# SPDX-License-Identifier: BUSL-1.1
"""Key/value registry adapter for asymmetric field encoding.

Wraps a PhaseRegistry (which handles key-space matching) and adds a
parallel value vector store. At build time, key vectors go into the
existing registry for matching; value vectors are stored separately
for use during forget (rank-1 subtraction) and field reconstruction.

This avoids modifying the existing PhaseRegistry — it remains 100%
unchanged, and the adapter adds the value dimension on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.registry import PhaseRegistry, SourcePointer


@dataclass
class KeyValueEntry:
    """Stored key and value vectors for a single source.

    Used for asymmetric field operations (forget, diff) that need
    both the key and value to perform exact rank-1 subtraction.
    """
    source_id: str
    key_vectors: NDArray[np.float32]    # (B, D_key) or list of per-band
    value_vectors: NDArray[np.float32]  # (B, D_value) or list of per-band
    salience: float = 1.0


class KeyValueRegistry:
    """Adapter that pairs a PhaseRegistry with a value vector store.

    The PhaseRegistry handles all key-space operations (registration,
    LSH hashing, ANN index, brute-force scoring). This adapter adds:
    - Value vector storage per source
    - Key/value pair retrieval for forget operations
    - Batch key/value registration
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        bands: int,
        **registry_kwargs,
    ) -> None:
        """
        Args:
            dim_key: Key dimensionality (used by PhaseRegistry for matching).
            dim_value: Value dimensionality (stored alongside for forget ops).
            bands: Number of frequency bands.
            **registry_kwargs: Passed to PhaseRegistry (num_tables, num_bits, etc).
        """
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.bands = bands

        # Key-space registry handles matching
        self.key_registry = PhaseRegistry(
            dim=dim_key, bands=bands, **registry_kwargs
        )

        # Value vector store (separate from key registry)
        self._value_store: dict[str, NDArray[np.float32]] = {}

    @property
    def source_count(self) -> int:
        return self.key_registry.source_count

    def register(
        self,
        source_id: str,
        key_vectors: NDArray[np.float32],
        value_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Register a source with separate key and value vectors.

        Args:
            source_id: Unique source identifier.
            key_vectors: Shape (B, D_key) — matching vectors.
            value_vectors: Shape (B, D_value) — retrieval vectors.
            salience: Source salience weight.
        """
        self.key_registry.register(source_id, key_vectors, salience)
        self._value_store[source_id] = value_vectors.copy()

    def unregister(self, source_id: str) -> None:
        """Remove a source from both key registry and value store."""
        self.key_registry.unregister(source_id)
        self._value_store.pop(source_id, None)

    def get_kv_entry(self, source_id: str) -> KeyValueEntry | None:
        """Retrieve the full key/value entry for a source.

        Used for forget operations that need both key and value vectors.
        """
        if source_id not in self.key_registry._source_index:
            return None
        reg_entry = self.key_registry._source_index[source_id]
        value_vectors = self._value_store.get(source_id)
        if value_vectors is None:
            return None
        return KeyValueEntry(
            source_id=source_id,
            key_vectors=reg_entry.phase_vectors,
            value_vectors=value_vectors,
            salience=reg_entry.salience,
        )

    def lookup(
        self,
        query_key: NDArray[np.float32],
        top_k: int = 20,
    ) -> list[SourcePointer]:
        """Look up sources by key-space matching.

        Delegates entirely to the wrapped PhaseRegistry.

        Args:
            query_key: Shape (B, D_key) — query key vectors.
            top_k: Number of results to return.
        """
        fused = query_key.mean(axis=0)
        return self.key_registry.lookup(fused, top_k=top_k, query_phase=query_key)

    def lookup_bruteforce(
        self,
        query_key: NDArray[np.float32],
        top_k: int = 20,
        band_weights: NDArray[np.float32] | None = None,
    ) -> list[SourcePointer]:
        """Brute-force key-space lookup. Delegates to PhaseRegistry."""
        return self.key_registry.lookup_bruteforce(
            query_key, top_k=top_k, band_weights=band_weights,
        )

    def get_value_vectors(self, source_id: str) -> NDArray[np.float32] | None:
        """Get stored value vectors for a source."""
        return self._value_store.get(source_id)

    def to_bytes(self, quantize: int = 0) -> tuple[bytes, bytes]:
        """Serialize both the key registry and value store.

        Returns:
            Tuple of (key_registry_bytes, value_store_bytes).
        """
        key_bytes = self.key_registry.to_bytes(quantize=quantize)

        # Serialize value store: [count:u32][entries...]
        import struct
        parts = [struct.pack("<I", len(self._value_store))]
        for source_id, value_vectors in self._value_store.items():
            id_bytes = source_id.encode("utf-8")
            parts.append(struct.pack("<H", len(id_bytes)))
            parts.append(id_bytes)
            val_bytes = value_vectors.astype(np.float32).tobytes()
            parts.append(struct.pack("<I", len(val_bytes)))
            parts.append(val_bytes)

        return key_bytes, b"".join(parts)

    @classmethod
    def from_bytes(
        cls,
        key_data: bytes,
        value_data: bytes,
        dim_key: int,
        dim_value: int,
        bands: int,
    ) -> KeyValueRegistry:
        """Deserialize a KeyValueRegistry.

        Args:
            key_data: Serialized key registry bytes.
            value_data: Serialized value store bytes.
            dim_key: Key dimensionality.
            dim_value: Value dimensionality.
            bands: Number of bands.
        """
        import struct

        kv_reg = cls(dim_key=dim_key, dim_value=dim_value, bands=bands)
        kv_reg.key_registry = PhaseRegistry.from_bytes(key_data, dim_key, bands)

        # Deserialize value store
        offset = 0
        (count,) = struct.unpack_from("<I", value_data, offset)
        offset += 4
        for _ in range(count):
            (id_len,) = struct.unpack_from("<H", value_data, offset)
            offset += 2
            source_id = value_data[offset:offset + id_len].decode("utf-8")
            offset += id_len
            (val_len,) = struct.unpack_from("<I", value_data, offset)
            offset += 4
            val_arr = np.frombuffer(
                value_data[offset:offset + val_len], dtype=np.float32
            ).reshape(bands, dim_value).copy()
            offset += val_len
            kv_reg._value_store[source_id] = val_arr

        return kv_reg
