# SPDX-License-Identifier: BUSL-1.1
"""Phase Registry — LSH-based index mapping resonance bright spots to sources.

The phase registry resolves high-amplitude positions in the resonance vector
back to specific source pointers. It uses Locality-Sensitive Hashing for O(1)
approximate lookup per bright spot.

Structure: HashMap<QuantisedPhaseRegion, Vec<SourcePointer>>
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.ann_index import (
    FAISSIndex,
)


@dataclass
class SourcePointer:
    """Pointer from the phase registry to a source in the source store."""
    source_id: str
    fidelity_score: float = 1.0
    band_signature: NDArray[np.float32] | None = None  # resonance per band at encode time


@dataclass
class RegistryEntry:
    """A single entry in an LSH bucket."""
    source_id: str
    phase_vectors: NDArray[np.float32]  # (B, D) — stored for exact re-scoring
    salience: float = 1.0
    fidelity_score: float = 1.0
    access_count: int = 0
    last_accessed: float = 0.0  # Unix timestamp of last retrieval hit


class SimHasher:
    """SimHash for locality-sensitive hashing of dense vectors.

    Uses random hyperplane projections: hash(v) = sign(R @ v) mapped to bits.
    Similar vectors produce similar hashes with high probability.
    """

    def __init__(self, dim: int, num_bits: int = 64, seed: int = 42) -> None:
        self.dim = dim
        self.num_bits = num_bits
        rng = np.random.default_rng(seed)
        # Random hyperplanes for hashing
        self.planes = rng.standard_normal((num_bits, dim)).astype(np.float32)

    def hash(self, vector: NDArray[np.float32]) -> int:
        """Compute the SimHash of a vector.

        Args:
            vector: Shape (D,).

        Returns:
            Integer hash value (num_bits bits).
        """
        return self.hash_multi(vector.reshape(1, -1))[0]

    def hash_multi(self, vectors: NDArray[np.float32]) -> list[int]:
        """Hash multiple vectors.

        Args:
            vectors: Shape (N, D).

        Returns:
            List of N hash values.
        """
        projections = vectors @ self.planes.T  # (N, num_bits)
        bits = (projections > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)  # (N, ceil(num_bits/8))
        # np.packbits places the first bit in the MSB of the first byte,
        # padding with zeros on the right.  When num_bits isn't a multiple
        # of 8 the resulting integer is left-shifted by the padding amount.
        # Right-shift to align the hash value to the lowest num_bits bits so
        # that _neighboring_hashes (which flips bits 0..num_bits-1) operates
        # on the correct positions.
        pad = (8 - self.num_bits % 8) % 8
        return [int.from_bytes(row.tobytes(), byteorder="big") >> pad for row in packed]


class PhaseRegistry:
    """LSH-based registry mapping phase regions to source pointers.

    Uses multiple hash tables (multi-probe LSH) for better recall.
    Each source is inserted into all tables; lookup probes all tables
    and deduplicates.
    """

    def __init__(
        self,
        dim: int,
        bands: int,
        num_tables: int = 8,
        num_bits: int = 10,
        num_probes: int = 2,
        seed: int = 42,
    ) -> None:
        """
        Args:
            dim: Dimensionality per band (D).
            bands: Number of frequency bands (B).
            num_tables: Number of LSH hash tables (more = better recall, more memory).
            num_bits: Bits per hash (fewer = coarser buckets = higher recall).
            num_probes: Multi-probe depth — check hashes within this Hamming distance.
            seed: Random seed for reproducible hashing.
        """
        self.dim = dim
        self.bands = bands
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.num_probes = num_probes

        # Create hashers — one per table, using the fused representation
        self._hashers = [
            SimHasher(dim=dim, num_bits=num_bits, seed=seed + t)
            for t in range(num_tables)
        ]

        # Hash tables: list of dict[int, list[RegistryEntry]]
        self._tables: list[dict[int, list[RegistryEntry]]] = [
            {} for _ in range(num_tables)
        ]

        # Source ID index for fast removal
        self._source_index: dict[str, RegistryEntry] = {}

        # Cached matrices for vectorised brute-force lookup
        self._cache_dirty = True
        self._cached_ids: list[str] = []
        self._cached_phases: NDArray[np.float32] | None = None  # (N, B*D) flattened
        self._cached_saliences: NDArray[np.float32] | None = None  # (N,)

        # ANN index for O(log N) approximate lookup
        self._ann_index: FAISSIndex | None = None
        self._ann_dirty = True

    def _neighboring_hashes(self, h: int) -> list[int]:
        """Generate hashes within Hamming distance `num_probes` of h.

        For num_probes=1: flip each bit individually -> num_bits neighbors.
        For num_probes=2: also flip pairs of bits.
        """
        neighbors = [h]
        if self.num_probes >= 1:
            for i in range(self.num_bits):
                neighbors.append(h ^ (1 << i))
        if self.num_probes >= 2:
            for i in range(self.num_bits):
                for j in range(i + 1, self.num_bits):
                    neighbors.append(h ^ (1 << i) ^ (1 << j))
        return neighbors

    @property
    def source_count(self) -> int:
        return len(self._source_index)

    def _fuse_for_hashing(self, phase_vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fuse multi-band phase vectors into a single vector for hashing.

        Simple approach: concatenate the mean across bands. This gives a
        D-dimensional summary for hashing.
        """
        return phase_vectors.mean(axis=0)  # (D,)

    def register(
        self,
        source_id: str,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Register a source in the phase registry.

        Args:
            source_id: Unique source identifier.
            phase_vectors: Shape (B, D) — the source's phase spectrum.
            salience: Source salience weight.
        """
        if source_id in self._source_index:
            self.unregister(source_id)

        entry = RegistryEntry(
            source_id=source_id,
            phase_vectors=phase_vectors.copy(),
            salience=salience,
        )

        fused = self._fuse_for_hashing(phase_vectors)

        for t in range(self.num_tables):
            h = self._hashers[t].hash(fused)
            bucket = self._tables[t].setdefault(h, [])
            bucket.append(entry)

        self._source_index[source_id] = entry
        self._cache_dirty = True
        self._ann_dirty = True

    def unregister(self, source_id: str) -> None:
        """Remove a source from the registry."""
        if source_id not in self._source_index:
            return

        entry = self._source_index.pop(source_id)
        self._cache_dirty = True
        self._ann_dirty = True
        fused = self._fuse_for_hashing(entry.phase_vectors)

        for t in range(self.num_tables):
            h = self._hashers[t].hash(fused)
            bucket = self._tables[t].get(h, [])
            self._tables[t][h] = [e for e in bucket if e.source_id != source_id]

    def touch(self, source_id: str, now: float | None = None) -> bool:
        """Record a retrieval hit on a source.

        Args:
            source_id: The source that was accessed.
            now: Unix timestamp (defaults to current time).

        Returns:
            True if the source exists and was touched.
        """
        entry = self._source_index.get(source_id)
        if entry is None:
            return False
        if now is None:
            import time
            now = time.time()
        entry.access_count += 1
        entry.last_accessed = now
        return True

    def lookup(
        self,
        resonance_vector: NDArray[np.float32],
        top_k: int = 20,
        query_phase: NDArray[np.float32] | None = None,
    ) -> list[SourcePointer]:
        """Find sources that match the resonance bright spots.

        Uses the query phase vectors for LSH lookup (same space as stored sources),
        then scores candidates using the resonance vector or exact cosine similarity.

        Args:
            resonance_vector: Shape (D,) — the fused resonance vector from the field.
            top_k: Number of top sources to return.
            query_phase: Shape (B, D) — if provided, used for both LSH lookup and exact re-scoring.

        Returns:
            Top-k SourcePointers sorted by score (descending).
        """
        # Use query phase for hashing (same space as stored sources)
        # Fall back to resonance vector if query_phase not provided
        if query_phase is not None:
            hash_vector = self._fuse_for_hashing(query_phase)
        else:
            hash_vector = resonance_vector

        # Collect candidates from all hash tables using multi-probe
        candidates: dict[str, RegistryEntry] = {}

        for t in range(self.num_tables):
            h = self._hashers[t].hash(hash_vector)
            # Probe the exact hash and neighboring hashes (flipped bits)
            for probe_h in self._neighboring_hashes(h):
                bucket = self._tables[t].get(probe_h, [])
                for entry in bucket:
                    if entry.source_id not in candidates:
                        candidates[entry.source_id] = entry

        if not candidates:
            return []

        # Score candidates
        scored: list[tuple[float, RegistryEntry]] = []
        for entry in candidates.values():
            if query_phase is not None:
                # Exact scoring: sum of per-band cosine similarities
                score = 0.0
                for b in range(self.bands):
                    score += float(np.dot(entry.phase_vectors[b], query_phase[b]))
                score *= entry.salience
            else:
                # Approximate scoring using fused resonance vector
                fused = self._fuse_for_hashing(entry.phase_vectors)
                score = float(np.dot(fused, resonance_vector)) * entry.salience
            scored.append((score, entry))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, entry in top:
            # Compute per-band signature
            band_sig = None
            if query_phase is not None:
                band_sig = np.array([
                    float(np.dot(entry.phase_vectors[b], query_phase[b]))
                    for b in range(self.bands)
                ], dtype=np.float32)

            results.append(SourcePointer(
                source_id=entry.source_id,
                fidelity_score=score,
                band_signature=band_sig,
            ))

        return results

    def _rebuild_cache(self) -> None:
        """Rebuild the cached matrices for brute-force lookup."""
        if not self._source_index:
            self._cached_ids = []
            self._cached_phases = None
            self._cached_saliences = None
            self._cache_dirty = False
            return

        self._cached_ids = list(self._source_index.keys())
        n = len(self._cached_ids)

        # Stack into (N, B, D) then reshape to (N, B*D) for single matmul
        phases_3d = np.array(
            [self._source_index[sid].phase_vectors for sid in self._cached_ids],
            dtype=np.float32,
        )  # (N, B, D)
        self._cached_phases = phases_3d.reshape(n, -1)  # (N, B*D)
        self._cached_saliences = np.array(
            [self._source_index[sid].salience for sid in self._cached_ids],
            dtype=np.float32,
        )
        self._cache_dirty = False

    def _rebuild_ann(self) -> None:
        """Mark the ANN index as stale.

        The ANN index is expensive to build (30+ seconds at 24K sources)
        so it is NOT rebuilt automatically at query time.  It is only
        built at knowledge model save time or loaded from the .rlat file.
        When stale, ``lookup_ann`` falls back to brute-force.
        """
        self._ann_index = None
        self._ann_dirty = False

    def set_ann_index(self, index: FAISSIndex | None) -> None:
        """Attach a pre-built ANN index (e.g. loaded from .rlat)."""
        self._ann_index = index
        self._ann_dirty = index is None

    @property
    def has_ann(self) -> bool:
        """True if an ANN index is available for fast lookup."""
        return self._ann_index is not None

    def lookup_ann(
        self,
        query_phase: NDArray[np.float32],
        top_k: int = 20,
        band_weights: NDArray[np.float32] | None = None,
        over_retrieve: int = 3,
        resonance_vectors: NDArray[np.float32] | None = None,
        resonance_alpha: float = 1.0,
        resonance_mode: str = "normalize",
    ) -> list[SourcePointer]:
        """ANN-accelerated lookup: HNSW probe then exact rescore.

        1. Probe HNSW for ``top_k * over_retrieve`` approximate candidates.
        2. Exact-rescore those candidates using per-band dot products.
        3. Return the final top_k with full band signatures.

        Falls back to ``lookup_bruteforce`` when no ANN index is available.

        Args:
            query_phase: (B, D) query phase vectors.
            top_k: Number of results to return.
            band_weights: Per-band weights (B,).
            over_retrieve: Multiplier for candidate over-retrieval (default 3x).
            resonance_vectors: Dense resonance vectors for blended scoring.
            resonance_alpha: Blend weight in [0, 1].
            resonance_mode: How to process resonance vectors.
        """
        if self._ann_dirty:
            self._rebuild_ann()

        if self._ann_index is None:
            return self.lookup_bruteforce(
                query_phase, top_k, band_weights,
                resonance_vectors, resonance_alpha, resonance_mode,
            )

        if self._cache_dirty:
            self._rebuild_cache()

        if not self._cached_ids:
            return []

        n = len(self._cached_ids)

        # 1. HNSW probe for approximate candidates
        q_flat = query_phase.ravel().astype(np.float32)
        q_norm = np.linalg.norm(q_flat)
        q_normalized = q_flat / max(q_norm, 1e-8)
        candidate_k = min(top_k * over_retrieve, n)
        indices, _scores = self._ann_index.query(q_normalized, candidate_k)

        # Filter invalid indices
        valid = indices[(indices >= 0) & (indices < n)]
        if len(valid) == 0:
            return []

        # 2. Exact rescore of candidates
        candidate_phases = self._cached_phases[valid]  # (C, B*D)
        candidate_saliences = self._cached_saliences[valid]  # (C,)

        if band_weights is not None:
            phases_3d = candidate_phases.reshape(len(valid), self.bands, -1)
            per_band = np.einsum("nbd,bd->nb", phases_3d, query_phase)
            scores = (per_band @ band_weights) * candidate_saliences
        else:
            scores = (candidate_phases @ q_flat) * candidate_saliences

        # Blend with resonance-guided scoring
        if resonance_vectors is not None and resonance_alpha < 1.0:
            r_processed = self._process_resonance(
                resonance_vectors, query_phase, resonance_mode,
            )
            r_flat = r_processed.ravel()
            res_scores = (candidate_phases @ r_flat) * candidate_saliences
            scores = resonance_alpha * scores + (1.0 - resonance_alpha) * res_scores

        # 3. Top-k from candidates
        if len(scores) > top_k:
            top_local = np.argpartition(scores, -top_k)[-top_k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
        else:
            top_local = np.argsort(scores)[::-1]

        # Build results with band signatures
        phases_3d = candidate_phases.reshape(len(valid), self.bands, -1)
        results = []
        for li in top_local[:top_k]:
            gi = int(valid[li])
            band_sig = np.array([
                float(np.dot(phases_3d[li, b], query_phase[b]))
                for b in range(self.bands)
            ], dtype=np.float32)
            results.append(SourcePointer(
                source_id=self._cached_ids[gi],
                fidelity_score=float(scores[li]),
                band_signature=band_sig,
            ))

        return results

    @staticmethod
    def _process_resonance(
        resonance_vectors: NDArray[np.float32],
        query_phase: NDArray[np.float32],
        mode: str,
    ) -> NDArray[np.float32]:
        """Apply resonance processing (shared by brute-force and ANN paths)."""
        r = resonance_vectors.copy()
        bands = r.shape[0]
        if mode == "residual":
            for b in range(bands):
                q_b = query_phase[b]
                q_dot_q = np.dot(q_b, q_b)
                if q_dot_q > 1e-8:
                    proj = np.dot(r[b], q_b) / q_dot_q
                    r[b] -= proj * q_b
                norm = np.linalg.norm(r[b])
                if norm > 1e-8:
                    r[b] /= norm
        elif mode == "raw":
            pass
        else:  # "normalize"
            for b in range(bands):
                norm = np.linalg.norm(r[b])
                if norm > 1e-8:
                    r[b] /= norm
        return r

    # Registry serialization magic bytes
    _MAGIC_QUANTIZED = b"RQNT"  # Quantized registry format
    # Legacy format has no magic — starts with u32 count directly

    def to_bytes(self, quantize: int = 0) -> bytes:
        """Serialise the registry to bytes for .rlat persistence.

        Args:
            quantize: Bits per value for quantization (0 = full float32, 4 = default quantized).
                4-bit gives ~87% compression. 8-bit gives ~50% compression.

        Format (quantized):
            [magic: 4 bytes "RQNT"][bits: u8][count: u32][entries...]
            Each entry: [id_len:u16][id_bytes][salience:f32][quantized phase bytes]

        Format (legacy/unquantized):
            [count:u32][entries...] where each entry is:
            [id_len:u16][id_bytes][salience:f32][B*D floats for phase_vectors]

        LSH tables are rebuilt from entries on load.
        """
        entries = list(self._source_index.values())

        if quantize > 0:
            return self._to_bytes_quantized(entries, bits=quantize)
        return self._to_bytes_float32(entries)

    def _to_bytes_float32(self, entries: list[RegistryEntry]) -> bytes:
        """Legacy full-precision serialization."""
        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(entries)))
        for entry in entries:
            id_bytes = entry.source_id.encode("utf-8")
            buf.write(struct.pack("<H", len(id_bytes)))
            buf.write(id_bytes)
            buf.write(struct.pack("<f", entry.salience))
            buf.write(entry.phase_vectors.astype(np.float32).tobytes())
        return buf.getvalue()

    def _to_bytes_quantized(self, entries: list[RegistryEntry], bits: int) -> bytes:
        """Quantized serialization with magic header."""
        from resonance_lattice.quantize import quantize_phases

        buf = io.BytesIO()
        buf.write(self._MAGIC_QUANTIZED)
        buf.write(struct.pack("<B", bits))
        buf.write(struct.pack("<I", len(entries)))

        for entry in entries:
            id_bytes = entry.source_id.encode("utf-8")
            buf.write(struct.pack("<H", len(id_bytes)))
            buf.write(id_bytes)
            buf.write(struct.pack("<f", entry.salience))
            qdata = quantize_phases(entry.phase_vectors, bits=bits)
            buf.write(struct.pack("<I", len(qdata)))
            buf.write(qdata)

        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, dim: int, bands: int,
                   num_tables: int = 8, num_bits: int = 10,
                   num_probes: int = 2, seed: int = 42) -> PhaseRegistry:
        """Deserialise a registry from bytes.

        Auto-detects format: quantized (starts with "RQNT") or legacy float32.
        Rebuilds LSH tables by re-registering each entry.
        Quantized phases are dequantized to float32 in memory for full-precision operations.
        """
        registry = cls(dim=dim, bands=bands, num_tables=num_tables,
                       num_bits=num_bits, num_probes=num_probes, seed=seed)

        if data[:4] == cls._MAGIC_QUANTIZED:
            cls._from_bytes_quantized(registry, data, bands, dim)
        else:
            cls._from_bytes_float32(registry, data, bands, dim)

        return registry

    @staticmethod
    def _from_bytes_float32(registry: PhaseRegistry, data: bytes, bands: int, dim: int) -> None:
        """Load legacy float32 format."""
        buf = io.BytesIO(data)
        count = struct.unpack("<I", buf.read(4))[0]
        phase_size = bands * dim * 4

        for _ in range(count):
            id_len = struct.unpack("<H", buf.read(2))[0]
            source_id = buf.read(id_len).decode("utf-8")
            salience = struct.unpack("<f", buf.read(4))[0]
            phase_vectors = np.frombuffer(buf.read(phase_size), dtype=np.float32).copy()
            phase_vectors = phase_vectors.reshape(bands, dim)
            registry.register(source_id, phase_vectors, salience)

    @staticmethod
    def _from_bytes_quantized(registry: PhaseRegistry, data: bytes, bands: int, dim: int) -> None:
        """Load quantized format — dequantize to float32 in memory."""
        from resonance_lattice.quantize import dequantize_phases

        buf = io.BytesIO(data)
        buf.read(4)  # Skip magic
        struct.unpack("<B", buf.read(1))[0]
        count = struct.unpack("<I", buf.read(4))[0]

        for _ in range(count):
            id_len = struct.unpack("<H", buf.read(2))[0]
            source_id = buf.read(id_len).decode("utf-8")
            salience = struct.unpack("<f", buf.read(4))[0]
            qlen = struct.unpack("<I", buf.read(4))[0]
            qdata = buf.read(qlen)
            phase_vectors = dequantize_phases(qdata, bands=bands, dim=dim)
            registry.register(source_id, phase_vectors, salience)

    def lookup_bruteforce(
        self,
        query_phase: NDArray[np.float32],
        top_k: int = 20,
        band_weights: NDArray[np.float32] | None = None,
        resonance_vectors: NDArray[np.float32] | None = None,
        resonance_alpha: float = 1.0,
        resonance_mode: str = "normalize",
        scoring_fn: str = "linear",
        eml_alpha: float = 1.0,
        eml_noise_floor: NDArray[np.float32] | float | None = None,
    ) -> list[SourcePointer]:
        """Vectorised brute-force lookup over all registered sources.

        Uses a cached (N, B*D) matrix and a single matmul for scoring.
        Cache is rebuilt only when the registry changes.

        Args:
            query_phase: (B, D) query phase vectors.
            top_k: Number of results to return.
            band_weights: Per-band weights (B,). If provided, each band's
                dot-product contribution is scaled by its weight before summing.
                Uniform weighting when None.
            resonance_vectors: (B, D) dense resonance vectors from field
                projection (F_b @ q_b). When provided with resonance_alpha < 1,
                scoring blends sparse dot products with resonance-guided scores.
            resonance_alpha: Blend weight in [0, 1]. 1.0 = pure sparse scoring
                (default/backward-compatible), 0.0 = pure resonance scoring.
            resonance_mode: How to process resonance vectors before scoring.
                "normalize" — L2-normalize per band (default).
                "residual" — remove query-parallel component, keep only the
                    field's unique contribution orthogonal to the query.
                "raw" — use resonance vectors as-is (no normalization).
        """
        if not self._source_index:
            return []

        if self._cache_dirty:
            self._rebuild_cache()

        source_ids = self._cached_ids
        n = len(source_ids)

        # Compute sparse dot-product scores
        # Always compute per-band scores (needed for EML and band_weights paths)
        phases_3d = self._cached_phases.reshape(n, self.bands, -1)
        per_band = np.einsum("nbd,bd->nb", phases_3d, query_phase)  # (N, B)

        # Compute linear total scores (always — used as baseline AND as the
        # emitted fidelity_score so downstream consumers (reranker, keyword
        # fusion) see the familiar distribution even when EML drives the ranking).
        if band_weights is not None:
            linear_total_scores = (per_band @ band_weights) * self._cached_saliences
        else:
            q_flat_for_linear = query_phase.ravel()
            linear_total_scores = (self._cached_phases @ q_flat_for_linear) * self._cached_saliences

        if scoring_fn == "eml":
            # EML per-band scoring: exp(α · sim_b) - ln(σ_b) per band, then sum.
            # This changes rankings because exp is applied BEFORE summation.
            if eml_noise_floor is None:
                sigma = np.ones(self.bands, dtype=np.float32)
            elif isinstance(eml_noise_floor, (int, float)):
                sigma = np.full(self.bands, max(float(eml_noise_floor), 1e-12), dtype=np.float32)
            else:
                sigma = np.maximum(eml_noise_floor, 1e-12)
            # Stabilised exp: subtract single global max before exp to prevent
            # overflow at high alpha while preserving rankings across all
            # sources and bands (monotone shift of ALL scores by same scalar).
            scaled = eml_alpha * per_band  # (N, B)
            global_shift = float(scaled.max())  # scalar across all N and B
            eml_per_band = np.exp(scaled - global_shift) - np.log(sigma)[np.newaxis, :]  # (N, B)
            if band_weights is not None:
                total_scores = (eml_per_band @ band_weights) * self._cached_saliences
            else:
                total_scores = eml_per_band.sum(axis=1) * self._cached_saliences
        elif band_weights is not None:
            total_scores = (per_band @ band_weights) * self._cached_saliences
        else:
            # Uniform: flatten query to (B*D,) and compute via single matmul
            q_flat = query_phase.ravel()  # (B*D,)
            total_scores = (self._cached_phases @ q_flat) * self._cached_saliences  # (N,)

        # Blend with resonance-guided scoring when available
        if resonance_vectors is not None and resonance_alpha < 1.0:
            r_processed = self._process_resonance(
                resonance_vectors, query_phase, resonance_mode,
            )
            r_flat = r_processed.ravel()  # (B*D,)
            resonance_scores = (self._cached_phases @ r_flat) * self._cached_saliences
            total_scores = resonance_alpha * total_scores + (1.0 - resonance_alpha) * resonance_scores

        # Partial sort for top-k
        if n > top_k:
            top_indices = np.argpartition(total_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(total_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(total_scores)[::-1]

        # Compute per-band scores only for the top-k results
        if band_weights is not None:
            # Already have phases_3d from above
            pass
        else:
            phases_3d = self._cached_phases.reshape(n, self.bands, -1)
        # Emit linear-calibrated scores for downstream consumers (reranker,
        # keyword fusion) while preserving the EML ranking. This prevents the
        # exp-skewed distribution from breaking z-score normalisation.
        emit_scores = linear_total_scores if scoring_fn == "eml" else total_scores

        results = []
        for i in top_indices[:top_k]:
            band_sig = np.array([
                float(np.dot(phases_3d[i, b], query_phase[b]))
                for b in range(self.bands)
            ], dtype=np.float32)
            results.append(SourcePointer(
                source_id=source_ids[i],
                fidelity_score=float(emit_scores[i]),
                band_signature=band_sig,
            ))

        return results
