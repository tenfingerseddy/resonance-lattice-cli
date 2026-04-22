# SPDX-License-Identifier: BUSL-1.1
"""Multi-vector field: each source stores a SET of phase vectors.

Replaces the lossy B×D×D outer-product accumulation with per-source
multi-vector storage. Retrieval uses soft-MaxSim (log-sum-exp over
source chunks per band), which is a natural EML composition.

This is a representation-level architecture change: instead of compressing
N sources into a single D×D matrix, each source retains its full set of
per-chunk phase vectors. The cost is O(N·T·B·D) storage instead of
O(B·D²), but the ceiling is higher because no per-source information
is lost to superposition interference.

Algebra:
    merge(A, B) = union of source dictionaries (commutative, associative)
    diff(A, B) = sources in A not in B
    forget(i) = delete source i's entry (exact)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class MultiVectorResonanceResult(NamedTuple):
    """Result of a multi-vector resonance query."""
    source_ids: list[str]
    scores: NDArray[np.float32]          # (K,) — top-K scores
    band_scores: NDArray[np.float32]     # (K, B) — per-band scores for top-K
    per_chunk_best: list[list[int]]      # for each top-K, which chunk was best per band


class MultiVectorField:
    """Multi-vector field: per-source set of phase vectors.

    Each source is stored as a list of (B, D) phase vectors (one per chunk).
    Retrieval scores each source by soft-MaxSim: for each band, the score is
    the log-sum-exp over chunk similarities.

    soft_max_τ(x_1, ..., x_T) = (1/τ) · log(Σ_t exp(τ · x_t))

    At τ→∞ this converges to hard MaxSim (max over chunks).
    At τ→0 it converges to log(T) + mean (average pooling).
    """

    def __init__(self, bands: int, dim: int) -> None:
        self.bands = bands
        self.dim = dim
        self._sources: dict[str, list[NDArray[np.float32]]] = {}
        # Cached flat arrays for vectorised scoring (invalidated on mutation)
        self._cache_dirty = True
        self._cached_ids: list[str] = []
        self._cached_offsets: NDArray[np.int64] = np.array([], dtype=np.int64)
        self._cached_phases: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def total_vectors(self) -> int:
        return sum(len(vecs) for vecs in self._sources.values())

    @property
    def size_bytes(self) -> int:
        return self.total_vectors * self.bands * self.dim * 4

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def superpose(
        self,
        source_id: str,
        phase_vectors: NDArray[np.float32],
    ) -> None:
        """Add a phase vector to a source's vector set.

        Args:
            source_id: Identifier for this source (document).
            phase_vectors: Shape (B, D) — one chunk's phase spectrum.
        """
        if phase_vectors.shape != (self.bands, self.dim):
            raise ValueError(
                f"phase shape {phase_vectors.shape} != ({self.bands}, {self.dim})"
            )
        if source_id not in self._sources:
            self._sources[source_id] = []
        self._sources[source_id].append(phase_vectors.copy())
        self._cache_dirty = True

    def superpose_batch(
        self,
        source_id: str,
        phase_batch: NDArray[np.float32],
    ) -> None:
        """Add multiple phase vectors to a source at once.

        Args:
            source_id: Identifier for this source.
            phase_batch: Shape (T, B, D) — T chunks' phase spectra.
        """
        if phase_batch.ndim != 3 or phase_batch.shape[1:] != (self.bands, self.dim):
            raise ValueError(
                f"batch shape {phase_batch.shape} != (T, {self.bands}, {self.dim})"
            )
        if source_id not in self._sources:
            self._sources[source_id] = []
        for t in range(phase_batch.shape[0]):
            self._sources[source_id].append(phase_batch[t].copy())
        self._cache_dirty = True

    def remove(self, source_id: str) -> None:
        """Remove a source and all its vectors. Algebraically exact."""
        if source_id in self._sources:
            del self._sources[source_id]
            self._cache_dirty = True

    def merge(self, other: MultiVectorField) -> MultiVectorField:
        """Merge two fields. Commutative for non-overlapping source_ids."""
        result = MultiVectorField(bands=self.bands, dim=self.dim)
        for sid, vecs in self._sources.items():
            result._sources[sid] = [v.copy() for v in vecs]
        for sid, vecs in other._sources.items():
            if sid in result._sources:
                result._sources[sid].extend(v.copy() for v in vecs)
            else:
                result._sources[sid] = [v.copy() for v in vecs]
        result._cache_dirty = True
        return result

    def diff(self, other: MultiVectorField) -> MultiVectorField:
        """Sources in self but not in other."""
        result = MultiVectorField(bands=self.bands, dim=self.dim)
        for sid, vecs in self._sources.items():
            if sid not in other._sources:
                result._sources[sid] = [v.copy() for v in vecs]
        result._cache_dirty = True
        return result

    def _rebuild_cache(self) -> None:
        """Build flat arrays for vectorised scoring."""
        if not self._sources:
            self._cached_ids = []
            self._cached_offsets = np.array([0], dtype=np.int64)
            self._cached_phases = np.empty((0, self.bands, self.dim), dtype=np.float32)
            self._cache_dirty = False
            return

        ids = []
        offsets = [0]
        all_phases = []
        for sid, vecs in self._sources.items():
            ids.append(sid)
            all_phases.extend(vecs)
            offsets.append(offsets[-1] + len(vecs))

        self._cached_ids = ids
        self._cached_offsets = np.array(offsets, dtype=np.int64)
        self._cached_phases = np.array(all_phases, dtype=np.float32)  # (V, B, D)
        self._cache_dirty = False

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        top_k: int = 20,
        tau: float = 20.0,
        band_weights: NDArray[np.float32] | None = None,
    ) -> MultiVectorResonanceResult:
        """Score all sources via soft-MaxSim and return top-K.

        For each source i with T_i chunks:
            score_i = Σ_b w_b · soft_max_τ(q_b · phi_{i,t,b} for t in 1..T_i)

        where soft_max_τ(x) = (1/τ) · log(Σ_t exp(τ · x_t))

        Args:
            query_phase: Shape (B, D).
            top_k: Number of results to return.
            tau: Soft-MaxSim temperature. Higher = closer to hard max.
            band_weights: Shape (B,). Uniform if None.

        Returns:
            MultiVectorResonanceResult with source_ids, scores, band_scores.
        """
        if query_phase.shape != (self.bands, self.dim):
            raise ValueError(
                f"query shape {query_phase.shape} != ({self.bands}, {self.dim})"
            )

        if self._cache_dirty:
            self._rebuild_cache()

        if not self._cached_ids:
            return MultiVectorResonanceResult(
                source_ids=[], scores=np.array([], dtype=np.float32),
                band_scores=np.empty((0, self.bands), dtype=np.float32),
                per_chunk_best=[],
            )

        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=np.float32) / self.bands

        N = len(self._cached_ids)

        # Per-band dot products for ALL vectors: (V, B)
        per_band_sims = np.einsum("vbd,bd->vb", self._cached_phases, query_phase)

        # Aggregate per-source via soft-MaxSim
        source_band_scores = np.zeros((N, self.bands), dtype=np.float32)
        best_chunks = []  # for diagnostics

        for i in range(N):
            start = self._cached_offsets[i]
            end = self._cached_offsets[i + 1]
            chunk_sims = per_band_sims[start:end]  # (T_i, B)

            if chunk_sims.shape[0] == 1:
                # Single chunk — no soft-max needed
                source_band_scores[i] = chunk_sims[0]
                best_chunks.append([0] * self.bands)
            else:
                # Soft-MaxSim per band: (1/τ) · log(Σ_t exp(τ · x_t))
                # Numerically stable: subtract max before exp
                scaled = tau * chunk_sims  # (T_i, B)
                max_per_band = scaled.max(axis=0, keepdims=True)  # (1, B)
                logsumexp = max_per_band.squeeze(0) + np.log(
                    np.sum(np.exp(scaled - max_per_band), axis=0)
                )  # (B,)
                source_band_scores[i] = logsumexp / tau
                best_chunks.append(chunk_sims.argmax(axis=0).tolist())

        # Weighted sum across bands
        total_scores = source_band_scores @ band_weights  # (N,)

        # Top-K
        if N > top_k:
            top_idx = np.argpartition(total_scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(total_scores[top_idx])[::-1]]
        else:
            top_idx = np.argsort(total_scores)[::-1]

        top_idx = top_idx[:top_k]

        return MultiVectorResonanceResult(
            source_ids=[self._cached_ids[i] for i in top_idx],
            scores=total_scores[top_idx],
            band_scores=source_band_scores[top_idx],
            per_chunk_best=[best_chunks[i] for i in top_idx],
        )

    def source_ids(self) -> list[str]:
        """All source IDs in the field."""
        return list(self._sources.keys())

    def get_vectors(self, source_id: str) -> list[NDArray[np.float32]] | None:
        """Get all phase vectors for a source. None if not found."""
        return self._sources.get(source_id)

    def chunk_count(self, source_id: str) -> int:
        """Number of chunk vectors stored for a source."""
        vecs = self._sources.get(source_id)
        return len(vecs) if vecs else 0

    def reset(self) -> None:
        """Clear all sources."""
        self._sources.clear()
        self._cache_dirty = True
