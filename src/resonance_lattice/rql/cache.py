# SPDX-License-Identifier: BUSL-1.1
"""Spectral Cache: avoid redundant eigendecompositions across chained operations.

Many operations share the same eigendecomposition. Computing eigh(F_b) is O(D³).
The cache computes it once and stores the result until the field changes.
"""

from __future__ import annotations

import numpy as np

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.types import EigenDecomp


class SpectralCache:
    """Caches eigendecompositions of field bands.

    Usage:
        cache = SpectralCache(field)
        eig = cache.get(band=0)  # Computes once
        eig = cache.get(band=0)  # Returns cached
        cache.invalidate()       # After field mutation
    """

    def __init__(self, field: DenseField):
        self._field = field
        self._cache: dict[int, EigenDecomp] = {}
        self._field_hash: int = self._compute_hash()

    def get(self, band: int) -> EigenDecomp:
        """Get eigendecomposition for a band, computing if needed."""
        current_hash = self._compute_hash()
        if current_hash != self._field_hash:
            self._cache.clear()
            self._field_hash = current_hash

        if band not in self._cache:
            F_b = self._field.F[band]
            F_sym = (F_b + F_b.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            # Sort descending by absolute value
            idx = np.argsort(np.abs(eigenvalues))[::-1]

            self._cache[band] = EigenDecomp(
                eigenvalues=eigenvalues[idx].astype(np.float32),
                eigenvectors=eigenvectors[:, idx].astype(np.float32),
                band=band,
            )

        return self._cache[band]

    def invalidate(self) -> None:
        """Clear the cache (call after field mutation)."""
        self._cache.clear()
        self._field_hash = self._compute_hash()

    def _compute_hash(self) -> int:
        """Fast hash of field state for change detection."""
        # Use a few diagonal samples + source count as a cheap fingerprint
        h = hash(self._field.source_count)
        for b in range(self._field.bands):
            h ^= hash(float(self._field.F[b, 0, 0]))
            d = self._field.dim
            if d > 1:
                h ^= hash(float(self._field.F[b, d // 2, d // 2]))
        return h
