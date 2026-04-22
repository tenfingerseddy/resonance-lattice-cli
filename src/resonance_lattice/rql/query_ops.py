# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain I: Query Operations.

All resonance and query operations — the interface between queries and fields.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField, ResonanceResult
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.types import Scalar


class QueryOps:

    @staticmethod
    def resonate(field: DenseField, query: NDArray[np.float32], band_weights: NDArray[np.float32] | None = None) -> ResonanceResult:
        """Standard resonance: r_b = F_b @ q_b. Cost: O(BD²). Delegates to DenseField."""
        return field.resonate(query, band_weights=band_weights)

    @staticmethod
    def resonate_batch(field: DenseField, queries: NDArray[np.float32]) -> list[ResonanceResult]:
        """Batch resonance: resonate N queries. queries: (N, B, D). Cost: O(NBD²)."""
        return [field.resonate(queries[i]) for i in range(queries.shape[0])]

    @staticmethod
    def optimal_query(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> NDArray[np.float32]:
        """Optimal query: q* = v₁ (dominant eigenvector). Maximises qᵀFq. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        if eig:
            v1 = eig.eigenvectors[:, 0]
        else:
            vals, vecs = np.linalg.eigh(field.F[band])
            v1 = vecs[:, np.argmax(np.abs(vals))]
        # Build full (B, D) query with v1 in the target band
        q = np.zeros((field.bands, field.dim), dtype=np.float32)
        q[band] = v1 / (np.linalg.norm(v1) + 1e-12)
        return q

    @staticmethod
    def energy(field: DenseField, query: NDArray[np.float32], band: int = 0) -> Scalar:
        """Query energy: E = qᵀ F_b q. Cost: O(D²)."""
        e = float(query[band] @ field.F[band] @ query[band])
        return Scalar(e, name="query_energy", band=band)

    @staticmethod
    def energy_all_bands(field: DenseField, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Energy per band: E_b = q_bᵀ F_b q_b. Cost: O(BD²)."""
        return np.array([float(query[b] @ field.F[b] @ query[b]) for b in range(field.bands)], dtype=np.float32)

    @staticmethod
    def gradient_at(field: DenseField, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Gradient: ∇E = F @ q (same as resonance, explicit name). Cost: O(BD²)."""
        grad = np.zeros_like(query)
        for b in range(field.bands):
            grad[b] = field.F[b] @ query[b]
        return grad

    @staticmethod
    def steepest_ascent(field: DenseField, query: NDArray[np.float32], alpha: float = 0.1) -> NDArray[np.float32]:
        """One step of steepest ascent: q' = normalise(q + α · F @ q). Cost: O(BD²)."""
        q = query.copy()
        for b in range(field.bands):
            g = field.F[b] @ q[b]
            q[b] = q[b] + alpha * g
            norm = np.linalg.norm(q[b])
            if norm > 1e-8:
                q[b] /= norm
        return q

    @staticmethod
    def probe(field: DenseField, query: NDArray[np.float32], threshold: float = 0.1) -> bool:
        """Probe: does the field 'know' about this query? (energy > threshold). Cost: O(BD²)."""
        total_energy = sum(float(query[b] @ field.F[b] @ query[b]) for b in range(field.bands))
        return total_energy > threshold

    @staticmethod
    def similarity_under_field(field: DenseField, q_a: NDArray[np.float32], q_b: NDArray[np.float32], band: int = 0) -> Scalar:
        """Field-weighted similarity: q_aᵀ F_b q_b. Cost: O(D²)."""
        s = float(q_a[band] @ field.F[band] @ q_b[band])
        return Scalar(s, name="field_similarity", band=band)

    @staticmethod
    def resonate_regularised(field: DenseField, query: NDArray[np.float32], lam: float = 0.01) -> NDArray[np.float32]:
        """Tikhonov-regularised resonance: r = (F + λI)⁻¹ F q. Cost: O(BD³)."""
        result = np.zeros_like(query)
        for b in range(field.bands):
            F_reg = field.F[b] + lam * np.eye(field.dim, dtype=np.float32)
            Fq = field.F[b] @ query[b]
            result[b] = np.linalg.solve(F_reg, Fq)
        return result

    @staticmethod
    def inverse_query(field: DenseField, target_resonance: NDArray[np.float32], band: int = 0, lam: float = 0.01) -> NDArray[np.float32]:
        """Inverse query: find q such that F @ q ≈ r (regularised least squares). Cost: O(D³)."""
        F_reg = field.F[band] + lam * np.eye(field.dim, dtype=np.float32)
        q = np.linalg.solve(F_reg, target_resonance)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q /= norm
        return q.astype(np.float32)

    @staticmethod
    def diversified_resonate(field: DenseField, query: NDArray[np.float32], n_diverse: int = 3, band: int = 0) -> list[NDArray[np.float32]]:
        """Diversified resonance: resonate, deflate, repeat. Returns n_diverse orthogonal resonances. Cost: O(nBD²)."""
        results = []
        F_temp = field.F[band].copy()
        q = query[band]

        for _ in range(n_diverse):
            r = F_temp @ q
            results.append(r.astype(np.float32))
            # Deflate: remove the component along r
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-8:
                r_hat = r / r_norm
                F_temp -= np.outer(F_temp @ r_hat, r_hat)

        return results

    @staticmethod
    def uncertainty_resonate(field: DenseField, query: NDArray[np.float32], band: int = 0) -> tuple[NDArray[np.float32], float]:
        """Resonate with uncertainty: return (r, σ) where σ = 1/√(qᵀFq). Cost: O(D²)."""
        r = field.F[band] @ query[band]
        energy = float(query[band] @ field.F[band] @ query[band])
        sigma = 1.0 / np.sqrt(max(energy, 1e-12))
        return r.astype(np.float32), sigma

    @staticmethod
    def anti_resonate(field: DenseField, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Anti-resonate: what the field does NOT know. r = (I - F/||F||) @ q. Cost: O(BD²)."""
        result = np.zeros_like(query)
        for b in range(field.bands):
            F_norm = np.linalg.norm(field.F[b], "fro") + 1e-12
            result[b] = query[b] - (field.F[b] @ query[b]) / F_norm
        return result
