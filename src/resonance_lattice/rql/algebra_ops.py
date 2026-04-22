# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain B: Algebraic Operations.

Arithmetic, decompositions, matrix functions, norms, projections.
Extends the existing algebra.py with additional operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.spectral import _apply_to_eigenvalues
from resonance_lattice.rql.types import EigenDecomp, Scalar


class AlgebraOps:

    # ── Arithmetic ───────────────────────────────────────

    @staticmethod
    def add(a: DenseField, b: DenseField) -> DenseField:
        """Field addition: F_a + F_b. PSD-preserving. Cost: O(BD²)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result.F = a.F + b.F
        result._source_count = a.source_count + b.source_count
        return result

    @staticmethod
    def subtract(a: DenseField, b: DenseField) -> DenseField:
        """Field subtraction: F_a - F_b. NOT PSD-preserving. Cost: O(BD²)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result.F = a.F - b.F
        result._source_count = abs(a.source_count - b.source_count)
        return result

    @staticmethod
    def scale(field: DenseField, alpha: float) -> DenseField:
        """Scale: α·F. PSD-preserving for α ≥ 0. Cost: O(BD²)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = alpha * field.F
        result._source_count = field.source_count
        return result

    @staticmethod
    def hadamard(a: DenseField, b: DenseField) -> DenseField:
        """Hadamard (element-wise) product: F_a ⊙ F_b. Cost: O(BD²)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result.F = a.F * b.F
        result._source_count = max(a.source_count, b.source_count)
        return result

    @staticmethod
    def weighted_sum(fields: list[DenseField], weights: list[float] | None = None) -> DenseField:
        """Weighted sum: Σ w_i F_i. PSD-preserving for w_i ≥ 0. Cost: O(NBD²)."""
        if not fields:
            raise ValueError("Empty field list")
        if weights is None:
            weights = [1.0 / len(fields)] * len(fields)
        result = DenseField(bands=fields[0].bands, dim=fields[0].dim)
        for f, w in zip(fields, weights):
            result.F += w * f.F
        result._source_count = sum(f.source_count for f in fields)
        return result

    @staticmethod
    def lerp(a: DenseField, b: DenseField, t: float) -> DenseField:
        """Linear interpolation: (1-t)·F_a + t·F_b. PSD-preserving for t ∈ [0,1]. Cost: O(BD²)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result.F = (1 - t) * a.F + t * b.F
        result._source_count = int((1 - t) * a.source_count + t * b.source_count)
        return result

    # ── Decompositions ───────────────────────────────────

    @staticmethod
    def eigen(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> EigenDecomp:
        """Eigendecomposition of band. Cost: O(D³)."""
        if cache:
            return cache.get(band)
        F_sym = (field.F[band] + field.F[band].T) / 2.0
        vals, vecs = np.linalg.eigh(F_sym)
        idx = np.argsort(np.abs(vals))[::-1]
        return EigenDecomp(vals[idx].astype(np.float32), vecs[:, idx].astype(np.float32), band)

    @staticmethod
    def svd(field: DenseField, band: int = 0) -> tuple[NDArray, NDArray, NDArray]:
        """SVD: F_b = U S Vᵀ. Cost: O(D³)."""
        U, S, Vt = np.linalg.svd(field.F[band], full_matrices=False)
        return U.astype(np.float32), S.astype(np.float32), Vt.astype(np.float32)

    @staticmethod
    def low_rank_approx(field: DenseField, k: int) -> DenseField:
        """Best rank-k approximation via SVD. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count
        for b in range(field.bands):
            U, S, Vt = np.linalg.svd(field.F[b], full_matrices=False)
            result.F[b] = ((U[:, :k] * S[:k]) @ Vt[:k, :]).astype(np.float32)
        return result

    # ── Matrix Functions ─────────────────────────────────

    @staticmethod
    def matrix_sqrt(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix square root: F^½. PSD-preserving. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.sqrt(np.maximum(lam, 0)), cache)

    @staticmethod
    def matrix_power(field: DenseField, p: float, cache: SpectralCache | None = None) -> DenseField:
        """Matrix power: F^p. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.sign(lam) * np.abs(lam) ** p, cache)

    # ── Norms ────────────────────────────────────────────

    @staticmethod
    def frobenius_norm(field: DenseField) -> Scalar:
        """Frobenius norm: ||F||_F = √(Σ F_ij²). Cost: O(BD²)."""
        return Scalar(float(np.linalg.norm(field.F)), name="frobenius_norm")

    @staticmethod
    def spectral_norm(field: DenseField, band: int = 0) -> Scalar:
        """Spectral norm: λ_max(F_b). Cost: O(D³)."""
        return Scalar(float(np.linalg.norm(field.F[band], 2)), name="spectral_norm", band=band)

    @staticmethod
    def nuclear_norm(field: DenseField, band: int = 0) -> Scalar:
        """Nuclear norm: Σ σ_i (sum of singular values). Cost: O(D³)."""
        S = np.linalg.svd(field.F[band], compute_uv=False)
        return Scalar(float(np.sum(S)), name="nuclear_norm", band=band)

    # ── Projections ──────────────────────────────────────

    @staticmethod
    def sandwich(field: DenseField, A: NDArray[np.float32]) -> DenseField:
        """Congruence transform: F' = A F Aᵀ. PSD-preserving. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count
        for b in range(field.bands):
            result.F[b] = (A @ field.F[b] @ A.T).astype(np.float32)
        return result

    @staticmethod
    def commutator(a: DenseField, b: DenseField) -> DenseField:
        """Commutator: [A, B] = AB - BA. NOT PSD-preserving. Anti-symmetric. Cost: O(BD³)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            result.F[bi] = (a.F[bi] @ b.F[bi] - b.F[bi] @ a.F[bi]).astype(np.float32)
        return result

    @staticmethod
    def anticommutator(a: DenseField, b: DenseField) -> DenseField:
        """Anti-commutator: {A, B} = AB + BA. PSD-preserving. Cost: O(BD³)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            result.F[bi] = (a.F[bi] @ b.F[bi] + b.F[bi] @ a.F[bi]).astype(np.float32)
        return result

    @staticmethod
    def jordan_product(a: DenseField, b: DenseField) -> DenseField:
        """Jordan product: (AB + BA) / 2. PSD-preserving symmetric product. Cost: O(BD³)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            result.F[bi] = ((a.F[bi] @ b.F[bi] + b.F[bi] @ a.F[bi]) / 2).astype(np.float32)
        return result

    @staticmethod
    def matrix_exp(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix exponential: exp(F). Always PSD. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, np.exp, cache)

    @staticmethod
    def matrix_log(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix logarithm: log(F). Requires PSD (λ > 0). Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.log(np.maximum(lam, 1e-12)), cache)

    @staticmethod
    def matmul(a: DenseField, b: DenseField) -> DenseField:
        """Matrix product: A @ B per band. NOT PSD-preserving. Cost: O(BD³)."""
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            result.F[bi] = (a.F[bi] @ b.F[bi]).astype(np.float32)
        return result

    @staticmethod
    def project_psd(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Project onto PSD cone: clamp negative eigenvalues to 0. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.maximum(lam, 0), cache)
