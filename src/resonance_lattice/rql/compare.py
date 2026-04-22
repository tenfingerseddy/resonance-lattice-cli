# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain H: Comparison Operations.

Distances, similarities, and alignments between fields.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.types import Scalar


class CompareOps:

    @staticmethod
    def frobenius_distance(a: DenseField, b: DenseField) -> Scalar:
        """Frobenius distance: ||F_a - F_b||_F. Cost: O(BD²)."""
        d = float(np.linalg.norm(a.F - b.F))
        return Scalar(d, name="frobenius_distance")

    @staticmethod
    def spectral_distance(a: DenseField, b: DenseField, band: int = 0, cache_a: SpectralCache | None = None, cache_b: SpectralCache | None = None) -> Scalar:
        """Spectral distance: ||λ_a - λ_b||₂. Cost: O(BD³)."""
        lam_a = cache_a.get(band).eigenvalues if cache_a else np.sort(np.linalg.eigvalsh(a.F[band]))[::-1]
        lam_b = cache_b.get(band).eigenvalues if cache_b else np.sort(np.linalg.eigvalsh(b.F[band]))[::-1]
        return Scalar(float(np.linalg.norm(lam_a - lam_b)), name="spectral_distance", band=band)

    @staticmethod
    def cosine_similarity(a: DenseField, b: DenseField) -> Scalar:
        """Cosine similarity: ⟨F_a, F_b⟩_F / (||F_a|| ||F_b||). Cost: O(BD²)."""
        inner = float(np.sum(a.F * b.F))
        norm_a = float(np.linalg.norm(a.F))
        norm_b = float(np.linalg.norm(b.F))
        return Scalar(inner / (norm_a * norm_b + 1e-12), name="cosine_similarity")

    @staticmethod
    def relative_change(new: DenseField, old: DenseField) -> Scalar:
        """Relative change: ||F_new - F_old||_F / ||F_old||_F. Cost: O(BD²)."""
        diff_norm = float(np.linalg.norm(new.F - old.F))
        old_norm = float(np.linalg.norm(old.F))
        return Scalar(diff_norm / (old_norm + 1e-12), name="relative_change")

    @staticmethod
    def band_similarity_matrix(a: DenseField, b: DenseField) -> NDArray[np.float32]:
        """Per-band cosine similarity matrix: S[i,j] = cos(F_a_i, F_b_j). Cost: O(B²D²)."""
        Ba, Bb = a.bands, b.bands
        S = np.zeros((Ba, Bb), dtype=np.float32)
        for i in range(Ba):
            ni = np.linalg.norm(a.F[i], "fro") + 1e-12
            for j in range(Bb):
                nj = np.linalg.norm(b.F[j], "fro") + 1e-12
                S[i, j] = float(np.sum(a.F[i] * b.F[j])) / (ni * nj)
        return S

    @staticmethod
    def energy_ratio(a: DenseField, b: DenseField) -> Scalar:
        """Energy ratio: Σ||F_a|| / Σ||F_b||. Cost: O(BD)."""
        ea = float(np.sum(a.energy()))
        eb = float(np.sum(b.energy()))
        return Scalar(ea / (eb + 1e-12), name="energy_ratio")

    @staticmethod
    def max_eigenvalue_ratio(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Max eigenvalue ratio: λ_max(A) / λ_max(B). Cost: O(BD³)."""
        la = float(np.max(np.abs(np.linalg.eigvalsh(a.F[band]))))
        lb = float(np.max(np.abs(np.linalg.eigvalsh(b.F[band]))))
        return Scalar(la / (lb + 1e-12), name="max_eigenvalue_ratio", band=band)

    @staticmethod
    def wasserstein_spectral(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Wasserstein-1 distance on eigenvalue distributions (sorted, 1D OT). Cost: O(D³)."""
        lam_a = np.sort(np.linalg.eigvalsh(a.F[band]))
        lam_b = np.sort(np.linalg.eigvalsh(b.F[band]))
        return Scalar(float(np.sum(np.abs(lam_a - lam_b))), name="wasserstein_spectral", band=band)

    @staticmethod
    def trace_distance(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Trace distance: ||ρ_a - ρ_b||₁ / 2 (nuclear norm). Cost: O(D³)."""
        def _density(F):
            tr = np.trace(F) + 1e-12
            return F / tr
        diff = _density(a.F[band]) - _density(b.F[band])
        S = np.linalg.svd(diff, compute_uv=False)
        return Scalar(float(np.sum(S)) / 2, name="trace_distance", band=band)

    @staticmethod
    def rank_correlation(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Spearman rank correlation of eigenvalue orderings. Cost: O(D³)."""
        lam_a = np.linalg.eigvalsh(a.F[band])
        lam_b = np.linalg.eigvalsh(b.F[band])
        rank_a = np.argsort(np.argsort(lam_a))
        rank_b = np.argsort(np.argsort(lam_b))
        n = len(rank_a)
        d_sq = np.sum((rank_a - rank_b) ** 2)
        rho = 1 - 6 * d_sq / (n * (n ** 2 - 1) + 1e-12)
        return Scalar(float(rho), name="rank_correlation", band=band)

    @staticmethod
    def intersection_over_union(a: DenseField, b: DenseField, band: int = 0, k: int = 20) -> Scalar:
        """IoU of top-k eigenvalue indices. Cost: O(D³)."""
        lam_a = np.linalg.eigvalsh(a.F[band])
        lam_b = np.linalg.eigvalsh(b.F[band])
        top_a = set(np.argsort(np.abs(lam_a))[-k:])
        top_b = set(np.argsort(np.abs(lam_b))[-k:])
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)
        return Scalar(intersection / (union + 1e-12), name="iou_top_k", band=band)

    @staticmethod
    def hellinger_distance(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Hellinger distance: √(1 - F(ρ_a, ρ_b)) where F is fidelity. Cost: O(D³)."""
        from resonance_lattice.rql.info import InfoOps
        fid = InfoOps.fidelity(a, b, band, epsilon).value
        return Scalar(float(np.sqrt(max(0, 1 - fid))), name="hellinger_distance", band=band)

    @staticmethod
    def symmetrized_kl(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Symmetrised KL: (KL(a||b) + KL(b||a)) / 2. Cost: O(D³)."""
        from resonance_lattice.rql.info import InfoOps
        kl_ab = InfoOps.kl_divergence(a, b, band).value
        kl_ba = InfoOps.kl_divergence(b, a, band).value
        return Scalar((kl_ab + kl_ba) / 2, name="symmetrized_kl", band=band)

    @staticmethod
    def edit_distance(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Approximate edit distance: minimum rank-1 updates to transform A to B. Cost: O(D³).

        Estimated as the nuclear norm of the difference (sum of singular values of A-B).
        """
        diff = a.F[band] - b.F[band]
        S = np.linalg.svd(diff, compute_uv=False)
        return Scalar(float(np.sum(S)), name="edit_distance", band=band)

    @staticmethod
    def drift_velocity(new: DenseField, old: DenseField, dt: float = 1.0) -> Scalar:
        """Drift velocity: ||F_new - F_old||_F / dt. Cost: O(BD²)."""
        d = float(np.linalg.norm(new.F - old.F))
        return Scalar(d / dt, name="drift_velocity")
