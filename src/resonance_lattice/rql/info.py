# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain C: Information-Theoretic Operations.

Entropy, divergence, capacity, purity — treating the field as a density matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.types import Scalar


class InfoOps:

    @staticmethod
    def von_neumann_entropy(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Von Neumann entropy: S = -tr(ρ log ρ) where ρ = F/tr(F). Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        total = np.sum(lam) + 1e-12
        p = lam / total
        p = p[p > 1e-12]
        S = -float(np.sum(p * np.log(p)))
        return Scalar(S, name="von_neumann_entropy", band=band)

    @staticmethod
    def channel_capacity(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Channel capacity: λ_max (maximum achievable resonance energy). Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        return Scalar(float(np.max(lam)), name="channel_capacity", band=band)

    @staticmethod
    def purity(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Purity: tr(ρ²). 1.0 = pure state (rank-1). 1/D = maximally mixed. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        total = np.sum(lam) + 1e-12
        p = lam / total
        return Scalar(float(np.sum(p ** 2)), name="purity", band=band)

    @staticmethod
    def mixedness(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Mixedness: 1 - purity. 0.0 = pure, ~1.0 = maximally mixed. Cost: O(BD³)."""
        p = InfoOps.purity(field, band, cache)
        return Scalar(1.0 - p.value, name="mixedness", band=band)

    @staticmethod
    def kl_divergence(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """KL divergence: D_KL(ρ_a || ρ_b) = tr(ρ_a (log ρ_a - log ρ_b)). Cost: O(D³)."""
        def _density(F, eps):
            F_sym = (F + F.T) / 2 + eps * np.eye(F.shape[0])
            tr = np.trace(F_sym)
            return F_sym / (tr + 1e-12)

        rho_a = _density(a.F[band], epsilon)
        rho_b = _density(b.F[band], epsilon)

        vals_a, vecs_a = np.linalg.eigh(rho_a)
        vals_a = np.maximum(vals_a, epsilon)
        log_rho_a = (vecs_a * np.log(vals_a)) @ vecs_a.T

        vals_b, vecs_b = np.linalg.eigh(rho_b)
        vals_b = np.maximum(vals_b, epsilon)
        log_rho_b = (vecs_b * np.log(vals_b)) @ vecs_b.T

        kl = float(np.trace(rho_a @ (log_rho_a - log_rho_b)))
        return Scalar(max(0.0, kl), name="kl_divergence", band=band)

    @staticmethod
    def js_divergence(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Jensen-Shannon divergence: (KL(a||m) + KL(b||m)) / 2, m = (a+b)/2. Cost: O(D³)."""
        mid = DenseField(bands=a.bands, dim=a.dim)
        mid.F = (a.F + b.F) / 2
        kl_am = InfoOps.kl_divergence(a, mid, band).value
        kl_bm = InfoOps.kl_divergence(b, mid, band).value
        return Scalar((kl_am + kl_bm) / 2, name="js_divergence", band=band)

    @staticmethod
    def mutual_information_bands(field: DenseField, band_a: int = 0, band_b: int = 1) -> Scalar:
        """Mutual information between two bands: I(a;b) = S(a) + S(b) - S(a,b). Cost: O(D³)."""
        S_a = InfoOps.von_neumann_entropy(field, band_a).value
        S_b = InfoOps.von_neumann_entropy(field, band_b).value
        # Joint entropy approximated via concatenated eigenvalues
        lam_a = np.abs(np.linalg.eigvalsh(field.F[band_a]))
        lam_b = np.abs(np.linalg.eigvalsh(field.F[band_b]))
        lam_joint = np.concatenate([lam_a, lam_b])
        total = np.sum(lam_joint) + 1e-12
        p = lam_joint / total
        p = p[p > 1e-12]
        S_joint = -float(np.sum(p * np.log(p)))
        mi = max(0.0, S_a + S_b - S_joint)
        return Scalar(mi, name="mutual_information")

    @staticmethod
    def information_content(field: DenseField, band: int = 0, alpha: float = 1.0) -> Scalar:
        """Information content: log₂(det(I + αF)). Cost: O(D³)."""
        F_b = field.F[band]
        M = np.eye(field.dim, dtype=np.float32) + alpha * F_b
        sign, logdet = np.linalg.slogdet(M)
        return Scalar(float(logdet / np.log(2)) if sign > 0 else 0.0, name="information_content", band=band)

    @staticmethod
    def redundancy(field: DenseField, band: int = 0) -> Scalar:
        """Redundancy: 1 - S(F)/log(D). How far from maximally mixed. Cost: O(D³)."""
        S = InfoOps.von_neumann_entropy(field, band).value
        max_S = np.log(field.dim)
        return Scalar(1.0 - S / max_S if max_S > 0 else 0.0, name="redundancy", band=band)

    @staticmethod
    def cross_entropy(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Cross entropy: H(a, b) = -tr(ρ_a log ρ_b). Cost: O(D³)."""
        def _density(F, eps):
            F_sym = (F + F.T) / 2 + eps * np.eye(F.shape[0])
            return F_sym / (np.trace(F_sym) + 1e-12)

        rho_a = _density(a.F[band], epsilon)
        rho_b = _density(b.F[band], epsilon)

        vals_b, vecs_b = np.linalg.eigh(rho_b)
        vals_b = np.maximum(vals_b, epsilon)
        log_rho_b = (vecs_b * np.log(vals_b)) @ vecs_b.T

        return Scalar(-float(np.trace(rho_a @ log_rho_b)), name="cross_entropy", band=band)

    @staticmethod
    def log_determinant(field: DenseField, band: int = 0) -> Scalar:
        """Log-determinant: log(det(F_b)). Cost: O(D³)."""
        sign, logdet = np.linalg.slogdet(field.F[band])
        return Scalar(float(logdet) if sign > 0 else float("-inf"), name="log_determinant", band=band)

    @staticmethod
    def renyi_entropy(field: DenseField, alpha: float = 2.0, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Rényi entropy: S_α = (1/(1-α)) log(tr(ρ^α)). α=2 gives collision entropy. Cost: O(D³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        total = np.sum(lam) + 1e-12
        p = lam / total
        p = p[p > 1e-12]
        if abs(alpha - 1.0) < 1e-6:
            return InfoOps.von_neumann_entropy(field, band, cache)
        S = float((1 / (1 - alpha)) * np.log(np.sum(p ** alpha) + 1e-12))
        return Scalar(S, name=f"renyi_entropy_a{alpha}", band=band)

    @staticmethod
    def tsallis_entropy(field: DenseField, q: float = 2.0, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Tsallis entropy: S_q = (1 - tr(ρ^q)) / (q - 1). Non-extensive generalisation. Cost: O(D³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        total = np.sum(lam) + 1e-12
        p = lam / total
        p = p[p > 1e-12]
        if abs(q - 1.0) < 1e-6:
            return InfoOps.von_neumann_entropy(field, band, cache)
        S = float((1 - np.sum(p ** q)) / (q - 1))
        return Scalar(S, name=f"tsallis_entropy_q{q}", band=band)

    @staticmethod
    def fidelity(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Fidelity: F(ρ_a, ρ_b) = (tr(√(√ρ_a · ρ_b · √ρ_a)))². Cost: O(D³).

        1.0 = identical states, 0.0 = orthogonal states.
        """
        def _density(F, eps):
            F_sym = (F + F.T) / 2 + eps * np.eye(F.shape[0])
            return F_sym / (np.trace(F_sym) + 1e-12)

        rho_a = _density(a.F[band], epsilon)
        rho_b = _density(b.F[band], epsilon)

        # √ρ_a
        vals_a, vecs_a = np.linalg.eigh(rho_a)
        vals_a = np.maximum(vals_a, epsilon)
        sqrt_a = (vecs_a * np.sqrt(vals_a)) @ vecs_a.T

        # √ρ_a · ρ_b · √ρ_a
        M = sqrt_a @ rho_b @ sqrt_a

        # tr(√M)
        vals_m = np.linalg.eigvalsh(M)
        vals_m = np.maximum(vals_m, 0)
        tr_sqrt = float(np.sum(np.sqrt(vals_m)))

        return Scalar(tr_sqrt ** 2, name="fidelity", band=band)

    @staticmethod
    def fisher_information(field: DenseField, query: NDArray[np.float32], band: int = 0, epsilon: float = 0.01) -> Scalar:
        """Fisher information: sensitivity of resonance to query perturbation. Cost: O(D³).

        Approximated as: J = ||∂r/∂q||² ≈ ||F_b||_F² (since ∂r/∂q = F_b).
        """
        return Scalar(float(np.linalg.norm(field.F[band], "fro") ** 2), name="fisher_information", band=band)

    @staticmethod
    def band_correlation_matrix(field: DenseField) -> NDArray[np.float32]:
        """Cross-band correlation: C_ab = tr(F_a F_b) / (||F_a|| ||F_b||). Cost: O(B²D²)."""
        B = field.bands
        C = np.zeros((B, B), dtype=np.float32)
        norms = np.array([np.linalg.norm(field.F[b], "fro") for b in range(B)])
        for a in range(B):
            for b in range(a, B):
                inner = float(np.sum(field.F[a] * field.F[b]))
                denom = norms[a] * norms[b] + 1e-12
                C[a, b] = C[b, a] = inner / denom
        return C
