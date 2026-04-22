# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain A: Spectral Operations.

Operations on the eigenvalue/eigenvector structure of field tensors.
All operations use SpectralCache to avoid redundant eigendecompositions.

Functional calculus: apply any f to eigenvalues → F' = V diag(f(λ)) Vᵀ
Filtering: threshold, bandpass, top-k, decay
Normalization: trace-normalize, spectral-normalize, whiten
Metrics: effective rank, entropy, gap, condition number, trace
Extraction: top-k, residual, deflate, split positive/negative
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.types import Scalar, Spectrum


def _apply_to_eigenvalues(
    field: DenseField,
    fn: Callable[[NDArray], NDArray],
    cache: SpectralCache | None = None,
    bands: list[int] | None = None,
) -> DenseField:
    """Core helper: apply a function to eigenvalues and reconstruct."""
    result = DenseField(bands=field.bands, dim=field.dim)
    result.F = field.F.copy()
    result._source_count = field.source_count

    for b in (bands or range(field.bands)):
        if cache:
            eig = cache.get(b)
            eigenvalues, eigenvectors = eig.eigenvalues.copy(), eig.eigenvectors
        else:
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        new_eigs = fn(eigenvalues).astype(np.float32)
        result.F[b] = (eigenvectors * new_eigs) @ eigenvectors.T

    return result


class SpectralOps:
    """Spectral operations on field tensors. All methods are static."""

    # ── Functional Calculus ──────────────────────────────

    @staticmethod
    def apply_fn(field: DenseField, fn: Callable[[NDArray], NDArray], cache: SpectralCache | None = None) -> DenseField:
        """Apply arbitrary function f to eigenvalues: F' = V diag(f(λ)) Vᵀ. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, fn, cache)

    @staticmethod
    def power(field: DenseField, p: float, cache: SpectralCache | None = None) -> DenseField:
        """Matrix power: F^p = V diag(λ^p) Vᵀ. Cost: O(BD³). Preserves PSD for p > 0."""
        return _apply_to_eigenvalues(field, lambda lam: np.sign(lam) * np.abs(lam) ** p, cache)

    @staticmethod
    def exp(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix exponential: exp(F) = V diag(exp(λ)) Vᵀ. Cost: O(BD³). Always PSD."""
        return _apply_to_eigenvalues(field, np.exp, cache)

    @staticmethod
    def log(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix logarithm: log(F) = V diag(log(λ)) Vᵀ. Cost: O(BD³). Requires PSD (λ > 0)."""
        return _apply_to_eigenvalues(field, lambda lam: np.log(np.maximum(lam, 1e-12)), cache)

    @staticmethod
    def sqrt(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Matrix square root: F^½ = V diag(√λ) Vᵀ. Cost: O(BD³). Preserves PSD."""
        return _apply_to_eigenvalues(field, lambda lam: np.sqrt(np.maximum(lam, 0)), cache)

    @staticmethod
    def inv(field: DenseField, epsilon: float = 1e-8, cache: SpectralCache | None = None) -> DenseField:
        """Pseudoinverse: F⁻¹ = V diag(1/λ) Vᵀ (regularised). Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.where(np.abs(lam) > epsilon, 1.0 / lam, 0.0), cache)

    # ── Filtering ────────────────────────────────────────

    @staticmethod
    def threshold_hard(field: DenseField, t: float, cache: SpectralCache | None = None) -> DenseField:
        """Hard threshold: λ' = λ · (|λ| > t). Cost: O(BD³). Idempotent."""
        return _apply_to_eigenvalues(field, lambda lam: lam * (np.abs(lam) > t), cache)

    @staticmethod
    def threshold_soft(field: DenseField, t: float, cache: SpectralCache | None = None) -> DenseField:
        """Soft threshold: λ' = sign(λ) · max(|λ| - t, 0). Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: np.sign(lam) * np.maximum(np.abs(lam) - t, 0), cache)

    @staticmethod
    def bandpass(field: DenseField, lo: float, hi: float, cache: SpectralCache | None = None) -> DenseField:
        """Bandpass: keep eigenvalues in [lo, hi], zero others. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: lam * ((np.abs(lam) >= lo) & (np.abs(lam) <= hi)), cache)

    @staticmethod
    def top_k(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """Keep top-k eigenvalues by magnitude, zero the rest. Cost: O(BD³)."""
        def _top_k(lam):
            if k >= len(lam):
                return lam
            threshold = np.sort(np.abs(lam))[::-1][k - 1] if k > 0 else float("inf")
            return lam * (np.abs(lam) >= threshold)
        return _apply_to_eigenvalues(field, _top_k, cache)

    @staticmethod
    def power_law_decay(field: DenseField, alpha: float = 1.0, cache: SpectralCache | None = None) -> DenseField:
        """Decay eigenvalues by rank: λ'_i = λ_i · (1 + rank_i)^{-α}. Cost: O(BD³)."""
        def _decay(lam):
            ranks = np.arange(len(lam), dtype=np.float32)
            return lam * (1 + ranks) ** (-alpha)
        return _apply_to_eigenvalues(field, _decay, cache)

    @staticmethod
    def exponential_decay(field: DenseField, alpha: float = 0.1, cache: SpectralCache | None = None) -> DenseField:
        """Exponential decay by rank: λ'_i = λ_i · exp(-α · rank_i). Cost: O(BD³)."""
        def _decay(lam):
            ranks = np.arange(len(lam), dtype=np.float32)
            return lam * np.exp(-alpha * ranks)
        return _apply_to_eigenvalues(field, _decay, cache)

    # ── Normalisation ────────────────────────────────────

    @staticmethod
    def normalize_trace(field: DenseField) -> DenseField:
        """Trace-normalise: F' = F / tr(F). Makes field a density matrix. Cost: O(BD)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count
        for b in range(field.bands):
            tr = np.trace(field.F[b])
            result.F[b] = field.F[b] / (tr + 1e-12)
        return result

    @staticmethod
    def normalize_spectral(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Spectral-normalise: F' = F / λ_max. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        for b in range(field.bands):
            if cache:
                lam_max = float(np.abs(cache.get(b).eigenvalues[0]))
            else:
                lam_max = float(np.linalg.norm(field.F[b], 2))
            if lam_max > 1e-12:
                result.F[b] /= lam_max
        return result

    # ── Metrics ──────────────────────────────────────────

    @staticmethod
    def effective_rank(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Effective rank: exp(H(|λ|/Σ|λ|)). Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        p = lam / (np.sum(lam) + 1e-12)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        return Scalar(float(np.exp(entropy)), name="effective_rank", band=band)

    @staticmethod
    def entropy(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Spectral entropy: H(|λ|/Σ|λ|) normalised to [0,1]. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        p = lam / (np.sum(lam) + 1e-12)
        p = p[p > 1e-12]
        H = -np.sum(p * np.log(p))
        H_max = np.log(len(p)) if len(p) > 1 else 1.0
        return Scalar(float(H / H_max) if H_max > 0 else 0.0, name="spectral_entropy", band=band)

    @staticmethod
    def spectral_gap(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Spectral gap: λ₁ - λ₂. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        gap = float(lam[0] - lam[1]) if len(lam) > 1 else float(lam[0])
        return Scalar(gap, name="spectral_gap", band=band)

    @staticmethod
    def condition_number(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Condition number: λ_max / λ_min(nonzero). Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        nonzero = lam[lam > 1e-12]
        cond = float(nonzero[0] / nonzero[-1]) if len(nonzero) > 0 else 0.0
        return Scalar(cond, name="condition_number", band=band)

    @staticmethod
    def trace(field: DenseField, band: int = 0) -> Scalar:
        """Trace: tr(F_b) = Σλ. Cost: O(D)."""
        return Scalar(float(np.trace(field.F[band])), name="trace", band=band)

    @staticmethod
    def participation_ratio(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Participation ratio: (Σλ)² / Σ(λ²). Measures eigenvalue spread. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        s1 = np.sum(lam)
        s2 = np.sum(lam ** 2)
        return Scalar(float(s1 ** 2 / (s2 + 1e-12)), name="participation_ratio", band=band)

    @staticmethod
    def numerical_rank(field: DenseField, band: int = 0, epsilon: float = 1e-6, cache: SpectralCache | None = None) -> Scalar:
        """Numerical rank: count(|λ| > ε). Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.abs(np.linalg.eigvalsh(field.F[band]))
        return Scalar(float(np.sum(lam > epsilon)), name="numerical_rank", band=band)

    @staticmethod
    def spectrum(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Spectrum:
        """Return sorted eigenvalue spectrum. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        if eig:
            return Spectrum(values=eig.eigenvalues.copy(), band=band)
        lam = np.linalg.eigvalsh(field.F[band])
        return Spectrum(values=np.sort(lam)[::-1].astype(np.float32), band=band)

    # ── Extraction ───────────────────────────────────────

    @staticmethod
    def extract_top_k(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """Extract top-k eigencomponents: V_k diag(λ_k) V_kᵀ. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count
        for b in range(field.bands):
            eig = cache.get(b) if cache else None
            if eig:
                vals, vecs = eig.eigenvalues, eig.eigenvectors
            else:
                F_sym = (field.F[b] + field.F[b].T) / 2.0
                vals, vecs = np.linalg.eigh(F_sym)
                idx = np.argsort(np.abs(vals))[::-1]
                vals, vecs = vals[idx], vecs[:, idx]
            k_actual = min(k, len(vals))
            V_k = vecs[:, :k_actual]
            L_k = vals[:k_actual]
            result.F[b] = ((V_k * L_k) @ V_k.T).astype(np.float32)
        return result

    @staticmethod
    def extract_residual(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """Extract residual after removing top-k: F - extract_top_k(F, k). Cost: O(BD³)."""
        top = SpectralOps.extract_top_k(field, k, cache)
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F - top.F
        result._source_count = field.source_count
        return result

    @staticmethod
    def deflate(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Remove dominant eigenmode: F' = F - λ₁v₁v₁ᵀ. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        for b in range(field.bands):
            eig = cache.get(b) if cache else None
            if eig:
                v1, l1 = eig.eigenvectors[:, 0], eig.eigenvalues[0]
            else:
                F_sym = (field.F[b] + field.F[b].T) / 2.0
                vals, vecs = np.linalg.eigh(F_sym)
                i = np.argmax(np.abs(vals))
                v1, l1 = vecs[:, i], vals[i]
            result.F[b] -= l1 * np.outer(v1, v1)
        return result

    @staticmethod
    def softmax(field: DenseField, temperature: float = 1.0, cache: SpectralCache | None = None) -> DenseField:
        """Spectral softmax: λ' = softmax(λ/T) · Σ|λ|. Redistributes energy via softmax. Cost: O(BD³)."""
        def _softmax(lam):
            scaled = lam / (temperature + 1e-12)
            scaled -= np.max(scaled)
            exp_s = np.exp(scaled)
            probs = exp_s / (np.sum(exp_s) + 1e-12)
            return probs * np.sum(np.abs(lam))
        return _apply_to_eigenvalues(field, _softmax, cache)

    @staticmethod
    def whiten(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Whitening: F^{-½} (decorrelate). Cost: O(BD³)."""
        return _apply_to_eigenvalues(
            field,
            lambda lam: np.where(np.abs(lam) > 1e-8, 1.0 / np.sqrt(np.abs(lam)), 0.0),
            cache,
        )

    @staticmethod
    def determinant(field: DenseField, band: int = 0) -> Scalar:
        """Determinant: det(F_b) = Π λ. Cost: O(D³)."""
        sign, logdet = np.linalg.slogdet(field.F[band])
        return Scalar(float(sign * np.exp(logdet)), name="determinant", band=band)

    @staticmethod
    def bottom_k(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """Keep bottom-k eigenvalues (noise/rare components). Cost: O(BD³)."""
        def _bottom_k(lam):
            if k >= len(lam):
                return lam
            abs_lam = np.abs(lam)
            threshold = np.sort(abs_lam)[k - 1] if k > 0 else 0
            return lam * (abs_lam <= threshold)
        return _apply_to_eigenvalues(field, _bottom_k, cache)

    @staticmethod
    def standardize(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Standardise eigenvalues: λ' = (λ - μ) / σ. Cost: O(BD³)."""
        def _std(lam):
            mu = np.mean(lam)
            sigma = np.std(lam) + 1e-12
            return (lam - mu) / sigma
        return _apply_to_eigenvalues(field, _std, cache)

    @staticmethod
    def split_pos_neg(field: DenseField, cache: SpectralCache | None = None) -> tuple[DenseField, DenseField]:
        """Split into positive and negative eigenvalue components. Cost: O(BD³)."""
        pos = DenseField(bands=field.bands, dim=field.dim)
        neg = DenseField(bands=field.bands, dim=field.dim)
        pos._source_count = field.source_count
        neg._source_count = field.source_count
        for b in range(field.bands):
            eig = cache.get(b) if cache else None
            if eig:
                vals, vecs = eig.eigenvalues, eig.eigenvectors
            else:
                F_sym = (field.F[b] + field.F[b].T) / 2.0
                vals, vecs = np.linalg.eigh(F_sym)
            pos_vals = np.maximum(vals, 0)
            neg_vals = np.minimum(vals, 0)
            pos.F[b] = ((vecs * pos_vals) @ vecs.T).astype(np.float32)
            neg.F[b] = ((vecs * neg_vals) @ vecs.T).astype(np.float32)
        return pos, neg
