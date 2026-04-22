# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain G: Signal Processing Operations.

Filtering, denoising, compression, and spectral analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.types import Scalar


class SignalOps:

    @staticmethod
    def lowpass(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """Low-pass: keep top-k eigenvalues (strong signals). Cost: O(BD³)."""
        return SpectralOps.top_k(field, k, cache)

    @staticmethod
    def highpass(field: DenseField, k: int, cache: SpectralCache | None = None) -> DenseField:
        """High-pass: remove top-k, keep the rest (fine detail). Cost: O(BD³)."""
        return SpectralOps.extract_residual(field, k, cache)

    @staticmethod
    def compress(field: DenseField, k: int, cache: SpectralCache | None = None) -> tuple[DenseField, Scalar]:
        """Compress to rank-k with distortion metric. Cost: O(BD³)."""
        compressed = SpectralOps.extract_top_k(field, k, cache)
        distortion = float(np.linalg.norm(field.F - compressed.F))
        relative = distortion / (float(np.linalg.norm(field.F)) + 1e-12)
        return compressed, Scalar(relative, name="compression_distortion")

    @staticmethod
    def denoise_hard(field: DenseField, sigma: float, cache: SpectralCache | None = None) -> DenseField:
        """Hard threshold denoising at σ√(2 log D). Cost: O(BD³)."""
        threshold = sigma * np.sqrt(2 * np.log(field.dim))
        return SpectralOps.threshold_hard(field, threshold, cache)

    @staticmethod
    def denoise_soft(field: DenseField, sigma: float, cache: SpectralCache | None = None) -> DenseField:
        """Soft threshold denoising at σ√(2 log D). Cost: O(BD³)."""
        threshold = sigma * np.sqrt(2 * np.log(field.dim))
        return SpectralOps.threshold_soft(field, threshold, cache)

    @staticmethod
    def energy_compaction(field: DenseField, k: int, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Fraction of energy in top-k eigenvalues. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        total = float(np.sum(lam)) + 1e-12
        top_k = float(np.sum(lam[:k]))
        return Scalar(top_k / total, name="energy_compaction", band=band)

    @staticmethod
    def denoise_optimal(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> DenseField:
        """Optimal Marchenko-Pastur threshold denoising. Cost: O(BD³).

        Uses random matrix theory to determine the optimal threshold
        separating signal from noise eigenvalues.
        """
        from resonance_lattice.rql.stats import StatsOps
        mp = StatsOps.marchenko_pastur_fit(field, band)
        threshold = mp["lambda_plus"]
        return SpectralOps.threshold_hard(field, threshold, cache)

    @staticmethod
    def spectral_convolution(a: DenseField, b: DenseField, cache_a: SpectralCache | None = None, cache_b: SpectralCache | None = None) -> DenseField:
        """Spectral convolution: pointwise multiply eigenvalues of two fields. Cost: O(BD³).

        Requires same eigenvectors (approximately). Uses field a's eigenvectors.
        """
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            eig_a = cache_a.get(bi) if cache_a else None
            if eig_a:
                vals_a, vecs_a = eig_a.eigenvalues, eig_a.eigenvectors
            else:
                F_sym = (a.F[bi] + a.F[bi].T) / 2
                vals_a, vecs_a = np.linalg.eigh(F_sym)

            # Project b onto a's eigenvectors to get b's eigenvalues in a's basis
            vals_b = np.array([float(vecs_a[:, i] @ b.F[bi] @ vecs_a[:, i]) for i in range(a.dim)])

            # Pointwise multiply
            conv_vals = vals_a * vals_b
            result.F[bi] = ((vecs_a * conv_vals) @ vecs_a.T).astype(np.float32)
        return result

    @staticmethod
    def spectral_derivative(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> NDArray[np.float32]:
        """Finite difference on sorted eigenvalue sequence. Cost: O(D³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        return np.diff(lam).astype(np.float32)

    @staticmethod
    def noise_floor(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Estimated noise level from small eigenvalues (bottom quartile median). Cost: O(D³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        bottom_quarter = lam[len(lam) * 3 // 4:]
        return Scalar(float(np.median(bottom_quarter)) if len(bottom_quarter) > 0 else 0.0, name="noise_floor", band=band)

    @staticmethod
    def moving_average_spectrum(field: DenseField, window: int = 5, cache: SpectralCache | None = None) -> DenseField:
        """Smooth eigenvalues with moving average window. Cost: O(BD³)."""
        from resonance_lattice.rql.spectral import _apply_to_eigenvalues

        def _smooth(lam):
            kernel = np.ones(window) / window
            # Pad for edge handling
            padded = np.pad(lam, (window // 2, window // 2), mode="edge")
            return np.convolve(padded, kernel, mode="valid")[:len(lam)]

        return _apply_to_eigenvalues(field, _smooth, cache)

    @staticmethod
    def snr_estimate(field: DenseField, band: int = 0, cache: SpectralCache | None = None) -> Scalar:
        """Estimate SNR from eigenvalue distribution. Signal = top eigenvalues, noise = bulk. Cost: O(BD³)."""
        eig = cache.get(band) if cache else None
        lam = np.abs(eig.eigenvalues) if eig else np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        if len(lam) < 2:
            return Scalar(float("inf"), name="snr_estimate", band=band)
        median_lam = float(np.median(lam))
        signal = float(np.sum(lam[lam > 2 * median_lam]))
        noise = float(np.sum(lam[lam <= 2 * median_lam]))
        return Scalar(signal / (noise + 1e-12), name="snr_estimate", band=band)
