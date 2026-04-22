# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain F: Statistical Operations.

Density matrices, moments, ensembles, covariance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.types import Scalar


class StatsOps:

    @staticmethod
    def density_matrix(field: DenseField, band: int = 0) -> NDArray[np.float32]:
        """Density matrix: ρ = F_b / tr(F_b). Cost: O(D²)."""
        tr = float(np.trace(field.F[band]))
        return (field.F[band] / (tr + 1e-12)).astype(np.float32)

    @staticmethod
    def covariance_bands(field: DenseField) -> NDArray[np.float32]:
        """Cross-band covariance: Cov(b_i, b_j) = tr(F_i F_j) - tr(F_i)tr(F_j). Cost: O(B²D²)."""
        B = field.bands
        traces = np.array([np.trace(field.F[b]) for b in range(B)])
        C = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            for j in range(i, B):
                cross = float(np.sum(field.F[i] * field.F[j]))
                C[i, j] = C[j, i] = cross - traces[i] * traces[j]
        return C

    @staticmethod
    def correlation_bands(field: DenseField) -> NDArray[np.float32]:
        """Cross-band correlation (normalised covariance). Cost: O(B²D²)."""
        cov = StatsOps.covariance_bands(field)
        std = np.sqrt(np.abs(np.diag(cov))) + 1e-12
        return (cov / np.outer(std, std)).astype(np.float32)

    @staticmethod
    def ensemble_mean(fields: list[DenseField]) -> DenseField:
        """Mean of an ensemble of fields. Cost: O(NBD²)."""
        if not fields:
            raise ValueError("Empty field list")
        result = DenseField(bands=fields[0].bands, dim=fields[0].dim)
        for f in fields:
            result.F += f.F
        result.F /= len(fields)
        result._source_count = sum(f.source_count for f in fields) // len(fields)
        return result

    @staticmethod
    def energy_percentile(field: DenseField, band: int = 0, percentile: float = 0.9) -> Scalar:
        """Eigenvalue at given percentile of cumulative energy. Cost: O(BD³)."""
        lam = np.sort(np.abs(np.linalg.eigvalsh(field.F[band])))[::-1]
        cumulative = np.cumsum(lam) / (np.sum(lam) + 1e-12)
        idx = np.searchsorted(cumulative, percentile)
        val = float(lam[min(idx, len(lam) - 1)])
        return Scalar(val, name=f"energy_p{int(percentile*100)}", band=band)

    @staticmethod
    def median_eigenvalue(field: DenseField, band: int = 0) -> Scalar:
        """Median eigenvalue. Cost: O(BD³)."""
        lam = np.linalg.eigvalsh(field.F[band])
        return Scalar(float(np.median(lam)), name="median_eigenvalue", band=band)

    @staticmethod
    def skewness_spectrum(field: DenseField, band: int = 0) -> Scalar:
        """Skewness of eigenvalue distribution. Cost: O(D³)."""
        lam = np.linalg.eigvalsh(field.F[band])
        mu = np.mean(lam)
        std = np.std(lam) + 1e-12
        return Scalar(float(np.mean(((lam - mu) / std) ** 3)), name="skewness", band=band)

    @staticmethod
    def kurtosis_spectrum(field: DenseField, band: int = 0) -> Scalar:
        """Kurtosis of eigenvalue distribution. 3.0 = Gaussian. Cost: O(D³)."""
        lam = np.linalg.eigvalsh(field.F[band])
        mu = np.mean(lam)
        std = np.std(lam) + 1e-12
        return Scalar(float(np.mean(((lam - mu) / std) ** 4)), name="kurtosis", band=band)

    @staticmethod
    def moment(field: DenseField, k: int = 2, band: int = 0) -> Scalar:
        """k-th spectral moment: Σ(λ^k) / D. Cost: O(D³)."""
        lam = np.linalg.eigvalsh(field.F[band])
        return Scalar(float(np.mean(lam ** k)), name=f"moment_{k}", band=band)

    @staticmethod
    def ensemble_variance(fields: list[DenseField]) -> DenseField:
        """Element-wise variance across an ensemble. Cost: O(NBD²)."""
        if not fields:
            raise ValueError("Empty field list")
        B, D = fields[0].bands, fields[0].dim
        mean_F = sum(f.F for f in fields) / len(fields)
        var_F = sum((f.F - mean_F) ** 2 for f in fields) / len(fields)
        result = DenseField(bands=B, dim=D)
        result.F = var_F.astype(np.float32)
        return result

    @staticmethod
    def mahalanobis_distance(field: DenseField, query: NDArray[np.float32], band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Mahalanobis distance: √(qᵀ F⁻¹ q). Cost: O(D³)."""
        F_reg = field.F[band] + epsilon * np.eye(field.dim, dtype=np.float32)
        q = query[band]
        F_inv_q = np.linalg.solve(F_reg, q)
        return Scalar(float(np.sqrt(max(0, np.dot(q, F_inv_q)))), name="mahalanobis", band=band)

    @staticmethod
    def marchenko_pastur_fit(field: DenseField, band: int = 0) -> dict[str, float]:
        """Fit Marchenko-Pastur distribution to eigenvalue bulk. Cost: O(D³).

        Returns estimated noise level (sigma²), aspect ratio (gamma),
        and bulk edges (lambda_minus, lambda_plus).
        """
        lam = np.sort(np.linalg.eigvalsh(field.F[band]))[::-1]
        D = len(lam)
        N = max(field.source_count, 1)
        gamma = D / N

        # Estimate sigma from the median eigenvalue (Marchenko-Pastur median ≈ sigma² * (1 + 1/sqrt(gamma)))
        median_lam = float(np.median(lam[lam > 0])) if np.any(lam > 0) else 0.0
        sigma_sq = median_lam / (1 + 1 / np.sqrt(gamma + 1e-12) + 1e-12)

        lambda_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
        lambda_minus = sigma_sq * max(0, (1 - np.sqrt(gamma)) ** 2)

        # Signal eigenvalues = those above bulk edge
        n_signal = int(np.sum(lam > lambda_plus))

        return {
            "sigma_squared": float(sigma_sq),
            "gamma": float(gamma),
            "lambda_plus": float(lambda_plus),
            "lambda_minus": float(lambda_minus),
            "n_signal_eigenvalues": n_signal,
            "n_bulk_eigenvalues": D - n_signal,
        }

    @staticmethod
    def outlier_eigenvalues(field: DenseField, band: int = 0) -> NDArray[np.float32]:
        """Eigenvalues beyond the Marchenko-Pastur bulk edge (signal eigenvalues). Cost: O(D³)."""
        mp = StatsOps.marchenko_pastur_fit(field, band)
        lam = np.sort(np.linalg.eigvalsh(field.F[band]))[::-1]
        return lam[lam > mp["lambda_plus"]].astype(np.float32)
