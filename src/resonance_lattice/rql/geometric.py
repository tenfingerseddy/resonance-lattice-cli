# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain D: Geometric Operations.

Operations on the Riemannian manifold of positive semi-definite matrices.
Distances, geodesics, interpolation, alignment, and manifold structure.

The space of PSD matrices has a natural Riemannian geometry where:
- The affine-invariant metric: ds² = tr((F⁻¹ dF)²)
- Geodesics: γ(t) = F^½ (F^{-½} G F^{-½})^t F^½
- The log-Euclidean metric: d(F,G) = ||log(F) - log(G)||_F
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.types import Scalar


class GeoOps:
    """Geometric operations on the PSD manifold."""

    @staticmethod
    def log_euclidean_distance(a: DenseField, b: DenseField, band: int = 0) -> Scalar:
        """Log-Euclidean distance: ||log(F_a) - log(F_b)||_F. Cost: O(D³)."""
        log_a = SpectralOps.log(a).F[band]
        log_b = SpectralOps.log(b).F[band]
        return Scalar(float(np.linalg.norm(log_a - log_b, "fro")), name="log_euclidean_distance", band=band)

    @staticmethod
    def affine_invariant_distance(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Affine-invariant distance: ||log(A^{-½} B A^{-½})||_F. Cost: O(D³).

        The natural metric on the PSD manifold. Invariant under congruence transforms.
        """
        Fa = (a.F[band] + a.F[band].T) / 2 + epsilon * np.eye(a.dim, dtype=np.float32)
        Fb = (b.F[band] + b.F[band].T) / 2 + epsilon * np.eye(b.dim, dtype=np.float32)

        # A^{-½}
        vals_a, vecs_a = np.linalg.eigh(Fa)
        vals_a = np.maximum(vals_a, epsilon)
        A_inv_sqrt = (vecs_a / np.sqrt(vals_a)) @ vecs_a.T

        # A^{-½} B A^{-½}
        M = A_inv_sqrt @ Fb @ A_inv_sqrt

        # log(M)
        vals_m, vecs_m = np.linalg.eigh(M)
        vals_m = np.maximum(vals_m, epsilon)
        log_M = (vecs_m * np.log(vals_m)) @ vecs_m.T

        return Scalar(float(np.linalg.norm(log_M, "fro")), name="affine_invariant_distance", band=band)

    @staticmethod
    def geodesic_interpolation(a: DenseField, b: DenseField, t: float = 0.5, band: int | None = None, epsilon: float = 1e-8) -> DenseField:
        """Geodesic interpolation: γ(t) = A^½ (A^{-½} B A^{-½})^t A^½. Cost: O(BD³).

        t=0 → A, t=1 → B, t=0.5 → geometric midpoint on PSD manifold.
        """
        bands = [band] if band is not None else list(range(a.bands))
        result = DenseField(bands=a.bands, dim=a.dim)
        result.F = a.F.copy()
        result._source_count = int((1 - t) * a.source_count + t * b.source_count)

        for bi in bands:
            Fa = (a.F[bi] + a.F[bi].T) / 2 + epsilon * np.eye(a.dim, dtype=np.float32)
            Fb = (b.F[bi] + b.F[bi].T) / 2 + epsilon * np.eye(b.dim, dtype=np.float32)

            vals_a, vecs_a = np.linalg.eigh(Fa)
            vals_a = np.maximum(vals_a, epsilon)

            A_sqrt = (vecs_a * np.sqrt(vals_a)) @ vecs_a.T
            A_inv_sqrt = (vecs_a / np.sqrt(vals_a)) @ vecs_a.T

            M = A_inv_sqrt @ Fb @ A_inv_sqrt
            vals_m, vecs_m = np.linalg.eigh(M)
            vals_m = np.maximum(vals_m, epsilon)

            M_t = (vecs_m * (vals_m ** t)) @ vecs_m.T
            result.F[bi] = (A_sqrt @ M_t @ A_sqrt).astype(np.float32)

        return result

    @staticmethod
    def log_euclidean_mean(fields: list[DenseField], epsilon: float = 1e-8) -> DenseField:
        """Log-Euclidean mean: exp(mean(log(F_i))). Cost: O(NBD³).

        A fast approximation to the Fréchet mean on the PSD manifold.
        """
        if not fields:
            raise ValueError("Empty field list")
        B, D = fields[0].bands, fields[0].dim
        log_sum = np.zeros((B, D, D), dtype=np.float64)

        for f in fields:
            for b in range(B):
                F_sym = (f.F[b] + f.F[b].T) / 2 + epsilon * np.eye(D, dtype=np.float32)
                vals, vecs = np.linalg.eigh(F_sym)
                vals = np.maximum(vals, epsilon)
                log_F = (vecs * np.log(vals)) @ vecs.T
                log_sum[b] += log_F

        log_mean = log_sum / len(fields)

        result = DenseField(bands=B, dim=D)
        for b in range(B):
            vals, vecs = np.linalg.eigh(log_mean[b])
            result.F[b] = ((vecs * np.exp(vals)) @ vecs.T).astype(np.float32)
        result._source_count = sum(f.source_count for f in fields) // len(fields)
        return result

    @staticmethod
    def tangent_vector(a: DenseField, b: DenseField) -> NDArray[np.float32]:
        """Euclidean tangent vector: V = B - A (approximation to log map). Cost: O(BD²)."""
        return (b.F - a.F).astype(np.float32)

    @staticmethod
    def stein_divergence(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Stein divergence (S-divergence): log(det((A+B)/2)) - ½log(det(A·B)). Cost: O(D³).

        A symmetrised divergence that is computationally simpler than affine-invariant.
        """
        Fa = a.F[band] + epsilon * np.eye(a.dim, dtype=np.float32)
        Fb = b.F[band] + epsilon * np.eye(b.dim, dtype=np.float32)

        avg = (Fa + Fb) / 2
        log_det_avg = np.linalg.slogdet(avg)[1]
        log_det_a = np.linalg.slogdet(Fa)[1]
        log_det_b = np.linalg.slogdet(Fb)[1]

        div = log_det_avg - 0.5 * (log_det_a + log_det_b)
        return Scalar(float(div), name="stein_divergence", band=band)

    @staticmethod
    def procrustes_align(a: DenseField, b: DenseField, band: int = 0) -> tuple[DenseField, NDArray[np.float32]]:
        """Procrustes alignment: find rotation R minimising ||F_a - R F_b Rᵀ||_F. Cost: O(D³).

        Returns the aligned field and the rotation matrix R.
        """
        # Use eigenvector alignment: find R that maps eigenvectors of b to a
        vals_a, vecs_a = np.linalg.eigh((a.F[band] + a.F[band].T) / 2)
        vals_b, vecs_b = np.linalg.eigh((b.F[band] + b.F[band].T) / 2)

        # Sort by eigenvalue magnitude
        idx_a = np.argsort(np.abs(vals_a))[::-1]
        idx_b = np.argsort(np.abs(vals_b))[::-1]
        vecs_a = vecs_a[:, idx_a]
        vecs_b = vecs_b[:, idx_b]

        # Optimal rotation via SVD of cross-covariance
        M = vecs_a @ vecs_b.T
        U, _, Vt = np.linalg.svd(M)
        R = (U @ Vt).astype(np.float32)

        # Apply rotation to all bands of b
        aligned = DenseField(bands=b.bands, dim=b.dim)
        for bi in range(b.bands):
            aligned.F[bi] = (R @ b.F[bi] @ R.T).astype(np.float32)
        aligned._source_count = b.source_count

        return aligned, R

    @staticmethod
    def principal_angles(a: DenseField, b: DenseField, band: int = 0, k: int = 10) -> NDArray[np.float32]:
        """Principal angles between eigenspaces: θ_i = arccos(σ_i(V_aᵀ V_b)). Cost: O(D³)."""
        vals_a, vecs_a = np.linalg.eigh((a.F[band] + a.F[band].T) / 2)
        vals_b, vecs_b = np.linalg.eigh((b.F[band] + b.F[band].T) / 2)

        idx_a = np.argsort(np.abs(vals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(vals_b))[::-1][:k]

        Va = vecs_a[:, idx_a]
        Vb = vecs_b[:, idx_b]

        S = np.linalg.svd(Va.T @ Vb, compute_uv=False)
        S = np.clip(S, -1, 1)
        return np.arccos(S).astype(np.float32)

    @staticmethod
    def grassmann_distance(a: DenseField, b: DenseField, band: int = 0, k: int = 10) -> Scalar:
        """Grassmann distance: ||θ||₂ where θ = principal angles. Cost: O(D³)."""
        angles = GeoOps.principal_angles(a, b, band, k)
        return Scalar(float(np.linalg.norm(angles)), name="grassmann_distance", band=band)

    @staticmethod
    def subspace_overlap(a: DenseField, b: DenseField, band: int = 0, k: int = 10) -> Scalar:
        """Subspace overlap: Σcos²(θ_i) / k. 1.0 = identical subspaces. Cost: O(D³)."""
        angles = GeoOps.principal_angles(a, b, band, k)
        overlap = float(np.sum(np.cos(angles) ** 2) / max(len(angles), 1))
        return Scalar(overlap, name="subspace_overlap", band=band)

    @staticmethod
    def natural_gradient(field: DenseField, gradient: NDArray[np.float32], band: int = 0, epsilon: float = 1e-8) -> NDArray[np.float32]:
        """Natural gradient: F⁻¹ · ∇ (Riemannian gradient on PSD manifold). Cost: O(D³)."""
        F_reg = field.F[band] + epsilon * np.eye(field.dim, dtype=np.float32)
        return np.linalg.solve(F_reg, gradient).astype(np.float32)

    @staticmethod
    def exp_map(field: DenseField, tangent: NDArray[np.float32], band: int = 0, epsilon: float = 1e-8) -> DenseField:
        """Riemannian exponential map: Exp_F(V) = F^½ exp(F^{-½} V F^{-½}) F^½. Cost: O(D³)."""
        F_b = field.F[band] + epsilon * np.eye(field.dim, dtype=np.float32)
        vals, vecs = np.linalg.eigh(F_b)
        vals = np.maximum(vals, epsilon)

        F_sqrt = (vecs * np.sqrt(vals)) @ vecs.T
        F_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T

        inner = F_inv_sqrt @ tangent @ F_inv_sqrt
        vals_i, vecs_i = np.linalg.eigh(inner)
        exp_inner = (vecs_i * np.exp(vals_i)) @ vecs_i.T

        result_band = F_sqrt @ exp_inner @ F_sqrt

        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result.F[band] = result_band.astype(np.float32)
        result._source_count = field.source_count
        return result

    @staticmethod
    def parallel_transport(field: DenseField, tangent: NDArray[np.float32], target: DenseField, band: int = 0, epsilon: float = 1e-8) -> NDArray[np.float32]:
        """Parallel transport of tangent V from field to target along geodesic. Cost: O(D³).

        Uses Schild's ladder approximation for the PSD manifold.
        """
        Fa = field.F[band] + epsilon * np.eye(field.dim, dtype=np.float32)
        Fb = target.F[band] + epsilon * np.eye(target.dim, dtype=np.float32)

        vals_a, vecs_a = np.linalg.eigh(Fa)
        vals_a = np.maximum(vals_a, epsilon)
        A_sqrt = (vecs_a * np.sqrt(vals_a)) @ vecs_a.T
        A_inv_sqrt = (vecs_a / np.sqrt(vals_a)) @ vecs_a.T

        # Transport matrix: E = (B/A)^½ = A^{-½} (A^½ B A^½)^½ A^{-½}
        inner = A_sqrt @ Fb @ A_sqrt
        vals_i, vecs_i = np.linalg.eigh(inner)
        vals_i = np.maximum(vals_i, epsilon)
        sqrt_inner = (vecs_i * np.sqrt(vals_i)) @ vecs_i.T

        E = A_inv_sqrt @ sqrt_inner @ A_inv_sqrt

        # Transported tangent: V' = E V Eᵀ
        return (E @ tangent @ E.T).astype(np.float32)

    @staticmethod
    def frechet_mean(fields: list[DenseField], band: int = 0, max_iterations: int = 20, tolerance: float = 1e-4, epsilon: float = 1e-8) -> DenseField:
        """Fréchet mean on PSD manifold via iterative log-map averaging. Cost: O(N·K·D³).

        The geometric mean that minimises sum of squared geodesic distances.
        """
        if not fields:
            raise ValueError("Empty field list")

        # Initialise with log-Euclidean mean (good starting point)
        current = GeoOps.log_euclidean_mean(fields, epsilon)

        for _ in range(max_iterations):
            # Compute mean tangent vector
            mean_tangent = np.zeros_like(current.F[band])
            for f in fields:
                log_v = GeoOps.log_map(current, f, band, epsilon)
                mean_tangent += log_v
            mean_tangent /= len(fields)

            # Check convergence
            if np.linalg.norm(mean_tangent) < tolerance:
                break

            # Step along mean tangent (small step for stability)
            current = GeoOps.exp_map(current, 0.5 * mean_tangent, band, epsilon)

        return current

    @staticmethod
    def curvature_scalar(field: DenseField, band: int = 0, epsilon: float = 1e-8) -> Scalar:
        """Scalar curvature of the PSD manifold at field point. Cost: O(D³).

        For the affine-invariant metric, scalar curvature at F is:
        R = -D(D+1)/4 (constant for the PSD manifold).
        But the EFFECTIVE curvature depends on eigenvalue spread.
        """
        lam = np.linalg.eigvalsh(field.F[band])
        lam = np.maximum(np.abs(lam), epsilon)
        D = len(lam)

        # Effective curvature: weighted by eigenvalue ratios
        # Higher spread → more curved (harder to interpolate)
        log_lam = np.log(lam)
        spread = float(np.std(log_lam))

        # Normalised curvature: D(D+1)/4 * spread
        R = -D * (D + 1) / 4 * spread
        return Scalar(float(R), name="scalar_curvature", band=band)

    @staticmethod
    def log_map(a: DenseField, b: DenseField, band: int = 0, epsilon: float = 1e-8) -> NDArray[np.float32]:
        """Riemannian logarithmic map: Log_A(B) = A^½ log(A^{-½} B A^{-½}) A^½. Cost: O(D³)."""
        Fa = a.F[band] + epsilon * np.eye(a.dim, dtype=np.float32)
        Fb = b.F[band] + epsilon * np.eye(b.dim, dtype=np.float32)

        vals_a, vecs_a = np.linalg.eigh(Fa)
        vals_a = np.maximum(vals_a, epsilon)

        A_sqrt = (vecs_a * np.sqrt(vals_a)) @ vecs_a.T
        A_inv_sqrt = (vecs_a / np.sqrt(vals_a)) @ vecs_a.T

        M = A_inv_sqrt @ Fb @ A_inv_sqrt
        vals_m, vecs_m = np.linalg.eigh(M)
        vals_m = np.maximum(vals_m, epsilon)
        log_M = (vecs_m * np.log(vals_m)) @ vecs_m.T

        return (A_sqrt @ log_M @ A_sqrt).astype(np.float32)
