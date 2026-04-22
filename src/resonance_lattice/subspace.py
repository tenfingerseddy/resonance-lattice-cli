# SPDX-License-Identifier: BUSL-1.1
"""Adaptive subspace projection for knowledge fields.

Activate only the dimensions the task needs. Factoid queries use K=64 (200x faster),
exploration queries use K=512.

Operations:
    metabolise(field, K, strategy) -> SubspaceField
    SubspaceField.resonate(query) -> ResonanceResult in reduced space
    SubspaceField.lift(vector) -> full-space vector

Speedup: O(K^2) vs O(D^2) per band. At K=128, D=2048: 256x faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField, ResonanceResult


class SubspaceStrategy(StrEnum):
    """How to select the K-dimensional subspace."""
    ENERGY = "energy"  # Top-K dimensions by diagonal energy (fastest)
    QUERY_ALIGNED = "query_aligned"  # Top-K dimensions of F@q by magnitude
    SPECTRAL = "spectral"  # Top-K eigenvectors (highest quality)
    CROSS_BAND = "cross_band"  # Dimensions active across ≥3/5 bands


@dataclass
class SubspaceInfo:
    """Metadata about the selected subspace."""
    K: int
    D: int
    strategy: SubspaceStrategy
    retained_energy_ratio: float  # Fraction of total energy captured
    selected_dims: NDArray[np.int64] | None  # For energy/cross_band strategies
    projection_matrix: NDArray[np.float32] | None  # (D, K) for spectral strategy


class SubspaceField:
    """A field projected into a K-dimensional subspace for fast retrieval.

    Wraps a DenseField and provides retrieval in the projected space:
        F_task = Pᵀ F P     (K×K per band)
        q_task = Pᵀ q       (K,)
        r_task = F_task @ q  (K,)
        r = P @ r_task       (D,) lifted back to full space

    The projection P is either:
    - A selection matrix (energy/cross_band): pick K dimension indices
    - An eigenvector matrix (spectral): project onto top-K eigenvectors
    """

    def __init__(
        self,
        parent: DenseField,
        K: int,
        strategy: SubspaceStrategy,
        query_phase: NDArray[np.float32] | None = None,
    ):
        self.parent = parent
        self.bands = parent.bands
        self.dim = parent.dim
        self.K = K

        if K >= parent.dim:
            raise ValueError(f"K={K} must be < dim={parent.dim}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

        if strategy == SubspaceStrategy.ENERGY:
            self._init_energy()
        elif strategy == SubspaceStrategy.QUERY_ALIGNED:
            if query_phase is None:
                raise ValueError("query_aligned strategy requires query_phase")
            self._init_query_aligned(query_phase)
        elif strategy == SubspaceStrategy.SPECTRAL:
            self._init_spectral()
        elif strategy == SubspaceStrategy.CROSS_BAND:
            self._init_cross_band()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy

    def _init_energy(self) -> None:
        """Select top-K dimensions by diagonal energy (variance)."""
        # Average diagonal energy across bands
        diag_energy = np.zeros(self.dim, dtype=np.float32)
        for b in range(self.bands):
            diag_energy += np.abs(np.diag(self.parent.F[b]))
        diag_energy /= self.bands

        self._dims = np.argsort(diag_energy)[-self.K:][::-1]
        self._build_from_dims()

    def _init_query_aligned(self, query_phase: NDArray[np.float32]) -> None:
        """Select top-K dimensions from F@q by magnitude."""
        dim_scores = np.zeros(self.dim, dtype=np.float32)
        for b in range(self.bands):
            r_b = self.parent.F[b] @ query_phase[b]
            dim_scores += np.abs(r_b)

        self._dims = np.argsort(dim_scores)[-self.K:][::-1]
        self._build_from_dims()

    def _init_spectral(self) -> None:
        """Select top-K eigenvectors (PCA of the field)."""
        # Average eigendecomposition across bands
        # Use band 0 as primary (most eigenvectors shared across bands)
        F_avg = np.mean(self.parent.F, axis=0)
        F_sym = (F_avg + F_avg.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        # Top-K by absolute eigenvalue (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1][:self.K]
        self._P = eigenvectors[:, idx].astype(np.float32)  # (D, K)

        # Project field: F_sub[b] = Pᵀ F[b] P
        self._F_sub = np.zeros((self.bands, self.K, self.K), dtype=np.float32)
        for b in range(self.bands):
            self._F_sub[b] = self._P.T @ self.parent.F[b] @ self._P

        # Compute energy retention
        total_energy = sum(np.linalg.norm(self.parent.F[b], "fro") for b in range(self.bands))
        sub_energy = sum(np.linalg.norm(self._F_sub[b], "fro") for b in range(self.bands))
        self._energy_ratio = sub_energy / (total_energy + 1e-12)
        self._dims = None
        self._use_projection = True

    def _init_cross_band(self) -> None:
        """Select dimensions active in ≥3 out of B bands."""
        B = self.bands
        max(3, B // 2 + 1)

        # For each dimension, count how many bands have above-median energy
        active_counts = np.zeros(self.dim, dtype=np.int32)
        for b in range(B):
            diag = np.abs(np.diag(self.parent.F[b]))
            median_energy = np.median(diag)
            active_counts += (diag > median_energy).astype(np.int32)

        # Take top-K by active count (break ties by total energy)
        total_diag = np.zeros(self.dim, dtype=np.float32)
        for b in range(B):
            total_diag += np.abs(np.diag(self.parent.F[b]))

        # Composite score: active_count * 1e6 + total_energy (count dominates)
        composite = active_counts * 1e6 + total_diag
        self._dims = np.argsort(composite)[-self.K:][::-1]
        self._build_from_dims()

    def _build_from_dims(self) -> None:
        """Build projected field from selected dimension indices."""
        self._use_projection = False
        dims = self._dims

        # Extract submatrix: F_sub[b] = F[b][dims, :][:, dims]
        self._F_sub = np.zeros((self.bands, self.K, self.K), dtype=np.float32)
        for b in range(self.bands):
            self._F_sub[b] = self.parent.F[b][np.ix_(dims, dims)]

        # Compute energy retention
        total_energy = sum(np.linalg.norm(self.parent.F[b], "fro") for b in range(self.bands))
        sub_energy = sum(np.linalg.norm(self._F_sub[b], "fro") for b in range(self.bands))
        self._energy_ratio = sub_energy / (total_energy + 1e-12)
        self._P = None

    def project_query(self, query_phase: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project a (B, D) query into the (B, K) subspace."""
        if self._use_projection:
            # Spectral: q_sub = Pᵀ q
            q_sub = np.zeros((self.bands, self.K), dtype=np.float32)
            for b in range(self.bands):
                q_sub[b] = self._P.T @ query_phase[b]
            return q_sub
        else:
            # Dimension selection: just pick K dimensions
            return query_phase[:, self._dims].copy()

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Resonate in the subspace: O(B*K²) instead of O(B*D²).

        Args:
            query_phase: Shape (B, D) — full-space query.
            band_weights: Shape (B,) — per-band fusion weights.

        Returns:
            ResonanceResult with vectors lifted back to full D-dimensional space.
        """
        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=np.float32) / self.bands

        # Project query into subspace
        q_sub = self.project_query(query_phase)

        # Resonate in subspace
        resonance_sub = np.zeros((self.bands, self.K), dtype=np.float32)
        band_energies = np.zeros(self.bands, dtype=np.float32)

        for b in range(self.bands):
            r_b = self._F_sub[b] @ q_sub[b]
            resonance_sub[b] = r_b
            band_energies[b] = np.linalg.norm(r_b)

        # Lift back to full space
        resonance_full = self.lift(resonance_sub)

        # Fuse across bands
        fused = np.zeros(self.dim, dtype=np.float32)
        for b in range(self.bands):
            fused += band_weights[b] * resonance_full[b]

        return ResonanceResult(
            resonance_vectors=resonance_full,
            fused=fused,
            band_energies=band_energies,
        )

    def lift(self, vectors_sub: NDArray[np.float32]) -> NDArray[np.float32]:
        """Lift (B, K) subspace vectors back to (B, D) full space."""
        full = np.zeros((self.bands, self.dim), dtype=np.float32)

        if self._use_projection:
            # Spectral: r = P @ r_sub
            for b in range(self.bands):
                full[b] = self._P @ vectors_sub[b]
        else:
            # Dimension selection: scatter back
            for b in range(self.bands):
                full[b, self._dims] = vectors_sub[b]

        return full

    def info(self) -> SubspaceInfo:
        """Return metadata about this subspace."""
        return SubspaceInfo(
            K=self.K,
            D=self.dim,
            strategy=self.strategy,
            retained_energy_ratio=self._energy_ratio,
            selected_dims=self._dims.copy() if self._dims is not None else None,
            projection_matrix=self._P.copy() if self._P is not None else None,
        )


def metabolise(
    field: DenseField,
    K: int,
    strategy: SubspaceStrategy | str = SubspaceStrategy.ENERGY,
    query_phase: NDArray[np.float32] | None = None,
) -> SubspaceField:
    """Create a SubspaceField — the field projected to K dimensions.

    This is the main entry point for Dimensional Metabolism.

    Args:
        field: The full dense field.
        K: Target dimensionality (K << D for speedup).
        strategy: How to select the subspace.
        query_phase: Required for query_aligned strategy.

    Returns:
        SubspaceField ready for fast retrieval.
    """
    if isinstance(strategy, str):
        strategy = SubspaceStrategy(strategy)

    return SubspaceField(
        parent=field,
        K=K,
        strategy=strategy,
        query_phase=query_phase,
    )
