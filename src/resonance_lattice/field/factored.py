# SPDX-License-Identifier: BUSL-1.1
"""Factored (incremental SVD) interference field implementation.

Stores the field as F_b = U_b · Sigma_b · V_b^T per band, where U, V are
D x K and Sigma is K x K diagonal. Each new source performs a rank-1 SVD
update (Brand 2006) in O(D·K) time instead of O(D^2) for the dense field.

Retrieval: r_b = U_b · Sigma_b · (V_b^T @ q_b) — O(D·K) per band.

Memory: B * (2*D*K + K) * precision_bytes. At B=5, D=2048, K=512, f32: ~20 MB.
Capacity: ~100K+ sources (natural language clusters in low-rank subspace).

References:
    Brand, M. (2006). "Fast low-rank modifications of the thin SVD."
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import ResonanceResult

logger = logging.getLogger(__name__)


class FactoredField:
    """Factored interference field: B sets of U·Sigma·V^T decompositions.

    Implements incremental SVD updates for rank-1 outer product storage.
    Each band's field F_b ≈ U_b @ diag(sigma_b) @ V_b.T, where U_b and V_b
    are D x K orthonormal matrices and sigma_b are K singular values.

    For symmetric outer-product storage (phi ⊗ phi), U_b == V_b, so we
    only store one basis matrix per band.
    """

    def __init__(
        self,
        bands: int,
        dim: int,
        rank: int = 512,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialise a zero factored field.

        Args:
            bands: Number of frequency bands (B).
            dim: Dimensionality per band (D).
            rank: Maximum rank K for the SVD approximation.
            dtype: NumPy dtype for storage.
        """
        self.bands = bands
        self.dim = dim
        self.rank = rank
        self.dtype = dtype
        self._source_count = 0

        # Per-band SVD components
        # U_b: (D, current_rank) orthonormal basis
        # sigma_b: (current_rank,) singular values
        # V_b: (D, current_rank) orthonormal basis (== U_b for symmetric case)
        self._U: list[NDArray | None] = [None] * bands
        self._sigma: list[NDArray | None] = [None] * bands
        self._V: list[NDArray | None] = [None] * bands
        self._current_rank: list[int] = [0] * bands

    @property
    def source_count(self) -> int:
        """Number of sources superposed into the field."""
        return self._source_count

    @property
    def size_bytes(self) -> int:
        """Approximate size of the factored representation in bytes."""
        elem_bytes = np.dtype(self.dtype).itemsize
        total = 0
        for b in range(self.bands):
            k = self._current_rank[b]
            if k > 0:
                # U[D,k] + sigma[k] + V[D,k]
                total += (2 * self.dim * k + k) * elem_bytes
        return total

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def superpose(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Add a source via rank-1 SVD update.

        For each band b, we update: F_b += alpha * phi_b ⊗ phi_b
        This is a rank-1 update to the SVD using Brand's algorithm.

        Args:
            phase_vectors: Shape (B, D) — the source's phase spectrum.
            salience: Scalar weight alpha_i.
        """
        self._validate_phase(phase_vectors)

        for b in range(self.bands):
            phi = phase_vectors[b].astype(self.dtype)
            self._rank1_update(b, phi, salience)

        self._source_count += 1

    def remove(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Remove a source via rank-1 SVD downdate.

        F_b -= alpha * phi_b ⊗ phi_b (negative rank-1 update).

        Args:
            phase_vectors: Shape (B, D) — same phase vectors used in superpose.
            salience: Same scalar weight used during superpose.
        """
        self._validate_phase(phase_vectors)

        for b in range(self.bands):
            phi = phase_vectors[b].astype(self.dtype)
            self._rank1_update(b, phi, -salience)

        self._source_count -= 1

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Project a query into the factored field.

        r_b = F_b @ q_b = U_b @ diag(sigma_b) @ (V_b.T @ q_b)

        This is O(D·K) per band instead of O(D^2).

        Args:
            query_phase: Shape (B, D) — the query's phase spectrum.
            band_weights: Shape (B,) — weights for fusion. Uniform if None.

        Returns:
            ResonanceResult.
        """
        self._validate_phase(query_phase)

        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=self.dtype) / self.bands
        else:
            band_weights = np.asarray(band_weights, dtype=self.dtype)

        resonance_vectors = np.zeros((self.bands, self.dim), dtype=self.dtype)
        band_energies = np.zeros(self.bands, dtype=self.dtype)

        for b in range(self.bands):
            if self._current_rank[b] == 0:
                continue

            q_b = query_phase[b].astype(self.dtype)
            U = self._U[b]
            sigma = self._sigma[b]
            V = self._V[b]

            # r_b = U @ diag(sigma) @ (V.T @ q_b)
            # Step 1: V.T @ q_b -> (k,)
            vt_q = V.T @ q_b
            # Step 2: diag(sigma) @ vt_q -> (k,)
            sv_q = sigma * vt_q
            # Step 3: U @ sv_q -> (D,)
            r_b = U @ sv_q

            resonance_vectors[b] = r_b
            band_energies[b] = np.linalg.norm(r_b)

        # Multi-band fusion
        fused = np.zeros(self.dim, dtype=self.dtype)
        for b in range(self.bands):
            fused += band_weights[b] * resonance_vectors[b]

        return ResonanceResult(
            resonance_vectors=resonance_vectors,
            fused=fused,
            band_energies=band_energies,
        )

    def compute_snr(
        self,
        num_sources: int | None = None,
        sparsity: float | None = None,
    ) -> float:
        """Estimate SNR accounting for sparsity.

        For sparse vectors: SNR = D / (s^2 * (N - 1))
        """
        n = num_sources if num_sources is not None else self._source_count
        if n <= 1:
            return float("inf")

        if sparsity is None:
            # Estimate from effective rank
            try:
                if self._sigma[0] is not None and len(self._sigma[0]) > 0:
                    svs = np.abs(self._sigma[0])
                    svs = svs[svs > 1e-8]
                    if len(svs) > 0:
                        effective_rank = (svs.sum() ** 2) / (svs ** 2).sum()
                        sparsity = max(0.01, min(1.0, effective_rank / self.dim))
                    else:
                        sparsity = 1.0
                else:
                    sparsity = 1.0
            except Exception:
                logger.warning("SNR sparsity estimation failed, defaulting to 1.0", exc_info=True)
                sparsity = 1.0

        return self.dim / (sparsity ** 2 * (n - 1))

    def energy(self) -> NDArray[np.float32]:
        """Compute energy per band (Frobenius norm ≈ sum of sigma^2)."""
        energies = np.zeros(self.bands, dtype=self.dtype)
        for b in range(self.bands):
            if self._current_rank[b] > 0 and self._sigma[b] is not None:
                energies[b] = np.sqrt(np.sum(self._sigma[b] ** 2))
        return energies

    def to_dense(self) -> NDArray[np.float32]:
        """Reconstruct the full B x D x D tensor (for testing/comparison).

        Returns:
            Shape (B, D, D) dense tensor.
        """
        F = np.zeros((self.bands, self.dim, self.dim), dtype=self.dtype)
        for b in range(self.bands):
            if self._current_rank[b] > 0:
                U = self._U[b]
                sigma = self._sigma[b]
                V = self._V[b]
                # F_b = U @ diag(sigma) @ V.T
                F[b] = U @ np.diag(sigma) @ V.T
        return F

    def reset(self) -> None:
        """Reset the field to zero."""
        self._U = [None] * self.bands
        self._sigma = [None] * self.bands
        self._V = [None] * self.bands
        self._current_rank = [0] * self.bands
        self._source_count = 0

    def _rank1_update(self, band: int, phi: NDArray, alpha: float) -> None:
        """Rank-1 SVD update for a single band.

        Updates F_b += alpha * phi ⊗ phi using Brand's incremental SVD.

        For symmetric outer products (phi ⊗ phi), we use:
            - a = sqrt(|alpha|) * phi
            - b = sign(alpha) * sqrt(|alpha|) * phi
            - So F += a ⊗ b = alpha * phi ⊗ phi

        Brand's algorithm:
            1. Project phi into current basis: p = U.T @ phi
            2. Residual: r = phi - U @ p
            3. Normalise residual: r_hat = r / ||r|| (if ||r|| > eps)
            4. Build (k+1) x (k+1) intermediate matrix and re-SVD
            5. Truncate to rank K
        """
        abs_alpha = abs(alpha)
        sign = 1.0 if alpha >= 0 else -1.0
        scale = np.sqrt(abs_alpha)

        if self._current_rank[band] == 0:
            # First source: trivial initialisation
            norm = np.linalg.norm(phi)
            if norm < 1e-10:
                return
            self._U[band] = (phi / norm).reshape(-1, 1).astype(self.dtype)
            self._sigma[band] = np.array([scale * norm * scale], dtype=self.dtype)
            # For symmetric: V == U (phi ⊗ phi is symmetric)
            self._V[band] = self._U[band].copy()
            if sign < 0:
                self._sigma[band] *= -1
            self._current_rank[band] = 1
            return

        U = self._U[band]
        sigma = self._sigma[band]
        V = self._V[band]
        k = self._current_rank[band]

        # Scale vectors
        a = scale * phi  # left vector
        b = sign * scale * phi  # right vector (same as a for positive alpha)

        # Project onto current basis
        p_a = U.T @ a  # (k,)
        r_a = a - U @ p_a  # residual
        r_a_norm = np.linalg.norm(r_a)

        p_b = V.T @ b  # (k,)
        r_b = b - V @ p_b
        r_b_norm = np.linalg.norm(r_b)

        # Build the (k+1) x (k+1) intermediate matrix
        # M = [diag(sigma) + p_a @ p_b.T,  r_b_norm * p_a]
        #     [r_a_norm * p_b.T,            r_a_norm * r_b_norm]

        extend_left = r_a_norm > 1e-10
        extend_right = r_b_norm > 1e-10

        if extend_left and extend_right:
            r_a_hat = r_a / r_a_norm
            r_b_hat = r_b / r_b_norm

            # Build (k+1) x (k+1) matrix
            M = np.zeros((k + 1, k + 1), dtype=self.dtype)
            M[:k, :k] = np.diag(sigma) + np.outer(p_a, p_b)
            M[:k, k] = r_b_norm * p_a
            M[k, :k] = r_a_norm * p_b
            M[k, k] = r_a_norm * r_b_norm

            # SVD of M
            Um, sm, Vmt = np.linalg.svd(M, full_matrices=False)

            # New rank (truncate to self.rank)
            new_k = min(k + 1, self.rank)
            Um = Um[:, :new_k]
            sm = sm[:new_k]
            Vmt = Vmt[:new_k, :]

            # Update basis: U_new = [U, r_a_hat] @ Um
            U_ext = np.column_stack([U, r_a_hat.reshape(-1, 1)])
            V_ext = np.column_stack([V, r_b_hat.reshape(-1, 1)])

            self._U[band] = (U_ext @ Um).astype(self.dtype)
            self._sigma[band] = sm.astype(self.dtype)
            self._V[band] = (V_ext @ Vmt.T).astype(self.dtype)
            self._current_rank[band] = new_k

        elif extend_left:
            # Only left residual is non-trivial
            r_a_hat = r_a / r_a_norm

            M = np.zeros((k + 1, k), dtype=self.dtype)
            M[:k, :k] = np.diag(sigma) + np.outer(p_a, p_b)
            M[k, :k] = r_a_norm * p_b

            Um, sm, Vmt = np.linalg.svd(M, full_matrices=False)
            new_k = min(len(sm), self.rank)

            U_ext = np.column_stack([U, r_a_hat.reshape(-1, 1)])
            self._U[band] = (U_ext @ Um[:, :new_k]).astype(self.dtype)
            self._sigma[band] = sm[:new_k].astype(self.dtype)
            self._V[band] = (V @ Vmt[:new_k, :].T).astype(self.dtype)
            self._current_rank[band] = new_k

        elif extend_right:
            r_b_hat = r_b / r_b_norm

            M = np.zeros((k, k + 1), dtype=self.dtype)
            M[:k, :k] = np.diag(sigma) + np.outer(p_a, p_b)
            M[:k, k] = r_b_norm * p_a

            Um, sm, Vmt = np.linalg.svd(M, full_matrices=False)
            new_k = min(len(sm), self.rank)

            self._U[band] = (U @ Um[:, :new_k]).astype(self.dtype)
            self._sigma[band] = sm[:new_k].astype(self.dtype)
            V_ext = np.column_stack([V, r_b_hat.reshape(-1, 1)])
            self._V[band] = (V_ext @ Vmt[:new_k, :].T).astype(self.dtype)
            self._current_rank[band] = new_k

        else:
            # Both residuals negligible — phi is in the current subspace
            M = np.diag(sigma) + np.outer(p_a, p_b)
            Um, sm, Vmt = np.linalg.svd(M, full_matrices=False)
            new_k = min(len(sm), self.rank)

            self._U[band] = (U @ Um[:, :new_k]).astype(self.dtype)
            self._sigma[band] = sm[:new_k].astype(self.dtype)
            self._V[band] = (V @ Vmt[:new_k, :].T).astype(self.dtype)
            self._current_rank[band] = new_k

    def _validate_phase(self, phase_vectors: NDArray) -> None:
        """Validate phase vector shape."""
        expected = (self.bands, self.dim)
        if phase_vectors.shape != expected:
            raise ValueError(
                f"phase_vectors shape {phase_vectors.shape} != expected {expected}"
            )
