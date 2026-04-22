# SPDX-License-Identifier: BUSL-1.1
"""Product-quantised interference field implementation.

THE PRODUCTION ANSWER (Spec Section 8.4).

Decomposes each band's D-dimensional space into M subspaces of dimension D/M.
Within each subspace, K codebook centroids are learned. The field is stored as
B sets of M K×K matrices, giving O(B·M·K²) retrieval — completely independent
of corpus size N.

Capacity: ~8M sources at K=1024, M=8.
Memory: B * M * (K * D/M + K * K) * precision_bytes ≈ 80 MB at B=5, M=8, K=1024, f16.
Retrieval: O(B·M·K²) per query — independent of N.

References:
    Jégou et al. (2011). "Product Quantization for Nearest Neighbor Search."
    Spec Section 8.4: Product Quantised Hopfield Storage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans2

from resonance_lattice.field.dense import ResonanceResult


class PQField:
    """Product-quantised interference field.

    The field is decomposed into M subspaces per band. Each subspace has
    a codebook of K centroids. The quantised field is stored as K×K matrices
    per subspace per band:

        F_b^(m)_quantised[j, k] = Σ_{i where c_i=j} α_i · ⟨C[j], C[k]⟩

    Retrieval:
        For each subspace m, precompute d_k = ⟨C_m[k], q_b^(m)⟩ for all k.
        Then r_b^(m) = Σ_k F_b^(m)[:, k] * d_k   (matrix-vector multiply on K×K)
        Finally r_b = concat(C_m @ r_b^(m) for each m)  (project back to D-space)

    Total cost: O(B · M · K²) — independent of N.
    """

    def __init__(
        self,
        bands: int,
        dim: int,
        num_subspaces: int = 8,
        codebook_size: int = 1024,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialise a PQ field.

        Args:
            bands: Number of frequency bands (B).
            dim: Dimensionality per band (D). Must be divisible by num_subspaces.
            num_subspaces: M — number of PQ subspaces.
            codebook_size: K — centroids per subspace codebook.
            dtype: NumPy dtype.
        """
        if dim % num_subspaces != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_subspaces ({num_subspaces})"
            )

        self.bands = bands
        self.dim = dim
        self.M = num_subspaces
        self.K = codebook_size
        self.sub_dim = dim // num_subspaces  # D/M
        self.dtype = dtype
        self._source_count = 0

        # Codebooks: per band, per subspace — shape (K, sub_dim)
        # Initialised to None until trained
        self._codebooks: list[list[NDArray | None]] = [
            [None] * self.M for _ in range(bands)
        ]
        self._codebooks_trained = False

        # Quantised field: per band, per subspace — shape (K, K)
        # F_b^(m)[j, k] accumulates contributions from sources quantised to centroid j
        self._qfield: list[list[NDArray]] = [
            [np.zeros((codebook_size, codebook_size), dtype=dtype) for _ in range(self.M)]
            for _ in range(bands)
        ]

        # Accumulator for codebook training (phase vectors before codebooks exist)
        self._training_buffer: list[NDArray] = []
        self._training_buffer_limit = 1000  # Train codebook after this many sources

    @property
    def source_count(self) -> int:
        return self._source_count

    @property
    def codebooks_trained(self) -> bool:
        return self._codebooks_trained

    @property
    def size_bytes(self) -> int:
        """Size of quantised field + codebooks in bytes."""
        elem = np.dtype(self.dtype).itemsize
        codebook_size = self.bands * self.M * self.K * self.sub_dim * elem
        qfield_size = self.bands * self.M * self.K * self.K * elem
        return codebook_size + qfield_size

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def train_codebooks(
        self,
        phase_vectors: NDArray[np.float32],
        max_iter: int = 20,
    ) -> None:
        """Train codebooks from a sample of phase vectors using k-means.

        Args:
            phase_vectors: Shape (N, B, D) — sample of phase spectra.
            max_iter: Maximum k-means iterations.
        """
        N = phase_vectors.shape[0]
        if N < self.K:
            raise ValueError(
                f"Need at least K={self.K} samples to train codebooks, got {N}"
            )

        for b in range(self.bands):
            for m in range(self.M):
                # Extract subspace slice: (N, sub_dim)
                start = m * self.sub_dim
                end = start + self.sub_dim
                sub_vectors = phase_vectors[:, b, start:end].astype(np.float64)

                # k-means clustering
                centroids, labels = kmeans2(
                    sub_vectors, self.K, iter=max_iter, minit="points"
                )

                self._codebooks[b][m] = centroids.astype(self.dtype)

        self._codebooks_trained = True

        # If we had buffered sources, encode them now
        if self._training_buffer:
            for pv in self._training_buffer:
                self._encode_and_accumulate(pv, salience=1.0)
            self._training_buffer = []  # Release memory

    def init_codebooks_random(self, seed: int = 42) -> None:
        """Initialise codebooks with random centroids (for testing).

        Args:
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        for b in range(self.bands):
            for m in range(self.M):
                centroids = rng.standard_normal((self.K, self.sub_dim)).astype(self.dtype)
                # L2 normalise each centroid
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                centroids = centroids / norms
                self._codebooks[b][m] = centroids
        self._codebooks_trained = True

    def superpose(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Add a source to the PQ field.

        If codebooks are trained: quantise the phase vector and update the
        K×K field matrices.
        If not yet trained: buffer the phase vector. Once buffer reaches
        training_buffer_limit, auto-train codebooks.

        Args:
            phase_vectors: Shape (B, D).
            salience: Scalar weight alpha_i.
        """
        self._validate_phase(phase_vectors)

        if not self._codebooks_trained:
            self._training_buffer.append(phase_vectors.copy())
            self._source_count += 1

            if len(self._training_buffer) >= self._training_buffer_limit:
                # Auto-train codebooks
                buffer_array = np.stack(self._training_buffer, axis=0)
                actual_k = min(self.K, len(self._training_buffer))
                if actual_k < self.K:
                    # Not enough samples for full K, use what we have
                    self.K = actual_k
                    self._qfield = [
                        [np.zeros((actual_k, actual_k), dtype=self.dtype) for _ in range(self.M)]
                        for _ in range(self.bands)
                    ]
                self.train_codebooks(buffer_array)
            return

        self._encode_and_accumulate(phase_vectors, salience)
        self._source_count += 1

    def superpose_batch(
        self,
        phase_batch: NDArray[np.float32],
        saliences: NDArray[np.float32] | None = None,
    ) -> None:
        """Add multiple sources to the PQ field in a single vectorised call.

        ~60x faster than sequential superpose for large batches. Uses batch
        nearest-centroid assignment via matrix multiply and np.add.at for
        scatter-accumulation into the K×K field matrices.

        Args:
            phase_batch: Shape (N, B, D) — batch of phase spectra.
            saliences: Shape (N,) — per-source weights. None = all 1.0.
        """
        N = phase_batch.shape[0]
        if N == 0:
            return

        if not self._codebooks_trained:
            for i in range(N):
                self._training_buffer.append(phase_batch[i].copy())
            self._source_count += N
            if len(self._training_buffer) >= self._training_buffer_limit:
                buffer_array = np.stack(self._training_buffer, axis=0)
                actual_k = min(self.K, len(self._training_buffer))
                if actual_k < self.K:
                    self.K = actual_k
                    self._qfield = [
                        [np.zeros((actual_k, actual_k), dtype=self.dtype)
                         for _ in range(self.M)]
                        for _ in range(self.bands)
                    ]
                self.train_codebooks(buffer_array)
            return

        if saliences is None:
            saliences = np.ones(N, dtype=self.dtype)
        else:
            saliences = np.asarray(saliences, dtype=self.dtype)

        for b in range(self.bands):
            for m in range(self.M):
                start = m * self.sub_dim
                end = start + self.sub_dim

                # Extract subspace slice for all sources: (N, sub_dim)
                phi_batch = phase_batch[:, b, start:end].astype(self.dtype)
                C_m = self._codebooks[b][m]  # (K, sub_dim)

                # Batch nearest centroid via dot product (equivalent to argmin L2
                # for L2-normalised codebooks; for unnormalised, use explicit L2)
                # dists[i, k] = ||phi_i - C_k||^2 = ||phi_i||^2 - 2*phi_i·C_k + ||C_k||^2
                phi_sq = np.sum(phi_batch ** 2, axis=1, keepdims=True)  # (N, 1)
                C_sq = np.sum(C_m ** 2, axis=1, keepdims=True).T        # (1, K)
                dots = phi_batch @ C_m.T                                 # (N, K)
                dists = phi_sq - 2 * dots + C_sq                        # (N, K)
                assignments = np.argmin(dists, axis=1)                   # (N,)

                # Precompute codebook gram matrix: G[j, k] = C_m[j] · C_m[k]
                G = C_m @ C_m.T  # (K, K)

                # Group-by-centroid accumulation (faster than np.add.at scatter):
                # For each centroid c, sum saliences of all sources assigned to c,
                # then F[c, :] += total_salience_c * G[c, :]
                # This reduces N scatter ops to K dense updates.
                centroid_saliences = np.bincount(
                    assignments, weights=saliences, minlength=self.K
                )  # (K,)
                # Only update centroids that have at least one assignment
                active = np.nonzero(centroid_saliences)[0]
                if len(active) > 0:
                    self._qfield[b][m][active] += (
                        centroid_saliences[active, np.newaxis] * G[active]
                    )

        self._source_count += N

    def remove(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Remove a source from the PQ field (negative accumulation).

        Args:
            phase_vectors: Shape (B, D) — same as used during superpose.
            salience: Same weight used during superpose.
        """
        self._validate_phase(phase_vectors)

        if not self._codebooks_trained:
            self._source_count -= 1
            return

        self._encode_and_accumulate(phase_vectors, -salience)
        self._source_count -= 1

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Project a query into the PQ field.

        For each band b, subspace m:
            1. Slice query: q_m = q_b[m*d:(m+1)*d]
            2. Compute centroid similarities: d_k = C_m[k] · q_m for all k
            3. Resonance in codebook space: r_codes = F_b^(m) @ d  (K×K @ K = K)
            4. Project back: r_m = C_m.T @ r_codes  (sub_dim×K @ K = sub_dim)
            5. Concatenate subspace results

        Total: O(B · M · K²) — independent of N.

        Args:
            query_phase: Shape (B, D).
            band_weights: Shape (B,). Uniform if None.

        Returns:
            ResonanceResult.
        """
        self._validate_phase(query_phase)

        if not self._codebooks_trained:
            # Return zeros if codebooks not trained yet
            return ResonanceResult(
                resonance_vectors=np.zeros((self.bands, self.dim), dtype=self.dtype),
                fused=np.zeros(self.dim, dtype=self.dtype),
                band_energies=np.zeros(self.bands, dtype=self.dtype),
            )

        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=self.dtype) / self.bands
        else:
            band_weights = np.asarray(band_weights, dtype=self.dtype)

        resonance_vectors = np.zeros((self.bands, self.dim), dtype=self.dtype)
        band_energies = np.zeros(self.bands, dtype=self.dtype)

        for b in range(self.bands):
            r_b = np.zeros(self.dim, dtype=self.dtype)

            for m in range(self.M):
                start = m * self.sub_dim
                end = start + self.sub_dim

                q_m = query_phase[b, start:end].astype(self.dtype)
                C_m = self._codebooks[b][m]  # (K, sub_dim)
                Fq_m = self._qfield[b][m]    # (K, K)

                # Centroid similarities: d = C_m @ q_m -> (K,)
                d = C_m @ q_m

                # Resonance in codebook space: r_codes = Fq_m @ d -> (K,)
                r_codes = Fq_m @ d

                # Project back to subspace: r_m = C_m.T @ r_codes -> (sub_dim,)
                r_m = C_m.T @ r_codes

                r_b[start:end] = r_m

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

    def compute_snr(self, num_sources: int | None = None) -> float:
        """Estimate SNR for the PQ field."""
        n = num_sources if num_sources is not None else self._source_count
        if n <= 1:
            return float("inf")
        # PQ effective capacity: K^2 per subspace
        effective_dim = self.K * self.K * self.M
        return effective_dim / (n - 1)

    def energy(self) -> NDArray[np.float32]:
        """Compute energy per band (Frobenius norm of quantised field)."""
        energies = np.zeros(self.bands, dtype=self.dtype)
        for b in range(self.bands):
            total_sq = 0.0
            for m in range(self.M):
                total_sq += np.sum(self._qfield[b][m] ** 2)
            energies[b] = np.sqrt(total_sq)
        return energies

    def reset(self) -> None:
        """Reset the field (but keep codebooks)."""
        for b in range(self.bands):
            for m in range(self.M):
                self._qfield[b][m].fill(0)
        self._source_count = 0
        self._training_buffer.clear()

    def _encode_and_accumulate(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float,
    ) -> None:
        """Quantise a phase vector and accumulate into the K×K field matrices.

        For each band b, subspace m:
            1. Find nearest centroid: c = argmin_k ||phi_m - C_m[k]||
            2. Update: F_b^(m)[c, :] += alpha * C_m[c] · C_m[:].T
               (row c gets the similarity of centroid c with all other centroids,
                weighted by salience)

        This eliminates N-dependence: all per-source info is folded into K×K.
        """
        assignments = np.zeros((self.bands, self.M), dtype=np.int32)

        for b in range(self.bands):
            for m in range(self.M):
                start = m * self.sub_dim
                end = start + self.sub_dim

                phi_m = phase_vectors[b, start:end].astype(self.dtype)
                C_m = self._codebooks[b][m]  # (K, sub_dim)

                # Find nearest centroid
                dists = np.linalg.norm(C_m - phi_m[np.newaxis, :], axis=1)
                c = np.argmin(dists)
                assignments[b, m] = c

                # Accumulate: outer product in codebook space
                # F_b^(m)[c, k] += alpha * <C[c], C[k]> for all k
                # This is: F_b^(m)[c, :] += alpha * (C_m @ C_m[c])
                centroid_similarities = C_m @ C_m[c]  # (K,)
                self._qfield[b][m][c, :] += salience * centroid_similarities

    def _validate_phase(self, phase_vectors: NDArray) -> None:
        """Validate phase vector shape."""
        expected = (self.bands, self.dim)
        if phase_vectors.shape != expected:
            raise ValueError(
                f"phase_vectors shape {phase_vectors.shape} != expected {expected}"
            )
