# SPDX-License-Identifier: BUSL-1.1
"""Dense interference field implementation.

The dense field stores the full B x D x D tensor. Each band's field is a D x D
matrix formed by the superposition (sum of outer products) of all encoded sources.

This is the simplest and most mathematically transparent implementation. It is
appropriate for development, testing, and corpora up to ~82K sources (with sparse
phase vectors at s=0.05, D=2048).

Memory: B * D^2 * precision_bytes. At B=5, D=2048, f32: ~80 MB.
Retrieval: O(B * D^2) per query — a matrix-vector multiply per band.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ResonanceResult(NamedTuple):
    """Result of a resonance projection."""
    resonance_vectors: NDArray[np.float32]  # shape: (B, D) — per-band resonance
    fused: NDArray[np.float32]              # shape: (D,) — weighted sum across bands
    band_energies: NDArray[np.float32]      # shape: (B,) — energy per band


class DenseField:
    """Dense interference field: B x D x D tensor.

    Implements the core Hopfield-style associative memory where each source
    contributes a rank-1 outer product weighted by salience.

    F_b += alpha_i * (phi_i_b (x) phi_i_b)

    Retrieval is a matrix-vector multiply:
    r_b = F_b @ q_b
    """

    def __init__(self, bands: int, dim: int, dtype: np.dtype = np.float32) -> None:
        """Initialise a zero field.

        Args:
            bands: Number of frequency bands (B).
            dim: Dimensionality per band (D).
            dtype: NumPy dtype for the field tensor.
        """
        self.bands = bands
        self.dim = dim
        self.dtype = dtype
        self.F: NDArray = np.zeros((bands, dim, dim), dtype=dtype)
        self._source_count = 0
        self._sparsity_sum: float = 0.0  # Accumulated sparsity for averaging

    @property
    def source_count(self) -> int:
        """Number of sources superposed into the field."""
        return self._source_count

    @property
    def size_bytes(self) -> int:
        """Size of the field tensor in bytes."""
        return self.F.nbytes

    @property
    def size_mb(self) -> float:
        """Size of the field tensor in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def superpose(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Add a source to the field via rank-1 outer product update.

        F_b += alpha * (phi_b (x) phi_b) for each band b.

        Args:
            phase_vectors: Shape (B, D) — the source's phase spectrum.
            salience: Scalar weight alpha_i (importance * recency * authority * novelty).
        """
        self._validate_phase(phase_vectors)
        for b in range(self.bands):
            phi = phase_vectors[b]
            # Rank-1 update: outer product of phi with itself, scaled by salience
            self.F[b] += salience * np.outer(phi, phi)
        # Track average sparsity (fraction of non-zero components)
        avg_s = np.mean([
            np.count_nonzero(np.abs(phase_vectors[b]) > 1e-6) / self.dim
            for b in range(self.bands)
        ])
        self._sparsity_sum += avg_s
        self._source_count += 1

    def superpose_batch(
        self,
        phase_batch: NDArray[np.float32],
        saliences: NDArray[np.float32] | None = None,
    ) -> None:
        """Add multiple sources to the field in a single BLAS call.

        Uses the identity: sum_i(alpha_i * phi_i @ phi_i^T) = X^T @ diag(alpha) @ X
        For uniform salience: F_b += X_b^T @ X_b (single matmul per band).

        This is ~500x faster than sequential superpose for large batches.

        Args:
            phase_batch: Shape (N, B, D) — batch of phase spectra.
            saliences: Shape (N,) — per-source salience weights. None = all 1.0.
        """
        N = phase_batch.shape[0]
        if saliences is None:
            for b in range(self.bands):
                X = phase_batch[:, b, :]  # (N, D)
                self.F[b] += X.T @ X  # (D, D) — single BLAS call
        else:
            for b in range(self.bands):
                X = phase_batch[:, b, :]  # (N, D)
                # Scale rows by sqrt(salience) so X^T @ X = sum(alpha_i * outer)
                sqrt_s = np.sqrt(saliences)[:, np.newaxis]  # (N, 1)
                Xs = X * sqrt_s
                self.F[b] += Xs.T @ Xs

        # Track stats
        avg_s = np.mean([
            np.mean(np.count_nonzero(np.abs(phase_batch[:, b, :]) > 1e-6, axis=1)) / self.dim
            for b in range(self.bands)
        ])
        self._sparsity_sum += avg_s * N
        self._source_count += N

    def remove(
        self,
        phase_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Remove a source from the field via rank-1 subtraction.

        F_b -= alpha * (phi_b (x) phi_b) for each band b.

        This is algebraically exact if the same phase_vectors and salience
        that were used for superpose are provided.

        Args:
            phase_vectors: Shape (B, D) — the source's phase spectrum (same as superpose).
            salience: Same scalar weight used during superpose.
        """
        self._validate_phase(phase_vectors)
        for b in range(self.bands):
            phi = phase_vectors[b]
            self.F[b] -= salience * np.outer(phi, phi)
        self._source_count -= 1

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Project a query into the field to find resonant sources.

        For each band b:
            r_b = F_b @ q_b

        Then fuse across bands:
            R = sum_b(w_b * r_b)

        Args:
            query_phase: Shape (B, D) — the query's phase spectrum.
            band_weights: Shape (B,) — weights for multi-band fusion.
                If None, uniform weights (1/B) are used.

        Returns:
            ResonanceResult with per-band resonance vectors, fused vector,
            and per-band energy levels.
        """
        self._validate_phase(query_phase)

        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=self.dtype) / self.bands
        else:
            band_weights = np.asarray(band_weights, dtype=self.dtype)
            if band_weights.shape != (self.bands,):
                raise ValueError(
                    f"band_weights shape {band_weights.shape} != ({self.bands},)"
                )

        # Per-band resonance: r_b = F_b @ q_b
        resonance_vectors = np.zeros((self.bands, self.dim), dtype=self.dtype)
        band_energies = np.zeros(self.bands, dtype=self.dtype)

        for b in range(self.bands):
            r_b = self.F[b] @ query_phase[b]
            resonance_vectors[b] = r_b
            band_energies[b] = np.linalg.norm(r_b)

        # Multi-band fusion: R = sum_b(w_b * r_b)
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
        """Estimate the signal-to-noise ratio.

        For sparse vectors: SNR = D / (s^2 * (N - 1))
        For dense vectors:  SNR = D / (N - 1)

        If sparsity is not provided, estimates it empirically from
        the field tensor's effective rank.

        Args:
            num_sources: Override source count. If None, uses self.source_count.
            sparsity: Average fraction of non-zero components per phase vector.
                If None, estimates from field tensor structure.

        Returns:
            Estimated SNR. Values > 10 indicate reliable retrieval.
        """
        n = num_sources if num_sources is not None else self._source_count
        if n <= 1:
            return float("inf")

        if sparsity is None:
            # Use tracked average sparsity from superpose calls
            if self._source_count > 0 and self._sparsity_sum > 0:
                sparsity = self._sparsity_sum / self._source_count
                sparsity = max(0.01, min(1.0, sparsity))
            else:
                sparsity = 1.0  # Assume dense if no tracking data

        return self.dim / (sparsity ** 2 * (n - 1))

    def energy(self) -> NDArray[np.float32]:
        """Compute the Frobenius norm (energy) of each band's field matrix.

        Returns:
            Shape (B,) — energy per band.
        """
        return np.array(
            [np.linalg.norm(self.F[b], "fro") for b in range(self.bands)],
            dtype=self.dtype,
        )

    def resonate_eml(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
        alpha: float = 1.0,
        noise_floor: NDArray[np.float32] | float | None = None,
        fusion: str = "linear",
        weights_exp: NDArray[np.float32] | None = None,
        weights_log: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """EML nonlinear resonance: exp amplifies matches, ln calibrates noise.

        Computes per-band resonance r_b = F_b @ q_b (identical to resonate),
        then applies EML scoring and optional EML band fusion.

        EML scoring per band:
            energy_b = exp(α · ||r_b||) - ln(σ_b)

        EML band fusion (when fusion="eml"):
            R = exp(Σ_b α_b · r_b) - ln(max(Σ_b β_b · |r_b|, ε))

        Args:
            query_phase: Shape (B, D) — the query's phase spectrum.
            band_weights: Shape (B,) — weights for linear fusion (used when fusion="linear").
            alpha: Exponential scaling for EML scoring. Higher = sharper discrimination.
            noise_floor: Per-band noise floor σ. Shape (B,) or scalar.
                If None, uses per-band mean energy as estimate.
            fusion: "linear" (default) or "eml" for nonlinear band fusion.
            weights_exp: Shape (B,) — exp channel weights for EML fusion.
            weights_log: Shape (B,) — log channel weights for EML fusion.

        Returns:
            ResonanceResult with EML-scored band energies and fused vector.
        """
        self._validate_phase(query_phase)

        # Per-band resonance (same as standard resonate)
        resonance_vectors = np.zeros((self.bands, self.dim), dtype=self.dtype)
        raw_energies = np.zeros(self.bands, dtype=self.dtype)

        for b in range(self.bands):
            r_b = self.F[b] @ query_phase[b]
            resonance_vectors[b] = r_b
            raw_energies[b] = np.linalg.norm(r_b)

        # Estimate noise floor if not provided
        if noise_floor is None:
            # Use mean energy across bands as noise floor estimate
            mean_e = max(float(np.mean(raw_energies)), 1e-12)
            sigma = np.full(self.bands, mean_e, dtype=self.dtype)
        elif isinstance(noise_floor, (int, float)):
            sigma = np.full(self.bands, max(float(noise_floor), 1e-12), dtype=self.dtype)
        else:
            sigma = np.maximum(noise_floor, 1e-12)

        # EML scoring: energy_b = exp(α · raw_energy_b) - ln(σ_b)
        band_energies = np.exp(alpha * raw_energies) - np.log(sigma)

        # Band fusion
        if fusion == "eml" and weights_exp is not None and weights_log is not None:
            # EML fusion: exp(Σ α_b · r_b) - ln(max(Σ β_b · |r_b|, ε))
            exp_input = np.einsum("bd,b->d", resonance_vectors, weights_exp)
            log_input = np.einsum("bd,b->d", np.abs(resonance_vectors), weights_log)
            log_input = np.maximum(log_input, 1e-12)
            fused = (np.exp(exp_input) - np.log(log_input)).astype(self.dtype)
        else:
            # Standard linear fusion
            if band_weights is None:
                band_weights = np.ones(self.bands, dtype=self.dtype) / self.bands
            else:
                band_weights = np.asarray(band_weights, dtype=self.dtype)
            fused = np.zeros(self.dim, dtype=self.dtype)
            for b in range(self.bands):
                fused += band_weights[b] * resonance_vectors[b]

        return ResonanceResult(
            resonance_vectors=resonance_vectors,
            fused=fused,
            band_energies=band_energies.astype(self.dtype),
        )

    def reset(self) -> None:
        """Reset the field to zero."""
        self.F.fill(0)
        self._source_count = 0
        self._sparsity_sum = 0.0

    def _validate_phase(self, phase_vectors: NDArray) -> None:
        """Validate phase vector shape and values."""
        expected = (self.bands, self.dim)
        if phase_vectors.shape != expected:
            raise ValueError(
                f"phase_vectors shape {phase_vectors.shape} != expected {expected}"
            )
        if not np.all(np.isfinite(phase_vectors)):
            bad_bands = [
                b for b in range(self.bands)
                if not np.all(np.isfinite(phase_vectors[b]))
            ]
            raise ValueError(
                f"phase_vectors contain NaN/Inf in band(s) {bad_bands}"
            )
