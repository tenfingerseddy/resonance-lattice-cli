# SPDX-License-Identifier: BUSL-1.1
"""Asymmetric key/value dense field implementation.

Separates matching (key) from retrieval (value) by storing asymmetric outer
products instead of symmetric ones:

    F_b += alpha_i * (key_i_b ⊗ value_i_b)    instead of    (phi_i ⊗ phi_i)

Query in key-space, result in value-space:
    r_b = F_b.T @ q_b = Σ alpha_i * value_i_b * <key_i_b, q_b>

This is the "star schema" for semantic fields: the key is the foreign key
(optimized for discrimination/matching), and the value is the measure
(optimized for evidence/reconstruction). They live in different spaces
and can have different dimensionalities.

When dim_key == dim_value, and key == value, this degenerates to the
symmetric DenseField — ensuring backward compatibility.

Memory: B * D_key * D_value * precision_bytes.
    At B=5, D_key=256, D_value=512, f32: ~2.5 MB (vs ~80 MB for symmetric D=2048).
Retrieval: O(B * D_key * D_value) per query.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class AsymmetricResonanceResult(NamedTuple):
    """Result of an asymmetric resonance projection.

    Unlike symmetric ResonanceResult, the resonance vectors live in
    value-space (D_value), not key-space (D_key). The query was in
    key-space; the response is in value-space.
    """
    resonance_vectors: NDArray[np.float32]  # shape: (B, D_value) — per-band resonance
    fused: NDArray[np.float32]              # shape: (D_value,) — weighted sum across bands
    band_energies: NDArray[np.float32]      # shape: (B,) — energy per band


class AsymmetricDenseField:
    """Asymmetric key/value dense field: B matrices of shape D_key x D_value.

    Each source contributes a rank-1 asymmetric outer product weighted by salience:
        F_b += alpha_i * (key_i_b ⊗ value_i_b)

    Retrieval projects a query through key-space into value-space:
        r_b = F_b.T @ q_b = Σ alpha_i * value_i_b * <key_i_b, q_b>

    The key vector determines WHEN a source responds (matching selectivity).
    The value vector determines WHAT a source contributes (evidence richness).

    Supports per-band dimensions: each band can have its own D_key and D_value,
    allowing right-sized encoding per semantic layer.

    Algebraic properties preserved:
        - Merge commutativity: F_a + F_b = F_b + F_a (sum of outer products)
        - Forget exactness: F -= alpha * (key ⊗ value) undoes superpose exactly
        - Diff queryability: (F_new - F_old) is queryable, result in value-space
    """

    def __init__(
        self,
        bands: int,
        dim_key: int | tuple[int, ...],
        dim_value: int | tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialise a zero asymmetric field.

        Args:
            bands: Number of frequency bands (B).
            dim_key: Key dimensionality per band. Either a single int (uniform)
                or a tuple of length B (per-band sizing).
            dim_value: Value dimensionality per band. Either a single int (uniform)
                or a tuple of length B (per-band sizing).
            dtype: NumPy dtype for the field matrices.
        """
        self.bands = bands
        self.dtype = dtype

        # Normalize to per-band tuples
        if isinstance(dim_key, int):
            self.dims_key: tuple[int, ...] = tuple(dim_key for _ in range(bands))
        else:
            if len(dim_key) != bands:
                raise ValueError(
                    f"dim_key tuple length {len(dim_key)} != bands {bands}"
                )
            self.dims_key = tuple(dim_key)

        if isinstance(dim_value, int):
            self.dims_value: tuple[int, ...] = tuple(dim_value for _ in range(bands))
        else:
            if len(dim_value) != bands:
                raise ValueError(
                    f"dim_value tuple length {len(dim_value)} != bands {bands}"
                )
            self.dims_value = tuple(dim_value)

        # Per-band field matrices: F[b] has shape (D_key_b, D_value_b)
        self.F: list[NDArray] = [
            np.zeros((dk, dv), dtype=dtype)
            for dk, dv in zip(self.dims_key, self.dims_value)
        ]
        self._source_count = 0
        self._key_sparsity_sum: float = 0.0
        self._value_sparsity_sum: float = 0.0

    @property
    def source_count(self) -> int:
        """Number of sources superposed into the field."""
        return self._source_count

    @property
    def uniform_key_dim(self) -> int | None:
        """Return the uniform key dimension if all bands share the same D_key, else None."""
        if len(set(self.dims_key)) == 1:
            return self.dims_key[0]
        return None

    @property
    def uniform_value_dim(self) -> int | None:
        """Return the uniform value dimension if all bands share the same D_value, else None."""
        if len(set(self.dims_value)) == 1:
            return self.dims_value[0]
        return None

    @property
    def size_bytes(self) -> int:
        """Total size of all field matrices in bytes."""
        return sum(f.nbytes for f in self.F)

    @property
    def size_mb(self) -> float:
        """Total size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def superpose(
        self,
        key_vectors: NDArray[np.float32],
        value_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Add a source to the field via asymmetric rank-1 outer product.

        F_b += alpha * (key_b ⊗ value_b) for each band b.

        Args:
            key_vectors: Per-band key vectors. For uniform dims: shape (B, D_key).
                For per-band dims: list/tuple of arrays, each (D_key_b,).
            value_vectors: Per-band value vectors. For uniform dims: shape (B, D_value).
                For per-band dims: list/tuple of arrays, each (D_value_b,).
            salience: Scalar weight alpha_i.
        """
        keys = self._normalize_vectors(key_vectors, self.dims_key, "key")
        values = self._normalize_vectors(value_vectors, self.dims_value, "value")

        for b in range(self.bands):
            self.F[b] += salience * np.outer(keys[b], values[b])

        # Track sparsity
        self._key_sparsity_sum += np.mean([
            np.count_nonzero(np.abs(keys[b]) > 1e-6) / self.dims_key[b]
            for b in range(self.bands)
        ])
        self._value_sparsity_sum += np.mean([
            np.count_nonzero(np.abs(values[b]) > 1e-6) / self.dims_value[b]
            for b in range(self.bands)
        ])
        self._source_count += 1

    def superpose_batch(
        self,
        key_batch: NDArray[np.float32],
        value_batch: NDArray[np.float32],
        saliences: NDArray[np.float32] | None = None,
    ) -> None:
        """Add multiple sources in a single BLAS call per band.

        Uses: F_b += K_b.T @ diag(alpha) @ V_b  (or K_b.T @ V_b for uniform salience).

        Args:
            key_batch: Shape (N, B, D_key) — batch of key spectra (uniform dims only).
            value_batch: Shape (N, B, D_value) — batch of value spectra (uniform dims only).
            saliences: Shape (N,) — per-source salience weights. None = all 1.0.
        """
        N = key_batch.shape[0]
        if saliences is None:
            for b in range(self.bands):
                K = key_batch[:, b, :]   # (N, D_key)
                V = value_batch[:, b, :] # (N, D_value)
                self.F[b] += K.T @ V     # (D_key, D_value)
        else:
            for b in range(self.bands):
                K = key_batch[:, b, :]   # (N, D_key)
                V = value_batch[:, b, :] # (N, D_value)
                # Scale keys by salience: (alpha * key) ⊗ value = key.T @ diag(alpha) @ value
                Ks = K * saliences[:, np.newaxis]  # (N, D_key)
                self.F[b] += Ks.T @ V

        # Track stats
        self._key_sparsity_sum += np.mean([
            np.mean(np.count_nonzero(np.abs(key_batch[:, b, :]) > 1e-6, axis=1))
            / self.dims_key[b]
            for b in range(self.bands)
        ]) * N
        self._value_sparsity_sum += np.mean([
            np.mean(np.count_nonzero(np.abs(value_batch[:, b, :]) > 1e-6, axis=1))
            / self.dims_value[b]
            for b in range(self.bands)
        ]) * N
        self._source_count += N

    def remove(
        self,
        key_vectors: NDArray[np.float32],
        value_vectors: NDArray[np.float32],
        salience: float = 1.0,
    ) -> None:
        """Remove a source via asymmetric rank-1 subtraction.

        F_b -= alpha * (key_b ⊗ value_b) for each band b.

        Algebraically exact if the same key_vectors, value_vectors, and salience
        that were used for superpose are provided.

        Args:
            key_vectors: Same key vectors used during superpose.
            value_vectors: Same value vectors used during superpose.
            salience: Same scalar weight used during superpose.
        """
        keys = self._normalize_vectors(key_vectors, self.dims_key, "key")
        values = self._normalize_vectors(value_vectors, self.dims_value, "value")

        for b in range(self.bands):
            self.F[b] -= salience * np.outer(keys[b], values[b])
        self._source_count -= 1

    def resonate(
        self,
        query_key: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> AsymmetricResonanceResult:
        """Project a query through key-space into value-space.

        For each band b:
            r_b = F_b.T @ q_b    (query in D_key, result in D_value)

        Then fuse across bands (only if all bands share the same D_value):
            R = sum_b(w_b * r_b)

        Args:
            query_key: Per-band query key vectors. For uniform dims: shape (B, D_key).
                For per-band dims: list/tuple of arrays, each (D_key_b,).
            band_weights: Shape (B,) — weights for multi-band fusion.
                If None, uniform weights (1/B) are used.

        Returns:
            AsymmetricResonanceResult with per-band resonance in value-space,
            fused vector (if uniform D_value), and per-band energies.
        """
        keys = self._normalize_vectors(query_key, self.dims_key, "query_key")

        if band_weights is None:
            band_weights = np.ones(self.bands, dtype=self.dtype) / self.bands
        else:
            band_weights = np.asarray(band_weights, dtype=self.dtype)
            if band_weights.shape != (self.bands,):
                raise ValueError(
                    f"band_weights shape {band_weights.shape} != ({self.bands},)"
                )

        # Per-band resonance: r_b = F_b.T @ q_b (key-space → value-space)
        resonance_vectors: list[NDArray] = []
        band_energies = np.zeros(self.bands, dtype=self.dtype)

        for b in range(self.bands):
            r_b = self.F[b].T @ keys[b]  # (D_value_b,)
            resonance_vectors.append(r_b)
            band_energies[b] = np.linalg.norm(r_b)

        # Multi-band fusion (uniform D_value required)
        if self.uniform_value_dim is not None:
            dv = self.uniform_value_dim
            res_array = np.stack(resonance_vectors)  # (B, D_value)
            fused = np.zeros(dv, dtype=self.dtype)
            for b in range(self.bands):
                fused += band_weights[b] * resonance_vectors[b]
        else:
            # Per-band dimensions differ — pack as ragged array, fused is zero-length
            max_dv = max(self.dims_value)
            res_array = np.zeros((self.bands, max_dv), dtype=self.dtype)
            for b in range(self.bands):
                res_array[b, :self.dims_value[b]] = resonance_vectors[b]
            fused = np.zeros(0, dtype=self.dtype)

        return AsymmetricResonanceResult(
            resonance_vectors=res_array,
            fused=fused,
            band_energies=band_energies,
        )

    def compute_snr(
        self,
        num_sources: int | None = None,
        key_sparsity: float | None = None,
    ) -> float:
        """Estimate the signal-to-noise ratio for the key-matching path.

        SNR = min(D_key_b) / (s_key^2 * (N - 1))

        Uses the minimum D_key across bands as the bottleneck dimension.

        Args:
            num_sources: Override source count. If None, uses self.source_count.
            key_sparsity: Average key sparsity. If None, uses tracked average.

        Returns:
            Estimated SNR. Values > 10 indicate reliable retrieval.
        """
        n = num_sources if num_sources is not None else self._source_count
        if n <= 1:
            return float("inf")

        if key_sparsity is None:
            if self._source_count > 0 and self._key_sparsity_sum > 0:
                key_sparsity = self._key_sparsity_sum / self._source_count
                key_sparsity = max(0.01, min(1.0, key_sparsity))
            else:
                key_sparsity = 1.0

        min_dk = min(self.dims_key)
        return min_dk / (key_sparsity ** 2 * (n - 1))

    def energy(self) -> NDArray[np.float32]:
        """Compute the Frobenius norm (energy) of each band's field matrix.

        Returns:
            Shape (B,) — energy per band.
        """
        return np.array(
            [np.linalg.norm(self.F[b], "fro") for b in range(self.bands)],
            dtype=self.dtype,
        )

    def reset(self) -> None:
        """Reset the field to zero."""
        for b in range(self.bands):
            self.F[b].fill(0)
        self._source_count = 0
        self._key_sparsity_sum = 0.0
        self._value_sparsity_sum = 0.0

    def _normalize_vectors(
        self,
        vectors: NDArray | list | tuple,
        expected_dims: tuple[int, ...],
        name: str,
    ) -> list[NDArray]:
        """Normalize input to a list of per-band vectors with correct dimensions.

        Accepts either:
        - ndarray of shape (B, D) for uniform dimensions
        - list/tuple of B arrays, each (D_b,) for per-band dimensions

        Returns:
            List of B arrays, each validated against expected_dims[b].
        """
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            if vectors.shape[0] != self.bands:
                raise ValueError(
                    f"{name}_vectors bands {vectors.shape[0]} != expected {self.bands}"
                )
            result = []
            for b in range(self.bands):
                v = vectors[b]
                if v.shape[0] != expected_dims[b]:
                    raise ValueError(
                        f"{name}_vectors band {b} dim {v.shape[0]} != expected {expected_dims[b]}"
                    )
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"{name}_vectors contain NaN/Inf in band {b}")
                result.append(v)
            return result
        elif isinstance(vectors, (list, tuple)):
            if len(vectors) != self.bands:
                raise ValueError(
                    f"{name}_vectors length {len(vectors)} != expected {self.bands}"
                )
            result = []
            for b in range(self.bands):
                v = np.asarray(vectors[b], dtype=self.dtype)
                if v.shape[0] != expected_dims[b]:
                    raise ValueError(
                        f"{name}_vectors band {b} dim {v.shape[0]} != expected {expected_dims[b]}"
                    )
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"{name}_vectors contain NaN/Inf in band {b}")
                result.append(v)
            return result
        else:
            raise TypeError(
                f"{name}_vectors must be ndarray (B, D) or list of per-band arrays, "
                f"got {type(vectors)}"
            )
