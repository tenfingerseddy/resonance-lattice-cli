# SPDX-License-Identifier: BUSL-1.1
"""Spectral Attention: multi-head resonance on eigenspaces.

Standard retrieval: r = F @ q  (one flat resonance)
Spectral attention: r = Σ_k α_k(q) · Head_k(q)  (structured, multi-resolution)

Each head attends to a different partition of the eigenspectrum:
- Head 1: dominant knowledge (top eigenvalues — main themes)
- Head 2: supporting details (mid eigenvalues — nuance)
- Head 3: rare/novel information (low eigenvalues — specific but unusual)
- Head 4: contradictory signals (negative contributions — disagreements)

Attention weights α_k(q) are query-dependent — derived from how much
energy the query produces in each spectral region.

This is transformer-style multi-head attention, but the heads are
spectral decompositions of the knowledge field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ..field.dense import DenseField


class PartitionStrategy(StrEnum):
    """How to partition eigenvalues into attention heads."""
    EQUAL_COUNT = "equal_count"  # Same number of eigenvalues per head
    EQUAL_ENERGY = "equal_energy"  # Same total energy per head
    LOGARITHMIC = "logarithmic"  # Log-spaced boundaries (more heads for large eigenvalues)
    ADAPTIVE = "adaptive"  # Gap-based: split at spectral gaps


@dataclass
class AttentionHead:
    """A single spectral attention head."""
    index: int
    eigenvalue_range: tuple[float, float]  # (min, max) eigenvalue in this head
    energy_fraction: float  # Fraction of total field energy
    eigenvector_indices: NDArray[np.int64]  # Which eigenvectors belong to this head


@dataclass
class AttentionResult:
    """Result of multi-head spectral attention."""
    fused: NDArray[np.float32]  # (D,) — final weighted resonance
    per_head: list[NDArray[np.float32]]  # List of (D,) head resonances
    head_weights: NDArray[np.float32]  # (n_heads,) — attention weights per head
    head_energies: NDArray[np.float32]  # (n_heads,) — energy per head
    band: int


class SpectralAttention:
    """Multi-head attention on the eigenspectrum of a field band.

    Partitions the eigenspectrum into n_heads groups. Each head computes
    resonance using only its assigned eigenvectors. Attention weights
    are computed per-query from the energy each head produces.
    """

    def __init__(
        self,
        field: DenseField,
        n_heads: int = 4,
        band: int = 0,
        strategy: PartitionStrategy | str = PartitionStrategy.EQUAL_ENERGY,
    ):
        self.field = field
        self.n_heads = n_heads
        self.band = band

        if isinstance(strategy, str):
            strategy = PartitionStrategy(strategy)

        # Eigendecompose once (cached)
        F_b = field.F[band]
        F_sym = (F_b + F_b.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        # Sort descending by absolute eigenvalue
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]

        # Build head partitions
        self._heads = self._partition(strategy)

        # Pre-compute per-head field projections: F_head_k = V_k Λ_k V_kᵀ
        self._head_fields: list[NDArray[np.float32]] = []
        for head in self._heads:
            idxs = head.eigenvector_indices
            V_k = self._eigenvectors[:, idxs]
            L_k = self._eigenvalues[idxs]
            # F_k = V_k · diag(L_k) · V_kᵀ
            F_k = (V_k * L_k) @ V_k.T
            self._head_fields.append(F_k.astype(np.float32))

    def _partition(self, strategy: PartitionStrategy) -> list[AttentionHead]:
        """Partition eigenvalues into n_heads groups."""
        n = len(self._eigenvalues)
        abs_eigs = np.abs(self._eigenvalues)
        total_energy = float(np.sum(abs_eigs)) + 1e-12

        if strategy == PartitionStrategy.EQUAL_COUNT:
            boundaries = np.linspace(0, n, self.n_heads + 1, dtype=int)
        elif strategy == PartitionStrategy.EQUAL_ENERGY:
            cumulative = np.cumsum(abs_eigs)
            target_per_head = cumulative[-1] / self.n_heads if len(cumulative) > 0 else 0
            boundaries = [0]
            for k in range(1, self.n_heads):
                target = k * target_per_head
                idx = np.searchsorted(cumulative, target)
                boundaries.append(max(idx, boundaries[-1] + 1))
            boundaries.append(n)
            boundaries = np.array(boundaries, dtype=int)
        elif strategy == PartitionStrategy.LOGARITHMIC:
            # Log-spaced: more heads for high eigenvalues
            log_boundaries = np.logspace(0, np.log10(n + 1), self.n_heads + 1) - 1
            boundaries = np.clip(log_boundaries.astype(int), 0, n)
            boundaries[0] = 0
            boundaries[-1] = n
            # Ensure monotonic
            for i in range(1, len(boundaries)):
                boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1)
                boundaries[i] = min(boundaries[i], n)
        elif strategy == PartitionStrategy.ADAPTIVE:
            # Split at largest spectral gaps
            if n <= self.n_heads:
                boundaries = np.arange(n + 1, dtype=int)
            else:
                gaps = np.abs(np.diff(abs_eigs))
                # Find top-(n_heads-1) gaps
                gap_indices = np.argsort(gaps)[::-1][: self.n_heads - 1]
                gap_indices = np.sort(gap_indices) + 1  # Convert to boundary
                boundaries = np.concatenate([[0], gap_indices, [n]])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        heads = []
        for k in range(min(self.n_heads, len(boundaries) - 1)):
            start = int(boundaries[k])
            end = int(boundaries[k + 1])
            if start >= end:
                continue

            idxs = np.arange(start, end)
            head_energy = float(np.sum(abs_eigs[start:end]))

            heads.append(AttentionHead(
                index=k,
                eigenvalue_range=(
                    float(abs_eigs[min(end - 1, n - 1)]),
                    float(abs_eigs[start]),
                ),
                energy_fraction=head_energy / total_energy,
                eigenvector_indices=idxs,
            ))

        return heads

    def attend(
        self,
        query_phase: NDArray[np.float32],
        temperature: float = 1.0,
    ) -> AttentionResult:
        """Compute multi-head spectral attention for a query.

        1. Each head computes resonance using its eigensubspace
        2. Attention weights derived from per-head query energy
        3. Fused output = weighted sum of head resonances

        Args:
            query_phase: Shape (B, D) — uses self.band.
            temperature: Softmax temperature for attention weights.
                < 1.0: sharper (concentrate on strongest head)
                > 1.0: flatter (more uniform across heads)

        Returns:
            AttentionResult with per-head resonances and fused output.
        """
        q = query_phase[self.band]
        n_heads = len(self._heads)

        # Compute per-head resonance
        head_resonances = []
        head_energies = np.zeros(n_heads, dtype=np.float32)

        for k in range(n_heads):
            r_k = self._head_fields[k] @ q
            head_resonances.append(r_k.astype(np.float32))
            head_energies[k] = np.linalg.norm(r_k)

        # Attention weights via softmax over head energies
        scaled = head_energies / (temperature + 1e-12)
        scaled -= np.max(scaled)  # Numerical stability
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / (np.sum(exp_scaled) + 1e-12)

        # Fused resonance
        fused = np.zeros(self.field.dim, dtype=np.float32)
        for k in range(n_heads):
            fused += weights[k] * head_resonances[k]

        return AttentionResult(
            fused=fused,
            per_head=head_resonances,
            head_weights=weights,
            head_energies=head_energies,
            band=self.band,
        )

    def attend_all_bands(
        self,
        query_phase: NDArray[np.float32],
        temperature: float = 1.0,
    ) -> list[AttentionResult]:
        """Convenience: create SpectralAttention per band and attend.

        Note: This creates a new SpectralAttention for each band (eigendecomposition).
        For repeated use, create per-band instances once and reuse.
        """
        results = []
        for b in range(self.field.bands):
            sa = SpectralAttention(
                self.field, self.n_heads, band=b,
                strategy=PartitionStrategy.EQUAL_ENERGY,
            )
            results.append(sa.attend(query_phase, temperature))
        return results

    @property
    def heads(self) -> list[AttentionHead]:
        """Access the head metadata."""
        return self._heads

    def head_info(self) -> list[dict]:
        """Return human-readable info about each head."""
        return [
            {
                "head": h.index,
                "eigenvalue_range": h.eigenvalue_range,
                "energy_fraction": f"{h.energy_fraction:.1%}",
                "n_eigenvectors": len(h.eigenvector_indices),
            }
            for h in self._heads
        ]
