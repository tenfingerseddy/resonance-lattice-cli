# SPDX-License-Identifier: BUSL-1.1
"""Resonance Cascades: multi-hop semantic reasoning via matrix powers.

Discover what's related to what's related to your query — each cascade
hop expands the "resonance horizon" from direct matches to conceptual
landscapes, all via pure linear algebra.

r₁ = F @ q          (1-hop: direct matches)
r₂ = F @ r₁ = F²@q  (2-hop: neighbors of matches)
r₃ = F @ r₂ = F³@q  (3-hop: conceptual landscape)

The closed form (resolvent operator):
    R = (I - αF)⁻¹ @ q  — converges when |α| < 1/λ_max

Cross-band cascades traverse entity → relation → topic without graphs.

Cost: K hops = K × O(BD²). For K=3, that's 3× a single retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField


@dataclass
class CascadeResult:
    """Result of a multi-hop resonance cascade."""
    fused_resonance: NDArray[np.float32]  # (D,) — final weighted cascade result
    per_hop: list[NDArray[np.float32]]  # List of (B, D) resonance vectors per hop
    hop_energies: NDArray[np.float32]  # (K,) — energy at each hop
    band_energies: NDArray[np.float32]  # (B,) — final band energies
    alpha: float  # Decay parameter used
    depth: int  # Number of hops


@dataclass
class CrossBandCascadeResult:
    """Result of a cross-band cascade (e.g. entity → relation → topic)."""
    final_resonance: NDArray[np.float32]  # (D,) — final resonance vector
    per_stage: list[tuple[int, NDArray[np.float32]]]  # [(band_idx, resonance_vector)]
    stage_energies: list[float]


class ResonanceCascade:
    """Multi-hop retrieval via matrix powers on the interference field.

    Standard cascade:
        R = Σ_k α^k · F^k @ q  (truncated Neumann series)

    Cross-band cascade:
        r_entity = F_Ω₄ @ q
        r_relations = F_Ω₃ @ r_entity
        r_topic = F_Ω₂ @ r_relations

    Resolvent (closed-form infinite cascade):
        R = (I - αF)⁻¹ @ q
    """

    @staticmethod
    def cascade(
        field: DenseField,
        query_phase: NDArray[np.float32],
        depth: int = 3,
        alpha: float = 0.1,
        band_weights: NDArray[np.float32] | None = None,
    ) -> CascadeResult:
        """Multi-hop resonance cascade via iterated matrix-vector products.

        Computes: R = Σ_{k=1}^{depth} α^k · F @ ... @ F @ q   (k times)

        Each hop discovers increasingly indirect associations:
        - Hop 1: Direct matches (same as standard retrieval)
        - Hop 2: What's related to the direct matches
        - Hop 3: The conceptual landscape around the query

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — the query.
            depth: Number of cascade hops (1-5 recommended).
            alpha: Decay factor per hop. Must satisfy |α| < 1/λ_max for convergence.
                Smaller α = more emphasis on direct matches.
                Larger α = more emphasis on distant associations.
            band_weights: Per-band fusion weights. Uniform if None.

        Returns:
            CascadeResult with the fused multi-hop resonance.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        B = field.bands
        D = field.dim

        if band_weights is None:
            band_weights = np.ones(B, dtype=np.float32) / B

        # Safety: check alpha vs lambda_max for convergence
        # (approximate — use diagonal norm as cheap upper bound)
        for b in range(B):
            diag_norm = np.max(np.abs(np.diag(field.F[b])))
            if alpha * diag_norm > 0.99:
                # Auto-scale alpha for safety
                safe_alpha = 0.9 / (diag_norm + 1e-12)
                alpha = min(alpha, safe_alpha)

        per_hop = []
        hop_energies = np.zeros(depth, dtype=np.float32)

        # Accumulator for the weighted cascade
        cascade_fused = np.zeros(D, dtype=np.float32)

        # r_0 = query (starting point for iteration)
        r = query_phase.copy()  # (B, D)

        for k in range(depth):
            # One hop: r_{k+1}_b = F_b @ r_k_b
            r_next = np.zeros((B, D), dtype=np.float32)
            hop_energy = 0.0

            for b in range(B):
                r_next[b] = field.F[b] @ r[b]
                hop_energy += np.linalg.norm(r_next[b])

            per_hop.append(r_next.copy())
            hop_energies[k] = hop_energy

            # Weight this hop: α^(k+1)
            weight = alpha ** (k + 1)
            for b in range(B):
                cascade_fused += weight * band_weights[b] * r_next[b]

            r = r_next

        # Final band energies from the last hop
        final_band_energies = np.array([
            np.linalg.norm(per_hop[-1][b]) for b in range(B)
        ], dtype=np.float32)

        return CascadeResult(
            fused_resonance=cascade_fused,
            per_hop=per_hop,
            hop_energies=hop_energies,
            band_energies=final_band_energies,
            alpha=alpha,
            depth=depth,
        )

    @staticmethod
    def cross_band_cascade(
        field: DenseField,
        query_phase: NDArray[np.float32],
        band_sequence: list[int] | None = None,
    ) -> CrossBandCascadeResult:
        """Cross-band cascade: traverse bands in sequence.

        Default sequence: Entity(4) → Relations(3) → Topic(2) → Domain(1)

        r_entity    = F_Ω₄ @ q_Ω₄         "What entities match?"
        r_relations = F_Ω₃ @ r_entity       "What relationships connect those?"
        r_topic     = F_Ω₂ @ r_relations    "What topics encompass those?"

        This achieves multi-hop reasoning across abstraction levels
        without any graph infrastructure.

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — the query.
            band_sequence: Ordered list of band indices to traverse.
                Default: [3, 2, 1, 0] (entity → relations → topic → domain)
                for a 5-band field, or [1, 0] for a 2-band field.

        Returns:
            CrossBandCascadeResult with per-stage resonance.
        """
        B = field.bands

        if band_sequence is None:
            # Default: high → low frequency (specific → general)
            band_sequence = list(range(min(B - 1, 3), -1, -1))

        # Validate
        for b in band_sequence:
            if b >= B:
                raise ValueError(f"Band index {b} >= field.bands ({B})")

        # Start with query projected into the first band
        first_band = band_sequence[0]
        r = field.F[first_band] @ query_phase[first_band]

        per_stage = [(first_band, r.copy())]
        stage_energies = [float(np.linalg.norm(r))]

        # Cascade through remaining bands
        for band_idx in band_sequence[1:]:
            r = field.F[band_idx] @ r
            per_stage.append((band_idx, r.copy()))
            stage_energies.append(float(np.linalg.norm(r)))

        return CrossBandCascadeResult(
            final_resonance=r,
            per_stage=per_stage,
            stage_energies=stage_energies,
        )

    @staticmethod
    def resolvent(
        field: DenseField,
        query_phase: NDArray[np.float32],
        alpha: float = 0.01,
        band: int = 0,
    ) -> NDArray[np.float32]:
        """Closed-form infinite cascade via the resolvent operator.

        R = (I - α·F_b)⁻¹ @ q_b

        This computes the infinite sum Σ_{k=0}^∞ α^k F^k @ q in one operation.
        Equivalent to a random walk that visits all transitive associations.

        CAUTION: Requires α < 1/λ_max(F_b) for convergence. If this condition
        is violated, the result is mathematically undefined.

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — the query.
            alpha: Decay parameter. Must be < 1/λ_max(F_b).
            band: Which band to compute the resolvent for.

        Returns:
            Shape (D,) — the resolvent resonance vector for the specified band.
        """
        D = field.dim
        F_b = field.F[band]

        # (I - αF)
        M = np.eye(D, dtype=np.float32) - alpha * F_b

        # Solve M @ r = q (more numerically stable than explicit inverse)
        r = np.linalg.solve(M, query_phase[band])

        return r.astype(np.float32)

    @staticmethod
    def optimal_alpha(field: DenseField, band: int = 0, safety_margin: float = 0.5) -> float:
        """Compute the optimal cascade decay parameter.

        α_opt = safety_margin / λ_max(F_b)

        A safety_margin of 0.5 means α is half the convergence limit,
        giving good cascade depth while maintaining numerical stability.

        Args:
            field: The dense field.
            band: Which band to compute for.
            safety_margin: Fraction of convergence limit. Default 0.5.

        Returns:
            Optimal alpha value.
        """
        F_b = field.F[band]
        # Approximate λ_max via power iteration (faster than full eigendecomp)
        v = np.random.default_rng(42).standard_normal(field.dim).astype(np.float32)
        v /= np.linalg.norm(v)

        for _ in range(20):  # Power iteration converges fast
            v_new = F_b @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                return 1.0  # Field is essentially zero
            v = v_new / norm

        lambda_max = float(np.dot(v, F_b @ v))
        if lambda_max <= 1e-12:
            return 1.0

        return safety_margin / lambda_max
