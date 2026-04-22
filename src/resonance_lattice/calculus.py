# SPDX-License-Identifier: BUSL-1.1
"""Field Calculus: derivatives and differential operators on knowledge fields.

The resonance vector IS the gradient of the field's energy landscape.
The field tensor IS the Hessian. This module makes those relationships
explicit and exploitable.

Capabilities:
    auto_band_weights(resonance) - Gradient-derived per-query optimal band weights
    expand_query(field, query, steps, eta) - Follow the energy gradient to discover related concepts
    field_confidence(field, band) - Eigenvalue-based uncertainty quantification
    knowledge_curvature(field, query) - Per-band curvature at a query point
    semantic_gradient(field, query) - The resonance vector reframed as a gradient direction

All operations are O(B*D²) or less — independent of corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField, ResonanceResult


@dataclass
class BandWeights:
    """Gradient-derived optimal band weights for a specific query."""
    weights: NDArray[np.float32]  # (B,) — normalised weights
    energies: NDArray[np.float32]  # (B,) — raw band energies (the gradient)
    dominant_band: int  # Index of the strongest band


@dataclass
class QueryExpansion:
    """Result of gradient-based query expansion."""
    expanded_query: NDArray[np.float32]  # (B, D) — the expanded query phase vectors
    original_query: NDArray[np.float32]  # (B, D) — the original
    steps_taken: int
    energy_trajectory: list[float]  # Energy at each step
    final_resonance: ResonanceResult | None = None  # Resonance at the expanded point


@dataclass
class FieldConfidence:
    """Uncertainty quantification via eigenvalue analysis."""
    condition_number: float  # λ_max / λ_min — high = confident in dominant direction
    effective_rank: float  # Number of significant eigenvalues
    spectral_entropy: float  # Shannon entropy of normalised eigenvalue distribution
    top_eigenvalues: NDArray[np.float32]  # Top-K eigenvalues
    explained_variance_ratio: NDArray[np.float32]  # Cumulative explained variance


@dataclass
class KnowledgeCurvature:
    """Curvature of the energy landscape at a query point."""
    per_band_curvature: NDArray[np.float32]  # (B,) — qᵀ F_b q per band
    total_curvature: float  # Sum across bands
    curvature_ratio: NDArray[np.float32]  # (B,) — normalised curvature per band
    is_confident: bool  # True if curvature is well-concentrated


class FieldCalculus:
    """Differential operators on Resonance Fields.

    The energy landscape of band b is: E_b(q) = ½ qᵀ F_b q
    Its gradient is: ∇E_b = F_b @ q = r_b (the resonance vector)
    Its Hessian is: H_b = F_b (the field tensor itself)

    These relationships enable:
    - Auto-tuning band weights per query (no MLP needed)
    - Query expansion by gradient ascent on the energy surface
    - Uncertainty quantification from the eigenvalue spectrum
    - Knowledge curvature analysis at any query point
    """

    @staticmethod
    def auto_band_weights(
        resonance: ResonanceResult,
        temperature: float = 1.0,
    ) -> BandWeights:
        """Compute gradient-derived optimal band weights for a query.

        The gradient of the retrieval energy w.r.t. band weight w_b is simply
        the energy of band b's resonance: ‖F_b @ q_b‖. Normalising these
        gives the optimal weighting that maximises total retrieval energy.

        With temperature:
            w_b = softmax(‖r_b‖ / τ)

        τ > 1: flatter (more uniform) weights — better for exploration
        τ < 1: sharper (more concentrated) weights — better for precision
        τ = 1: energy-proportional weights (default)

        Args:
            resonance: Result from field.resonate() — contains band_energies.
            temperature: Softmax temperature. Default 1.0.

        Returns:
            BandWeights with normalised weights and diagnostics.
        """
        energies = resonance.band_energies
        len(energies)

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Softmax over band energies / temperature
        scaled = energies / temperature
        scaled -= np.max(scaled)  # Numerical stability
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / (np.sum(exp_scaled) + 1e-12)

        return BandWeights(
            weights=weights.astype(np.float32),
            energies=energies.astype(np.float32),
            dominant_band=int(np.argmax(energies)),
        )

    @staticmethod
    def expand_query(
        field: DenseField,
        query_phase: NDArray[np.float32],
        steps: int = 1,
        eta: float = 0.2,
        normalise: bool = True,
    ) -> QueryExpansion:
        """Expand a query by following the energy gradient.

        q' = (1 - η) q + η · normalise(F @ q)

        Each step moves the query toward the centroid of relevant documents
        in the field, discovering related concepts the original query didn't
        mention. This is principled query expansion via gradient ascent on
        the field's energy landscape.

        Args:
            field: The dense field to expand against.
            query_phase: Shape (B, D) — the original query.
            steps: Number of gradient steps. 1 is usually sufficient.
            eta: Step size / interpolation weight. 0.2 = 80% original + 20% gradient.
            normalise: Whether to L2-normalise the expanded query per band.

        Returns:
            QueryExpansion with the expanded query and energy trajectory.
        """
        if not 0 < eta < 1:
            raise ValueError(f"eta must be in (0, 1), got {eta}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        q = query_phase.copy()
        original = query_phase.copy()
        energy_trajectory = []

        for _ in range(steps):
            # Compute resonance (this IS the gradient)
            resonance = field.resonate(q)
            energy_trajectory.append(float(np.sum(resonance.band_energies)))

            # Gradient step: interpolate query toward resonance direction
            for b in range(field.bands):
                r_b = resonance.resonance_vectors[b]
                r_norm = np.linalg.norm(r_b)
                if r_norm > 1e-8:
                    direction = r_b / r_norm
                    q[b] = (1 - eta) * q[b] + eta * direction

                    # Re-normalise to stay on the unit sphere
                    if normalise:
                        q_norm = np.linalg.norm(q[b])
                        if q_norm > 1e-8:
                            q[b] /= q_norm

        # Final energy
        final_resonance = field.resonate(q)
        energy_trajectory.append(float(np.sum(final_resonance.band_energies)))

        return QueryExpansion(
            expanded_query=q.astype(np.float32),
            original_query=original,
            steps_taken=steps,
            energy_trajectory=energy_trajectory,
            final_resonance=final_resonance,
        )

    @staticmethod
    def field_confidence(
        field: DenseField,
        band: int = 0,
        top_k: int = 50,
    ) -> FieldConfidence:
        """Quantify the field's confidence via eigenvalue analysis.

        The eigenvalue spectrum of F_b reveals the knowledge structure:
        - Large eigenvalue gap → strong dominant knowledge axis (confident)
        - Flat spectrum → diffuse, uncertain knowledge
        - Few large eigenvalues → low-rank, well-structured field

        Metrics:
        - condition_number: λ_max / λ_min — high means strong directionality
        - effective_rank: exp(entropy(λ/Σλ)) — number of "real" dimensions
        - spectral_entropy: normalised entropy — 0 = one dominant direction, 1 = uniform

        Args:
            field: The dense field.
            band: Which band to analyse.
            top_k: Number of eigenvalues to return.

        Returns:
            FieldConfidence with uncertainty metrics.
        """
        F_b = field.F[band]
        # Symmetrise for numerical stability
        F_sym = (F_b + F_b.T) / 2.0

        eigenvalues = np.linalg.eigvalsh(F_sym)
        # Sort descending by absolute value
        eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]

        # Condition number
        abs_eigs = np.abs(eigenvalues)
        max_eig = abs_eigs[0] if len(abs_eigs) > 0 else 0
        min_nonzero = abs_eigs[abs_eigs > 1e-12]
        min_eig = min_nonzero[-1] if len(min_nonzero) > 0 else 1e-12
        condition = max_eig / min_eig

        # Effective rank via spectral entropy
        positive_eigs = abs_eigs[abs_eigs > 1e-12]
        if len(positive_eigs) > 0:
            p = positive_eigs / np.sum(positive_eigs)
            entropy = -np.sum(p * np.log(p + 1e-12))
            max_entropy = np.log(len(positive_eigs)) if len(positive_eigs) > 1 else 1.0
            spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            effective_rank = np.exp(entropy)
        else:
            spectral_entropy = 0.0
            effective_rank = 0.0

        # Explained variance ratio
        total = np.sum(abs_eigs) + 1e-12
        explained = np.cumsum(abs_eigs[:top_k]) / total

        return FieldConfidence(
            condition_number=float(condition),
            effective_rank=float(effective_rank),
            spectral_entropy=float(spectral_entropy),
            top_eigenvalues=abs_eigs[:top_k].astype(np.float32),
            explained_variance_ratio=explained.astype(np.float32),
        )

    @staticmethod
    def knowledge_curvature(
        field: DenseField,
        query_phase: NDArray[np.float32],
    ) -> KnowledgeCurvature:
        """Compute the curvature of the energy landscape at a query point.

        Curvature at q for band b: κ_b = qᵀ F_b q

        High curvature → the field has strong knowledge in this direction
        Low curvature → weak/sparse knowledge in this direction
        The ratio across bands reveals which bands are most informative for this query.

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — the query point.

        Returns:
            KnowledgeCurvature with per-band and total curvature.
        """
        B = field.bands
        curvatures = np.zeros(B, dtype=np.float32)

        for b in range(B):
            # κ_b = qᵀ F_b q = dot(q, F_b @ q)
            r_b = field.F[b] @ query_phase[b]
            curvatures[b] = float(np.dot(query_phase[b], r_b))

        total = float(np.sum(np.abs(curvatures)))
        ratio = np.abs(curvatures) / (total + 1e-12)

        # Confidence: curvature is well-concentrated if max ratio > 0.5
        is_confident = bool(np.max(ratio) > 0.5)

        return KnowledgeCurvature(
            per_band_curvature=curvatures,
            total_curvature=total,
            curvature_ratio=ratio.astype(np.float32),
            is_confident=is_confident,
        )

    @staticmethod
    def semantic_gradient(
        field: DenseField,
        query_phase: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return the semantic gradient: the direction in embedding space
        where more relevant information lives.

        This IS the resonance vector, reframed: ∇E = F @ q.
        Each band's gradient points toward the centroid of aligned sources.

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — the query.

        Returns:
            Shape (B, D) — normalised gradient direction per band.
        """
        B = field.bands
        D = field.dim
        gradient = np.zeros((B, D), dtype=np.float32)

        for b in range(B):
            g = field.F[b] @ query_phase[b]
            norm = np.linalg.norm(g)
            if norm > 1e-8:
                gradient[b] = g / norm
            else:
                gradient[b] = g

        return gradient
