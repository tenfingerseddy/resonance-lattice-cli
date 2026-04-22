# SPDX-License-Identifier: BUSL-1.1
"""Field confidence estimation and meta-analysis.

Measures what the field knows well, what it doesn't know, where it
contradicts itself, and how confident it is — all encoded as field
patterns alongside the corpus content.

F_confident = F_corpus + F_meta

F_meta is built from:
1. Confidence map — from eigenvalue analysis
2. Gap indicators — from topological invariants (negative patterns = "holes")
3. Contradiction markers — from interference scoring
4. Structure model — from cluster analysis

Returns confidence, gaps, and contradictions for any queried topic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .calculus import FieldCalculus, FieldConfidence
from .field.dense import DenseField, ResonanceResult
from .topology import FieldTopology, TopologicalAnalyzer


@dataclass
class IntrospectionResult:
    """What the field knows about its own knowledge on a topic."""
    confidence: float  # 0-1: how confident the field is about this topic
    coverage: float  # 0-1: how much of the query space is covered
    contradiction_risk: float  # 0-1: likelihood of contradictory sources
    knowledge_depth: float  # 0-1: how deep the knowledge goes (eigenvalue concentration)
    gap_count: int  # Number of detected knowledge gaps near this topic
    recommendation: str  # "high_confidence" | "low_confidence" | "contradictory" | "insufficient"


@dataclass
class ConsciousnessState:
    """The field's current self-model."""
    global_confidence: FieldConfidence  # Overall field confidence
    topology: FieldTopology  # Topological structure
    meta_energy: float  # Energy in the meta-field
    introspection_count: int  # How many introspections performed


class FieldConsciousness:
    """A self-aware wrapper around a knowledge field.

    Maintains a meta-model of the field's own knowledge:
    - Confidence landscape (where is knowledge strong/weak?)
    - Topological structure (clusters, gaps, loops)
    - Contradiction map (where do sources disagree?)

    The meta-model is refreshed on demand and used to provide
    honest, calibrated assessments of retrieval confidence.
    """

    def __init__(self, field: DenseField, analysis_band: int = 0):
        self.field = field
        self.analysis_band = analysis_band
        self._state: ConsciousnessState | None = None
        self._introspection_count = 0

    def refresh(self) -> ConsciousnessState:
        """Rebuild the self-model from current field state.

        Runs Field Calculus confidence + Topological analysis.
        Call this after significant field changes (new sources, sculpting, etc.)
        """
        confidence = FieldCalculus.field_confidence(
            self.field, band=self.analysis_band, top_k=50,
        )
        topology = TopologicalAnalyzer.analyze(
            self.field, band=self.analysis_band,
            persistence_threshold=0.01, max_eigenvectors=100,
        )

        meta_energy = float(np.sum(confidence.top_eigenvalues))

        self._state = ConsciousnessState(
            global_confidence=confidence,
            topology=topology,
            meta_energy=meta_energy,
            introspection_count=self._introspection_count,
        )
        return self._state

    def introspect(
        self,
        query_phase: NDArray[np.float32],
    ) -> IntrospectionResult:
        """Introspect: what does the field know about this query's topic?

        Uses the field's self-model to assess:
        - Confidence: how strongly does the field resonate? (high energy = confident)
        - Coverage: what fraction of the query's dimensions are well-covered?
        - Contradiction risk: do nearby sources destructively interfere?
        - Knowledge depth: is the knowledge concentrated or diffuse?

        Args:
            query_phase: Shape (B, D) — the topic to introspect about.

        Returns:
            IntrospectionResult with confidence, gaps, and recommendation.
        """
        self._introspection_count += 1

        # Ensure state exists
        if self._state is None:
            self.refresh()

        D = self.field.dim
        b = self.analysis_band

        # 1. Resonance energy → confidence
        resonance = self.field.resonate(query_phase)
        query_energy = float(np.sum(resonance.band_energies))

        # Normalise by field's total energy
        field_energy = float(np.linalg.norm(self.field.F[b], "fro")) + 1e-12
        confidence = min(1.0, query_energy / (field_energy * 0.1 + 1e-12))

        # 2. Coverage: what fraction of query dimensions have strong field support?
        q = query_phase[b]
        r = resonance.resonance_vectors[b]

        # Dimensions where both query and resonance are active
        q_active = np.abs(q) > 1e-6
        r_active = np.abs(r) > np.median(np.abs(r)) * 0.1
        if np.sum(q_active) > 0:
            coverage = float(np.sum(q_active & r_active) / np.sum(q_active))
        else:
            coverage = 0.0

        # 3. Contradiction risk: check curvature around query
        # High curvature variance = contradictory sources pulling in different directions
        curvature = FieldCalculus.knowledge_curvature(self.field, query_phase)
        curvature_variance = float(np.var(curvature.curvature_ratio))
        contradiction_risk = min(1.0, curvature_variance * 10)  # Scale to 0-1

        # 4. Knowledge depth: concentration of resonance energy
        # If energy is concentrated in few dimensions → deep knowledge
        # If spread uniformly → shallow/broad knowledge
        r_sorted = np.sort(np.abs(r))[::-1]
        r_total = float(np.sum(r_sorted)) + 1e-12
        r_cumulative = np.cumsum(r_sorted) / r_total
        # How many dimensions capture 80% of energy?
        depth_dims = np.searchsorted(r_cumulative, 0.8) + 1
        knowledge_depth = 1.0 - (depth_dims / D)  # Higher = more concentrated

        # 5. Gap count: topological features near this query
        gap_count = self._state.topology.knowledge_clusters  # Proxy for gap detection

        # 6. Recommendation
        if confidence > 0.7 and contradiction_risk < 0.3:
            recommendation = "high_confidence"
        elif confidence > 0.3 and contradiction_risk < 0.5:
            recommendation = "moderate_confidence"
        elif contradiction_risk > 0.5:
            recommendation = "contradictory"
        else:
            recommendation = "insufficient"

        return IntrospectionResult(
            confidence=confidence,
            coverage=coverage,
            contradiction_risk=contradiction_risk,
            knowledge_depth=knowledge_depth,
            gap_count=gap_count,
            recommendation=recommendation,
        )

    def honest_resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> tuple[ResonanceResult, IntrospectionResult]:
        """Resonate AND introspect: return results with confidence assessment.

        This is the "conscious" retrieval — every result comes with the
        field's honest assessment of how much to trust it.

        Args:
            query_phase: Shape (B, D).
            band_weights: Per-band weights for resonance.

        Returns:
            (resonance_result, introspection) — results + self-assessment.
        """
        resonance = self.field.resonate(query_phase, band_weights=band_weights)
        introspection = self.introspect(query_phase)

        return resonance, introspection

    @property
    def state(self) -> ConsciousnessState | None:
        return self._state
