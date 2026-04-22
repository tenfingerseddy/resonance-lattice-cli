# SPDX-License-Identifier: BUSL-1.1
"""Probabilistic interference scoring for knowledge retrieval.

Replaces deterministic scoring with probabilistic scoring where:
- Amplitude: psi_i = phi_i^T F q  (same as current resonance — can be negative)
- Probability: P(relevant_i) = |psi_i|^2 / sum_j |psi_j|^2  (Born rule)

Sources that constructively interfere (aligned phases) get amplified.
Sources that destructively interfere (anti-aligned) get cancelled.
Contradictions fall out of the math — no explicit detection needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField


@dataclass
class QuantumScore:
    """A single source's quantum-inspired score."""
    source_id: str
    amplitude: float  # ψ — can be negative (phase carries meaning)
    probability: float  # |ψ|² / Z — always non-negative, sums to 1
    phase_sign: int  # +1 or -1 (constructive or destructive)


@dataclass
class Contradiction:
    """A pair of sources with destructive interference."""
    source_a: str
    source_b: str
    interference: float  # Negative = destructive (contradiction)
    strength: float  # Absolute magnitude of interference


@dataclass
class QuantumRetrievalResult:
    """Complete quantum scoring result."""
    scores: list[QuantumScore]  # Sorted by probability descending
    contradictions: list[Contradiction]  # Pairs with destructive interference
    partition_function: float  # Z = Σ |ψ_j|² (normalisation constant)
    constructive_fraction: float  # Fraction of sources with positive amplitude


class QuantumScorer:
    """Born rule scoring with constructive/destructive interference analysis.

    Standard scoring: s_i = φᵢᵀ F q (deterministic, can be dominated by high-energy sources)
    Quantum scoring: P_i = |φᵢᵀ F q|² / Z (probabilistic, naturally normalised)

    The Born rule has two key advantages:
    1. Squaring amplitudes amplifies high-quality matches and suppresses noise
    2. Negative amplitudes (contradictions) are visible before squaring
    """

    @staticmethod
    def score(
        field: DenseField,
        query_phase: NDArray[np.float32],
        source_phases: dict[str, NDArray[np.float32]],
        saliences: dict[str, float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
    ) -> QuantumRetrievalResult:
        """Score sources using Born rule probabilities.

        Args:
            field: The interference field.
            query_phase: Shape (B, D) — the query.
            source_phases: Dict of source_id → (B, D) phase vectors.
            saliences: Optional per-source salience weights.
            band_weights: Per-band fusion weights. Uniform if None.

        Returns:
            QuantumRetrievalResult with probabilities and contradictions.
        """
        B = field.bands

        if band_weights is None:
            band_weights = np.ones(B, dtype=np.float32) / B

        # Compute resonance vectors once
        resonance_vectors = np.zeros((B, field.dim), dtype=np.float32)
        for b in range(B):
            resonance_vectors[b] = field.F[b] @ query_phase[b]

        # Compute amplitudes for each source
        amplitudes: dict[str, float] = {}
        for sid, phase in source_phases.items():
            amplitude = 0.0
            for b in range(B):
                # Band-weighted dot product: φᵢ_b · r_b
                amplitude += band_weights[b] * float(np.dot(phase[b], resonance_vectors[b]))

            # Apply salience if provided
            if saliences and sid in saliences:
                amplitude *= saliences[sid]

            amplitudes[sid] = amplitude

        # Born rule: P_i = |ψ_i|² / Z
        amplitude_sq = {sid: amp ** 2 for sid, amp in amplitudes.items()}
        Z = sum(amplitude_sq.values()) + 1e-12

        scores = []
        n_constructive = 0
        for sid, amp in amplitudes.items():
            prob = amplitude_sq[sid] / Z
            sign = 1 if amp >= 0 else -1
            if sign > 0:
                n_constructive += 1
            scores.append(QuantumScore(
                source_id=sid,
                amplitude=amp,
                probability=prob,
                phase_sign=sign,
            ))

        scores.sort(key=lambda s: s.probability, reverse=True)

        constructive_frac = n_constructive / max(len(scores), 1)

        return QuantumRetrievalResult(
            scores=scores,
            contradictions=[],  # Filled by detect_contradictions
            partition_function=Z,
            constructive_fraction=constructive_frac,
        )

    @staticmethod
    def detect_contradictions(
        field: DenseField,
        source_phases: dict[str, NDArray[np.float32]],
        threshold: float = -0.1,
        band_weights: NDArray[np.float32] | None = None,
    ) -> list[Contradiction]:
        """Find source pairs with destructive interference.

        Cross-interference: I_ij = Σ_b w_b · (φᵢ_b)ᵀ F_b φⱼ_b

        If I_ij < threshold: sources i and j CONTRADICT on the field.

        Args:
            field: The interference field.
            source_phases: Dict of source_id → (B, D) phase vectors.
            threshold: Negative threshold for contradiction. More negative = stricter.
            band_weights: Per-band weights. Uniform if None.

        Returns:
            List of Contradiction pairs, sorted by strength descending.
        """
        B = field.bands
        if band_weights is None:
            band_weights = np.ones(B, dtype=np.float32) / B

        source_ids = list(source_phases.keys())
        N = len(source_ids)
        contradictions = []

        for i in range(N):
            for j in range(i + 1, N):
                sid_a = source_ids[i]
                sid_b = source_ids[j]
                phase_a = source_phases[sid_a]
                phase_b = source_phases[sid_b]

                # Cross-interference: φᵢᵀ F φⱼ (field-mediated interaction)
                interference = 0.0
                for b in range(B):
                    Fphi_j = field.F[b] @ phase_b[b]
                    interference += band_weights[b] * float(np.dot(phase_a[b], Fphi_j))

                if interference < threshold:
                    contradictions.append(Contradiction(
                        source_a=sid_a,
                        source_b=sid_b,
                        interference=interference,
                        strength=abs(interference),
                    ))

        contradictions.sort(key=lambda c: c.strength, reverse=True)
        return contradictions

    @staticmethod
    def score_with_contradictions(
        field: DenseField,
        query_phase: NDArray[np.float32],
        source_phases: dict[str, NDArray[np.float32]],
        saliences: dict[str, float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
        contradiction_threshold: float = -0.1,
    ) -> QuantumRetrievalResult:
        """Full quantum scoring with contradiction detection in one call.

        Combines score() and detect_contradictions() for convenience.
        """
        result = QuantumScorer.score(
            field, query_phase, source_phases,
            saliences=saliences, band_weights=band_weights,
        )
        contradictions = QuantumScorer.detect_contradictions(
            field, source_phases,
            threshold=contradiction_threshold,
            band_weights=band_weights,
        )
        result.contradictions = contradictions
        return result
