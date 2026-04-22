# SPDX-License-Identifier: BUSL-1.1
"""Swarm Intelligence: emergent field consensus from usage patterns.

Inspired by ant colony optimisation — ants deposit pheromones on paths
they traverse. Paths with more traffic get stronger. Unused paths evaporate.

The field becomes a dynamic pheromone landscape:
    F(t+1) = (1 - ρ) · F(t) + Σ_queries feedback · (φ ⊗ φ)

Frequently-accessed knowledge strengthens. Rarely-used knowledge fades.
"Highways" emerge between commonly co-queried concepts — discovered
structure, not engineered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..field.dense import DenseField, ResonanceResult


class SwarmEvent(NamedTuple):
    """A single query + feedback event."""
    query_phase: NDArray[np.float32]  # (B, D)
    feedback: float  # +1.0 = useful, -0.5 = rejected, 0.0 = neutral


@dataclass
class Highway:
    """A high-conductance path in the field (frequently co-queried concept pair)."""
    eigenvector: NDArray[np.float32]  # (D,) — the concept direction
    eigenvalue: float  # Strength (larger = more frequently activated)
    band: int
    rank: int  # Position in eigenvalue ordering


@dataclass
class SwarmStats:
    """Statistics from swarm evolution."""
    total_events: int
    total_reinforcements: int
    total_evaporations: int
    energy_before: float
    energy_after: float
    highways_detected: int


class SwarmField:
    """A field that adapts to usage patterns via pheromone dynamics.

    Wraps a DenseField and adds:
    - Feedback-driven reinforcement (strengthen paths that lead to useful results)
    - Evaporation (gradually decay unused knowledge)
    - Highway detection (find emergent high-conductance concept pairs)
    """

    def __init__(
        self,
        field: DenseField,
        rho: float = 0.01,
        reinforcement_strength: float = 0.1,
    ):
        """
        Args:
            field: The underlying dense field (modified in place by learn/evaporate).
            rho: Evaporation rate per step. 0.01 = 1% decay per evaporation.
            reinforcement_strength: How much feedback modifies the field.
        """
        self.field = field
        self.rho = rho
        self.reinforcement_strength = reinforcement_strength
        self._events: list[SwarmEvent] = []
        self._total_reinforcements = 0
        self._total_evaporations = 0

    @property
    def event_count(self) -> int:
        return len(self._events)

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Standard resonance (delegates to underlying field)."""
        return self.field.resonate(query_phase, band_weights=band_weights)

    def resonate_and_learn(
        self,
        query_phase: NDArray[np.float32],
        feedback: float = 0.0,
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Resonate, then update the field based on feedback.

        feedback > 0: reinforce this query pattern (user found it useful)
        feedback < 0: anti-reinforce (user rejected the results)
        feedback = 0: neutral (no learning signal)

        Args:
            query_phase: Shape (B, D) — the query.
            feedback: Feedback signal. +1.0 = very useful, -0.5 = rejected.
            band_weights: Per-band fusion weights.

        Returns:
            ResonanceResult from the current field state.
        """
        result = self.field.resonate(query_phase, band_weights=band_weights)

        if abs(feedback) > 1e-6:
            self._reinforce(query_phase, feedback)
            self._events.append(SwarmEvent(
                query_phase=query_phase.copy(),
                feedback=feedback,
            ))

        return result

    def _reinforce(self, query_phase: NDArray[np.float32], feedback: float) -> None:
        """Apply pheromone reinforcement: F += strength · feedback · (φ ⊗ φ)."""
        weight = self.reinforcement_strength * feedback
        for b in range(self.field.bands):
            phi = query_phase[b]
            self.field.F[b] += weight * np.outer(phi, phi)
        self._total_reinforcements += 1

    def evaporate(self) -> float:
        """Apply one step of pheromone evaporation: F *= (1 - ρ).

        Returns:
            Energy lost to evaporation.
        """
        energy_before = float(sum(
            np.linalg.norm(self.field.F[b], "fro") for b in range(self.field.bands)
        ))

        self.field.F *= (1 - self.rho)
        self._total_evaporations += 1

        energy_after = float(sum(
            np.linalg.norm(self.field.F[b], "fro") for b in range(self.field.bands)
        ))

        return energy_before - energy_after

    def detect_highways(
        self,
        band: int = 0,
        top_k: int = 10,
        sigma_threshold: float = 2.0,
    ) -> list[Highway]:
        """Find high-conductance paths (concept directions with unusually high energy).

        A "highway" is an eigenvector with eigenvalue > μ + σ_threshold·σ.
        These represent concept directions that are disproportionately strong —
        either from the original corpus or from reinforcement via usage.

        Args:
            band: Which band to analyse.
            top_k: Maximum highways to return.
            sigma_threshold: Number of standard deviations above mean for highway.

        Returns:
            List of Highway objects, sorted by eigenvalue descending.
        """
        F_b = self.field.F[band]
        F_sym = (F_b + F_b.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Highway threshold: μ + σ_threshold · σ
        mu = np.mean(eigenvalues)
        sigma = np.std(eigenvalues)
        threshold = mu + sigma_threshold * sigma

        highways = []
        for i in range(min(len(eigenvalues), top_k)):
            if eigenvalues[i] > threshold:
                highways.append(Highway(
                    eigenvector=eigenvectors[:, i].astype(np.float32),
                    eigenvalue=float(eigenvalues[i]),
                    band=band,
                    rank=i,
                ))
            else:
                break  # Sorted descending, so no more above threshold

        return highways

    def stats(self) -> SwarmStats:
        """Return current swarm statistics."""
        energy = float(sum(
            np.linalg.norm(self.field.F[b], "fro") for b in range(self.field.bands)
        ))
        highways = len(self.detect_highways(band=0))

        return SwarmStats(
            total_events=len(self._events),
            total_reinforcements=self._total_reinforcements,
            total_evaporations=self._total_evaporations,
            energy_before=0.0,  # Would need to track from init
            energy_after=energy,
            highways_detected=highways,
        )
