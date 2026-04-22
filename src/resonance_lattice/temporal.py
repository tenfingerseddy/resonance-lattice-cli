# SPDX-License-Identifier: BUSL-1.1
"""Temporal windowed fields for time-horizon-aware retrieval.

Maintains multiple field tensors at different time horizons:
  F_recent   (last 30 days, high salience weight)
  F_medium   (last 6 months, moderate salience weight)
  F_archive  (all time, low salience weight)

Query-time fusion:
  R = w_r*(F_recent @ q) + w_m*(F_medium @ q) + w_a*(F_archive @ q)

Weights can be query-dependent:
  "current Fabric limitations" -> weight F_recent
  "foundational database theory" -> weight F_archive

Spec reference: Section 7.2 (Temporal Windowed Retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.config import LatticeConfig
from resonance_lattice.encoder import Encoder, PhaseSpectrum
from resonance_lattice.field.dense import ResonanceResult
from resonance_lattice.lattice import Lattice, MaterialisedResult, RetrievalResult
from resonance_lattice.store import SourceContent


@dataclass
class TemporalConfig:
    """Configuration for temporal windowed fields."""
    windows: list[str] = field(default_factory=lambda: ["recent", "medium", "archive"])
    window_days: list[int] = field(default_factory=lambda: [30, 180, 36500])
    default_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])


class TemporalLattice:
    """Multi-temporal-window lattice for time-aware retrieval.

    Maintains separate field tensors per time window. Sources are routed
    to the appropriate window based on their timestamp. Queries are fused
    across windows with configurable temporal weighting.
    """

    def __init__(
        self,
        config: LatticeConfig,
        temporal_config: TemporalConfig | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.config = config
        self.temporal = temporal_config or TemporalConfig()
        self.encoder = encoder

        # Create one lattice per time window
        self.windows: dict[str, Lattice] = {}
        for name in self.temporal.windows:
            self.windows[name] = Lattice(config=config, encoder=encoder)

        # Track source timestamps for window routing
        self._timestamps: dict[str, datetime] = {}

    @property
    def total_sources(self) -> int:
        return sum(w.source_count for w in self.windows.values())

    def _route_to_window(self, timestamp: datetime | None = None) -> str:
        """Determine which window a source belongs to based on timestamp."""
        if timestamp is None:
            return self.temporal.windows[0]  # Default to most recent

        now = datetime.now(UTC)
        age_days = (now - timestamp).days

        for i, max_days in enumerate(self.temporal.window_days):
            if age_days <= max_days:
                return self.temporal.windows[i]

        return self.temporal.windows[-1]  # Archive fallback

    def superpose(
        self,
        phase_spectrum: PhaseSpectrum | NDArray[np.float32],
        salience: float = 1.0,
        source_id: str = "",
        content: SourceContent | dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Add a source to the appropriate temporal window.

        Args:
            phase_spectrum: Phase vectors.
            salience: Importance weight.
            source_id: Unique identifier.
            content: Source content.
            timestamp: When this source was created/updated.

        Returns:
            The source_id used.
        """
        window_name = self._route_to_window(timestamp)
        sid = self.windows[window_name].superpose(
            phase_spectrum=phase_spectrum,
            salience=salience,
            source_id=source_id,
            content=content,
        )
        if timestamp is not None:
            self._timestamps[sid] = timestamp
        return sid

    def superpose_text(
        self,
        text: str,
        salience: float = 1.0,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Encode and add text to the appropriate temporal window."""
        window_name = self._route_to_window(timestamp)
        sid = self.windows[window_name].superpose_text(
            text=text,
            salience=salience,
            source_id=source_id,
            metadata=metadata,
        )
        if timestamp is not None:
            self._timestamps[sid] = timestamp
        return sid

    def resonate(
        self,
        query_phase: PhaseSpectrum | NDArray[np.float32],
        temporal_weights: NDArray[np.float32] | list[float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
    ) -> RetrievalResult:
        """Query across all temporal windows with weighted fusion.

        R = w_r*(F_recent @ q) + w_m*(F_medium @ q) + w_a*(F_archive @ q)

        Args:
            query_phase: Query phase spectrum.
            temporal_weights: Per-window weights. Defaults to config.
            band_weights: Per-band weights.
            top_k: Number of results.

        Returns:
            Fused RetrievalResult across all windows.
        """
        if temporal_weights is None:
            temporal_weights = np.array(self.temporal.default_weights, dtype=np.float32)
        else:
            temporal_weights = np.array(temporal_weights, dtype=np.float32)

        # Normalise weights
        total_weight = temporal_weights.sum()
        if total_weight > 0:
            temporal_weights = temporal_weights / total_weight

        # Resonate in each window
        all_results: list[MaterialisedResult] = []
        combined_resonance = None

        for i, (name, lattice) in enumerate(self.windows.items()):
            if lattice.source_count == 0:
                continue

            result = lattice.resonate(
                query_phase=query_phase,
                band_weights=band_weights,
                top_k=top_k,
            )

            # Weight the scores by temporal weight
            w = temporal_weights[i] if i < len(temporal_weights) else 0.1
            for r in result.results:
                all_results.append(MaterialisedResult(
                    source_id=r.source_id,
                    score=r.score * w,
                    band_scores=r.band_scores * w if r.band_scores is not None else None,
                    content=r.content,
                ))

            # Fuse resonance vectors
            if combined_resonance is None:
                combined_resonance = ResonanceResult(
                    resonance_vectors=result.resonance.resonance_vectors * w,
                    fused=result.resonance.fused * w,
                    band_energies=result.resonance.band_energies * w,
                )
            else:
                combined_resonance = ResonanceResult(
                    resonance_vectors=combined_resonance.resonance_vectors + result.resonance.resonance_vectors * w,
                    fused=combined_resonance.fused + result.resonance.fused * w,
                    band_energies=combined_resonance.band_energies + result.resonance.band_energies * w,
                )

        # Sort by score and take top-k
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:top_k]

        if combined_resonance is None:
            from resonance_lattice.field.dense import ResonanceResult as RR
            combined_resonance = RR(
                resonance_vectors=np.zeros((self.config.bands, self.config.dim)),
                fused=np.zeros(self.config.dim),
                band_energies=np.zeros(self.config.bands),
            )

        return RetrievalResult(
            results=top_results,
            resonance=combined_resonance,
            source_pointers=[],
            timings_ms=None,
        )

    def resonate_text(
        self,
        query: str,
        temporal_weights: NDArray[np.float32] | list[float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
    ) -> RetrievalResult:
        """Encode query text and resonate across temporal windows."""
        if self.encoder is None:
            raise RuntimeError("No encoder configured")
        phase = self.encoder.encode_query(query)
        return self.resonate(phase, temporal_weights=temporal_weights,
                            band_weights=band_weights, top_k=top_k)

    def info(self) -> dict[str, Any]:
        """Summary of all temporal windows."""
        return {
            "total_sources": self.total_sources,
            "windows": {
                name: {
                    "sources": lattice.source_count,
                    "field_size_mb": lattice.field.size_mb,
                    "energy": lattice.field.energy().tolist(),
                }
                for name, lattice in self.windows.items()
            },
            "temporal_weights": self.temporal.default_weights,
        }
