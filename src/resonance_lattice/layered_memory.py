# SPDX-License-Identifier: BUSL-1.1
"""Layered memory — function-tier memory system over knowledge model primitives.

Three tiers (working / episodic / semantic), each backed by its own
`Lattice` instance (and thus its own ``.rlat`` file). Retrieval fuses
tier-weighted resonance scores across all tiers.

Structurally modeled on ``TemporalLattice`` (temporal.py) but tiers on
*function* rather than time. The two are composable — a semantic tier
could itself use temporal windows internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.config import LatticeConfig
from resonance_lattice.encoder import Encoder, PhaseSpectrum
from resonance_lattice.field.dense import ResonanceResult
from resonance_lattice.lattice import (
    EnrichedResult,
    Lattice,
    MaterialisedResult,
    RetrievalResult,
)
from resonance_lattice.retention import RetentionPolicy
from resonance_lattice.store import SourceContent

TIER_NAMES = ("working", "episodic", "semantic")

DEFAULT_WEIGHTS: dict[str, float] = {
    "working": 0.5,
    "episodic": 0.3,
    "semantic": 0.2,
}


@dataclass
class LayeredMemoryConfig:
    """Configuration for the three-tier memory system."""

    tier_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    retention: dict[str, RetentionPolicy] = field(default_factory=lambda: {
        "working": RetentionPolicy.WORKING,
        "episodic": RetentionPolicy.EPISODIC,
        "semantic": RetentionPolicy.SEMANTIC,
    })


class LayeredMemory:
    """Three-tier memory orchestrator.

    Each tier is a full ``Lattice`` backed by its own ``.rlat`` file inside
    a memory-root directory::

        memory_root/
          working.rlat
          episodic.rlat
          semantic.rlat
    """

    def __init__(
        self,
        tiers: dict[str, Lattice],
        config: LayeredMemoryConfig | None = None,
        encoder: Encoder | None = None,
        memory_root: Path | None = None,
    ) -> None:
        self.tiers = tiers
        self.config = config or LayeredMemoryConfig()
        self.encoder = encoder
        self.memory_root = memory_root

    # ── Construction helpers ──────────────────────────────────────────────

    @classmethod
    def init(
        cls,
        memory_root: str | Path,
        lattice_config: LatticeConfig | None = None,
        encoder: Encoder | None = None,
        config: LayeredMemoryConfig | None = None,
    ) -> LayeredMemory:
        """Create a new layered memory root with empty tier knowledge models."""
        root = Path(memory_root)
        root.mkdir(parents=True, exist_ok=True)

        if lattice_config is None:
            from resonance_lattice.config import LatticeConfig
            lattice_config = LatticeConfig()

        tiers: dict[str, Lattice] = {}
        for name in TIER_NAMES:
            store_path = root / f"{name}.db"
            tiers[name] = Lattice(
                config=lattice_config,
                store_path=str(store_path),
                encoder=encoder,
            )

        return cls(
            tiers=tiers,
            config=config,
            encoder=encoder,
            memory_root=root,
        )

    @classmethod
    def open(
        cls,
        memory_root: str | Path,
        encoder: Encoder | None = None,
        config: LayeredMemoryConfig | None = None,
        restore_encoder: bool = True,
    ) -> LayeredMemory:
        """Open an existing layered memory root."""
        root = Path(memory_root)
        tiers: dict[str, Lattice] = {}
        loaded_encoder = encoder

        for name in TIER_NAMES:
            path = root / f"{name}.rlat"
            if path.exists():
                lattice = Lattice.load(
                    path,
                    encoder=encoder,
                    restore_encoder=restore_encoder,
                )
                tiers[name] = lattice
                if loaded_encoder is None and lattice.encoder is not None:
                    loaded_encoder = lattice.encoder

        # Create missing tiers with config from the first loaded one
        if tiers:
            ref = next(iter(tiers.values()))
            for name in TIER_NAMES:
                if name not in tiers:
                    tiers[name] = Lattice(
                        config=ref.config,
                        encoder=loaded_encoder,
                    )

        return cls(
            tiers=tiers,
            config=config,
            encoder=loaded_encoder,
            memory_root=root,
        )

    def save(self) -> None:
        """Persist all tiers to their .rlat files."""
        if self.memory_root is None:
            raise RuntimeError("Cannot save: no memory_root set")
        for name, lattice in self.tiers.items():
            path = self.memory_root / f"{name}.rlat"
            lattice.save(str(path))

    # ── Write ─────────────────────────────────────────────────────────────

    def write(
        self,
        phase_spectrum: PhaseSpectrum | NDArray[np.float32],
        salience: float = 1.0,
        source_id: str = "",
        content: SourceContent | dict[str, Any] | None = None,
        tier: str = "working",
    ) -> str:
        """Write a source to a specific tier (default: working)."""
        if tier not in self.tiers:
            raise ValueError(f"Unknown tier: {tier!r}. Valid: {list(self.tiers)}")
        return self.tiers[tier].superpose(
            phase_spectrum=phase_spectrum,
            salience=salience,
            source_id=source_id,
            content=content,
        )

    def write_text(
        self,
        text: str,
        salience: float = 1.0,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
        tier: str = "working",
    ) -> str:
        """Encode text and write to a tier."""
        if tier not in self.tiers:
            raise ValueError(f"Unknown tier: {tier!r}. Valid: {list(self.tiers)}")
        return self.tiers[tier].superpose_text(
            text=text,
            salience=salience,
            source_id=source_id,
            metadata=metadata,
        )

    # ── Recall (tier-fused retrieval) ─────────────────────────────────────

    def recall(
        self,
        query_phase: PhaseSpectrum | NDArray[np.float32],
        tier_weights: dict[str, float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
        tiers: list[str] | None = None,
    ) -> RetrievalResult:
        """Query across memory tiers with weighted fusion.

        score(src) = tier_weight[t] * resonance(src) * salience(src)

        Args:
            query_phase: Query phase spectrum.
            tier_weights: Per-tier weights (defaults to config).
            band_weights: Per-band weights passed to each tier.
            top_k: Number of results.
            tiers: Subset of tiers to query (default: all).

        Returns:
            Fused RetrievalResult across selected tiers.
        """
        weights = tier_weights or self.config.tier_weights
        active_tiers = tiers or list(self.tiers.keys())

        # Normalise weights for active tiers
        raw_weights = {t: weights.get(t, 0.1) for t in active_tiers}
        total = sum(raw_weights.values())
        if total > 0:
            norm_weights = {t: w / total for t, w in raw_weights.items()}
        else:
            norm_weights = {t: 1.0 / len(active_tiers) for t in active_tiers}

        all_results: list[MaterialisedResult] = []
        combined_resonance: ResonanceResult | None = None

        for tier_name in active_tiers:
            lattice = self.tiers.get(tier_name)
            if lattice is None or lattice.source_count == 0:
                continue

            result = lattice.resonate(
                query_phase=query_phase,
                band_weights=band_weights,
                top_k=top_k,
            )

            w = norm_weights[tier_name]
            for r in result.results:
                all_results.append(MaterialisedResult(
                    source_id=r.source_id,
                    score=r.score * w,
                    band_scores=r.band_scores * w if r.band_scores is not None else None,
                    content=r.content,
                ))

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

        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:top_k]

        if combined_resonance is None:
            ref = next(iter(self.tiers.values()))
            combined_resonance = ResonanceResult(
                resonance_vectors=np.zeros((ref.config.bands, ref.config.dim)),
                fused=np.zeros(ref.config.dim),
                band_energies=np.zeros(ref.config.bands),
            )

        return RetrievalResult(
            results=top_results,
            resonance=combined_resonance,
            source_pointers=[],
            timings_ms=None,
        )

    def recall_text(
        self,
        query: str,
        tier_weights: dict[str, float] | None = None,
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
        tiers: list[str] | None = None,
    ) -> RetrievalResult:
        """Encode query text and recall across tiers."""
        if self.encoder is None:
            raise RuntimeError("No encoder configured")
        phase = self.encoder.encode_query(query)
        return self.recall(
            phase, tier_weights=tier_weights,
            band_weights=band_weights, top_k=top_k, tiers=tiers,
        )

    def recall_enriched(
        self,
        query: str,
        *,
        tier_weights: dict[str, float] | None = None,
        tiers: list[str] | None = None,
        top_k: int = 20,
        **enriched_kwargs: Any,
    ) -> EnrichedResult:
        """Per-tier ``enriched_query`` with weighted score fusion.

        Fans out the query to each active tier's full retrieval pipeline
        (dense + lexical + rerank + optional cross-encoder / cascade /
        contradictions / subgraph), then merges results across tiers by
        source_id using ``max(norm_weight[t] * score_t)``. Max (not sum)
        avoids double-counting sources that appear in multiple tiers, while
        preserving the tier-weight ordering signal.

        Unlike ``recall``/``recall_text`` which are pure-dense fused readouts,
        this method preserves every feature baked into ``Lattice.enriched_query``
        — it is the LayeredMemory analogue of the single-Lattice query path.

        Args:
            query: Query text.
            tier_weights: Per-tier weights (defaults to config). Normalised
                across active tiers; missing tiers get 0.1.
            tiers: Subset of tiers to query (default: all).
            top_k: Number of fused results to return.
            **enriched_kwargs: Passthrough to ``Lattice.enriched_query`` on
                each tier (e.g. ``mode``, ``enable_lexical``, ``enable_rerank``,
                ``lexical_weight``, ``question_date``).

        Returns:
            ``EnrichedResult`` with fused top-k results; ``coverage``,
            ``related``, ``contradictions`` are merged best-effort from the
            contributing tiers.
        """
        weights = tier_weights or self.config.tier_weights
        active_tiers = tiers or list(self.tiers.keys())

        raw_weights = {t: weights.get(t, 0.1) for t in active_tiers}
        total = sum(raw_weights.values())
        if total > 0:
            norm_weights = {t: w / total for t, w in raw_weights.items()}
        else:
            norm_weights = {t: 1.0 / len(active_tiers) for t in active_tiers}

        # Fan out per-tier enriched_query. Each tier returns an EnrichedResult
        # with top_k fused results already — we re-fuse across tiers below.
        per_tier: list[tuple[str, float, EnrichedResult]] = []
        for tier_name in active_tiers:
            lattice = self.tiers.get(tier_name)
            if lattice is None or lattice.source_count == 0:
                continue
            er = lattice.enriched_query(
                text=query, top_k=top_k, **enriched_kwargs,
            )
            per_tier.append((tier_name, norm_weights[tier_name], er))

        # Empty across all tiers — return a zero EnrichedResult shell that
        # downstream consumers can still introspect.
        if not per_tier:
            from resonance_lattice.lattice import CoverageProfile
            ref_bands = next(iter(self.tiers.values())).config.bands
            empty_coverage = CoverageProfile(
                band_energies=np.zeros(ref_bands, dtype=np.float32),
                band_names=[f"band_{i}" for i in range(ref_bands)],
                total_energy=0.0,
                confidence=0.0,
                gaps=[],
            )
            return EnrichedResult(
                query=query, results=[],
                coverage=empty_coverage,
                related=[], contradictions=[],
                latency_ms=0.0, timings_ms={"total": 0.0},
                assessment=None,
            )

        # Merge results: per source_id keep the max tier-weighted score,
        # and remember which tier/result supplied it (so we can carry the
        # original MaterialisedResult fields — content, band_scores, etc.).
        best_per_source: dict[str, tuple[float, MaterialisedResult, str]] = {}
        for tier_name, w, er in per_tier:
            for r in er.results:
                weighted = float(r.score) * w
                existing = best_per_source.get(r.source_id)
                if existing is None or weighted > existing[0]:
                    rescored = MaterialisedResult(
                        source_id=r.source_id,
                        score=weighted,
                        band_scores=r.band_scores,
                        content=r.content,
                        raw_score=float(r.score),
                        provenance=f"{getattr(r, 'provenance', 'dense')}:{tier_name}",
                    )
                    best_per_source[r.source_id] = (weighted, rescored, tier_name)

        fused_results = sorted(
            (v[1] for v in best_per_source.values()),
            key=lambda r: r.score, reverse=True,
        )[:top_k]

        # Coverage / related / contradictions: take from the tier that supplied
        # the top-ranked result (best signal about what the query actually hit).
        # This is diagnostic-level output; full cross-tier fusion of these
        # structural signals is not required for the core results ranking.
        top_tier = best_per_source[fused_results[0].source_id][2] if fused_results else per_tier[0][0]
        top_er = next(er for t, _, er in per_tier if t == top_tier)

        merged_related = []
        merged_contradictions = []
        seen_related = set()
        for _, _, er in per_tier:
            for rel in er.related:
                if rel.source_id not in seen_related:
                    seen_related.add(rel.source_id)
                    merged_related.append(rel)
            merged_contradictions.extend(er.contradictions)

        total_latency = sum(er.latency_ms for _, _, er in per_tier)
        merged_timings: dict[str, float] = {}
        for _, _, er in per_tier:
            for k, v in (er.timings_ms or {}).items():
                merged_timings[k] = merged_timings.get(k, 0.0) + float(v)
        merged_timings["total"] = total_latency

        return EnrichedResult(
            query=query,
            results=fused_results,
            coverage=top_er.coverage,
            related=merged_related,
            contradictions=merged_contradictions,
            latency_ms=total_latency,
            timings_ms=merged_timings,
            assessment=top_er.assessment,
        )

    # ── Entry access (for primer & diagnostics) ────────────────────────────

    def get_registry_entry(self, source_id: str) -> tuple[str, Any] | None:
        """Find a source's registry entry across tiers.

        Returns ``(tier_name, RegistryEntry)`` or ``None``.
        """
        for tier_name, lattice in self.tiers.items():
            entry = lattice.registry._source_index.get(source_id)
            if entry is not None:
                return tier_name, entry
        return None

    def iter_tier_entries(self, tier: str) -> list[tuple[str, Any, SourceContent | None]]:
        """Iterate all entries in a tier with their content.

        Returns list of ``(source_id, RegistryEntry, SourceContent | None)``.
        """
        lattice = self.tiers.get(tier)
        if lattice is None:
            return []
        results = []
        for source_id, entry in lattice.registry._source_index.items():
            content = lattice.store.retrieve(source_id)
            results.append((source_id, entry, content))
        return results

    # ── Inspection ────────────────────────────────────────────────────────

    @property
    def total_sources(self) -> int:
        return sum(t.source_count for t in self.tiers.values())

    def info(self) -> dict[str, Any]:
        """Per-tier summary."""
        return {
            "total_sources": self.total_sources,
            "memory_root": str(self.memory_root) if self.memory_root else None,
            "tier_weights": self.config.tier_weights,
            "tiers": {
                name: {
                    "sources": lattice.source_count,
                    "field_size_mb": lattice.field.size_mb,
                    "energy": lattice.field.energy().tolist(),
                }
                for name, lattice in self.tiers.items()
            },
        }
