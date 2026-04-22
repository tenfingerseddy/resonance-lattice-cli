# SPDX-License-Identifier: BUSL-1.1
"""Band-aware context materialiser for the Resonance Lattice.

Assembles multi-resolution context from retrieval results, allocating
tokens to different abstraction levels based on which bands resonated:

  Omega_1-2 -> Landscape summary (~300 tokens)
  Omega_3   -> Structural relationships (~400 tokens)
  Omega_4-5 -> Evidence passages (~2000 tokens)

This gives the consuming LLM hierarchical context: a map of the territory,
the roads between landmarks, and the specific buildings it needs.

Spec references: Section 11 (Integration Architecture).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.config import MaterialiserConfig


@dataclass
class MaterialisedContext:
    """Assembled multi-resolution context ready for LLM consumption."""
    landscape: str           # Omega_1-2: domain/topic overview
    structure: str           # Omega_3: relationships
    evidence: list[str]      # Omega_4-5: exact passages with attribution
    total_tokens_est: int    # Estimated total token count
    band_distribution: dict[str, float]  # Which bands contributed most
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, include_headers: bool = True) -> str:
        """Format as a string suitable for LLM context injection.

        Args:
            include_headers: Whether to include section headers.

        Returns:
            Formatted context string.
        """
        parts = []

        if self.landscape.strip():
            if include_headers:
                parts.append("## Corpus Context")
            parts.append(self.landscape)

        if self.structure.strip():
            if include_headers:
                parts.append("\n## Key Relationships")
            parts.append(self.structure)

        if self.evidence:
            if include_headers:
                parts.append("\n## Evidence")
            for i, passage in enumerate(self.evidence, 1):
                parts.append(f"[{i}] {passage}")

        return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Try to truncate at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space]
    return truncated


class Materialiser:
    """Assembles multi-resolution context from resonance retrieval results.

    Takes retrieval results (with band scores) and constructs a structured
    context that gives the LLM hierarchical understanding: overview, relationships,
    and evidence.
    """

    def __init__(self, config: MaterialiserConfig | None = None) -> None:
        self.config = config or MaterialiserConfig()

    def assemble(
        self,
        results: list,
        band_weights: NDArray[np.float32] | None = None,
    ) -> MaterialisedContext:
        """Assemble multi-resolution context from retrieval results.

        Args:
            results: List of MaterialisedResult objects (from Lattice.resonate).
            band_weights: Optional band weights used during retrieval
                (for determining which bands contributed most).

        Returns:
            MaterialisedContext with landscape, structure, and evidence.
        """
        if not results:
            return MaterialisedContext(
                landscape="No relevant information found in the corpus.",
                structure="",
                evidence=[],
                total_tokens_est=10,
                band_distribution={},
            )

        # Analyse band distribution to understand what kind of results we have
        band_dist = self._compute_band_distribution(results)

        # Partition results by their dominant band contributions
        landscape_results = []  # Omega_1-2 dominant
        structure_results = []  # Omega_3 dominant
        evidence_results = []   # Omega_4-5 dominant

        for i, r in enumerate(results):
            if len(results) <= 3:
                # Few results: put first in landscape, rest in evidence
                if i == 0:
                    landscape_results.append(r)
                else:
                    evidence_results.append(r)
            elif r.band_scores is not None and len(r.band_scores) >= 3:
                num_bands = len(r.band_scores)
                # 3+ bands: partition by dominant band group
                low_energy = sum(r.band_scores[:min(2, num_bands)])
                mid_energy = r.band_scores[2] if num_bands > 2 else 0.0
                high_energy = sum(r.band_scores[min(3, num_bands):]) if num_bands > 3 else 0.0

                if low_energy >= mid_energy and low_energy >= high_energy:
                    landscape_results.append(r)
                elif mid_energy >= low_energy and mid_energy >= high_energy:
                    structure_results.append(r)
                else:
                    evidence_results.append(r)
            else:
                # 2 bands or no band scores: split by rank position
                # Top 20% landscape, rest evidence (most useful for LLM)
                threshold = max(1, len(results) // 5)
                if i < threshold:
                    landscape_results.append(r)
                else:
                    evidence_results.append(r)

        # If partitioning left a section empty, borrow from evidence
        if not landscape_results and evidence_results:
            landscape_results = evidence_results[:2]
        if not structure_results and evidence_results:
            structure_results = evidence_results[:2]

        # Build each section
        landscape = self._build_landscape(landscape_results)
        structure = self._build_structure(structure_results)
        evidence = self._build_evidence(evidence_results)

        total_tokens = (
            _estimate_tokens(landscape)
            + _estimate_tokens(structure)
            + sum(_estimate_tokens(e) for e in evidence)
        )

        return MaterialisedContext(
            landscape=landscape,
            structure=structure,
            evidence=evidence,
            total_tokens_est=total_tokens,
            band_distribution=band_dist,
        )

    def _compute_band_distribution(self, results: list) -> dict[str, float]:
        """Compute the average band energy distribution across results."""
        if not results:
            return {}

        band_sums: dict[int, float] = {}
        count = 0
        for r in results:
            if r.band_scores is not None:
                for b, score in enumerate(r.band_scores):
                    band_sums[b] = band_sums.get(b, 0.0) + float(score)
                count += 1

        if count == 0:
            return {}

        total = sum(band_sums.values())
        if total < 1e-8:
            return {f"band_{b}": 0.0 for b in band_sums}

        return {f"band_{b}": v / total for b, v in sorted(band_sums.items())}

    def _build_landscape(self, results: list) -> str:
        """Build landscape context from Omega_1-2 results."""
        if not results:
            return ""

        summaries = []
        for r in results:
            if r.content is not None and hasattr(r.content, "summary") and r.content.summary:
                summaries.append(r.content.summary)

        if not summaries:
            return ""

        landscape = "The corpus contains the following relevant context:\n"
        for s in summaries[:5]:  # Max 5 summaries
            landscape += f"- {s}\n"

        return _truncate_to_tokens(landscape, self.config.landscape_tokens)

    def _build_structure(self, results: list) -> str:
        """Build structural context from Omega_3 results."""
        if not results:
            return ""

        relations = []
        for r in results:
            if r.content is not None and hasattr(r.content, "relations") and r.content.relations:
                for rel in r.content.relations:
                    if len(rel) >= 3:
                        relations.append(f"{rel[0]} {rel[1]} {rel[2]}")

        if not relations:
            return ""

        structure = "Key relationships:\n"
        for rel in relations[:10]:  # Max 10 relations
            structure += f"- {rel}\n"

        return _truncate_to_tokens(structure, self.config.structure_tokens)

    def _build_evidence(self, results: list) -> list[str]:
        """Build evidence passages from Omega_4-5 results."""
        evidence = []
        remaining_tokens = self.config.evidence_tokens

        for r in results:
            if remaining_tokens <= 0:
                break

            text = ""
            if r.content is not None:
                if hasattr(r.content, "full_text") and r.content.full_text:
                    text = r.content.full_text
                elif hasattr(r.content, "summary") and r.content.summary:
                    text = r.content.summary

            if text:
                truncated = _truncate_to_tokens(text, remaining_tokens)
                source_attr = f"(source: {r.source_id}, score: {r.score:.3f})"
                passage = f"{truncated} {source_attr}"
                evidence.append(passage)
                remaining_tokens -= _estimate_tokens(passage)

        return evidence
