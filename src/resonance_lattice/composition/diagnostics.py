# SPDX-License-Identifier: BUSL-1.1
"""Composition Diagnostics: pre-search analysis of composed knowledge models.

Provides information-theoretic and algebraic analysis of how
knowledge models relate before you search them. Answers questions like:
- "How much do these knowledge models overlap?"
- "What does B add that A doesn't have?"
- "Where do they contradict each other?"
- "Is the composition richer or more confused than either alone?"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.field.dense import DenseField


@dataclass
class PairwiseDiagnostic:
    """Diagnostics for a pair of knowledge models."""
    name_a: str
    name_b: str
    overlap_fraction: float         # how much shared knowledge (0-1)
    novelty_a_to_b: float           # how much A would add to B (0-1)
    novelty_b_to_a: float           # how much B would add to A (0-1)
    contradiction_ratio: float      # how much they disagree (0-1)
    per_band_overlap: NDArray       # (B,) overlap per band
    per_band_contradiction: NDArray  # (B,) contradiction per band


@dataclass
class CompositionDiagnostics:
    """Full diagnostics for a set of knowledge models."""
    constituent_names: list[str]
    per_constituent_energy: dict[str, float]
    per_constituent_sources: dict[str, int]
    pairwise: list[PairwiseDiagnostic]
    composed_energy: float
    composed_entropy_estimate: float  # rough spectral entropy of composed field
    total_sources: int


def _spectral_entropy(field: DenseField) -> float:
    """Estimate spectral entropy of a field (average across bands)."""
    entropies = []
    for b in range(field.bands):
        F_sym = (field.F[b] + field.F[b].T) / 2.0
        eigvals = np.linalg.eigvalsh(F_sym)
        eigvals = np.abs(eigvals)
        total = eigvals.sum()
        if total < 1e-12:
            entropies.append(0.0)
            continue
        probs = eigvals / total
        probs = probs[probs > 1e-12]
        entropy = -float(np.sum(probs * np.log2(probs)))
        entropies.append(entropy)
    return float(np.mean(entropies))


def diagnose_composition(
    constituents: dict[str, Lattice],  # noqa: F821
    composed_field: DenseField | None = None,
) -> CompositionDiagnostics:
    """Analyze a set of knowledge models before composing them.

    Computes pairwise overlap, novelty, and contradiction between
    all pairs of knowledge models, plus global diagnostics.

    Args:
        constituents: Named lattices to analyze.
        composed_field: If provided, also analyze the composed result.
            If None, a simple merge is computed for global stats.
    """
    names = list(constituents.keys())
    lattices = list(constituents.values())

    # Per-constituent stats
    per_energy = {}
    per_sources = {}
    for name, lattice in zip(names, lattices):
        per_energy[name] = float(sum(
            np.linalg.norm(lattice.field.F[b], "fro")
            for b in range(lattice.field.bands)
        ))
        per_sources[name] = lattice.source_count

    # Pairwise diagnostics
    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_field = lattices[i].field
            b_field = lattices[j].field

            # Overlap
            inter = FieldAlgebra.intersect(a_field, b_field)

            # Contradiction
            contra = FieldAlgebra.contradict(a_field, b_field)

            # Novelty (approximate: use a random probe)
            # What fraction of B's energy is novel relative to A?
            rng = np.random.default_rng(42)
            probe_phases = rng.standard_normal(
                (a_field.bands, a_field.dim)
            ).astype(np.float32)
            probe_phases /= np.linalg.norm(probe_phases, axis=1, keepdims=True)

            novelty_b_to_a = FieldAlgebra.novelty(a_field, probe_phases).score
            novelty_a_to_b = FieldAlgebra.novelty(b_field, probe_phases).score

            pairwise.append(PairwiseDiagnostic(
                name_a=names[i],
                name_b=names[j],
                overlap_fraction=inter.overlap_fraction,
                novelty_a_to_b=novelty_a_to_b,
                novelty_b_to_a=novelty_b_to_a,
                contradiction_ratio=contra.contradiction_ratio,
                per_band_overlap=inter.per_band_overlap,
                per_band_contradiction=contra.per_band_contradiction,
            ))

    # Composed field stats
    if composed_field is None:
        # Simple merge for global stats
        composed_field = DenseField(
            bands=lattices[0].field.bands,
            dim=lattices[0].field.dim,
        )
        for lattice in lattices:
            composed_field.F += lattice.field.F

    composed_energy = float(sum(
        np.linalg.norm(composed_field.F[b], "fro")
        for b in range(composed_field.bands)
    ))

    return CompositionDiagnostics(
        constituent_names=names,
        per_constituent_energy=per_energy,
        per_constituent_sources=per_sources,
        pairwise=pairwise,
        composed_energy=composed_energy,
        composed_entropy_estimate=_spectral_entropy(composed_field),
        total_sources=sum(per_sources.values()),
    )


def format_diagnostics(diag: CompositionDiagnostics) -> str:
    """Format diagnostics as human-readable text for --explain output."""
    lines = []
    lines.append("-- Composition Diagnostics --")
    lines.append("")

    # Per-constituent
    lines.append("Constituents:")
    for name in diag.constituent_names:
        e = diag.per_constituent_energy[name]
        n = diag.per_constituent_sources[name]
        lines.append(f"  {name}: {n} sources, energy={e:.2f}")

    lines.append("")

    # Pairwise
    if diag.pairwise:
        lines.append("Pairwise analysis:")
        for p in diag.pairwise:
            lines.append(f"  {p.name_a} <-> {p.name_b}:")
            lines.append(f"    Overlap:       {p.overlap_fraction:.1%}")
            lines.append(f"    Contradiction: {p.contradiction_ratio:.1%}")
            lines.append(f"    {p.name_a} adds to {p.name_b}: novelty={p.novelty_a_to_b:.1%}")
            lines.append(f"    {p.name_b} adds to {p.name_a}: novelty={p.novelty_b_to_a:.1%}")
        lines.append("")

    # Global
    lines.append("Composed field:")
    lines.append(f"  Total sources:    {diag.total_sources}")
    lines.append(f"  Composed energy:  {diag.composed_energy:.2f}")
    lines.append(f"  Spectral entropy: {diag.composed_entropy_estimate:.2f} bits")
    sum_energy = sum(diag.per_constituent_energy.values())
    if sum_energy > 0:
        synergy = (diag.composed_energy - sum_energy) / sum_energy
        if synergy > 0.05:
            lines.append(f"  Synergy:          +{synergy:.1%} (cross-reinforcement)")
        elif synergy < -0.05:
            lines.append(f"  Interference:     {synergy:.1%} (partial cancellation)")
        else:
            lines.append(f"  Interaction:      {synergy:+.1%} (near-additive)")

    lines.append("")
    return "\n".join(lines)
