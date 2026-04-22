# SPDX-License-Identifier: BUSL-1.1
"""Adaptive Composition: query-aware dynamic weighting across knowledge models.

The system learns which knowledge model to trust for which question.
Three strategies:

    energy   — weight by resonance strength (existing behavior)
    novelty  — weight by unique information each knowledge model offers
    band     — per-band routing (different knowledge models serve different bands)

Usage:
    weights = adaptive_weights(fields, query, strategy="novelty")
    # weights = {"docs": 0.6, "code": 0.4}

    band_weights = adaptive_band_weights(fields, query)
    # band_weights["docs"] = [0.8, 0.3, 0.1, 0.5, 0.2]
    # band_weights["code"] = [0.2, 0.7, 0.9, 0.5, 0.8]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField


@dataclass
class AdaptiveWeights:
    """Query-aware weights for multi-knowledge model composition."""
    strategy: str                          # "energy", "novelty", or "band"
    global_weights: dict[str, float]       # per-cartridge scalar weights
    band_weights: dict[str, NDArray] | None  # per-cartridge per-band weights (band strategy only)
    diagnostics: dict[str, float]          # per-cartridge diagnostic values


def _softmax(x: NDArray, temperature: float = 1.0) -> NDArray:
    """Numerically stable softmax."""
    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-12)


def adaptive_weights(
    fields: dict[str, DenseField],
    query_phase: NDArray[np.float32],
    strategy: str = "energy",
    temperature: float = 1.0,
) -> AdaptiveWeights:
    """Compute query-aware weights for a set of fields.

    Args:
        fields: Named fields to weight.
        query_phase: Shape (B, D) query.
        strategy: "energy", "novelty", or "band".
        temperature: Softmax temperature. Lower = more decisive.

    Returns:
        AdaptiveWeights with per-field weights.
    """
    names = list(fields.keys())
    field_list = list(fields.values())

    if strategy == "energy":
        return _energy_weights(names, field_list, query_phase, temperature)
    elif strategy == "novelty":
        return _novelty_weights(names, field_list, query_phase, temperature)
    elif strategy == "band":
        return _band_weights(names, field_list, query_phase, temperature)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'energy', 'novelty', or 'band'.")


def _energy_weights(
    names: list[str],
    fields: list[DenseField],
    query_phase: NDArray,
    temperature: float,
) -> AdaptiveWeights:
    """Weight by resonance energy: fields that resonate more get more weight."""
    energies = np.zeros(len(fields))
    for i, field in enumerate(fields):
        resonance = field.resonate(query_phase)
        energies[i] = float(np.sum(resonance.band_energies))

    weights = _softmax(energies, temperature)
    global_w = {name: float(w) for name, w in zip(names, weights)}
    diag = {name: float(e) for name, e in zip(names, energies)}

    return AdaptiveWeights(
        strategy="energy",
        global_weights=global_w,
        band_weights=None,
        diagnostics=diag,
    )


def _novelty_weights(
    names: list[str],
    fields: list[DenseField],
    query_phase: NDArray,
    temperature: float,
) -> AdaptiveWeights:
    """Weight by novelty × energy: fields offering unique info get more weight.

    For each field, measure how much it resonates AND how different its
    resonance is from the average of all other fields.
    """
    # Compute resonance for each field
    resonances = []
    energies = np.zeros(len(fields))
    for i, field in enumerate(fields):
        r = field.resonate(query_phase)
        resonances.append(r.fused)
        energies[i] = float(np.sum(r.band_energies))

    # Novelty: cosine distance from average
    novelty_scores = np.zeros(len(fields))
    for i, r in enumerate(resonances):
        # Leave-one-out average
        if len(resonances) > 1:
            others = [resonances[j] for j in range(len(resonances)) if j != i]
            other_avg = np.mean(others, axis=0)
            cos_sim = float(np.dot(r, other_avg)) / (
                np.linalg.norm(r) * np.linalg.norm(other_avg) + 1e-12
            )
            novelty_scores[i] = 1.0 - max(0.0, cos_sim)  # higher = more novel
        else:
            novelty_scores[i] = 1.0

    # Combined score: novelty × energy
    combined = novelty_scores * energies
    weights = _softmax(combined, temperature)

    global_w = {name: float(w) for name, w in zip(names, weights)}
    diag = {name: float(n) for name, n in zip(names, novelty_scores)}

    return AdaptiveWeights(
        strategy="novelty",
        global_weights=global_w,
        band_weights=None,
        diagnostics=diag,
    )


def _band_weights(
    names: list[str],
    fields: list[DenseField],
    query_phase: NDArray,
    temperature: float,
) -> AdaptiveWeights:
    """Per-band routing: different fields serve different bands.

    "Code knowledge model handles entity questions, docs handles topic questions."
    """
    B = fields[0].bands

    # Per-band per-field energy
    band_energies = np.zeros((len(fields), B))
    for i, field in enumerate(fields):
        resonance = field.resonate(query_phase)
        band_energies[i] = resonance.band_energies

    # Softmax across fields for each band
    per_band_weights = {}
    for f_idx, name in enumerate(names):
        bw = np.zeros(B)
        for b in range(B):
            col = band_energies[:, b]
            sm = _softmax(col, temperature)
            bw[b] = sm[f_idx]
        per_band_weights[name] = bw.astype(np.float32)

    # Global weight: average across bands
    global_w = {
        name: float(np.mean(per_band_weights[name]))
        for name in names
    }

    # Diagnostics: which band each field is strongest at
    diag = {}
    for f_idx, name in enumerate(names):
        strongest = int(np.argmax(per_band_weights[name]))
        diag[name] = float(per_band_weights[name][strongest])

    return AdaptiveWeights(
        strategy="band",
        global_weights=global_w,
        band_weights=per_band_weights,
        diagnostics=diag,
    )


def format_adaptive_weights(weights: AdaptiveWeights, band_names: list[str] | None = None) -> str:
    """Format adaptive weights as human-readable text."""
    lines = [f"Adaptive weights ({weights.strategy}):"]
    for name, w in sorted(weights.global_weights.items(), key=lambda x: -x[1]):
        diag_val = weights.diagnostics.get(name, 0)
        lines.append(f"  {name}: {w:.1%} (score={diag_val:.4f})")

    if weights.band_weights and band_names:
        lines.append("")
        lines.append("Per-band routing:")
        for name, bw in weights.band_weights.items():
            parts = [f"{bn}={w:.0%}" for bn, w in zip(band_names, bw)]
            lines.append(f"  {name}: {', '.join(parts)}")

    return "\n".join(lines)
