# SPDX-License-Identifier: BUSL-1.1
"""Temporal Knowledge Algebra: query the evolution of knowledge over time.

Treats time-ordered knowledge model snapshots as a trajectory on the PSD
manifold and provides operations to analyze how knowledge changes.

Operations:
    temporal_derivative  — rate of knowledge change (Riemannian or Euclidean)
    knowledge_trend      — is a topic growing or shrinking over time?
    temporal_diff_chain  — search each consecutive diff for a query
    temporal_extrapolate — predict next knowledge state

Uses existing FieldAlgebra.diff() for Euclidean diffs and GeoOps for
Riemannian operations when precision matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.field.dense import DenseField


@dataclass
class TemporalDerivative:
    """Rate of knowledge change between two snapshots."""
    derivative_field: DenseField  # signed — not PSD
    added_energy: float           # energy of positive eigenvalues (growth)
    removed_energy: float         # energy of negative eigenvalues (loss)
    net_change: float             # added - removed


@dataclass
class KnowledgeTrend:
    """How a query's relevance changes across snapshots."""
    timestamps: list[float]
    scores: list[float]     # resonance score at each snapshot
    deltas: list[float]     # score change between consecutive snapshots
    trend_direction: str    # "growing", "shrinking", "stable"
    trend_magnitude: float  # absolute average delta


@dataclass
class TemporalDiffChainResult:
    """Result of searching each consecutive diff in a snapshot series."""
    pairs: list[tuple[str, str]]           # (older_label, newer_label)
    diff_fields: list[DenseField]          # queryable delta per pair
    per_diff_energy: list[float]           # energy of each diff
    total_evolution_energy: float          # sum of all diff energies


def temporal_derivative(
    newer: DenseField,
    older: DenseField,
    dt: float = 1.0,
    mode: str = "euclidean",
) -> TemporalDerivative:
    """Compute the rate of knowledge change between two snapshots.

    Euclidean mode: dF/dt = (F_new - F_old) / dt
    Riemannian mode: dF/dt = Log_{F_old}(F_new) / dt  (preserves manifold geometry)

    The derivative field is signed (not PSD) — positive eigenvalues
    represent knowledge growth, negative represent knowledge loss.

    Args:
        newer: More recent field.
        older: Earlier field.
        dt: Time step between snapshots (for scaling).
        mode: "euclidean" (fast, approximate) or "riemannian" (precise, O(BD^3)).
    """
    assert newer.bands == older.bands and newer.dim == older.dim
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if mode not in ("euclidean", "riemannian"):
        raise ValueError(f"Unknown mode: {mode}. Use 'euclidean' or 'riemannian'.")

    deriv = DenseField(bands=newer.bands, dim=newer.dim)
    added = 0.0
    removed = 0.0

    if mode == "riemannian":
        from resonance_lattice.rql.geometric import GeoOps
        for b in range(newer.bands):
            log_tangent = GeoOps.log_map(older, newer, band=b)
            deriv.F[b] = (log_tangent / dt).astype(np.float32)
            eigvals = np.linalg.eigvalsh(deriv.F[b])
            added += float(np.sum(eigvals[eigvals > 0]))
            removed += float(np.sum(np.abs(eigvals[eigvals < 0])))
    else:
        # Euclidean: simple difference
        diff_result = FieldAlgebra.diff(newer, older)
        deriv.F = diff_result.delta_field.F / dt
        added = diff_result.added_energy / dt
        removed = diff_result.removed_energy / dt

    return TemporalDerivative(
        derivative_field=deriv,
        added_energy=added,
        removed_energy=removed,
        net_change=added - removed,
    )


def knowledge_trend(
    snapshots: list[DenseField],
    query_phase: NDArray[np.float32],
    timestamps: list[float] | None = None,
) -> KnowledgeTrend:
    """Track how a query's relevance changes across time-ordered snapshots.

    For each snapshot, resonates the query and records the score.
    Returns the trajectory and whether the topic is growing or shrinking.

    Args:
        snapshots: Time-ordered list of fields (oldest first).
        query_phase: Shape (B, D) — the query to track.
        timestamps: Optional timestamps. Defaults to [0, 1, 2, ...].
    """
    if timestamps is None:
        timestamps = list(range(len(snapshots)))

    scores = []
    for field in snapshots:
        resonance = field.resonate(query_phase)
        scores.append(float(np.linalg.norm(resonance.fused)))

    deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

    if not deltas:
        return KnowledgeTrend(
            timestamps=timestamps,
            scores=scores,
            deltas=deltas,
            trend_direction="stable",
            trend_magnitude=0.0,
        )

    avg_delta = float(np.mean(deltas))
    magnitude = float(np.mean(np.abs(deltas)))

    if avg_delta > 0.01 * magnitude + 1e-8:
        direction = "growing"
    elif avg_delta < -0.01 * magnitude - 1e-8:
        direction = "shrinking"
    else:
        direction = "stable"

    return KnowledgeTrend(
        timestamps=timestamps,
        scores=scores,
        deltas=deltas,
        trend_direction=direction,
        trend_magnitude=magnitude,
    )


def temporal_diff_chain(
    snapshots: list[DenseField],
    labels: list[str] | None = None,
) -> TemporalDiffChainResult:
    """Compute consecutive diffs across a series of snapshots.

    Each diff is a queryable DenseField representing what changed
    between consecutive snapshots. Search the diffs to find topics
    that evolved at each step.

    Args:
        snapshots: Time-ordered list of fields (oldest first).
        labels: Optional names for each snapshot (e.g. ["v1", "v2", "v3"]).
    """
    if labels is None:
        labels = [f"t{i}" for i in range(len(snapshots))]

    pairs = []
    diffs = []
    energies = []

    for i in range(len(snapshots) - 1):
        diff = FieldAlgebra.diff(snapshots[i + 1], snapshots[i])
        pairs.append((labels[i], labels[i + 1]))
        diffs.append(diff.delta_field)
        energies.append(abs(diff.net_change))

    return TemporalDiffChainResult(
        pairs=pairs,
        diff_fields=diffs,
        per_diff_energy=energies,
        total_evolution_energy=sum(energies),
    )


def temporal_extrapolate(
    snapshots: list[DenseField],
    dt: float = 1.0,
    mode: str = "euclidean",
) -> DenseField:
    """Predict the next knowledge state from a snapshot series.

    Uses the last two snapshots to estimate velocity, then extrapolates.

    Euclidean: F_next = F_last + dt * (F_last - F_prev)
    Riemannian: F_next = Exp_{F_last}(dt * Log_{F_prev}(F_last))

    Args:
        snapshots: At least 2 time-ordered fields.
        dt: Extrapolation step size.
        mode: "euclidean" or "riemannian".
    """
    if len(snapshots) < 2:
        raise ValueError("Need at least 2 snapshots to extrapolate")

    prev = snapshots[-2]
    last = snapshots[-1]

    if mode == "riemannian":
        from resonance_lattice.rql.geometric import GeoOps
        result = DenseField(bands=last.bands, dim=last.dim)
        result._source_count = last.source_count
        for b in range(last.bands):
            tangent = GeoOps.log_map(prev, last, band=b)
            # Exp map from last in direction of tangent
            F_last = last.F[b] + 1e-8 * np.eye(last.dim, dtype=np.float32)
            vals, vecs = np.linalg.eigh(F_last)
            vals = np.maximum(vals, 1e-8)
            A_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
            A_sqrt = (vecs * np.sqrt(vals)) @ vecs.T
            M = A_inv_sqrt @ (dt * tangent) @ A_inv_sqrt
            vals_m, vecs_m = np.linalg.eigh(M)
            exp_M = (vecs_m * np.exp(vals_m)) @ vecs_m.T
            result.F[b] = (A_sqrt @ exp_M @ A_sqrt).astype(np.float32)
        return result
    else:
        # Euclidean: linear extrapolation
        deriv = temporal_derivative(last, prev, dt=1.0, mode="euclidean")
        result = DenseField(bands=last.bands, dim=last.dim)
        result._source_count = last.source_count
        result.F = last.F + dt * deriv.derivative_field.F
        return result
