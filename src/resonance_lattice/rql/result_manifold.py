# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain: Result Manifold.

The bridge between search results and field mathematics. result_field()
constructs a mini-DenseField from top-K phase vectors, making all 271 RQL
operations instantly applicable to result sets.

    Result set → result_field() → DenseField → any RQL operation

Stages:
    1. Core bridge + diagnostics (diversity, coverage, concentration, signal_strength)
    2. EML result operations (contrast, sharpen, filter)
    3. Query refinement (refine_query, iterative_deepen)

All functions are stateless. They construct a DenseField from phases and
call existing RQL operations — no new mathematical machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.algebra_ops import AlgebraOps
from resonance_lattice.rql.compare import CompareOps
from resonance_lattice.rql.eml import EmlOps
from resonance_lattice.rql.info import InfoOps
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.types import Scalar

# ── Stage 1: Core Bridge ──────────────────────────────────


def result_field(
    phases: NDArray[np.float32],
    scores: NDArray[np.float32] | None = None,
) -> DenseField:
    """Construct a DenseField from top-K result phase vectors.

    This is the bridge that makes all 271 RQL operations applicable
    to search result sets. Each result's phase spectrum is superposed
    into a mini-field, optionally weighted by retrieval score.

    Args:
        phases: Shape (K, B, D) — phase spectra for K results.
        scores: Shape (K,) — retrieval scores as superposition weights.
            Negative scores are clamped to 0. If None, uniform weights.

    Returns:
        DenseField with the result set's semantic structure.

    Examples:
        >>> rf = result_field(phases)
        >>> from resonance_lattice.rql import spectral, info
        >>> spectral.effective_rank(rf)     # how many topics?
        >>> info.von_neumann_entropy(rf)    # how diverse?
        >>> EmlOps.sharpen(rf)              # enhance contrast
    """
    if phases.ndim != 3:
        raise ValueError(f"phases must be (K, B, D), got shape {phases.shape}")

    K, B, D = phases.shape
    if K == 0:
        raise ValueError("Cannot build result field from empty result set")

    field = DenseField(bands=B, dim=D)

    if scores is not None:
        if scores.shape != (K,):
            raise ValueError(f"scores shape {scores.shape} != expected ({K},)")
        saliences = np.maximum(scores, 0.0).astype(np.float32)
        field.superpose_batch(phases, saliences=saliences)
    else:
        field.superpose_batch(phases)

    return field


# ── Stage 1: Convenience Diagnostics ──────────────────────


def diversity(rf: DenseField, band: int = 0) -> Scalar:
    """Semantic diversity of the result set.

    Uses effective rank: the exponential of the von Neumann entropy
    of the normalised eigenvalue spectrum. Measures how many independent
    semantic directions the results span.

    High → results cover many distinct topics.
    Low  → results cluster in a narrow subspace.
    """
    return SpectralOps.effective_rank(rf, band=band)


def coverage(rf: DenseField, corpus: DenseField) -> Scalar:
    """How much of the corpus's semantic space the results cover.

    Cosine similarity between the result field and corpus field tensors.
    High → results are representative of the full corpus.
    Low  → results occupy a niche subspace.
    """
    sim = CompareOps.cosine_similarity(rf, corpus)
    return Scalar(value=sim.value, name="coverage", metadata=sim.metadata)


def concentration(rf: DenseField, band: int = 0) -> Scalar:
    """How focused the result set is around a single theme.

    Uses purity (Tr(rho^2) where rho is the density matrix).
    High purity → results are tightly clustered.
    Low purity  → results are diffuse across topics.
    """
    return InfoOps.purity(rf, band=band)


def signal_strength(rf: DenseField) -> Scalar:
    """Total energy in the result field (Frobenius norm).

    Higher → stronger collective signal from results.
    Scales with both result count and retrieval score magnitude.
    """
    return AlgebraOps.frobenius_norm(rf)


# ── Eigenbasis Reranking Signal ────────────────────────────


def eigenbasis_scores(
    phases: NDArray[np.float32],
    scores: NDArray[np.float32] | None = None,
    band: int = 0,
    minor_bonus: float = 0.3,
) -> NDArray[np.float32]:
    """Per-result manifold score from eigenbasis decomposition.

    Decomposes each result's projection onto the result field's eigenbasis
    and derives a scalar score capturing structural uniqueness — something
    neither dense similarity nor keyword overlap measures.

    The score rewards results that:
    - Span multiple eigenmodes (high coverage)
    - Are NOT purely aligned with the dominant theme (low consensus)
    - Contribute energy in minor eigenvectors (long-tail facets)

    Formula per band:
        c_j = v_j^T @ phi_i[b]                        # projection coefficients
        alpha_j = c_j^2 / sum(c_k^2)                  # normalised energy dist
        coverage = exp(-sum(alpha_j * ln(alpha_j)))    # effective modes
        consensus = c_top^2 / sum(c_k^2)              # top-eigenvector fraction
        minor_energy = sum(c_j^2 for bottom half) / sum(c_k^2)

        score = (1 - consensus) * coverage + minor_bonus * minor_energy

    Args:
        phases: Shape (K, B, D) — phase spectra for K results.
        scores: Shape (K,) — retrieval scores as superposition weights.
        band: Which band to compute on (default 0, broadest semantic band).
        minor_bonus: Weight for minor-eigenvector energy. Default 0.3.

    Returns:
        Shape (K,) — manifold scores. Higher = more structurally unique/broad.
    """
    if phases.ndim != 3:
        raise ValueError(f"phases must be (K, B, D), got shape {phases.shape}")

    K, B, D = phases.shape
    if K == 0:
        return np.zeros(0, dtype=np.float32)

    # Build result field and eigendecompose the target band
    rf = result_field(phases, scores)
    F_sym = (rf.F[band] + rf.F[band].T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(F_sym)
    # eigh returns ascending order; eigenvectors[:, -1] is largest

    half = max(1, D // 2)
    manifold_scores = np.zeros(K, dtype=np.float32)
    eps = 1e-12

    for i in range(K):
        phi = phases[i, band]

        # Project onto eigenbasis: c_j = v_j^T @ phi
        coefficients = eigenvectors.T @ phi  # shape (D,)
        c_sq = coefficients ** 2
        total_energy = c_sq.sum()

        if total_energy < eps:
            # Zero-energy result: no meaningful projection
            manifold_scores[i] = 0.0
            continue

        # Normalised energy distribution
        alpha = c_sq / total_energy

        # Coverage: exponential entropy of alpha (effective number of modes)
        # Clamp alpha > 0 for safe log
        alpha_safe = np.maximum(alpha, eps)
        entropy = -np.sum(alpha * np.log(alpha_safe))
        coverage_val = np.exp(entropy)

        # Consensus: fraction of energy in the top eigenvector
        consensus_val = c_sq[-1] / total_energy

        # Minor energy: fraction in bottom half of eigenspectrum
        minor_val = c_sq[:half].sum() / total_energy

        manifold_scores[i] = (
            (1.0 - consensus_val) * coverage_val
            + minor_bonus * minor_val
        )

    return manifold_scores


# ── Stage 2: EML Result Operations ────────────────────────


def eml_result_contrast(
    rf: DenseField,
    corpus: DenseField,
) -> DenseField:
    """REML contrast: what's unique in results vs corpus.

    REML(result, corpus) = expm(result) - logm(corpus)

    exp amplifies the result field's dominant directions.
    log compresses the corpus's background structure.
    The difference highlights what this result set adds beyond
    the corpus baseline.

    Args:
        rf: Result field from result_field().
        corpus: Full corpus DenseField (same B, D dimensions).

    Returns:
        Contrast field emphasising result-specific structure.
    """
    return EmlOps.reml(rf, corpus)


def eml_result_sharpen(rf: DenseField) -> DenseField:
    """Self-EML sharpening of the result field.

    expm(F) - logm(F) on the result field's eigenvalues.
    exp amplifies dominant eigenvalues (strong themes).
    log compresses weak eigenvalues (noise/diffuse topics).

    Returns:
        Sharpened result field with enhanced contrast between
        primary themes and background noise.
    """
    return EmlOps.self_eml(rf)


def eml_result_filter(
    rf: DenseField,
    a: float = 0.0,
    b: float = 0.0,
    c: float = 1.0,
    d: float = 1.0,
) -> DenseField:
    """Apply universal EML spectral filter to result field eigenvalues.

    f(lambda) = exp(a*lambda + b) - ln(c*lambda + d)

    The 4-parameter family covers:
        - Identity (a=0, b=0, c=0, d=1)
        - Pure exp amplification (c=0, d=1)
        - Pure log compression (a=0, b=0)
        - Power law, threshold, bandpass, and compositions

    Args:
        rf: Result field from result_field().
        a, b, c, d: EML filter parameters.

    Returns:
        Filtered result field.
    """
    return EmlOps.filter(rf, a=a, b=b, c=c, d=d)


# ── Stage 3: Query Refinement ─────────────────────────────


def refine_query(
    rf: DenseField,
    original_query: NDArray[np.float32],
    blend: float = 0.3,
) -> NDArray[np.float32]:
    """Refine query using the result field's dominant eigenvector.

    Extracts the top eigenvector per band from the result field and
    blends it with the original query:

        q' = normalize((1 - blend) * q + blend * v1)

    The refined query tilts toward the consensus direction of the
    results while preserving the original intent.

    Args:
        rf: Result field from result_field().
        original_query: Shape (B, D) — original query phase vectors.
        blend: How much to shift toward result consensus (0-1).
            0 = no change, 1 = pure result direction.

    Returns:
        Shape (B, D) — refined query phase vectors, L2-normalised per band.
    """
    B, D = original_query.shape
    if rf.bands != B or rf.dim != D:
        raise ValueError(
            f"Result field ({rf.bands}, {rf.dim}) vs "
            f"query ({B}, {D}) dimension mismatch"
        )

    refined = np.zeros_like(original_query)

    for b in range(B):
        # Symmetric eigendecomposition; eigh returns ascending order
        F_sym = (rf.F[b] + rf.F[b].T) / 2.0
        _, eigenvectors = np.linalg.eigh(F_sym)
        v1 = eigenvectors[:, -1]  # largest eigenvalue's eigenvector

        # Align sign with query direction
        if np.dot(v1, original_query[b]) < 0:
            v1 = -v1

        # Blend original query with result consensus
        q_new = (1.0 - blend) * original_query[b] + blend * v1

        # L2 normalise
        norm = np.linalg.norm(q_new)
        if norm > 1e-8:
            q_new /= norm

        refined[b] = q_new

    return refined.astype(np.float32)


def iterative_deepen(
    rf: DenseField,
    query: NDArray[np.float32],
    rounds: int = 2,
    blend: float = 0.3,
    decay: float = 0.7,
) -> NDArray[np.float32]:
    """Iteratively refine query through the result field.

    Each round applies refine_query() with a decaying blend factor:
        blend_i = blend * decay^i

    This is fixed-field refinement — it does not re-search. Each round
    projects the query further into the result field's dominant subspace,
    with diminishing influence to prevent overshooting.

    For full search→refine→search loops, see Lattice --refine (Stage 4).

    Args:
        rf: Result field from result_field().
        query: Shape (B, D) — original query phase vectors.
        rounds: Number of refinement iterations (default 2).
        blend: Initial blend factor (0-1).
        decay: Per-round blend decay (blend shrinks by this factor each round).

    Returns:
        Shape (B, D) — refined query after all rounds.
    """
    current = query.copy()
    for i in range(rounds):
        current_blend = blend * (decay ** i)
        current = refine_query(rf, current, blend=current_blend)
    return current


# ── Stage 4 Support: Result Assessment ────────────────────


@dataclass
class ResultAssessment:
    """Manifold metrics for a search result set.

    Computed from the result field — a DenseField constructed from the
    top-K result phase vectors. All metrics are derived from existing
    RQL operations applied to this field.
    """
    diversity_score: float
    concentration_score: float
    signal_strength_score: float
    coverage_score: float | None = None
    result_field_ref: DenseField | None = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON consumers (excludes heavy field reference)."""
        d: dict[str, Any] = {
            "diversity": round(self.diversity_score, 4),
            "concentration": round(self.concentration_score, 6),
            "signal_strength": round(self.signal_strength_score, 4),
        }
        if self.coverage_score is not None:
            d["coverage"] = round(self.coverage_score, 4)
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def assess_results(
    phases: NDArray[np.float32],
    scores: NDArray[np.float32] | None = None,
    corpus_field: DenseField | None = None,
    band: int = 0,
) -> ResultAssessment:
    """Compute manifold diagnostics for a result set.

    Builds a result field and computes diversity, concentration,
    signal strength, and optionally coverage against the corpus.

    Args:
        phases: Shape (K, B, D) — phase spectra for K results.
        scores: Shape (K,) — retrieval scores as weights.
        corpus_field: Full corpus field for coverage computation.
        band: Which band to compute per-band metrics on.

    Returns:
        ResultAssessment with all computed metrics.
    """
    rf = result_field(phases, scores)

    div = diversity(rf, band=band)
    conc = concentration(rf, band=band)
    sig = signal_strength(rf)

    cov_score = None
    if corpus_field is not None:
        cov = coverage(rf, corpus_field)
        cov_score = cov.value

    return ResultAssessment(
        diversity_score=div.value,
        concentration_score=conc.value,
        signal_strength_score=sig.value,
        coverage_score=cov_score,
        result_field_ref=rf,
    )
