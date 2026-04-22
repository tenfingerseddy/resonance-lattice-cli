# SPDX-License-Identifier: BUSL-1.1
"""RQL Phase 1: Python DSL with fluent API and operator factories.

This module provides:
1. Field wrapper with arithmetic operators (+, -, *) and .pipe() composition
2. Operator factory functions (boost, suppress, cascade, etc.)
3. Fluent chaining that compiles to Knowledge Compiler chains
4. Module-level encoder registry for text-based operations

The Field class wraps DenseField and adds:
    - __add__: merge two fields (resolves pending pipe chains first)
    - __sub__: diff two fields (resolves pending pipe chains first)
    - __mul__/__rmul__: scale a field (resolves pending pipe chains first)
    - pipe(op): apply an operator, return new Field
    - resonate(query): project query into the (possibly transformed) field
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.compiler import (
    AutoTune,
    Cascade,
    Chain,
    CompilationContext,
    CrossBandCouple,
    EmlContrast,
    EmlSharpen,
    EmlSoften,
    EmlTune,
    ExpandQuery,
    Metabolise,
    Operator,
    Sculpt,
    SpectralFilter,
)
from resonance_lattice.field.dense import DenseField, ResonanceResult

if TYPE_CHECKING:
    from resonance_lattice.encoder import Encoder

# ═══════════════════════════════════════════════════════
# Module-level encoder registry for text-based operations
# ═══════════════════════════════════════════════════════

_encoder: Encoder | None = None


def set_encoder(encoder: Encoder) -> None:
    """Register an encoder for text-based RQL operations (boost, suppress).

    Call this once after loading a knowledge model or initialising an encoder.
    The CLI, MCP server, and worker set this automatically.
    """
    global _encoder
    _encoder = encoder


def get_encoder() -> Encoder | None:
    """Return the registered encoder, or None if not set."""
    return _encoder


class Field:
    """Fluent wrapper around DenseField for RQL-style composition.

    Supports arithmetic:
        merged = field_a + field_b   # merge
        delta = field_a - field_b    # diff
        scaled = 0.5 * field_a       # scale

    Supports piping:
        result = field.pipe(boost("topic", 0.5)).pipe(cascade(3)).resonate(query)

    The pipe() method builds up a Chain internally. The chain is compiled
    lazily when resonate() or compile() is called.
    """

    def __init__(self, dense_field: DenseField):
        self._field = dense_field
        self._chain = Chain([], name="rql")

    @property
    def inner(self) -> DenseField:
        """Access the underlying DenseField."""
        return self._field

    @property
    def bands(self) -> int:
        return self._field.bands

    @property
    def dim(self) -> int:
        return self._field.dim

    def pipe(self, operator: Operator) -> Field:
        """Apply an operator to the chain. Returns a NEW Field (immutable).

        Usage:
            result = field.pipe(boost("topic", 0.5)).pipe(cascade(3))
        """
        new = Field(self._field)
        new._chain = Chain(
            operators=self._chain.operators + [operator],
            name=self._chain.name,
        )
        return new

    def compile(
        self,
        query_phase: NDArray[np.float32] | None = None,
    ) -> tuple[DenseField, CompilationContext]:
        """Compile the accumulated chain against the field.

        Returns the transformed field and compilation context.
        """
        return self._chain.compile(self._field, query_phase)

    def resonate(
        self,
        query_phase: NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
    ) -> ResonanceResult:
        """Compile the chain and resonate the query through the result.

        This is the terminal operation — it executes everything.
        """
        compiled_field, ctx = self.compile(query_phase)

        # Use context band weights if set by AutoTune, otherwise use provided
        weights = ctx.band_weights if ctx.band_weights is not None else band_weights

        # Use possibly-expanded query from context
        q = ctx.query_phase if ctx.query_phase is not None else query_phase

        # If a subspace was created by Metabolise, use it
        if ctx.subspace is not None:
            return ctx.subspace.resonate(q, band_weights=weights)

        return compiled_field.resonate(q, band_weights=weights)

    def _resolve(self) -> DenseField:
        """Resolve any pending pipe chain, returning the effective field.

        If no operators are queued, returns the raw field (no copy).
        If operators exist, compiles them with query_phase=None.
        Query-dependent operators (AutoTune, ExpandQuery, Cascade) may
        produce degraded results without a query — this is documented.
        """
        if not self._chain.operators:
            return self._field
        compiled, _ = self._chain.compile(self._field, query_phase=None)
        return compiled

    def __add__(self, other: Field) -> Field:
        """Merge two fields: F_merged = F_a + F_b.

        Resolves any pending pipe chains on both operands before merging.
        """
        result = FieldAlgebra.merge(self._resolve(), other._resolve())
        return Field(result.field)

    def __sub__(self, other: Field) -> Field:
        """Diff two fields: delta = F_a - F_b.

        Resolves any pending pipe chains on both operands before diffing.
        """
        result = FieldAlgebra.diff(self._resolve(), other._resolve())
        return Field(result.delta_field)

    def __mul__(self, scalar: float) -> Field:
        """Scale a field: F_scaled = s * F.

        Resolves any pending pipe chain before scaling.
        """
        resolved = self._resolve()
        new_field = DenseField(bands=resolved.bands, dim=resolved.dim)
        new_field.F = scalar * resolved.F.copy()
        new_field._source_count = resolved.source_count
        return Field(new_field)

    def __rmul__(self, scalar: float) -> Field:
        """Support scalar * field syntax."""
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        chain_repr = f" |> {self._chain}" if self._chain.operators else ""
        return f"Field(B={self.bands}, D={self.dim}, N={self._field.source_count}{chain_repr})"


# ═══════════════════════════════════════════════════════
# Operator Factory Functions
# ═══════════════════════════════════════════════════════

def _encode_text(text: str, encoder: Encoder | None = None) -> NDArray[np.float32]:
    """Encode text to phase vectors using the provided or registered encoder."""
    enc = encoder or _encoder
    if enc is None:
        raise ValueError(
            "Text-based boost/suppress requires an encoder. Either:\n"
            "  1. Call rql.set_encoder(encoder) first, or\n"
            "  2. Pass encoder= to boost()/suppress(), or\n"
            "  3. Pass pre-encoded (B, D) phase vectors instead of text."
        )
    spectrum = enc.encode(text)
    return spectrum.vectors


def boost(
    text_or_phase: str | NDArray[np.float32],
    beta: float = 0.5,
    encoder: Encoder | None = None,
) -> Operator:
    """Create a boost sculpting operator.

    Amplifies a semantic direction in the field via rank-1 update:
    F' = F + beta * (phi x phi)

    Args:
        text_or_phase: Text string (encoded via registered/provided encoder)
            or pre-encoded (B, D) phase vectors.
        beta: Boost strength (positive).
        encoder: Optional encoder override. If None, uses the module-level
            encoder set via set_encoder().
    """
    if isinstance(text_or_phase, str):
        text_or_phase = _encode_text(text_or_phase, encoder)
    return Sculpt(text_or_phase, beta=abs(beta), label=f"boost:{abs(beta):.2f}")


def suppress(
    text_or_phase: str | NDArray[np.float32],
    beta: float = 0.3,
    encoder: Encoder | None = None,
) -> Operator:
    """Create a suppress sculpting operator.

    Attenuates a semantic direction in the field via rank-1 subtraction:
    F' = F - beta * (phi x phi)

    Args:
        text_or_phase: Text string (encoded via registered/provided encoder)
            or pre-encoded (B, D) phase vectors.
        beta: Suppression strength (will be negated).
        encoder: Optional encoder override. If None, uses the module-level
            encoder set via set_encoder().
    """
    if isinstance(text_or_phase, str):
        text_or_phase = _encode_text(text_or_phase, encoder)
    return Sculpt(text_or_phase, beta=-abs(beta), label=f"suppress:{abs(beta):.2f}")


def cascade(depth: int = 3, alpha: float = 0.1) -> Operator:
    """Create a multi-hop cascade operator."""
    return Cascade(depth=depth, alpha=alpha)


def metabolise(K: int = 128, strategy: str = "energy") -> Operator:
    """Create an adaptive subspace projection operator."""
    return Metabolise(K=K, strategy=strategy)


def autotune(temperature: float = 1.0) -> Operator:
    """Create a gradient-based band weight auto-tuning operator."""
    return AutoTune(temperature=temperature)


def expand(steps: int = 1, eta: float = 0.2) -> Operator:
    """Create a gradient-based query expansion operator."""
    return ExpandQuery(steps=steps, eta=eta)


def spectral_filter(
    transform: Callable[[NDArray], NDArray],
    label: str = "",
) -> Operator:
    """Create a spectral filter operator."""
    return SpectralFilter(transform=transform, label=label)


def crossband(mixing_matrix: NDArray[np.float32] | None = None) -> Operator:
    """Create a cross-band coupling operator."""
    return CrossBandCouple(mixing_matrix=mixing_matrix)


# ── EML Corpus Transform Factories ────────────────────────


def eml_sharpen(strength: float = 1.0) -> Operator:
    """Sharpen the corpus field for more precise retrieval.

    Nonlinear contrast enhancement via self-EML: amplifies dominant
    topics exponentially while compressing noise logarithmically.
    Higher strength = sharper discrimination.

    Usage: Field(f).pipe(eml_sharpen(1.5)).resonate(q)
    """
    return EmlSharpen(strength=strength)


def eml_soften(strength: float = 1.0) -> Operator:
    """Soften the corpus field for broader exploration.

    Flattens the eigenvalue spectrum so buried topics surface
    alongside dominant ones. Higher strength = flatter spectrum.

    Usage: Field(f).pipe(eml_soften(0.8)).resonate(q)
    """
    return EmlSoften(strength=strength)


def eml_contrast(background: DenseField) -> Operator:
    """Contrast the corpus against a background field via REML.

    REML(primary, background) = expm(primary) - logm(background).
    Amplifies what's unique to the primary field while compressing
    the background's structure. "Show me what A knows that B doesn't."

    Usage: Field(primary).pipe(eml_contrast(background_field)).resonate(q)
    """
    return EmlContrast(background=background)


def eml_tune(preset: str) -> Operator:
    """Tune the corpus field for a specific retrieval task.

    Presets:
        focus   — Precision mode for factoid lookups and specific questions.
        explore — Breadth mode for research and trade-off questions.
        denoise — Clean mode for noisy corpora with boilerplate.

    Usage: Field(f).pipe(eml_tune("focus")).resonate(q)
    """
    return EmlTune(preset=preset)
