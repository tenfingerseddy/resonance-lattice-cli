# SPDX-License-Identifier: BUSL-1.1
"""Knowledge Compilers: composable algebraic transformation chains.

A declarative algebra for programming knowledge retrieval — SQL for semantic fields.

A Knowledge Compiler is an ordered sequence of operators applied to a field:
    F_task = O_n(...(O₂(O₁(F_base))))

Each operator is a pure mathematical transformation. Chains can be hand-crafted,
learned from feedback, or discovered by evolutionary search.

Algebraic properties:
    - Closure: every operator maps Field → Field
    - Identity: empty chain = no transformation
    - Associativity: (A then B) then C = A then (B then C)
    - Invertibility: most operators have inverses

This module subsumes all four other breakthroughs as operators:
    - AutoTune (Field Calculus)
    - Sculpt/Suppress (Interference Sculpting)
    - Metabolise (Dimensional Metabolism)
    - Cascade (Resonance Cascades)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .calculus import FieldCalculus
from .cascade import ResonanceCascade
from .field.dense import DenseField, ResonanceResult
from .subspace import SubspaceField, SubspaceStrategy, metabolise

# ═══════════════════════════════════════════════════════
# Base Operator
# ═══════════════════════════════════════════════════════

class Operator(ABC):
    """Base class for all knowledge compilation operators.

    Each operator transforms a field (or the retrieval process) in a
    mathematically well-defined way. Operators are composable via chains.
    """

    @abstractmethod
    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        """Apply this operator to a field.

        Args:
            field: The input field. May be modified in place or a new field returned.
            context: Shared context for the compilation chain.

        Returns:
            The transformed field.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


@dataclass
class CompilationContext:
    """Shared context passed through a compilation chain.

    Operators can read/write to this context to communicate with
    downstream operators. For example, AutoTune writes band_weights
    that Cascade reads.
    """
    query_phase: NDArray[np.float32] | None = None
    band_weights: NDArray[np.float32] | None = None
    resonance: ResonanceResult | None = None
    subspace: SubspaceField | None = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# Core Operators (Ideas 1-4 as composable units)
# ═══════════════════════════════════════════════════════

class AutoTune(Operator):
    """Gradient-derived per-query band weights (Field Calculus).

    Uses the resonance energy gradient to compute optimal band weights.
    Writes the computed weights to context.band_weights.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        if context.query_phase is None:
            return field

        # Compute resonance to get band energies
        resonance = field.resonate(context.query_phase)
        context.resonance = resonance

        # Derive optimal band weights from gradient
        bw = FieldCalculus.auto_band_weights(resonance, temperature=self.temperature)
        context.band_weights = bw.weights

        return field  # AutoTune doesn't modify the field itself

    def __repr__(self) -> str:
        return f"AutoTune(τ={self.temperature})"


class ExpandQuery(Operator):
    """Gradient-based query expansion (Field Calculus).

    Moves the query toward the field's energy gradient to discover
    related concepts. Modifies context.query_phase.
    """

    def __init__(self, steps: int = 1, eta: float = 0.2):
        self.steps = steps
        self.eta = eta

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        if context.query_phase is None:
            return field

        expansion = FieldCalculus.expand_query(
            field, context.query_phase,
            steps=self.steps, eta=self.eta,
        )
        context.query_phase = expansion.expanded_query
        return field

    def __repr__(self) -> str:
        return f"ExpandQuery(steps={self.steps}, η={self.eta})"


class Sculpt(Operator):
    """Add a sculpting pattern to the field (Interference Sculpting).

    Modifies the field tensor directly with a rank-1 update.
    """

    def __init__(
        self,
        phase_vectors: NDArray[np.float32],
        beta: float,
        label: str = "",
    ):
        self.phase_vectors = phase_vectors
        self.beta = beta
        self.label = label or ("boost" if beta > 0 else "suppress")

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        for b in range(field.bands):
            psi = self.phase_vectors[b]
            field.F[b] += self.beta * np.outer(psi, psi)
        return field

    def __repr__(self) -> str:
        return f"Sculpt({self.label}, β={self.beta})"


class SpectralFilter(Operator):
    """Modify eigenvalues of the field for task-specific emphasis.

    g: eigenvalue → transformed eigenvalue

    Examples:
        lambda x: x * (x > median)   — threshold: keep strong signals only
        lambda x: x ** 0.5           — flatten: surface weak signals
        lambda x: min(x, cap)        — clamp: prevent dominance
    """

    def __init__(self, transform: Callable[[NDArray], NDArray], label: str = ""):
        self.transform = transform
        self.label = label or "spectral_filter"

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        for b in range(field.bands):
            F_b = field.F[b]
            F_sym = (F_b + F_b.T) / 2.0

            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            # Apply the transform function to eigenvalues
            transformed = self.transform(eigenvalues)

            # Reconstruct
            field.F[b] = (eigenvectors * transformed) @ eigenvectors.T

        return field

    def __repr__(self) -> str:
        return f"SpectralFilter({self.label})"


class CrossBandCouple(Operator):
    """Mix signals across frequency bands.

    F'_b = Σ_c M_{bc} · F_c

    Current system uses independent bands (M = identity). This operator
    introduces controlled cross-band resonance.
    """

    def __init__(self, mixing_matrix: NDArray[np.float32] | None = None):
        self.mixing_matrix = mixing_matrix

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        B = field.bands
        M = self.mixing_matrix

        if M is None:
            # Default: uniform mixing with self-bias
            M = np.full((B, B), 0.1 / B, dtype=np.float32)
            np.fill_diagonal(M, 0.9)

        # Validate
        if M.shape != (B, B):
            raise ValueError(f"Mixing matrix shape {M.shape} != ({B}, {B})")

        # Apply mixing: F'_b = Σ_c M_{bc} · F_c
        old_F = field.F.copy()
        for b in range(B):
            field.F[b] = sum(M[b, c] * old_F[c] for c in range(B))

        return field

    def __repr__(self) -> str:
        return "CrossBandCouple()"


class Metabolise(Operator):
    """Project field to K-dimensional subspace (Dimensional Metabolism).

    After this operator, subsequent operations work in reduced space.
    Stores the SubspaceField in context.subspace.
    """

    def __init__(self, K: int = 128, strategy: str = "energy"):
        self.K = K
        self.strategy = SubspaceStrategy(strategy)

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        context.subspace = metabolise(
            field, self.K, self.strategy,
            query_phase=context.query_phase,
        )
        # The field itself is unchanged — the subspace is used at retrieval time
        return field

    def __repr__(self) -> str:
        return f"Metabolise(K={self.K}, {self.strategy.value})"


class Cascade(Operator):
    """Multi-hop resonance cascade (Resonance Cascades).

    Applies the cascade at retrieval time, storing the result in context.
    """

    def __init__(self, depth: int = 3, alpha: float = 0.1):
        self.depth = depth
        self.alpha = alpha

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        if context.query_phase is None:
            return field

        result = ResonanceCascade.cascade(
            field, context.query_phase,
            depth=self.depth, alpha=self.alpha,
            band_weights=context.band_weights,
        )
        context.metadata["cascade_result"] = result
        return field

    def __repr__(self) -> str:
        return f"Cascade(depth={self.depth}, α={self.alpha})"


# ═══════════════════════════════════════════════════════
# EML Corpus Transforms
# ═══════════════════════════════════════════════════════


class EmlSharpen(Operator):
    """Nonlinear corpus contrast enhancement via self-EML.

    exp(strength * λ_norm) - ln(λ_norm + eps) per band, where eigenvalues
    are normalised to [0, 1] to keep the transform scale-invariant.
    Exponentially amplifies dominant eigenvalues while logarithmically
    compressing weak ones. Makes the field more discriminative.

    strength=0: pure log compression (softer).
    strength=1: full self-EML sharpening.
    strength>1: aggressive amplification.
    """

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        eps = 1e-12

        for b in range(field.bands):
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            lam_max = float(np.max(np.abs(eigenvalues)))
            if lam_max < eps:
                continue

            lam_norm = eigenvalues / lam_max
            exp_term = np.exp(self.strength * lam_norm)
            log_term = np.log(np.maximum(np.abs(lam_norm), eps))
            filtered = (exp_term - log_term) * lam_max

            result.F[b] = ((eigenvectors * filtered.astype(np.float32))
                           @ eigenvectors.T)

        return result

    def __repr__(self) -> str:
        return f"EmlSharpen(strength={self.strength})"


class EmlSoften(Operator):
    """Logarithmic spectrum flattening for broader exploration.

    Compresses eigenvalue gaps so weak signals surface alongside
    dominant ones. The inverse of sharpening — reveals buried topics
    that strong clusters normally overshadow.

    Operates on normalised eigenvalues for scale invariance:
        f(λ_norm) = -ln(strength * λ_norm + 1), then rescaled by λ_max
    """

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        eps = 1e-12

        for b in range(field.bands):
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            lam_max = float(np.max(np.abs(eigenvalues)))
            if lam_max < eps:
                continue

            lam_norm = eigenvalues / lam_max
            log_arg = np.maximum(self.strength * lam_norm + 1.0, eps)
            filtered = -np.log(log_arg) * lam_max

            result.F[b] = ((eigenvectors * filtered.astype(np.float32))
                           @ eigenvectors.T)

        return result

    def __repr__(self) -> str:
        return f"EmlSoften(strength={self.strength})"


class EmlContrast(Operator):
    """Asymmetric corpus contrast via REML.

    REML(primary, background) = expm(primary) - logm(background)

    Exponentially amplifies what's unique to the primary field
    while logarithmically compressing the background's structure.
    Unlike linear diff (A - B), REML never creates pathological
    negative eigenvalues and provides tunable asymmetric contrast.

    Use case: "Show me what my project docs know that the vendor
    docs don't."
    """

    def __init__(self, background: DenseField):
        self.background = background

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        from resonance_lattice.rql.eml import EmlOps
        return EmlOps.reml(field, self.background)

    def __repr__(self) -> str:
        return "EmlContrast()"


# EML tune presets: (a, b, c, d) for f(λ) = exp(aλ+b) - ln(cλ+d)
EML_TUNE_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "focus": (2.0, 0.0, 0.0, 1.0),      # exp(2λ) — exponential sharpening
    "explore": (0.0, 0.0, 1.0, 1.0),     # -ln(λ+1) — log flattening
    "denoise": (0.5, 0.0, 1.0, 0.0),     # exp(0.5λ) - ln(λ) — bandpass
}


class EmlTune(Operator):
    """EML retrieval tuning presets.

    Configures the corpus field for different retrieval tasks using
    a single parameter:

    - focus:   Precision mode. Exponentially amplifies strong signals.
               Best for factoid lookups and specific questions.
    - explore: Breadth mode. Flattens the spectrum to surface buried topics.
               Best for research and "what are the trade-offs?" questions.
    - denoise: Clean mode. Bandpass filter that amplifies signal above the
               noise floor and compresses noise below it.
               Best for noisy corpora with lots of boilerplate.

    The operator normalises eigenvalues to [0, 1] before applying the filter
    and rescales the result, so presets work on any corpus size (small test
    fields or huge production knowledge models).
    """

    def __init__(self, preset: str):
        if preset not in EML_TUNE_PRESETS:
            raise ValueError(
                f"Unknown tune preset '{preset}'. "
                f"Choose from: {', '.join(EML_TUNE_PRESETS)}"
            )
        self.preset = preset
        self.a, self.b, self.c, self.d = EML_TUNE_PRESETS[preset]

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        # Apply scale-invariant EML filter: normalise eigenvalues to [0, 1],
        # apply the preset filter, then rescale the result to preserve
        # the field's overall energy scale.
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        eps = 1e-12

        for b in range(field.bands):
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            lam_max = float(np.max(np.abs(eigenvalues)))
            if lam_max < eps:
                continue  # empty band, leave as-is

            # Normalise to [-1, 1] range
            lam_norm = eigenvalues / lam_max

            # Apply EML filter in the normalised space
            exp_term = np.exp(self.a * lam_norm + self.b)
            log_arg = np.maximum(self.c * lam_norm + self.d, eps)
            log_term = np.log(log_arg)
            filtered = exp_term - log_term

            # Rescale to preserve original energy magnitude
            filtered = filtered * lam_max

            result.F[b] = ((eigenvectors * filtered.astype(np.float32))
                           @ eigenvectors.T)

        return result

    def __repr__(self) -> str:
        return f"EmlTune('{self.preset}')"


# ═══════════════════════════════════════════════════════
# The Chain (Knowledge Compiler)
# ═══════════════════════════════════════════════════════

class Chain:
    """A composable sequence of operators — the Knowledge Compiler.

    Usage:
        chain = Chain([
            Sculpt(encode("Python"), beta=0.3),
            SpectralFilter(lambda x: x * (x > np.median(x))),
            AutoTune(temperature=0.5),
        ])

        # Compile: apply chain to field
        compiled_field, context = chain.compile(field, query_phase)

        # Retrieve using the compiled field + context
        result = compiled_field.resonate(context.query_phase, context.band_weights)

    Chains are composable:
        chain_ab = chain_a + chain_b  # Concatenation
    """

    def __init__(self, operators: list[Operator] | None = None, name: str = ""):
        self.operators = operators or []
        self.name = name or "chain"

    def compile(
        self,
        field: DenseField,
        query_phase: NDArray[np.float32] | None = None,
        copy_field: bool = True,
    ) -> tuple[DenseField, CompilationContext]:
        """Apply the operator chain to a field.

        Args:
            field: The base field.
            query_phase: Optional query for query-dependent operators.
            copy_field: If True, copies the field before modifying.
                Set to False for in-place modification (faster but destructive).

        Returns:
            (transformed_field, context) — the compiled field and metadata.
        """
        if copy_field:
            compiled = DenseField(bands=field.bands, dim=field.dim)
            compiled.F = field.F.copy()
            compiled._source_count = field.source_count
        else:
            compiled = field

        context = CompilationContext(
            query_phase=query_phase.copy() if query_phase is not None else None,
        )

        for op in self.operators:
            compiled = op.apply(compiled, context)

        return compiled, context

    def __add__(self, other: Chain) -> Chain:
        """Concatenate two chains: chain_a + chain_b."""
        return Chain(
            operators=self.operators + other.operators,
            name=f"{self.name}+{other.name}",
        )

    def __repr__(self) -> str:
        ops = " → ".join(repr(op) for op in self.operators)
        return f"Chain({self.name})[{ops}]"

    def __len__(self) -> int:
        return len(self.operators)


# ═══════════════════════════════════════════════════════
# Pre-built Chains (ready-to-use recipes)
# ═══════════════════════════════════════════════════════

def precision_chain(K: int = 128) -> Chain:
    """Optimised for factoid queries — fast, focused, strong signals only.

    1. Metabolise to K dimensions (spectral strategy)
    2. Filter to keep only strong eigenvalues
    3. Auto-tune band weights with low temperature (sharp)
    """
    return Chain(
        operators=[
            Metabolise(K=K, strategy="spectral"),
            SpectralFilter(
                transform=lambda x: x * (x > np.median(x[x > 0]) if np.any(x > 0) else 0),
                label="threshold_median",
            ),
            AutoTune(temperature=0.5),
        ],
        name="precision",
    )


def exploration_chain(depth: int = 3, alpha: float = 0.1) -> Chain:
    """Optimised for discovery queries — broad, deep, surfaces weak signals.

    1. Cross-band coupling (mix signals across abstraction levels)
    2. Multi-hop cascade (traverse conceptual neighborhoods)
    3. Flatten eigenvalue spectrum (surface weak signals)
    """
    return Chain(
        operators=[
            CrossBandCouple(),
            Cascade(depth=depth, alpha=alpha),
            SpectralFilter(
                transform=lambda x: np.sqrt(np.maximum(x, 0)),
                label="sqrt_flatten",
            ),
        ],
        name="exploration",
    )


def focused_chain(
    boost_vectors: list[tuple[NDArray[np.float32], float, str]] | None = None,
    suppress_vectors: list[tuple[NDArray[np.float32], float, str]] | None = None,
    K: int = 256,
) -> Chain:
    """Optimised for domain-specific queries — sculpt then metabolise.

    1. Boost target patterns
    2. Suppress unwanted patterns
    3. Metabolise to K dimensions
    4. Auto-tune band weights
    """
    ops: list[Operator] = []

    if boost_vectors:
        for pv, beta, label in boost_vectors:
            ops.append(Sculpt(pv, beta=abs(beta), label=f"boost:{label}"))

    if suppress_vectors:
        for pv, beta, label in suppress_vectors:
            ops.append(Sculpt(pv, beta=-abs(beta), label=f"suppress:{label}"))

    ops.append(Metabolise(K=K, strategy="energy"))
    ops.append(AutoTune(temperature=1.0))

    return Chain(operators=ops, name="focused")
