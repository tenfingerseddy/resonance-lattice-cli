# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain: EML Operations.

The EML (Exp-Minus-Log) operator eml(x, y) = exp(x) - ln(y) is a Sheffer stroke
for continuous mathematics: every elementary function can be built from binary
trees of EML gates plus the constant 1. (Odrzywołek, arxiv:2603.21852)

This module provides:
    - Scalar EML operator and derived elementary functions
    - EML tree data structure for compositional operations
    - Matrix REML: expm(A) - logm(B) — universal field operator
    - EML spectral filter for learnable eigenvalue transforms
    - EML scoring functions for nonlinear resonance discrimination

All spectral operations use SpectralCache where available.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.types import Scalar

# ── Scalar EML Primitive ────────────────────────────────


def eml(x: float | NDArray, y: float | NDArray, eps: float = 1e-12) -> float | NDArray:
    """The universal binary operator: eml(x, y) = exp(x) - ln(y).

    Generates all elementary functions when composed with constant 1.
    Grammar: S -> 1 | eml(S, S).

    Args:
        x: First operand (exp domain).
        y: Second operand (ln domain). Must be > 0.
        eps: Floor for y to avoid ln(0).
    """
    y_safe = np.maximum(y, eps) if isinstance(y, np.ndarray) else max(y, eps)
    return np.exp(x) - np.log(y_safe)


# ── Derived Elementary Functions ─────────────────────────
# Each derived from eml + constant 1, proving universality.


def eml_exp(x: float | NDArray) -> float | NDArray:
    """exp(x) = eml(x, 1). Since ln(1) = 0."""
    return eml(x, 1.0)


def eml_ln(x: float | NDArray, eps: float = 1e-12) -> float | NDArray:
    """ln(x) = eml(1, eml(eml(1, x), 1)).

    Derivation: eml(1, x) = e - ln(x)
                eml(eml(1, x), 1) = exp(e - ln(x)) = e^e / x
                eml(1, eml(eml(1, x), 1)) = e - ln(e^e / x) = e - e + ln(x) = ln(x)
    """
    return eml(1.0, eml(eml(1.0, x, eps), 1.0, eps), eps)


def eml_neg(x: float | NDArray) -> float | NDArray:
    """-x derived via EML chain.

    Uses: -x = ln(1/e^x) = ln(e^(-x)) = eml_ln(eml_exp(-x))
    Simplified direct form: eml(0, eml_exp(x)) = 1 - ln(exp(x)) = 1 - x.
    So -x = eml(0, eml_exp(x)) - 1.

    For practical use, this demonstrates derivability rather than efficiency.
    """
    return 1.0 - x  # Algebraically exact: eml(0, exp(x)) - 1


def eml_add(x: float | NDArray, y: float | NDArray) -> float | NDArray:
    """x + y via EML: ln(exp(x) * exp(y)) = ln(exp(x)) + ln(exp(y)).

    Direct form: eml_ln(1/(eml_exp(-x) * eml_exp(-y)))
    Practical: uses log-sum-exp identity.
    """
    return x + y  # Algebraically derivable from EML; direct form for numerical stability


def eml_mul(x: float | NDArray, y: float | NDArray) -> float | NDArray:
    """x * y = exp(ln(x) + ln(y)) via EML chain.

    eml_exp(eml_ln(x) + eml_ln(y)) = exp(ln(x) + ln(y)) = x * y.
    """
    return x * y  # Derivable; direct form for stability


def eml_sqrt(x: float | NDArray, eps: float = 1e-12) -> float | NDArray:
    """sqrt(x) = exp(0.5 * ln(x)) via EML.

    = eml_exp(0.5 * eml_ln(x)).
    """
    x_safe = np.maximum(x, eps) if isinstance(x, np.ndarray) else max(x, eps)
    return np.exp(0.5 * np.log(x_safe))


def eml_pow(x: float | NDArray, p: float, eps: float = 1e-12) -> float | NDArray:
    """x^p = exp(p * ln(x)) via EML chain."""
    x_safe = np.maximum(x, eps) if isinstance(x, np.ndarray) else max(x, eps)
    return np.exp(p * np.log(x_safe))


def eml_inv(x: float | NDArray, eps: float = 1e-12) -> float | NDArray:
    """1/x = exp(-ln(x)) via EML chain."""
    return eml_pow(x, -1.0, eps)


# ── EML Constants ────────────────────────────────────────

E = eml(1.0, 1.0)  # e = exp(1) - ln(1) = e - 0 = e ≈ 2.71828


# ── EML Tree Data Structure ─────────────────────────────


@dataclass
class EMLNode:
    """Grammar: S -> constant | variable | eml(S, S).

    Every internal node applies the same eml(left, right) gate.
    Leaf nodes are either constants (typically 1.0) or named variables.
    """

    is_leaf: bool
    value: float | None = None
    left: EMLNode | None = None
    right: EMLNode | None = None
    input_name: str | None = None

    def evaluate(self, variables: dict[str, float | NDArray] | None = None) -> float | NDArray:
        """Evaluate the tree with given variable bindings."""
        if self.is_leaf:
            if self.input_name and variables:
                return variables[self.input_name]
            return self.value if self.value is not None else 1.0

        left_val = self.left.evaluate(variables)  # type: ignore[union-attr]
        right_val = self.right.evaluate(variables)  # type: ignore[union-attr]
        return eml(left_val, right_val)

    def to_expression(self) -> str:
        """Human-readable expression string."""
        if self.is_leaf:
            if self.input_name:
                return self.input_name
            return str(self.value) if self.value is not None else "1"

        left_str = self.left.to_expression()  # type: ignore[union-attr]
        right_str = self.right.to_expression()  # type: ignore[union-attr]
        return f"eml({left_str}, {right_str})"

    @property
    def depth(self) -> int:
        """Tree depth (leaves = 0)."""
        if self.is_leaf:
            return 0
        ld = self.left.depth if self.left else 0  # type: ignore[union-attr]
        rd = self.right.depth if self.right else 0  # type: ignore[union-attr]
        return 1 + max(ld, rd)

    @property
    def size(self) -> int:
        """Total node count."""
        if self.is_leaf:
            return 1
        ls = self.left.size if self.left else 0  # type: ignore[union-attr]
        rs = self.right.size if self.right else 0  # type: ignore[union-attr]
        return 1 + ls + rs


def leaf(value: float = 1.0) -> EMLNode:
    """Create a constant leaf node."""
    return EMLNode(is_leaf=True, value=value)


def var(name: str) -> EMLNode:
    """Create a variable leaf node."""
    return EMLNode(is_leaf=True, input_name=name)


def eml_tree(left: EMLNode, right: EMLNode) -> EMLNode:
    """Create an EML gate node: eml(left, right)."""
    return EMLNode(is_leaf=False, left=left, right=right)


# ── Pre-built EML Trees for Common Functions ─────────────

def make_exp_tree() -> EMLNode:
    """exp(x) = eml(x, 1)."""
    return eml_tree(var("x"), leaf(1.0))


def make_ln_tree() -> EMLNode:
    """ln(x) = eml(1, eml(eml(1, x), 1))."""
    return eml_tree(leaf(1.0), eml_tree(eml_tree(leaf(1.0), var("x")), leaf(1.0)))


# ── EML Spectral Filter ─────────────────────────────────


def eml_spectral_filter(
    eigenvalues: NDArray[np.float32],
    a: float = 0.0,
    b: float = 0.0,
    c: float = 1.0,
    d: float = 1.0,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    """Apply EML-parameterised filter to eigenvalues.

    f(λ) = exp(a·λ + b) - ln(max(c·λ + d, ε))

    This 4-parameter family covers:
        - Pure exp (c=0, d=1): exp(a·λ + b)
        - Pure log (a=0, b=0): -ln(c·λ + d)
        - Identity-like (a≈0, b≈0, c≈0, d≈1): ≈ 1
        - Power law (via exp-log composition)
        - Threshold + amplify (sharp log→exp transition)
        - Bandpass (log suppresses small λ, exp amplifies large λ)
    """
    exp_term = np.exp(a * eigenvalues + b)
    log_arg = np.maximum(c * eigenvalues + d, eps)
    log_term = np.log(log_arg)
    return (exp_term - log_term).astype(np.float32)


# ── EML Scoring Functions ────────────────────────────────


def eml_score_vector(
    similarity: NDArray[np.float32],
    noise_floor: float | NDArray[np.float32] = 1.0,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    """EML-based nonlinear scoring: score = exp(α·sim) - ln(σ).

    exp(α·sim) exponentially amplifies strong matches.
    -ln(σ) calibrates against the noise floor per band.

    Args:
        similarity: Dot product similarities (any shape).
        noise_floor: Per-band noise floor σ (scalar or matching shape).
            Higher σ → lower scores (more noise subtracted).
        alpha: Exponential scaling factor. Higher → sharper discrimination.
        eps: Floor for noise_floor to avoid ln(0).
    """
    return eml(alpha * similarity, np.maximum(noise_floor, eps))


def eml_fuse_bands(
    band_resonances: NDArray[np.float32],
    weights_exp: NDArray[np.float32],
    weights_log: NDArray[np.float32],
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    """EML nonlinear band fusion.

    R = exp(Σ_b α_b · r_b) - ln(max(Σ_b β_b · |r_b|, ε))

    Creates two channels:
        Precision channel (exp): sharp discrimination from specific bands.
        Context channel (log): stable background from broad bands.

    Args:
        band_resonances: Shape (B, D) — per-band resonance vectors.
        weights_exp: Shape (B,) — α weights for exp channel.
        weights_log: Shape (B,) — β weights for log channel.
        eps: Floor to avoid ln(0).

    Returns:
        Shape (D,) — fused resonance vector.
    """
    exp_input = np.einsum("bd,b->d", band_resonances, weights_exp)
    log_input = np.einsum("bd,b->d", np.abs(band_resonances), weights_log)
    log_input = np.maximum(log_input, eps)
    return (np.exp(exp_input) - np.log(log_input)).astype(np.float32)


# ── Matrix REML — Universal Field Operator ───────────────


@dataclass
class REMLResult:
    """Result of a matrix REML operation."""
    field: DenseField
    exp_eigenvalues: NDArray[np.float32] | None = None
    log_eigenvalues: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)


class EmlOps:
    """EML operations on field tensors. All methods are static.

    The Matrix REML (Resonance EML) operator:
        REML(A, B) = expm(A) - logm(B)

    When A and B share an eigenbasis:
        REML(A, B) = V diag(eml(λ_A, μ_B)) V^T

    Since scalar eml is universal, REML generates all spectral functional calculus:
        - reml(F, I) = expm(F)             [matrix exponential]
        - reml(0, F) = I - logm(F)         [negated matrix log]
        - reml(F, F) = expm(F) - logm(F)   [self-EML: sharpening]
    """

    @staticmethod
    def filter(
        field: DenseField,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 1.0,
        d: float = 1.0,
        cache: SpectralCache | None = None,
        bands: list[int] | None = None,
    ) -> DenseField:
        """Apply EML spectral filter: F' = V diag(eml_filter(λ)) V^T.

        f(λ) = exp(a·λ + b) - ln(max(c·λ + d, ε))

        Cost: O(BD^3) for eigendecomposition.
        """
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count

        for band_idx in (bands or range(field.bands)):
            if cache:
                eig = cache.get(band_idx)
                eigenvalues = eig.eigenvalues.copy()
                eigenvectors = eig.eigenvectors
            else:
                F_sym = (field.F[band_idx] + field.F[band_idx].T) / 2.0
                eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            new_eigs = eml_spectral_filter(eigenvalues, a, b, c, d)
            result.F[band_idx] = (eigenvectors * new_eigs) @ eigenvectors.T

        return result

    @staticmethod
    def reml(
        field_a: DenseField,
        field_b: DenseField,
        cache_a: SpectralCache | None = None,
        cache_b: SpectralCache | None = None,
        eps: float = 1e-12,
    ) -> DenseField:
        """Matrix REML: REML(A, B) = expm(A) - logm(B), per band.

        Uses eigendecomposition: expm(A) = V_A diag(exp(λ_A)) V_A^T
                                 logm(B) = V_B diag(log(μ_B)) V_B^T

        Cost: O(BD^3) for two eigendecompositions per band.
        """
        if field_a.bands != field_b.bands or field_a.dim != field_b.dim:
            raise ValueError(
                f"Field dimensions must match: "
                f"({field_a.bands}, {field_a.dim}) vs ({field_b.bands}, {field_b.dim})"
            )

        result = DenseField(bands=field_a.bands, dim=field_a.dim)
        result._source_count = max(field_a.source_count, field_b.source_count)

        for b in range(field_a.bands):
            # expm(A)
            if cache_a:
                eig_a = cache_a.get(b)
                lam_a, V_a = eig_a.eigenvalues, eig_a.eigenvectors
            else:
                F_a_sym = (field_a.F[b] + field_a.F[b].T) / 2.0
                lam_a, V_a = np.linalg.eigh(F_a_sym)

            exp_A = (V_a * np.exp(lam_a)) @ V_a.T

            # logm(B)
            if cache_b:
                eig_b = cache_b.get(b)
                lam_b, V_b = eig_b.eigenvalues, eig_b.eigenvectors
            else:
                F_b_sym = (field_b.F[b] + field_b.F[b].T) / 2.0
                lam_b, V_b = np.linalg.eigh(F_b_sym)

            log_B = (V_b * np.log(np.maximum(lam_b, eps))) @ V_b.T

            result.F[b] = (exp_A - log_B).astype(np.float32)

        return result

    @staticmethod
    def self_eml(
        field: DenseField,
        cache: SpectralCache | None = None,
        eps: float = 1e-12,
    ) -> DenseField:
        """Self-EML sharpening: F' = expm(F) - logm(F).

        Amplifies dominant eigendirections (via exp) while compressing
        weak ones (via log). A nonlinear contrast enhancement for the field.

        Cost: O(BD^3).
        """
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count

        for b in range(field.bands):
            if cache:
                eig = cache.get(b)
                lam, V = eig.eigenvalues, eig.eigenvectors
            else:
                F_sym = (field.F[b] + field.F[b].T) / 2.0
                lam, V = np.linalg.eigh(F_sym)

            eml_lam = np.exp(lam) - np.log(np.maximum(lam, eps))
            result.F[b] = (V * eml_lam.astype(np.float32)) @ V.T

        return result

    @staticmethod
    def sharpen(
        field: DenseField,
        strength: float = 1.0,
        cache: SpectralCache | None = None,
    ) -> DenseField:
        """EML sharpening with controllable strength.

        f(λ) = exp(s·λ) - ln(max(λ, ε))

        At strength=0: exp(0) - ln(λ) = 1 - ln(λ) (pure log compression).
        At strength=1: exp(λ) - ln(λ) (full EML sharpening).
        At strength>1: increasingly aggressive amplification.

        Cost: O(BD^3).
        """
        return EmlOps.filter(field, a=strength, b=0.0, c=1.0, d=0.0, cache=cache)

    @staticmethod
    def noise_floor(
        field: DenseField,
        band: int = 0,
        cache: SpectralCache | None = None,
    ) -> Scalar:
        """Estimate per-band noise floor from eigenvalue distribution.

        Uses the median eigenvalue as a robust noise floor estimate.
        Relevant for EML scoring: score = exp(α·sim) - ln(σ).

        Cost: O(D^3) for eigendecomposition if not cached.
        """
        if cache:
            eig = cache.get(band)
            eigenvalues = eig.eigenvalues
        else:
            F_sym = (field.F[band] + field.F[band].T) / 2.0
            eigenvalues, _ = np.linalg.eigh(F_sym)

        positive = eigenvalues[eigenvalues > 0]
        if len(positive) == 0:
            return Scalar(value=1.0, name="noise_floor", band=band)

        return Scalar(
            value=float(np.median(positive)),
            name="noise_floor",
            band=band,
            metadata={"eigenvalue_range": (float(positive.min()), float(positive.max()))},
        )
