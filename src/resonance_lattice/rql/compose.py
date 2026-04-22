# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain J: Compositional Operations.

Higher-order operations on operators and fields — parallel execution,
conditional branching, fixed-point iteration, band mapping.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.compiler import Chain
from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.types import Scalar


class ComposeOps:

    @staticmethod
    def map_bands(field: DenseField, fn: Callable[[NDArray, int], NDArray]) -> DenseField:
        """Apply a function independently to each band: F'_b = fn(F_b, b). Cost: B × fn_cost."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count
        for b in range(field.bands):
            result.F[b] = fn(field.F[b], b).astype(np.float32)
        return result

    @staticmethod
    def reduce_bands(field: DenseField, fn: Callable[[NDArray, NDArray], NDArray]) -> NDArray[np.float32]:
        """Reduce bands with a binary function: fold(fn, F_bands). Cost: B × fn_cost."""
        result = field.F[0].copy()
        for b in range(1, field.bands):
            result = fn(result, field.F[b])
        return result.astype(np.float32)

    @staticmethod
    def parallel(field: DenseField, chains: list[Chain], weights: list[float] | None = None, query_phase: NDArray[np.float32] | None = None) -> DenseField:
        """Apply multiple chains in parallel, weighted-merge results. Cost: Σ chain_costs."""
        if not chains:
            return field
        if weights is None:
            weights = [1.0 / len(chains)] * len(chains)

        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count

        for chain, w in zip(chains, weights):
            compiled, _ = chain.compile(field, query_phase)
            result.F += w * compiled.F

        return result

    @staticmethod
    def branch_merge(
        field: DenseField,
        predicate: Callable[[DenseField], bool],
        chain_true: Chain,
        chain_false: Chain,
        query_phase: NDArray[np.float32] | None = None,
    ) -> DenseField:
        """Conditional: if predicate(field) then chain_true else chain_false."""
        chain = chain_true if predicate(field) else chain_false
        compiled, _ = chain.compile(field, query_phase)
        return compiled

    @staticmethod
    def fixed_point(
        field: DenseField,
        chain: Chain,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        query_phase: NDArray[np.float32] | None = None,
    ) -> tuple[DenseField, int]:
        """Iterate chain until convergence: F* such that chain(F*) ≈ F*. Cost: K × chain_cost."""
        current = field
        for i in range(max_iterations):
            compiled, _ = chain.compile(current, query_phase)
            diff = float(np.linalg.norm(compiled.F - current.F))
            if diff < tolerance:
                return compiled, i + 1
            current = compiled
        return current, max_iterations

    @staticmethod
    def power_iterate(
        field: DenseField,
        chain: Chain,
        iterations: int = 5,
        query_phase: NDArray[np.float32] | None = None,
    ) -> list[DenseField]:
        """Apply chain repeatedly, return trajectory: [F, T(F), T²(F), ...]. Cost: K × chain_cost."""
        trajectory = [field]
        current = field
        for _ in range(iterations):
            compiled, _ = chain.compile(current, query_phase)
            trajectory.append(compiled)
            current = compiled
        return trajectory

    @staticmethod
    def conditional(
        field: DenseField,
        metric_fn: Callable[[DenseField], float],
        threshold: float,
        chain_above: Chain,
        chain_below: Chain,
        query_phase: NDArray[np.float32] | None = None,
    ) -> DenseField:
        """Apply chain_above if metric > threshold, else chain_below."""
        metric = metric_fn(field)
        chain = chain_above if metric > threshold else chain_below
        compiled, _ = chain.compile(field, query_phase)
        return compiled

    @staticmethod
    def zip_bands(a: DenseField, b: DenseField, fn: Callable[[NDArray, NDArray], NDArray]) -> DenseField:
        """Pair bands from two fields, apply binary function. Cost: B × fn_cost."""
        assert a.bands == b.bands and a.dim == b.dim
        result = DenseField(bands=a.bands, dim=a.dim)
        result._source_count = max(a.source_count, b.source_count)
        for bi in range(a.bands):
            result.F[bi] = fn(a.F[bi], b.F[bi]).astype(np.float32)
        return result

    @staticmethod
    def commutativity_test(field: DenseField, chain_a: Chain, chain_b: Chain, query_phase: NDArray[np.float32] | None = None) -> Scalar:
        """Test commutativity: ||A(B(F)) - B(A(F))||_F. 0 = commutative. Cost: 2 × (chain_a + chain_b)."""
        ab, _ = chain_b.compile(chain_a.compile(field, query_phase)[0], query_phase)
        ba, _ = chain_a.compile(chain_b.compile(field, query_phase)[0], query_phase)
        return Scalar(float(np.linalg.norm(ab.F - ba.F)), name="commutativity_error")
