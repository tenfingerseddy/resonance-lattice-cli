# SPDX-License-Identifier: BUSL-1.1
"""Differential Fields: end-to-end differentiable knowledge operations.

The entire pipeline (sculpt → metabolise → cascade → resonate → score) is
differentiable. Backpropagate through the field to learn optimal chain
parameters from retrieval outcomes.

No neural network needed — pure calculus on algebraic operations.
Uses finite differences for gradient computation (fast for <20 params).

learn_chain(chain, field, queries, targets, steps, lr) → optimized chain
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..compiler import (
    AutoTune,
    Cascade,
    Chain,
    ExpandQuery,
    Metabolise,
)
from ..field.dense import DenseField


@dataclass
class DiffResult:
    """Result of differentiable chain optimization."""
    optimized_chain: Chain
    initial_loss: float
    final_loss: float
    loss_trajectory: list[float]
    steps_taken: int
    params_learned: int


def _extract_params(chain: Chain) -> list[float]:
    """Extract all differentiable parameters from a chain."""
    params = []
    for op in chain.operators:
        if isinstance(op, AutoTune):
            params.append(op.temperature)
        elif isinstance(op, ExpandQuery):
            params.append(op.eta)
        elif isinstance(op, Cascade):
            params.append(op.alpha)
        elif isinstance(op, Metabolise):
            params.append(float(op.K))
    return params


def _inject_params(chain: Chain, params: list[float]) -> Chain:
    """Create a new chain with injected parameter values."""
    ops = []
    param_idx = 0
    for op in chain.operators:
        op = copy.deepcopy(op)
        if isinstance(op, AutoTune):
            op.temperature = max(0.01, params[param_idx])
            param_idx += 1
        elif isinstance(op, ExpandQuery):
            op.eta = float(np.clip(params[param_idx], 0.01, 0.95))
            param_idx += 1
        elif isinstance(op, Cascade):
            op.alpha = float(np.clip(params[param_idx], 0.001, 1.0))
            param_idx += 1
        elif isinstance(op, Metabolise):
            op.K = max(4, int(round(params[param_idx])))
            param_idx += 1
        ops.append(op)
    return Chain(ops, name=chain.name)


def _evaluate_loss(
    chain: Chain,
    field: DenseField,
    queries: list[NDArray[np.float32]],
    targets: list[NDArray[np.float32]],
) -> float:
    """Compute loss: negative average cosine similarity."""
    total = 0.0
    for q, t in zip(queries, targets):
        try:
            compiled, ctx = chain.compile(field, q)
            bw = ctx.band_weights
            qp = ctx.query_phase if ctx.query_phase is not None else q

            if ctx.subspace is not None:
                result = ctx.subspace.resonate(qp, band_weights=bw)
            else:
                result = compiled.resonate(qp, band_weights=bw)

            r = result.fused
            cos_sim = float(np.dot(r, t) / (np.linalg.norm(r) * np.linalg.norm(t) + 1e-12))
            total += cos_sim
        except Exception:
            total -= 1.0

    return -total / max(len(queries), 1)  # Negative because we minimize


def compute_gradient(
    chain: Chain,
    field: DenseField,
    queries: list[NDArray[np.float32]],
    targets: list[NDArray[np.float32]],
    epsilon: float = 0.01,
) -> NDArray[np.float64]:
    """Compute gradient of loss w.r.t. chain parameters via finite differences.

    ∂L/∂θ_i ≈ (L(θ + ε·e_i) - L(θ - ε·e_i)) / (2ε)

    Args:
        chain: The chain to differentiate.
        field: The knowledge field.
        queries: Training queries.
        targets: Target resonance vectors.
        epsilon: Finite difference step size.

    Returns:
        Gradient vector (same length as number of differentiable params).
    """
    params = _extract_params(chain)
    n_params = len(params)
    gradient = np.zeros(n_params, dtype=np.float64)

    for i in range(n_params):
        # Forward perturbation
        params_plus = params.copy()
        params_plus[i] += epsilon
        chain_plus = _inject_params(chain, params_plus)
        loss_plus = _evaluate_loss(chain_plus, field, queries, targets)

        # Backward perturbation
        params_minus = params.copy()
        params_minus[i] -= epsilon
        chain_minus = _inject_params(chain, params_minus)
        loss_minus = _evaluate_loss(chain_minus, field, queries, targets)

        # Central difference
        gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient


def learn_chain(
    chain: Chain,
    field: DenseField,
    queries: list[NDArray[np.float32]],
    targets: list[NDArray[np.float32]],
    steps: int = 50,
    lr: float = 0.1,
    epsilon: float = 0.01,
) -> DiffResult:
    """Optimize chain parameters via gradient descent.

    Args:
        chain: Starting chain.
        field: The knowledge field.
        queries: Training queries [(B, D), ...].
        targets: Target resonance vectors [(D,), ...].
        steps: Number of gradient steps.
        lr: Learning rate.
        epsilon: Finite difference epsilon.

    Returns:
        DiffResult with the optimized chain and loss trajectory.
    """
    current_chain = chain
    params = _extract_params(current_chain)
    n_params = len(params)

    if n_params == 0:
        initial_loss = _evaluate_loss(current_chain, field, queries, targets)
        return DiffResult(
            optimized_chain=current_chain,
            initial_loss=initial_loss,
            final_loss=initial_loss,
            loss_trajectory=[initial_loss],
            steps_taken=0,
            params_learned=0,
        )

    initial_loss = _evaluate_loss(current_chain, field, queries, targets)
    loss_trajectory = [initial_loss]

    for step in range(steps):
        # Compute gradient
        grad = compute_gradient(current_chain, field, queries, targets, epsilon)

        # Gradient descent step
        params = _extract_params(current_chain)
        new_params = [p - lr * g for p, g in zip(params, grad)]

        # Update chain
        current_chain = _inject_params(current_chain, new_params)

        loss = _evaluate_loss(current_chain, field, queries, targets)
        loss_trajectory.append(loss)

        # Simple learning rate decay
        if step > 0 and loss > loss_trajectory[-2]:
            lr *= 0.5  # Reduce lr if loss increased

    return DiffResult(
        optimized_chain=current_chain,
        initial_loss=initial_loss,
        final_loss=loss_trajectory[-1],
        loss_trajectory=loss_trajectory,
        steps_taken=steps,
        params_learned=n_params,
    )
