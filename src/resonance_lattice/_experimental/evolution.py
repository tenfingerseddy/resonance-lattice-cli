# SPDX-License-Identifier: BUSL-1.1
"""Field Genetics: Darwinian evolution of Knowledge Compiler chains.

Chains are genomes. Operators are genes. User feedback is selection pressure.
The system discovers retrieval programs no human would design.

Supports:
- Mutation: perturb parameters, insert/delete/swap operators
- Crossover: splice two chains at a random cut point
- Selection: tournament selection on fitness
- Lamarckian gradient refinement: acquired traits inherited by offspring

evolve(population, field, queries, targets, generations) → best chain
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..compiler import (
    AutoTune,
    Cascade,
    Chain,
    CrossBandCouple,
    ExpandQuery,
    Metabolise,
    Operator,
)
from ..field.dense import DenseField


@dataclass
class Individual:
    """A single individual in the population."""
    chain: Chain
    fitness: float = 0.0
    generation: int = 0


@dataclass
class EvolutionResult:
    """Result of an evolutionary run."""
    best_chain: Chain
    best_fitness: float
    fitness_history: list[float]  # Best fitness per generation
    generations_run: int
    population_size: int
    total_evaluations: int


# ═══════════════════════════════════════════════════════
# Operator Pool (genes available for mutation)
# ═══════════════════════════════════════════════════════

def _random_operator(rng: np.random.Generator, bands: int, dim: int) -> Operator:
    """Generate a random operator from the available pool."""
    op_type = rng.choice(5)

    if op_type == 0:
        return AutoTune(temperature=float(rng.uniform(0.1, 5.0)))
    elif op_type == 1:
        return ExpandQuery(steps=int(rng.integers(1, 4)), eta=float(rng.uniform(0.05, 0.5)))
    elif op_type == 2:
        return Cascade(depth=int(rng.integers(1, 6)), alpha=float(rng.uniform(0.001, 0.5)))
    elif op_type == 3:
        K = int(rng.choice([8, 16, 32, 64]))
        K = min(K, dim - 1)
        return Metabolise(K=K, strategy=rng.choice(["energy", "spectral"]))
    else:
        return CrossBandCouple()


# ═══════════════════════════════════════════════════════
# Mutation Operators
# ═══════════════════════════════════════════════════════

def mutate_params(chain: Chain, rng: np.random.Generator, sigma: float = 0.2) -> Chain:
    """Perturb numerical parameters of operators."""
    new_ops = []
    for op in chain.operators:
        op = copy.deepcopy(op)
        if isinstance(op, AutoTune):
            op.temperature = max(0.01, op.temperature + float(rng.normal(0, sigma)))
        elif isinstance(op, ExpandQuery):
            op.eta = float(np.clip(op.eta + rng.normal(0, sigma * 0.5), 0.01, 0.95))
        elif isinstance(op, Cascade):
            op.alpha = float(np.clip(op.alpha + rng.normal(0, sigma * 0.1), 0.001, 1.0))
            op.depth = max(1, op.depth + int(rng.choice([-1, 0, 1])))
        elif isinstance(op, Metabolise):
            delta = int(rng.choice([-16, -8, 0, 8, 16]))
            op.K = max(4, op.K + delta)
        new_ops.append(op)
    return Chain(new_ops, name=f"{chain.name}_mutated")


def insert_op(chain: Chain, rng: np.random.Generator, bands: int, dim: int) -> Chain:
    """Insert a random operator at a random position."""
    ops = list(chain.operators)
    new_op = _random_operator(rng, bands, dim)
    pos = rng.integers(0, len(ops) + 1)
    ops.insert(pos, new_op)
    return Chain(ops, name=f"{chain.name}_inserted")


def delete_op(chain: Chain, rng: np.random.Generator) -> Chain:
    """Delete a random operator (if chain has > 1 operator)."""
    if len(chain.operators) <= 1:
        return chain
    ops = list(chain.operators)
    pos = rng.integers(0, len(ops))
    ops.pop(pos)
    return Chain(ops, name=f"{chain.name}_deleted")


def swap_ops(chain: Chain, rng: np.random.Generator) -> Chain:
    """Swap two random operators."""
    if len(chain.operators) < 2:
        return chain
    ops = list(chain.operators)
    i, j = rng.choice(len(ops), size=2, replace=False)
    ops[i], ops[j] = ops[j], ops[i]
    return Chain(ops, name=f"{chain.name}_swapped")


def splice(chain_a: Chain, chain_b: Chain, rng: np.random.Generator) -> Chain:
    """Crossover: splice two chains at a random cut point."""
    if len(chain_a.operators) == 0:
        return chain_b
    if len(chain_b.operators) == 0:
        return chain_a

    cut_a = rng.integers(0, len(chain_a.operators) + 1)
    cut_b = rng.integers(0, len(chain_b.operators) + 1)

    ops = list(chain_a.operators[:cut_a]) + list(chain_b.operators[cut_b:])
    return Chain(ops, name=f"splice_{chain_a.name}_{chain_b.name}")


# ═══════════════════════════════════════════════════════
# Fitness Evaluation
# ═══════════════════════════════════════════════════════

def default_fitness(
    chain: Chain,
    field: DenseField,
    queries: list[NDArray[np.float32]],
    targets: list[NDArray[np.float32]],
) -> float:
    """Default fitness function: cosine similarity between resonance and target.

    Args:
        chain: The chain to evaluate.
        field: The knowledge field.
        queries: List of (B, D) query phase vectors.
        targets: List of (D,) target resonance vectors.

    Returns:
        Average cosine similarity (higher = better).
    """
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
            total -= 1.0  # Penalty for crashes

    return total / max(len(queries), 1)


# ═══════════════════════════════════════════════════════
# Evolution Engine
# ═══════════════════════════════════════════════════════

def evolve(
    field: DenseField,
    queries: list[NDArray[np.float32]],
    targets: list[NDArray[np.float32]],
    seed_chain: Chain | None = None,
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.3,
    tournament_size: int = 3,
    max_chain_length: int = 8,
    fitness_fn: Callable | None = None,
    seed: int = 42,
) -> EvolutionResult:
    """Evolve a population of chains to maximize fitness.

    Args:
        field: The knowledge field.
        queries: Training queries [(B, D), ...].
        targets: Target resonance vectors [(D,), ...].
        seed_chain: Optional starting chain. If None, starts with random population.
        population_size: Number of individuals.
        generations: Number of generations to evolve.
        mutation_rate: Probability of mutation per individual.
        crossover_rate: Probability of crossover per generation.
        tournament_size: Tournament selection size.
        max_chain_length: Maximum operators per chain.
        fitness_fn: Custom fitness function. Uses default_fitness if None.
        seed: Random seed.

    Returns:
        EvolutionResult with the best chain found.
    """
    rng = np.random.default_rng(seed)
    bands = field.bands
    dim = field.dim
    evaluate = fitness_fn or default_fitness

    # Initialize population
    population: list[Individual] = []

    if seed_chain is not None:
        population.append(Individual(chain=seed_chain, generation=0))

    while len(population) < population_size:
        n_ops = rng.integers(1, 5)
        ops = [_random_operator(rng, bands, dim) for _ in range(n_ops)]
        population.append(Individual(
            chain=Chain(ops, name=f"random_{len(population)}"),
            generation=0,
        ))

    # Evaluate initial fitness
    for ind in population:
        ind.fitness = evaluate(ind.chain, field, queries, targets)

    fitness_history = [max(ind.fitness for ind in population)]
    total_evals = population_size

    for gen in range(generations):
        new_population: list[Individual] = []

        # Elitism: keep best individual
        best = max(population, key=lambda x: x.fitness)
        new_population.append(Individual(
            chain=best.chain, fitness=best.fitness, generation=gen + 1,
        ))

        while len(new_population) < population_size:
            # Tournament selection
            tournament = [population[i] for i in rng.choice(len(population), size=tournament_size)]
            parent = max(tournament, key=lambda x: x.fitness)

            child_chain = parent.chain

            # Crossover
            if rng.random() < crossover_rate:
                tournament2 = [population[i] for i in rng.choice(len(population), size=tournament_size)]
                parent2 = max(tournament2, key=lambda x: x.fitness)
                child_chain = splice(child_chain, parent2.chain, rng)

            # Mutation
            if rng.random() < mutation_rate:
                mutation_type = rng.choice(4)
                if mutation_type == 0:
                    child_chain = mutate_params(child_chain, rng)
                elif mutation_type == 1 and len(child_chain) < max_chain_length:
                    child_chain = insert_op(child_chain, rng, bands, dim)
                elif mutation_type == 2:
                    child_chain = delete_op(child_chain, rng)
                else:
                    child_chain = swap_ops(child_chain, rng)

            # Enforce max length
            if len(child_chain) > max_chain_length:
                child_chain = Chain(
                    list(child_chain.operators[:max_chain_length]),
                    name=child_chain.name,
                )

            # Evaluate
            fitness = evaluate(child_chain, field, queries, targets)
            total_evals += 1

            new_population.append(Individual(
                chain=child_chain, fitness=fitness, generation=gen + 1,
            ))

        population = new_population
        best_fitness = max(ind.fitness for ind in population)
        fitness_history.append(best_fitness)

    best = max(population, key=lambda x: x.fitness)

    return EvolutionResult(
        best_chain=best.chain,
        best_fitness=best.fitness,
        fitness_history=fitness_history,
        generations_run=generations,
        population_size=population_size,
        total_evaluations=total_evals,
    )
