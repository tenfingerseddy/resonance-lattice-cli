# SPDX-License-Identifier: BUSL-1.1
"""Reversible cascades: information-preserving knowledge traversal.

Uses Hamiltonian (symplectic) integration to preserve phase-space volume
exactly, preventing the eigenvalue blowup/collapse of naive cascading.

Störmer-Verlet integrator:
    p_{n+½} = p_n + (h/2) · F · q_n
    q_{n+1} = q_n + h · p_{n+½}
    p_{n+1} = p_{n+½} + (h/2) · F · q_{n+1}

Guarantee: det(Jacobian) = 1.0 at every step.
→ No information lost regardless of cascade depth.
→ Perfectly reversible: recover original query from any trajectory point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField


@dataclass
class SymplecticState:
    """A (position, momentum) pair in the Hamiltonian phase space."""
    q: NDArray[np.float32]  # Position (query-like vector)
    p: NDArray[np.float32]  # Momentum (resonance-like vector)


@dataclass
class SymplecticTrajectory:
    """Complete trajectory from a symplectic cascade."""
    positions: list[NDArray[np.float32]]  # q at each step
    momenta: list[NDArray[np.float32]]  # p at each step
    energies: list[float]  # Hamiltonian H at each step (should be ~constant)
    energy_drift: float  # |H_final - H_initial| / |H_initial|
    depth: int
    step_size: float
    band: int


class SymplecticCascade:
    """Information-preserving cascade via Hamiltonian dynamics.

    Treats (query, resonance) as conjugate variables:
        H(q, p) = ½ pᵀp - ½ qᵀFq

    The Störmer-Verlet integrator is:
    - Symplectic: preserves phase space volume exactly
    - Time-reversible: run backwards to recover q₀ from q_n
    - Energy-conserving to O(h²): bounded error that does NOT grow with steps
    """

    @staticmethod
    def cascade(
        field: DenseField,
        query_phase: NDArray[np.float32],
        depth: int = 10,
        h: float = 0.1,
        band: int = 0,
    ) -> SymplecticTrajectory:
        """Symplectic cascade on a single band.

        Evolves (q, p) through `depth` Störmer-Verlet steps:
            p_{n+½} = p_n + (h/2) · F_b · q_n
            q_{n+1} = q_n + h · p_{n+½}
            p_{n+1} = p_{n+½} + (h/2) · F_b · q_{n+1}

        Args:
            field: The dense field.
            query_phase: Shape (B, D) — uses the specified band.
            depth: Number of integration steps.
            h: Step size. Smaller = more accurate energy conservation.
            band: Which band to cascade on.

        Returns:
            SymplecticTrajectory with positions, momenta, and energy diagnostics.
        """
        F_b = field.F[band]

        # Initial conditions
        q = query_phase[band].copy().astype(np.float32)
        p = (F_b @ q).astype(np.float32)  # Initial momentum = first resonance

        positions = [q.copy()]
        momenta = [p.copy()]

        H_initial = SymplecticCascade._hamiltonian(q, p, F_b)
        energies = [H_initial]

        for _ in range(depth):
            # Störmer-Verlet (leapfrog) integration
            # Half-step momentum
            p_half = p + (h / 2) * (F_b @ q)

            # Full-step position
            q_new = q + h * p_half

            # Half-step momentum
            p_new = p_half + (h / 2) * (F_b @ q_new)

            q = q_new.astype(np.float32)
            p = p_new.astype(np.float32)

            positions.append(q.copy())
            momenta.append(p.copy())
            energies.append(SymplecticCascade._hamiltonian(q, p, F_b))

        # Energy drift (should be bounded O(h²), NOT growing with depth)
        H_final = energies[-1]
        drift = abs(H_final - H_initial) / (abs(H_initial) + 1e-12)

        return SymplecticTrajectory(
            positions=positions,
            momenta=momenta,
            energies=energies,
            energy_drift=drift,
            depth=depth,
            step_size=h,
            band=band,
        )

    @staticmethod
    def cascade_all_bands(
        field: DenseField,
        query_phase: NDArray[np.float32],
        depth: int = 10,
        h: float = 0.1,
    ) -> list[SymplecticTrajectory]:
        """Run symplectic cascade on all bands independently.

        Args:
            field: The dense field.
            query_phase: Shape (B, D).
            depth: Number of steps per band.
            h: Step size.

        Returns:
            List of SymplecticTrajectory, one per band.
        """
        return [
            SymplecticCascade.cascade(field, query_phase, depth, h, band=b)
            for b in range(field.bands)
        ]

    @staticmethod
    def reverse(
        trajectory: SymplecticTrajectory,
        field: DenseField,
    ) -> NDArray[np.float32]:
        """Reverse the cascade to recover the original query.

        Validates information preservation: the reversed trajectory
        should match the initial position to high precision.

        Args:
            trajectory: A forward trajectory.
            field: The same field used for the forward cascade.

        Returns:
            Recovered initial position (should match trajectory.positions[0]).
        """
        F_b = field.F[trajectory.band]
        h = -trajectory.step_size  # Negative step = time reversal

        # Start from final state
        q = trajectory.positions[-1].copy()
        p = trajectory.momenta[-1].copy()

        for _ in range(trajectory.depth):
            # Same Störmer-Verlet, but with negative h
            p_half = p + (h / 2) * (F_b @ q)
            q_new = q + h * p_half
            p_new = p_half + (h / 2) * (F_b @ q_new)
            q = q_new.astype(np.float32)
            p = p_new.astype(np.float32)

        return q

    @staticmethod
    def fuse_trajectory(
        trajectory: SymplecticTrajectory,
        weights: str = "exponential",
        decay: float = 0.5,
    ) -> NDArray[np.float32]:
        """Fuse trajectory positions into a single resonance vector.

        Args:
            trajectory: The symplectic trajectory.
            weights: Weighting scheme:
                "exponential" — decay^k weighting (favours early hops)
                "uniform" — equal weight per hop
                "final" — only use the last position
            decay: Decay rate for exponential weighting.

        Returns:
            Shape (D,) — fused resonance vector.
        """
        positions = trajectory.positions
        D = len(positions[0])

        if weights == "final":
            return positions[-1].copy()

        fused = np.zeros(D, dtype=np.float32)

        if weights == "uniform":
            for pos in positions:
                fused += pos
            fused /= len(positions)

        elif weights == "exponential":
            total_weight = 0.0
            for k, pos in enumerate(positions):
                w = decay ** k
                fused += w * pos
                total_weight += w
            fused /= total_weight + 1e-12

        return fused

    @staticmethod
    def _hamiltonian(
        q: NDArray[np.float32],
        p: NDArray[np.float32],
        F_b: NDArray[np.float32],
    ) -> float:
        """Compute the Hamiltonian H(q, p) = ½pᵀp - ½qᵀFq."""
        kinetic = 0.5 * float(np.dot(p, p))
        potential = -0.5 * float(q @ F_b @ q)
        return kinetic + potential

    @staticmethod
    def optimal_step_size(
        field: DenseField,
        band: int = 0,
        target_drift: float = 0.01,
    ) -> float:
        """Estimate optimal step size for a given energy drift tolerance.

        Energy error scales as O(h²). Estimate λ_max and compute h such that
        the expected drift is below target_drift.

        Args:
            field: The dense field.
            band: Which band.
            target_drift: Maximum acceptable relative energy drift.

        Returns:
            Recommended step size h.
        """
        F_b = field.F[band]

        # Approximate λ_max via power iteration
        rng = np.random.default_rng(42)
        v = rng.standard_normal(field.dim).astype(np.float32)
        v /= np.linalg.norm(v)

        for _ in range(20):
            v_new = F_b @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                return 0.1  # Field is near-zero
            v = v_new / norm

        lambda_max = abs(float(np.dot(v, F_b @ v)))

        if lambda_max < 1e-12:
            return 0.1

        # h ∝ sqrt(target_drift / λ_max)
        h = np.sqrt(target_drift / lambda_max)
        return float(np.clip(h, 0.001, 1.0))
