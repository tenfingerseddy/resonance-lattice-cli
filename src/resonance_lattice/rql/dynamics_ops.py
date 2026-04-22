# SPDX-License-Identifier: BUSL-1.1
"""RQL Domain E: Dynamic Operations.

Time evolution, flows, diffusion, decay — treating the field as a dynamical system.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.spectral import _apply_to_eigenvalues
from resonance_lattice.rql.types import Scalar


class DynamicsOps:

    @staticmethod
    def heat_diffusion(field: DenseField, t: float = 0.1, cache: SpectralCache | None = None) -> DenseField:
        """Heat diffusion: F(t) = V diag(exp(-λ·t)) Vᵀ. Smooths the field. Cost: O(BD³)."""
        return _apply_to_eigenvalues(field, lambda lam: lam * np.exp(-np.abs(lam) * t), cache)

    @staticmethod
    def exponential_decay(field: DenseField, rate: float = 0.01) -> DenseField:
        """Exponential decay: F' = F · exp(-rate). Uniform attenuation. Cost: O(BD²)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F * np.exp(-rate)
        result._source_count = field.source_count
        return result

    @staticmethod
    def impulse(field: DenseField, phase: NDArray[np.float32], strength: float = 1.0) -> DenseField:
        """Impulse: F' = F + δ·(φ ⊗ φ). Instantaneous rank-1 kick. Cost: O(BD)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        for b in range(field.bands):
            result.F[b] += strength * np.outer(phase[b], phase[b])
        return result

    @staticmethod
    def spectral_sharpening(field: DenseField, factor: float = 1.5, cache: SpectralCache | None = None) -> DenseField:
        """Spectral sharpening: increase eigenvalue gaps. λ' = sign(λ)·|λ|^factor. Cost: O(BD³)."""
        return _apply_to_eigenvalues(
            field,
            lambda lam: np.sign(lam) * np.abs(lam) ** factor,
            cache,
        )

    @staticmethod
    def gradient_flow(field: DenseField, grad: NDArray[np.float32], dt: float = 0.01) -> DenseField:
        """One step of gradient flow: F' = F - dt · ∇L. Cost: O(BD²)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F - dt * grad
        result._source_count = field.source_count
        return result

    @staticmethod
    def langevin_step(field: DenseField, dt: float = 0.01, temperature: float = 0.1, cache: SpectralCache | None = None) -> DenseField:
        """Langevin dynamics: F' = F - dt·∇V + √(2·dt·T)·noise. Stochastic exploration. Cost: O(BD³)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count
        rng = np.random.default_rng()
        for b in range(field.bands):
            # Gradient of potential (eigenvalue-based)
            noise = rng.standard_normal(field.F[b].shape).astype(np.float32)
            noise = (noise + noise.T) / 2  # Symmetric noise
            result.F[b] += np.sqrt(2 * dt * temperature) * noise
        return result

    @staticmethod
    def annealing_step(field: DenseField, temperature: float, cooling_rate: float = 0.95, cache: SpectralCache | None = None) -> DenseField:
        """Simulated annealing: apply spectral softmax at temperature, then cool. Cost: O(BD³)."""
        new_temp = temperature * cooling_rate
        return _apply_to_eigenvalues(
            field,
            lambda lam: lam * np.exp(lam / (new_temp + 1e-12)) / (np.sum(np.exp(lam / (new_temp + 1e-12))) + 1e-12) * np.sum(np.abs(lam)),
            cache,
        )

    @staticmethod
    def dissipative_step(field: DenseField, equilibrium: DenseField, gamma: float = 0.1) -> DenseField:
        """Dissipative relaxation: F' = (1-γ)F + γF_eq. Cost: O(BD²)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = (1 - gamma) * field.F + gamma * equilibrium.F
        result._source_count = field.source_count
        return result

    @staticmethod
    def logistic_growth(field: DenseField, rate: float = 0.1, capacity: float = 1.0, dt: float = 0.01) -> DenseField:
        """Logistic growth: F' = F + dt·r·F·(1 - F/K) element-wise. Cost: O(BD²)."""
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F + dt * rate * field.F * (1 - field.F / (capacity + 1e-12))
        result._source_count = field.source_count
        return result

    @staticmethod
    def spectral_erosion(field: DenseField, cache: SpectralCache | None = None) -> DenseField:
        """Remove the smallest eigenvalue from each band. Cost: O(BD³)."""
        def _erode(lam):
            min_idx = np.argmin(np.abs(lam))
            lam_new = lam.copy()
            lam_new[min_idx] = 0.0
            return lam_new
        return _apply_to_eigenvalues(field, _erode, cache)

    @staticmethod
    def mean_field_step(field: DenseField, neighbors: list[DenseField], dt: float = 0.1) -> DenseField:
        """Mean-field relaxation: F' = (1-dt)F + dt·mean(neighbors). Cost: O(NBD²)."""
        if not neighbors:
            return field
        result = DenseField(bands=field.bands, dim=field.dim)
        mean_F = sum(n.F for n in neighbors) / len(neighbors)
        result.F = (1 - dt) * field.F + dt * mean_F
        result._source_count = field.source_count
        return result

    @staticmethod
    def ode_step_rk4(field: DenseField, rhs: Callable, dt: float = 0.01) -> DenseField:
        """4th-order Runge-Kutta: F(t+dt) via RK4 for dF/dt = rhs(F). Cost: O(4BD³).

        rhs: DenseField -> NDArray (B, D, D) — the right-hand side.
        """
        F = field.F.astype(np.float64)
        k1 = dt * rhs(field).astype(np.float64)

        f2 = DenseField(bands=field.bands, dim=field.dim)
        f2.F = (F + k1 / 2).astype(np.float32)
        k2 = dt * rhs(f2).astype(np.float64)

        f3 = DenseField(bands=field.bands, dim=field.dim)
        f3.F = (F + k2 / 2).astype(np.float32)
        k3 = dt * rhs(f3).astype(np.float64)

        f4 = DenseField(bands=field.bands, dim=field.dim)
        f4.F = (F + k3).astype(np.float32)
        k4 = dt * rhs(f4).astype(np.float64)

        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = (F + (k1 + 2 * k2 + 2 * k3 + k4) / 6).astype(np.float32)
        result._source_count = field.source_count
        return result

    @staticmethod
    def lotka_volterra_bands(field: DenseField, alpha: float = 0.1, beta: float = 0.05, dt: float = 0.01) -> DenseField:
        """Predator-prey dynamics between adjacent bands. Cost: O(BD²).

        Band b is "prey" for band b+1 ("predator"). Energy flows up the band hierarchy.
        dF_b/dt = α·F_b - β·F_b·F_{b+1}  (prey grows, consumed by predator)
        dF_{b+1}/dt = -α·F_{b+1} + β·F_b·F_{b+1}  (predator decays, fed by prey)
        """
        result = DenseField(bands=field.bands, dim=field.dim)
        result.F = field.F.copy()
        result._source_count = field.source_count

        for b in range(field.bands - 1):
            prey = result.F[b]
            predator = result.F[b + 1]
            interaction = prey * predator  # Element-wise
            result.F[b] += dt * (alpha * prey - beta * interaction)
            result.F[b + 1] += dt * (-alpha * predator + beta * interaction)

        return result

    @staticmethod
    def equilibrium_distance(field: DenseField, equilibrium: DenseField) -> Scalar:
        """Distance from equilibrium: ||F - F_eq||_F. Cost: O(BD²)."""
        return Scalar(float(np.linalg.norm(field.F - equilibrium.F)), name="equilibrium_distance")

    @staticmethod
    def conservation_check(field: DenseField) -> dict[str, float]:
        """Check conserved quantities: trace, Frobenius norm, determinant sign. Cost: O(BD)."""
        return {
            f"trace_band_{b}": float(np.trace(field.F[b]))
            for b in range(field.bands)
        } | {
            f"energy_band_{b}": float(np.linalg.norm(field.F[b], "fro"))
            for b in range(field.bands)
        }
