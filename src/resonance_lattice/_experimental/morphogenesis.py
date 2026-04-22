# SPDX-License-Identifier: BUSL-1.1
"""Morphogenetic Fields: reaction-diffusion knowledge growth.

Inspired by embryonic development — Turing patterns where chemical gradients
create biological form from uniform tissue. Dense knowledge regions self-amplify
(local activation) while sparse regions are suppressed (long-range inhibition).

The field self-organises into stable knowledge structures WITHOUT labeled data.

Mathematical model (FitzHugh-Nagumo on the eigenspectrum):
    A(t+dt) = A(t) + dt · [D_A · L(A) + f(A, B)]    (activator)
    B(t+dt) = B(t) + dt · [D_B · L(B) + g(A, B)]    (inhibitor)

    L(X) = spectral Laplacian on eigenvalues
    f(A, B) = A - A³ - B + σ    (cubic activator)
    g(A, B) = ε · (A - γ·B)     (slow linear inhibitor)
    D_B > D_A ensures Turing instability (long-range inhibition)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..field.dense import DenseField


@dataclass
class MorphResult:
    """Result of morphogenetic field evolution."""
    evolved_field: DenseField
    steps_taken: int
    energy_trajectory: list[float]  # Total field energy at each step
    activator_energy: list[float]  # Activator energy at each step
    inhibitor_energy: list[float]  # Inhibitor energy at each step
    pattern_count: int  # Estimated number of stable patterns formed


class MorphogeneticEvolution:
    """Reaction-diffusion dynamics on knowledge fields.

    Evolves the field through a coupled activator-inhibitor system where:
    - The ACTIVATOR amplifies regions of strong knowledge (positive feedback)
    - The INHIBITOR suppresses noise and prevents runaway growth (negative feedback)
    - The balance between D_A (activator diffusion) and D_B (inhibitor diffusion)
      determines whether Turing patterns form

    D_B > D_A is the Turing instability condition: the inhibitor diffuses faster
    than the activator, creating LOCAL activation with LONG-RANGE inhibition.
    """

    @staticmethod
    def evolve(
        field: DenseField,
        steps: int = 10,
        dt: float = 0.01,
        D_A: float = 1.0,
        D_B: float = 4.0,
        epsilon: float = 0.1,
        gamma: float = 0.5,
        sigma: float = 0.1,
        band: int | None = None,
    ) -> MorphResult:
        """Evolve the field through reaction-diffusion dynamics.

        Args:
            field: The field to evolve (NOT modified — new field returned).
            steps: Number of evolution steps.
            dt: Time step size (smaller = more stable, slower).
            D_A: Activator diffusion coefficient.
            D_B: Inhibitor diffusion coefficient. Must be > D_A for Turing patterns.
            epsilon: Inhibitor timescale (smaller = slower inhibitor response).
            gamma: Inhibitor coupling strength.
            sigma: Baseline activation level.
            band: If specified, only evolve this band. None = evolve all bands.

        Returns:
            MorphResult with the evolved field and diagnostics.
        """
        bands_to_evolve = [band] if band is not None else list(range(field.bands))

        # Create evolved field (copy)
        evolved = DenseField(bands=field.bands, dim=field.dim)
        evolved.F = field.F.copy()
        evolved._source_count = field.source_count

        energy_traj = []
        act_energy = []
        inh_energy = []

        for b in bands_to_evolve:
            F_b = evolved.F[b]
            F_sym = (F_b + F_b.T) / 2.0

            # Eigendecompose
            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            # Activator = eigenvalues (knowledge strength per direction)
            A = eigenvalues.copy().astype(np.float64)
            # Inhibitor = starts proportional to activator
            B = gamma * A.copy()

            for step in range(steps):
                # Spectral Laplacian: L(x) = x² - x (amplifies large, suppresses small)
                L_A = A * A - A
                L_B = B * B - B

                # Reaction terms (FitzHugh-Nagumo)
                f_AB = A - A ** 3 - B + sigma  # Cubic activator
                g_AB = epsilon * (A - gamma * B)  # Linear inhibitor

                # Update
                A_new = A + dt * (D_A * L_A + f_AB)
                B_new = B + dt * (D_B * L_B + g_AB)

                # Clamp to prevent divergence
                A = np.clip(A_new, -2.0, 2.0)
                B = np.clip(B_new, -2.0, 2.0)

                energy_traj.append(float(np.sum(np.abs(A))))
                act_energy.append(float(np.sum(np.maximum(A, 0))))
                inh_energy.append(float(np.sum(np.maximum(B, 0))))

            # Reconstruct field from evolved eigenvalues
            evolved.F[b] = (eigenvectors * A.astype(np.float32)) @ eigenvectors.T

        # Estimate pattern count: number of eigenvalues that are
        # significantly above the mean (stable activation peaks)
        final_eigs = np.linalg.eigvalsh((evolved.F[bands_to_evolve[0]] + evolved.F[bands_to_evolve[0]].T) / 2)
        mean_eig = np.mean(np.abs(final_eigs))
        pattern_count = int(np.sum(np.abs(final_eigs) > 2 * mean_eig))

        return MorphResult(
            evolved_field=evolved,
            steps_taken=steps,
            energy_trajectory=energy_traj,
            activator_energy=act_energy,
            inhibitor_energy=inh_energy,
            pattern_count=pattern_count,
        )

    @staticmethod
    def amplify_knowledge(
        field: DenseField,
        threshold: float = 0.5,
    ) -> DenseField:
        """Simple morphogenetic amplification: boost eigenvalues above threshold,
        suppress those below.

        This is the fast-path version of evolve() — a single-step spectral
        sharpening that amplifies strong knowledge and suppresses noise.

        Args:
            field: The field to amplify (NOT modified).
            threshold: Eigenvalue threshold (relative to max). Values below
                threshold * max_eigenvalue are suppressed toward zero.

        Returns:
            New DenseField with sharpened eigenvalue spectrum.
        """
        amplified = DenseField(bands=field.bands, dim=field.dim)
        amplified.F = field.F.copy()
        amplified._source_count = field.source_count

        for b in range(field.bands):
            F_b = amplified.F[b]
            F_sym = (F_b + F_b.T) / 2.0

            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            max_eig = np.max(np.abs(eigenvalues)) + 1e-12
            cutoff = threshold * max_eig

            # Amplify above threshold, suppress below
            mask = np.abs(eigenvalues) > cutoff
            eigenvalues[~mask] *= 0.1  # Suppress noise (don't zero — preserve structure)
            eigenvalues[mask] *= 1.2  # Slightly amplify strong patterns

            amplified.F[b] = (eigenvectors * eigenvalues) @ eigenvectors.T

        return amplified
