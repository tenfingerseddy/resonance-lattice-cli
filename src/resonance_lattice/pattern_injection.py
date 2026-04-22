# SPDX-License-Identifier: BUSL-1.1
"""Pattern injection: reversible retrieval boosting and suppression.

Add synthetic patterns to the field that amplify desired retrieval
and suppress unwanted retrieval — all reversible, continuous, and instant.

Operations:
    sculpt(field, vectors, beta) - Add a synthetic pattern (boost or suppress)
    unsculpt(state, field) - Reverse all sculpting, restoring the original field exactly
    orthogonalise(target, avoid) - Remove overlap between sculpting vectors for precision

All sculpting is reversible: subtract what you added. Exact in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField


@dataclass
class SculptingVector:
    """A single sculpting pattern with its weight."""
    label: str  # Human-readable name (e.g. "boost:Python", "suppress:Java")
    phase_vectors: NDArray[np.float32]  # (B, D)
    beta: float  # Positive = boost, negative = suppress


@dataclass
class SculptingState:
    """Tracks all active sculpting patterns for reversibility."""
    patterns: list[SculptingVector] = dc_field(default_factory=list)
    original_energy: NDArray[np.float32] | None = None  # (B,) energy before sculpting

    @property
    def pattern_count(self) -> int:
        return len(self.patterns)


class InterferenceSculptor:
    """Active interference pattern design for knowledge fields.

    Instead of treating cross-source interference as noise, DESIGN it:
    - Constructive amplification: boost retrieval of target patterns
    - Destructive cancellation: suppress unwanted retrieval patterns
    - Orthogonal sculpting: precision mode that avoids collateral damage

    All operations are reversible: unsculpt() restores the original field.
    """

    @staticmethod
    def sculpt(
        field: DenseField,
        phase_vectors: NDArray[np.float32],
        beta: float,
        label: str = "",
        state: SculptingState | None = None,
    ) -> SculptingState:
        """Apply a sculpting pattern to the field.

        F'_b = F_b + β · (ψ_b ⊗ ψ_b) for each band b.

        β > 0: Boost — amplify retrieval of patterns similar to ψ
        β < 0: Suppress — cancel retrieval of patterns similar to ψ

        Args:
            field: The field to sculpt (MODIFIED in place).
            phase_vectors: Shape (B, D) — the sculpting direction.
            beta: Sculpting weight. Positive = boost, negative = suppress.
            label: Human-readable name for this pattern.
            state: Existing sculpting state to append to. Creates new if None.

        Returns:
            Updated SculptingState tracking this and prior sculpting operations.
        """
        if state is None:
            state = SculptingState(
                original_energy=field.energy().copy(),
            )

        # Validate shape
        if phase_vectors.shape != (field.bands, field.dim):
            raise ValueError(
                f"phase_vectors shape {phase_vectors.shape} != "
                f"expected ({field.bands}, {field.dim})"
            )

        # Apply rank-1 update per band
        for b in range(field.bands):
            psi = phase_vectors[b]
            field.F[b] += beta * np.outer(psi, psi)

        # Track for reversal
        sv = SculptingVector(
            label=label or f"sculpt_{state.pattern_count}",
            phase_vectors=phase_vectors.copy(),
            beta=beta,
        )
        state.patterns.append(sv)

        return state

    @staticmethod
    def sculpt_multi(
        field: DenseField,
        targets: list[tuple[NDArray[np.float32], float, str]],
        state: SculptingState | None = None,
    ) -> SculptingState:
        """Apply multiple sculpting patterns at once.

        F'_b = F_b + Σ_j β_j · (ψ_j_b ⊗ ψ_j_b)

        Args:
            field: The field to sculpt (MODIFIED in place).
            targets: List of (phase_vectors, beta, label) tuples.
            state: Existing sculpting state. Creates new if None.

        Returns:
            Updated SculptingState.
        """
        if state is None:
            state = SculptingState(original_energy=field.energy().copy())

        for phase_vectors, beta, label in targets:
            InterferenceSculptor.sculpt(field, phase_vectors, beta, label, state)

        return state

    @staticmethod
    def unsculpt(
        field: DenseField,
        state: SculptingState,
    ) -> None:
        """Reverse ALL sculpting operations, restoring the original field.

        Subtracts each sculpting pattern in reverse order.
        Exact in fp32 — the field returns to its pre-sculpting state.

        Args:
            field: The sculpted field (MODIFIED in place — restored to original).
            state: The sculpting state to reverse.
        """
        # Reverse in LIFO order for maximum numerical precision
        for sv in reversed(state.patterns):
            for b in range(field.bands):
                psi = sv.phase_vectors[b]
                field.F[b] -= sv.beta * np.outer(psi, psi)

        state.patterns.clear()

    @staticmethod
    def unsculpt_one(
        field: DenseField,
        state: SculptingState,
        label: str,
    ) -> bool:
        """Remove a single sculpting pattern by label.

        Args:
            field: The sculpted field (MODIFIED in place).
            state: The sculpting state.
            label: Label of the pattern to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        for i, sv in enumerate(state.patterns):
            if sv.label == label:
                # Reverse this specific pattern
                for b in range(field.bands):
                    psi = sv.phase_vectors[b]
                    field.F[b] -= sv.beta * np.outer(psi, psi)
                state.patterns.pop(i)
                return True
        return False

    @staticmethod
    def orthogonalise(
        target: NDArray[np.float32],
        avoid: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Remove the component of `target` that overlaps with `avoid`.

        ψ_orthogonal = ψ_target - proj(ψ_target, ψ_avoid)

        Use this to suppress Java WITHOUT affecting JavaScript:
            psi_java_only = orthogonalise(psi_java, psi_javascript)

        Args:
            target: Shape (B, D) — the vector to orthogonalise.
            avoid: Shape (B, D) — the direction to avoid.

        Returns:
            Shape (B, D) — target with the avoid-direction removed, L2-normalised.
        """
        B = target.shape[0]
        result = np.zeros_like(target)

        for b in range(B):
            t = target[b]
            a = avoid[b]

            # Project target onto avoid direction
            a_norm_sq = np.dot(a, a)
            if a_norm_sq > 1e-12:
                projection = (np.dot(t, a) / a_norm_sq) * a
                result[b] = t - projection
            else:
                result[b] = t.copy()

            # Re-normalise
            norm = np.linalg.norm(result[b])
            if norm > 1e-8:
                result[b] /= norm

        return result

    @staticmethod
    def clamp_eigenvalues(field: DenseField, min_eigenvalue: float = 0.0) -> int:
        """Clamp negative eigenvalues to prevent field instability from over-suppression.

        After aggressive sculpting, the field may have negative eigenvalues
        (corresponding to "holes" in the interference pattern). This clamps
        them to the specified minimum.

        Args:
            field: The field to clamp (MODIFIED in place).
            min_eigenvalue: Minimum allowed eigenvalue. 0.0 makes the field PSD.

        Returns:
            Number of eigenvalues that were clamped.
        """
        clamped_count = 0

        for b in range(field.bands):
            F_b = field.F[b]
            F_sym = (F_b + F_b.T) / 2.0

            eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

            negative_mask = eigenvalues < min_eigenvalue
            n_negative = int(np.sum(negative_mask))

            if n_negative > 0:
                eigenvalues[negative_mask] = min_eigenvalue
                field.F[b] = (eigenvectors * eigenvalues) @ eigenvectors.T
                clamped_count += n_negative

        return clamped_count
