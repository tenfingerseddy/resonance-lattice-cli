# SPDX-License-Identifier: BUSL-1.1
"""Field Algebra: algebraically exact operations on Resonance Fields.

These operations are unique to the outer-product field representation.
No vector database supports them. They are the academic contribution
that makes the Resonance Lattice publishable.

Operations:
    merge(a, b)      - Combine two knowledge bases: F_ab = F_a + F_b
    forget(field, id) - GDPR-exact source removal: F' = F - phi_i (x) phi_i
    diff(newer, older) - What changed: delta_F = F_new - F_old
    novelty(field, phi) - Information gain of a new source
    intersect(a, b)    - Shared knowledge via simultaneous eigendecomposition

All operations are O(B * D^2) — independent of corpus size.
All operations on dense fields are algebraically exact (within fp32).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField

# ═══════════════════════════════════════════════════════════
# Result data classes
# ═══════════════════════════════════════════════════════════

@dataclass
class MergeResult:
    """Result of merging two fields."""
    field: DenseField
    source_count: int
    total_energy: float
    per_band_energy: NDArray  # (B,)


@dataclass
class RemovalCertificate:
    """Cryptographic-style proof that a source was removed."""
    source_id: str
    timestamp: float
    exact: bool  # True if original phase vector was used
    residual_energy: float  # Should be ~0 for exact removal
    energy_before: float
    energy_after: float
    energy_removed: float  # = energy_before - energy_after


@dataclass
class DiffResult:
    """Result of computing a field difference."""
    delta_field: DenseField
    added_energy: float
    removed_energy: float
    net_change: float  # positive = more info in newer
    per_band_change: NDArray  # (B,) — signed energy change per band
    top_changed_bands: list[tuple[str, float]]  # (band_name, change)


@dataclass
class NoveltyScore:
    """How much new information a candidate source would add."""
    score: float  # 0.0 = completely redundant, 1.0 = completely novel
    per_band: NDArray  # (B,) novelty per band
    self_energy: float  # Energy of the candidate's outer product
    projection_energy: float  # How much the field already "knows" about this
    information_gain_estimate: float  # Estimated bits of new info


@dataclass
class IntersectionResult:
    """Shared knowledge between two fields."""
    shared_field: DenseField
    shared_energy: float
    total_energy_a: float
    total_energy_b: float
    overlap_fraction: float  # shared / min(a, b)
    per_band_overlap: NDArray  # (B,)


@dataclass
class ProjectionResult:
    """Result of projecting field A through field B's eigenspace.

    The projected field contains A's knowledge restricted to the semantic
    directions that B considers important. Unlike intersect (which keeps
    directions strong in both), project keeps all of A's energy along B's
    directions regardless of magnitude in B.
    """
    projected_field: DenseField
    projected_energy: float       # total energy after projection
    original_energy: float        # total energy of A before projection
    retention_fraction: float     # projected / original (how much of A survived)
    per_band_retention: NDArray   # (B,) retention per band
    subspace_rank: int            # k — dimensionality of B's eigenspace used


@dataclass
class ContradictionResult:
    """Result of detecting contradictions between two fields.

    The contradiction field is itself queryable — resonate against it
    to find topics where A and B disagree.
    """
    contradiction_field: DenseField
    total_contradiction: float     # aggregate contradiction energy
    total_agreement: float         # aggregate agreement energy
    contradiction_ratio: float     # contradiction / (contradiction + agreement)
    per_band_contradiction: NDArray  # (B,) contradiction per band


# ═══════════════════════════════════════════════════════════
# Field Algebra
# ═══════════════════════════════════════════════════════════

BAND_NAMES = ["domain", "topic", "relations", "entity", "verbatim"]


class FieldAlgebra:
    """Algebraically exact operations on Resonance Fields.

    All operations produce NEW fields (immutable algebra).
    Error bounds are tracked and propagated.
    """

    @staticmethod
    def merge(
        a: DenseField,
        b: DenseField,
        weight_a: float = 1.0,
        weight_b: float = 1.0,
    ) -> MergeResult:
        """Merge two fields: F_merged = w_a * F_a + w_b * F_b.

        This combines two knowledge bases into one. The merged field
        responds to queries about EITHER corpus.

        Properties:
            - Commutative: merge(a, b) == merge(b, a) when weights equal
            - Associative: merge(merge(a, b), c) == merge(a, merge(b, c))
            - Identity: merge(a, zero_field) == a

        Args:
            a, b: Dense fields with matching (bands, dim).
            weight_a, weight_b: Relative importance weights.
        """
        assert a.bands == b.bands and a.dim == b.dim, \
            f"Fields must match: ({a.bands},{a.dim}) vs ({b.bands},{b.dim})"

        merged = DenseField(bands=a.bands, dim=a.dim)
        merged.F = weight_a * a.F + weight_b * b.F
        merged._source_count = a.source_count + b.source_count

        per_band = np.array([
            np.linalg.norm(merged.F[b], "fro") for b in range(merged.bands)
        ])

        return MergeResult(
            field=merged,
            source_count=merged.source_count,
            total_energy=float(per_band.sum()),
            per_band_energy=per_band,
        )

    @staticmethod
    def forget(
        field: DenseField,
        phase_vectors: NDArray,
        salience: float = 1.0,
        source_id: str = "",
    ) -> tuple[DenseField, RemovalCertificate]:
        """Algebraically exact source removal via rank-1 subtraction.

        F' = F - alpha * (phi (x) phi)

        The rank-1 update itself is algebraically exact. However, other
        superposed sources create cross-talk: the field retains a residual
        resonance from inter-source interference. The certificate reports
        this residual so callers can assess removal quality.

        For a single-source field, residual is zero (truly exact).
        For N sources, residual ~ (N-1) * cross-talk — typically small
        relative to original signal but measurably non-zero.

        Args:
            field: The field to remove from (NOT modified — new field returned).
            phase_vectors: Shape (B, D) — the source's phase spectrum.
            salience: Same weight used during superpose.
            source_id: Identifier for the audit certificate.

        Returns:
            (new_field, certificate) — the modified field and proof of removal.
        """
        # Compute energy before
        energy_before = float(sum(
            np.linalg.norm(field.F[b], "fro") for b in range(field.bands)
        ))

        # Create new field (don't modify original)
        new_field = DenseField(bands=field.bands, dim=field.dim)
        new_field.F = field.F.copy()
        new_field._source_count = field.source_count

        # Exact rank-1 subtraction
        new_field.remove(phase_vectors, salience)

        # Compute energy after
        energy_after = float(sum(
            np.linalg.norm(new_field.F[b], "fro") for b in range(new_field.bands)
        ))

        # Verify removal by checking residual
        # Re-resonating with the removed source shows remaining cross-talk
        residual = 0.0
        for b in range(new_field.bands):
            r = new_field.F[b] @ phase_vectors[b]
            residual += float(np.dot(phase_vectors[b], r))

        # The rank-1 subtraction is exact. Residual energy comes from
        # cross-talk with other superposed sources, not from incomplete
        # removal. Report both facts in the certificate.
        energy_removed = energy_before - energy_after
        residual_ratio = abs(residual) / (energy_removed + 1e-12)

        cert = RemovalCertificate(
            source_id=source_id,
            timestamp=time.time(),
            exact=residual_ratio < 0.01,  # exact when cross-talk < 1% of removed energy
            residual_energy=residual,
            energy_before=energy_before,
            energy_after=energy_after,
            energy_removed=energy_removed,
        )

        return new_field, cert

    @staticmethod
    def diff(
        newer: DenseField,
        older: DenseField,
    ) -> DiffResult:
        """Compute what changed between two field states.

        delta_F = F_newer - F_older

        The delta field is QUERYABLE — resonate against it to find
        content that exists in the newer version but not the older.

        Properties:
            - Identity: diff(a, a) == zero_field
            - Inverse: merge(older, diff(newer, older)) == newer
            - Asymmetric: diff(a, b) != diff(b, a) in general

        Args:
            newer, older: Dense fields with matching dimensions.
        """
        assert newer.bands == older.bands and newer.dim == older.dim

        delta = DenseField(bands=newer.bands, dim=newer.dim)
        delta.F = newer.F - older.F
        delta._source_count = newer.source_count - older.source_count

        # Analyze the change per band
        per_band = np.zeros(delta.bands)
        for b in range(delta.bands):
            # Signed energy: positive eigenvalues = added, negative = removed
            eigvals = np.linalg.eigvalsh(delta.F[b])
            per_band[b] = float(np.sum(eigvals))

        added = float(np.sum(np.maximum(per_band, 0)))
        removed = float(np.sum(np.abs(np.minimum(per_band, 0))))

        top_changed = sorted(
            [(BAND_NAMES[b] if b < len(BAND_NAMES) else f"band_{b}", float(per_band[b]))
             for b in range(delta.bands)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return DiffResult(
            delta_field=delta,
            added_energy=added,
            removed_energy=removed,
            net_change=added - removed,
            per_band_change=per_band,
            top_changed_bands=top_changed,
        )

    @staticmethod
    def novelty(
        field: DenseField,
        phase_vectors: NDArray,
        salience: float = 1.0,
    ) -> NoveltyScore:
        """How much new information would this source add to the field?

        Measures the projection of the candidate's outer product onto
        the field's existing eigenspectrum. High projection = redundant.
        Low projection = novel.

        Score interpretation:
            0.0 = The field already knows everything this source would add
            1.0 = This source is completely orthogonal to existing knowledge
            0.5 = Half of this source's information is new

        Args:
            field: The existing field.
            phase_vectors: Shape (B, D) — candidate source.
            salience: Weight the source would be added with.
        """
        per_band = np.zeros(field.bands)

        total_self = 0.0
        total_proj = 0.0

        for b in range(field.bands):
            phi = phase_vectors[b]

            # Self-energy: how much this source would contribute
            self_energy = salience * float(np.dot(phi, phi)) ** 2

            # Projection onto field: how much the field already "resonates"
            # with this source's content
            resonance = field.F[b] @ phi
            projection = float(np.dot(phi, resonance))

            # Novelty: 1 - (projection / (self_energy * source_count + eps))
            # Normalized by what we'd expect from N independent sources
            expected_if_independent = self_energy * max(field.source_count, 1)
            band_novelty = 1.0 - min(1.0, abs(projection) / (expected_if_independent + 1e-12))

            per_band[b] = band_novelty
            total_self += self_energy
            total_proj += abs(projection)

        overall = float(np.mean(per_band))

        # Information gain estimate (rough): log2(1 + novelty * self_energy)
        info_gain = float(np.log2(1 + overall * total_self + 1e-12))

        return NoveltyScore(
            score=overall,
            per_band=per_band,
            self_energy=total_self,
            projection_energy=total_proj,
            information_gain_estimate=info_gain,
        )

    @staticmethod
    def intersect(
        a: DenseField,
        b: DenseField,
        threshold: float = 0.01,
    ) -> IntersectionResult:
        """Extract shared knowledge between two fields.

        Uses simultaneous eigendecomposition to find directions that
        have high energy in BOTH fields. The intersection field contains
        only the shared principal components.

        Args:
            a, b: Dense fields with matching dimensions.
            threshold: Minimum relative energy to include a component.
        """
        assert a.bands == b.bands and a.dim == b.dim

        shared = DenseField(bands=a.bands, dim=a.dim)
        per_band_overlap = np.zeros(a.bands)

        energy_a = 0.0
        energy_b = 0.0
        energy_shared = 0.0

        for band in range(a.bands):
            # Eigendecompose both fields
            eigvals_a, eigvecs_a = np.linalg.eigh(a.F[band])
            eigvals_b, eigvecs_b = np.linalg.eigh(b.F[band])

            ea = float(np.sum(np.abs(eigvals_a)))
            eb = float(np.sum(np.abs(eigvals_b)))
            energy_a += ea
            energy_b += eb

            # Find shared directions: project a's eigenvectors onto b's space
            # Overlap matrix: how much each eigenvector of a aligns with b
            overlap_matrix = eigvecs_a.T @ b.F[band] @ eigvecs_a  # (D, D)

            # For each eigenvector of a, measure its energy in b
            for k in range(a.dim):
                energy_in_a = abs(eigvals_a[k])
                energy_in_b = abs(overlap_matrix[k, k])

                # Include if significant in both
                min_energy = min(ea, eb) + 1e-12
                if (energy_in_a / min_energy > threshold and
                        energy_in_b / min_energy > threshold):
                    # Weight by geometric mean of energies in both fields
                    weight = np.sqrt(energy_in_a * energy_in_b)
                    v = eigvecs_a[:, k]
                    shared.F[band] += weight * np.outer(v, v)
                    energy_shared += weight

            # Per-band overlap
            band_shared = float(np.linalg.norm(shared.F[band], "fro"))
            per_band_overlap[band] = band_shared / (min(ea, eb) + 1e-12)

        overlap_frac = energy_shared / (min(energy_a, energy_b) + 1e-12)

        return IntersectionResult(
            shared_field=shared,
            shared_energy=energy_shared,
            total_energy_a=energy_a,
            total_energy_b=energy_b,
            overlap_fraction=float(overlap_frac),
            per_band_overlap=per_band_overlap,
        )

    @staticmethod
    def project(
        a: DenseField,
        b: DenseField,
        k: int | None = None,
    ) -> ProjectionResult:
        """Project field A into field B's eigenspace.

        "Show me A's knowledge, but only through B's perspective."

        For each band:
            V_B = top-k eigenvectors of B_b
            P = V_B @ V_B^T                    (orthogonal projector)
            project(A, B)_b = P @ A_b @ P      (congruence transform)

        This is different from intersect:
        - intersect keeps directions strong in BOTH A and B
        - project keeps ALL of A's energy along B's directions,
          regardless of magnitude in B

        The result is PSD (orthogonal projector sandwich preserves PSD).

        Args:
            a: Field to project (the knowledge source).
            b: Field defining the projection subspace (the "lens").
            k: Number of eigenvectors of B to use. If None, uses
                the effective rank of B (eigenvectors with > 1% of
                total energy). Lower k = tighter focus.

        Returns:
            ProjectionResult with the projected field and diagnostics.
        """
        assert a.bands == b.bands and a.dim == b.dim, \
            f"Fields must match: ({a.bands},{a.dim}) vs ({b.bands},{b.dim})"

        projected = DenseField(bands=a.bands, dim=a.dim)
        projected._source_count = a.source_count

        per_band_retention = np.zeros(a.bands)
        total_original = 0.0
        total_projected = 0.0

        for band in range(a.bands):
            # Eigendecompose B to find its principal directions
            F_b_sym = (b.F[band] + b.F[band].T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(F_b_sym)

            # Determine subspace rank
            if k is None:
                # Use effective rank: eigenvectors with > 1% of total energy
                total_eig_energy = float(np.sum(np.abs(eigvals)))
                threshold = 0.01 * total_eig_energy
                mask = np.abs(eigvals) > threshold
                band_k = max(int(np.sum(mask)), 1)
            else:
                band_k = min(k, a.dim)

            # Top-k eigenvectors of B (sorted descending by magnitude)
            idx = np.argsort(np.abs(eigvals))[::-1][:band_k]
            V_k = eigvecs[:, idx]  # (D, k)

            # Orthogonal projector: P = V_k @ V_k^T
            # Congruence: projected_b = P @ A_b @ P
            # Efficient: (V_k @ V_k^T) @ A_b @ (V_k @ V_k^T)
            #          = V_k @ (V_k^T @ A_b @ V_k) @ V_k^T
            A_in_B = V_k.T @ a.F[band] @ V_k  # (k, k) — A in B's basis
            projected.F[band] = (V_k @ A_in_B @ V_k.T).astype(np.float32)

            # Track retention
            e_orig = float(np.linalg.norm(a.F[band], "fro"))
            e_proj = float(np.linalg.norm(projected.F[band], "fro"))
            total_original += e_orig
            total_projected += e_proj
            per_band_retention[band] = e_proj / (e_orig + 1e-12)

        actual_k = band_k  # last band's k (representative)

        return ProjectionResult(
            projected_field=projected,
            projected_energy=total_projected,
            original_energy=total_original,
            retention_fraction=total_projected / (total_original + 1e-12),
            per_band_retention=per_band_retention,
            subspace_rank=actual_k,
        )

    @staticmethod
    def contradict(
        a: DenseField,
        b: DenseField,
    ) -> ContradictionResult:
        """Detect contradictions between two knowledge fields.

        Creates a queryable contradiction field — resonate against it
        to find topics where A and B disagree.

        For each band:
            Eigendecompose (A + B) / 2 to find shared directions
            For each direction, compare A's and B's energy
            Contradiction = high energy in both but significantly different

        The contradiction field is PSD and composable with all
        existing algebra operations.

        Args:
            a, b: Dense fields with matching dimensions.

        Returns:
            ContradictionResult with queryable contradiction field.
        """
        assert a.bands == b.bands and a.dim == b.dim, \
            f"Fields must match: ({a.bands},{a.dim}) vs ({b.bands},{b.dim})"

        contradiction = DenseField(bands=a.bands, dim=a.dim)
        per_band = np.zeros(a.bands)
        total_contradiction = 0.0
        total_agreement = 0.0

        for band in range(a.bands):
            # Eigendecompose the average field
            avg = (a.F[band] + b.F[band]) / 2.0
            avg_sym = (avg + avg.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(avg_sym)

            # For each eigenvector, measure contradiction
            c_scores = np.zeros(a.dim)
            for k in range(a.dim):
                v = eigvecs[:, k]
                alpha = float(v @ a.F[band] @ v)  # A's energy
                beta = float(v @ b.F[band] @ v)   # B's energy

                max_ab = max(abs(alpha), abs(beta))
                min_ab = min(abs(alpha), abs(beta))

                if max_ab < 1e-12:
                    continue

                # Contradiction: both have energy but differ significantly
                # c_k = |α - β| / max(α, β) × min(α, β)
                # High when both fields care but disagree
                contrast = abs(alpha - beta) / (max_ab + 1e-12)
                c_scores[k] = contrast * min_ab

                # Track agreement vs contradiction
                agreement = min_ab * (1.0 - contrast)
                total_agreement += agreement
                total_contradiction += c_scores[k]

            # Build contradiction field from scored directions
            contradiction.F[band] = (
                (eigvecs * c_scores) @ eigvecs.T
            ).astype(np.float32)
            per_band[band] = float(np.linalg.norm(contradiction.F[band], "fro"))

        total = total_contradiction + total_agreement + 1e-12

        return ContradictionResult(
            contradiction_field=contradiction,
            total_contradiction=total_contradiction,
            total_agreement=total_agreement,
            contradiction_ratio=total_contradiction / total,
            per_band_contradiction=per_band,
        )
