# SPDX-License-Identifier: BUSL-1.1
"""Algebraic Access Control: role-based knowledge views as field algebra.

Uses orthogonal projection to create role-specific views of knowledge.
Each view is deterministic, auditable, and verifiable:

    restricted + hidden = original  (within floating-point tolerance)

Usage:
    policy = AccessPolicy.from_exemplars("engineering", encoder, ["code", "architecture"])
    restricted = policy.apply(field)       # only engineering-relevant knowledge
    audit = policy.audit(field)            # certificate proving what was hidden
    hidden = policy.hidden_view(field)     # the complementary (hidden) view

Properties:
    - Composable: policy_a.intersect(policy_b) = intersection of allowed subspaces
    - Auditable: deterministic projection with hash-based certificates
    - Reversible: hidden_view() shows the complement
    - GDPR-compatible: prove that certain knowledge is invisible to certain roles
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.lens import EigenspaceLens, LensBuilder


@dataclass
class AccessCertificate:
    """Auditable proof of what knowledge was visible and hidden."""
    policy_name: str
    policy_hash: str          # deterministic hash of the projection subspace
    timestamp: float
    original_energy: float
    visible_energy: float
    hidden_energy: float
    retention_fraction: float  # visible / original
    per_band_retention: NDArray[np.float32]
    verifiable: bool          # True if visible + hidden ≈ original


class AccessPolicy:
    """A named, auditable projection that restricts knowledge visibility.

    Internally wraps an EigenspaceLens. The policy defines an "allowed
    subspace" — knowledge projected into this subspace is visible,
    the complement is hidden.
    """

    def __init__(
        self,
        name: str,
        lens: EigenspaceLens,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self._lens = lens
        self.metadata = metadata or {}
        self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Deterministic hash of the projection subspace."""
        raw = self._lens._subspace.tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def policy_hash(self) -> str:
        return self._hash

    @property
    def rank(self) -> int:
        return self._lens.rank

    def apply(self, field: DenseField) -> DenseField:
        """Apply the access policy — returns only visible knowledge."""
        return self._lens.apply(field)

    def hidden_view(self, field: DenseField) -> DenseField:
        """Return the hidden (complementary) knowledge."""
        return self._lens.invert(field)

    def audit(self, field: DenseField) -> AccessCertificate:
        """Generate an audit certificate for this policy applied to a field.

        The certificate proves:
        - What fraction of knowledge is visible/hidden
        - That visible + hidden = original (verifiable)
        - The policy hash (for reproducibility)
        """
        visible = self.apply(field)
        hidden = self.hidden_view(field)

        original_energy = float(sum(
            np.linalg.norm(field.F[b], "fro") for b in range(field.bands)
        ))
        visible_energy = float(sum(
            np.linalg.norm(visible.F[b], "fro") for b in range(visible.bands)
        ))
        hidden_energy = float(sum(
            np.linalg.norm(hidden.F[b], "fro") for b in range(hidden.bands)
        ))

        per_band = np.zeros(field.bands, dtype=np.float32)
        for b in range(field.bands):
            e_orig = float(np.linalg.norm(field.F[b], "fro"))
            e_vis = float(np.linalg.norm(visible.F[b], "fro"))
            per_band[b] = e_vis / (e_orig + 1e-12)

        # Verify: visible + hidden ≈ original
        reconstructed = visible.F + hidden.F
        error = float(np.linalg.norm(field.F - reconstructed))
        verifiable = error < 1e-4 * original_energy

        return AccessCertificate(
            policy_name=self.name,
            policy_hash=self._hash,
            timestamp=time.time(),
            original_energy=original_energy,
            visible_energy=visible_energy,
            hidden_energy=hidden_energy,
            retention_fraction=visible_energy / (original_energy + 1e-12),
            per_band_retention=per_band,
            verifiable=verifiable,
        )

    def intersect(self, other: AccessPolicy) -> AccessPolicy:
        """Intersect two policies: only knowledge visible to BOTH is allowed.

        The resulting subspace is the intersection of both allowed subspaces.
        """
        # Apply both lenses sequentially — projects into shared subspace
        from resonance_lattice.lens import CompoundLens
        compound = CompoundLens(
            lenses=[self._lens, other._lens],
            name=f"{self.name}&{other.name}",
        )
        # Convert back to eigenspace lens via a dummy field
        B, k, D = self._lens._subspace.shape
        dummy = DenseField(bands=B, dim=D)
        # Fill with identity-like content to extract the compound projection
        for b in range(B):
            dummy.F[b] = np.eye(D, dtype=np.float32)
        projected = compound.apply(dummy)
        result_lens = LensBuilder.from_field(projected, name=f"{self.name}&{other.name}")
        return AccessPolicy(
            name=f"{self.name}&{other.name}",
            lens=result_lens,
            metadata={"parents": [self.name, other.name]},
        )

    # ── Factory methods ──────────────────────────────────────

    @classmethod
    def from_exemplars(
        cls,
        name: str,
        phase_vectors_list: list[NDArray[np.float32]],
        k: int | None = None,
    ) -> AccessPolicy:
        """Build an access policy from exemplar phase vectors.

        Args:
            name: Policy name (e.g. "engineering", "marketing").
            phase_vectors_list: Encoded exemplars defining the allowed domain.
            k: Subspace rank.
        """
        lens = LensBuilder.from_exemplars(name, phase_vectors_list, k=k)
        return cls(name=name, lens=lens)

    @classmethod
    def from_field(
        cls,
        name: str,
        field: DenseField,
        k: int | None = None,
    ) -> AccessPolicy:
        """Build an access policy from a field's eigenspace.

        Everything in this field's principal directions is "allowed".
        """
        lens = LensBuilder.from_field(field, name=name, k=k)
        return cls(name=name, lens=lens)

    @classmethod
    def from_text(
        cls,
        name: str,
        texts: list[str],
        encoder: Any,
        k: int | None = None,
    ) -> AccessPolicy:
        """Build an access policy from exemplar texts.

        Args:
            name: Policy name.
            texts: Texts defining the allowed semantic domain.
            encoder: Encoder with .encode(text) method.
            k: Subspace rank.
        """
        phases = [encoder.encode(t).vectors for t in texts]
        return cls.from_exemplars(name, phases, k=k)

    def __repr__(self) -> str:
        return f"AccessPolicy(name={self.name!r}, rank={self.rank}, hash={self._hash})"
