# SPDX-License-Identifier: BUSL-1.1
"""Knowledge Lenses: named, reusable semantic perspectives.

A lens is not a filter. It's a perspective that reshapes what a field reveals.
A "security lens" applied to code reveals security-relevant connections
invisible in the raw field. A "performance lens" highlights latency-critical
paths.

Three lens types:
    EigenspaceLens — project to a learned semantic subspace
    SpectralLens   — reweight eigenvalues by a function
    CompoundLens   — chain of existing compiler operators

Key properties:
    - Serialisable (.rlens files) — save alongside .rlat knowledge models
    - Composable — lens_a.compose(lens_b) = subspace intersection
    - Invertible — lens.invert() shows the complementary view
    - Integrates with compiler as LensOp operator

Building a lens:
    LensBuilder.from_exemplars("security", encoder, ["authentication", "encryption", ...])
    LensBuilder.from_field(compliance_field, k=32)
    LensBuilder.from_directions(phase_vectors)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.compiler import CompilationContext, Operator
from resonance_lattice.field.dense import DenseField

# ═══════════════════════════════════════════════════════════
# Lens base class
# ═══════════════════════════════════════════════════════════

class Lens(ABC):
    """Base class for knowledge lenses."""

    def __init__(self, name: str = "", metadata: dict[str, Any] | None = None):
        self.name = name
        self.metadata = metadata or {}

    @abstractmethod
    def apply(self, field: DenseField) -> DenseField:
        """Apply the lens to a field, returning a new (viewed) field."""
        ...

    @abstractmethod
    def invert(self, field: DenseField) -> DenseField:
        """Apply the complementary view — everything this lens hides."""
        ...

    def compose(self, other: Lens) -> CompoundLens:
        """Compose two lenses: apply self, then other."""
        return CompoundLens(
            lenses=[self, other],
            name=f"{self.name}+{other.name}",
        )

    def as_operator(self) -> LensOp:
        """Convert to a compiler Operator for use in Chain.pipe()."""
        return LensOp(self)

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise lens to a JSON-compatible dict."""
        ...

    def save(self, path: str | Path) -> None:
        """Save lens to a .rlens file."""
        path = Path(path)
        data = self.to_dict()
        # Store numpy arrays as binary alongside the JSON
        arrays = {}
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                arrays[key] = value
                data[key] = f"__array__{key}"

        # Write JSON metadata
        json_path = path.with_suffix(".rlens")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Write arrays
        if arrays:
            npz_path = path.with_suffix(".rlens.npz")
            np.savez_compressed(npz_path, **arrays)

    @staticmethod
    def load(path: str | Path) -> Lens:
        """Load a lens from a .rlens file."""
        path = Path(path).with_suffix(".rlens")
        with open(path) as f:
            data = json.load(f)

        # Load arrays if present
        npz_path = path.with_suffix(".rlens.npz")
        if npz_path.exists():
            arrays = dict(np.load(npz_path))
            for key in list(data.keys()):
                if isinstance(data[key], str) and data[key].startswith("__array__"):
                    arr_key = data[key].replace("__array__", "")
                    data[key] = arrays[arr_key]

        lens_type = data.pop("type")
        if lens_type == "eigenspace":
            return EigenspaceLens._from_dict(data)
        elif lens_type == "spectral":
            return SpectralLens._from_dict(data)
        elif lens_type == "compound":
            return CompoundLens._from_dict(data)
        else:
            raise ValueError(f"Unknown lens type: {lens_type}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ═══════════════════════════════════════════════════════════
# EigenspaceLens — project to a learned semantic subspace
# ═══════════════════════════════════════════════════════════

class EigenspaceLens(Lens):
    """A lens that projects fields into a semantic subspace.

    Built from exemplar documents/queries. The exemplars define
    the subspace (via their aggregate outer product's eigenvectors).
    Applying the lens keeps only field content in this subspace.

    Math:
        M = sum_i phi_i (x) phi_i   (aggregate outer product of exemplars)
        V_k = top-k eigenvectors of M
        apply(F)_b = V_k^T @ F_b @ V_k   (project into subspace)

    The lens is PSD-preserving.
    """

    def __init__(
        self,
        subspace: NDArray[np.float32],
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            subspace: Shape (B, k, D) — per-band subspace basis vectors.
                Each subspace[b] is a (k, D) matrix of orthonormal rows.
            name: Human-readable lens name.
            metadata: Additional metadata (exemplar texts, creation date, etc.)
        """
        super().__init__(name=name, metadata=metadata)
        self._subspace = subspace  # (B, k, D)

    @property
    def bands(self) -> int:
        return self._subspace.shape[0]

    @property
    def rank(self) -> int:
        return self._subspace.shape[1]

    @property
    def dim(self) -> int:
        return self._subspace.shape[2]

    def apply(self, field: DenseField) -> DenseField:
        """Project field into the lens subspace."""
        assert field.bands == self.bands and field.dim == self.dim

        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count

        for b in range(field.bands):
            V = self._subspace[b]  # (k, D)
            # Project: V @ F @ V^T gives (k,k), then lift back: V^T @ (k,k) @ V
            F_proj = V @ field.F[b] @ V.T  # (k, k)
            result.F[b] = (V.T @ F_proj @ V).astype(np.float32)  # (D, D)

        return result

    def invert(self, field: DenseField) -> DenseField:
        """Return the complementary view — everything this lens hides."""
        viewed = self.apply(field)
        complement = DenseField(bands=field.bands, dim=field.dim)
        complement._source_count = field.source_count
        complement.F = field.F - viewed.F
        return complement

    def to_dict(self) -> dict:
        return {
            "type": "eigenspace",
            "name": self.name,
            "metadata": self.metadata,
            "subspace": self._subspace,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> EigenspaceLens:
        return cls(
            subspace=data["subspace"],
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════
# SpectralLens — reweight eigenvalues
# ═══════════════════════════════════════════════════════════

class SpectralLens(Lens):
    """A lens that reshapes a field by reweighting eigenvalues.

    Math:
        F = V diag(lambda) V^T
        apply(F) = V diag(fn(lambda)) V^T

    Common uses:
        sharp_lens = SpectralLens(lambda x: x ** 1.5)    # sharpen dominant modes
        flat_lens  = SpectralLens(lambda x: log(1 + x))  # reveal quiet signals
        denoise    = SpectralLens(lambda x: x * (x > threshold))  # hard threshold
    """

    def __init__(
        self,
        transform: Callable[[NDArray], NDArray],
        name: str = "",
        metadata: dict[str, Any] | None = None,
        transform_name: str = "",
    ):
        super().__init__(name=name, metadata=metadata)
        self._transform = transform
        self._transform_name = transform_name or name

    def apply(self, field: DenseField) -> DenseField:
        result = DenseField(bands=field.bands, dim=field.dim)
        result._source_count = field.source_count

        for b in range(field.bands):
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(F_sym)
            new_eigs = self._transform(eigvals).astype(np.float32)
            result.F[b] = (eigvecs * new_eigs) @ eigvecs.T

        return result

    def invert(self, field: DenseField) -> DenseField:
        """Approximate inverse: field - apply(field)."""
        viewed = self.apply(field)
        complement = DenseField(bands=field.bands, dim=field.dim)
        complement._source_count = field.source_count
        complement.F = field.F - viewed.F
        return complement

    def to_dict(self) -> dict:
        return {
            "type": "spectral",
            "name": self.name,
            "metadata": self.metadata,
            "transform_name": self._transform_name,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> SpectralLens:
        # Pre-built transforms
        name = data.get("transform_name", "")
        transform = _NAMED_TRANSFORMS.get(name, lambda x: x)
        return cls(
            transform=transform,
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
            transform_name=name,
        )


# Pre-built spectral transforms (serialisable by name)
_NAMED_TRANSFORMS: dict[str, Callable] = {
    "sharpen": lambda x: np.sign(x) * np.abs(x) ** 1.5,
    "flatten": lambda x: np.log1p(np.abs(x)) * np.sign(x),
    "denoise": lambda x: x * (np.abs(x) > np.median(np.abs(x))),
    "top_half": lambda x: x * (x > np.median(x)),
    "normalize": lambda x: x / (np.max(np.abs(x)) + 1e-12),
}


# ═══════════════════════════════════════════════════════════
# CompoundLens — chain of lenses or operators
# ═══════════════════════════════════════════════════════════

class CompoundLens(Lens):
    """A lens composed of multiple sub-lenses applied in sequence."""

    def __init__(
        self,
        lenses: list[Lens],
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(name=name, metadata=metadata)
        self._lenses = lenses

    def apply(self, field: DenseField) -> DenseField:
        result = field
        for lens in self._lenses:
            result = lens.apply(result)
        return result

    def invert(self, field: DenseField) -> DenseField:
        """Approximate: field - apply(field)."""
        viewed = self.apply(field)
        complement = DenseField(bands=field.bands, dim=field.dim)
        complement._source_count = field.source_count
        complement.F = field.F - viewed.F
        return complement

    def to_dict(self) -> dict:
        return {
            "type": "compound",
            "name": self.name,
            "metadata": self.metadata,
            "lenses": [l.to_dict() for l in self._lenses],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> CompoundLens:
        lenses = []
        for ld in data.get("lenses", []):
            lens_type = ld.pop("type")
            if lens_type == "eigenspace":
                lenses.append(EigenspaceLens._from_dict(ld))
            elif lens_type == "spectral":
                lenses.append(SpectralLens._from_dict(ld))
        return cls(lenses=lenses, name=data.get("name", ""), metadata=data.get("metadata", {}))


# ═══════════════════════════════════════════════════════════
# LensOp — compiler operator wrapper
# ═══════════════════════════════════════════════════════════

class LensOp(Operator):
    """Compiler operator that applies a lens within a Chain."""

    def __init__(self, lens: Lens):
        self._lens = lens

    def apply(self, field: DenseField, context: CompilationContext) -> DenseField:
        return self._lens.apply(field)

    @property
    def name(self) -> str:
        return f"Lens({self._lens.name})"


# ═══════════════════════════════════════════════════════════
# LensBuilder — construct lenses from various sources
# ═══════════════════════════════════════════════════════════

class LensBuilder:
    """Factory methods for building lenses from different sources."""

    @staticmethod
    def from_exemplars(
        name: str,
        phase_vectors_list: list[NDArray[np.float32]],
        k: int | None = None,
    ) -> EigenspaceLens:
        """Build a lens from exemplar phase vectors.

        The exemplars define a semantic region. Their aggregate outer
        product's eigenvectors form the lens subspace.

        Args:
            name: Lens name (e.g. "security", "performance").
            phase_vectors_list: List of (B, D) phase vectors from
                encoded exemplar texts.
            k: Subspace rank. None = auto (effective rank).
        """
        if not phase_vectors_list:
            raise ValueError("Need at least one exemplar")

        B, D = phase_vectors_list[0].shape

        # Aggregate outer product per band
        M = np.zeros((B, D, D), dtype=np.float32)
        for phi in phase_vectors_list:
            for b in range(B):
                M[b] += np.outer(phi[b], phi[b])

        # Eigendecompose each band, extract top-k
        subspace_list = []
        for b in range(B):
            M_sym = (M[b] + M[b].T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(M_sym)

            if k is None:
                # Auto: effective rank (eigenvalues > 1% of total)
                total = float(np.sum(np.abs(eigvals)))
                threshold = 0.01 * total
                mask = np.abs(eigvals) > threshold
                band_k = max(int(np.sum(mask)), 1)
            else:
                band_k = min(k, D)

            # Top-k eigenvectors (sorted descending)
            idx = np.argsort(np.abs(eigvals))[::-1][:band_k]
            V_k = eigvecs[:, idx].T  # (k, D)
            subspace_list.append(V_k)

        # Pad to uniform k across bands
        max_k = max(v.shape[0] for v in subspace_list)
        subspace = np.zeros((B, max_k, D), dtype=np.float32)
        for b, V in enumerate(subspace_list):
            subspace[b, :V.shape[0], :] = V

        return EigenspaceLens(
            subspace=subspace,
            name=name,
            metadata={"n_exemplars": len(phase_vectors_list), "rank": max_k},
        )

    @staticmethod
    def from_field(
        field: DenseField,
        name: str = "",
        k: int | None = None,
    ) -> EigenspaceLens:
        """Build a lens from a field's eigenspace.

        Projects through the principal directions of the given field.
        Equivalent to project(target, field) but as a reusable lens.

        Args:
            field: The field whose eigenspace defines the lens.
            name: Lens name.
            k: Subspace rank. None = effective rank.
        """
        B = field.bands
        D = field.dim

        subspace_list = []
        for b in range(B):
            F_sym = (field.F[b] + field.F[b].T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(F_sym)

            if k is None:
                total = float(np.sum(np.abs(eigvals)))
                threshold = 0.01 * total
                mask = np.abs(eigvals) > threshold
                band_k = max(int(np.sum(mask)), 1)
            else:
                band_k = min(k, D)

            idx = np.argsort(np.abs(eigvals))[::-1][:band_k]
            V_k = eigvecs[:, idx].T
            subspace_list.append(V_k)

        max_k = max(v.shape[0] for v in subspace_list)
        subspace = np.zeros((B, max_k, D), dtype=np.float32)
        for b, V in enumerate(subspace_list):
            subspace[b, :V.shape[0], :] = V

        return EigenspaceLens(
            subspace=subspace,
            name=name,
            metadata={"source": "field", "rank": max_k},
        )

    @staticmethod
    def from_text(
        name: str,
        texts: list[str],
        encoder: Any,
        k: int | None = None,
    ) -> EigenspaceLens:
        """Build a lens from exemplar texts using an encoder.

        Convenience method: encodes texts then calls from_exemplars().

        Args:
            name: Lens name (e.g. "security").
            texts: Exemplar texts defining the semantic focus.
            encoder: Encoder with .encode(text) -> PhaseSpectrum.
            k: Subspace rank.
        """
        phases = [encoder.encode(t).vectors for t in texts]
        return LensBuilder.from_exemplars(name, phases, k=k)

    @staticmethod
    def sharpen(name: str = "sharpen") -> SpectralLens:
        """Pre-built: emphasise dominant modes, suppress noise."""
        return SpectralLens(
            _NAMED_TRANSFORMS["sharpen"],
            name=name,
            transform_name="sharpen",
        )

    @staticmethod
    def flatten(name: str = "flatten") -> SpectralLens:
        """Pre-built: equalise modes, reveal quiet signals."""
        return SpectralLens(
            _NAMED_TRANSFORMS["flatten"],
            name=name,
            transform_name="flatten",
        )

    @staticmethod
    def denoise(name: str = "denoise") -> SpectralLens:
        """Pre-built: hard threshold at median eigenvalue."""
        return SpectralLens(
            _NAMED_TRANSFORMS["denoise"],
            name=name,
            transform_name="denoise",
        )
