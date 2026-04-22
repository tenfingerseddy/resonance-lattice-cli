# SPDX-License-Identifier: BUSL-1.1
"""RQL Type System: typed results for all operations.

Every RQL operation returns a typed result, not a raw NDArray.
This enables the operation registry, auto-documentation, and
property-based testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Scalar:
    """A scalar result with metadata."""
    value: float
    name: str = ""
    band: int | None = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Scalar({self.value:.6f}, name='{self.name}')"


@dataclass
class Spectrum:
    """Sorted eigenvalue spectrum for one band."""
    values: NDArray[np.float32]  # Sorted descending by absolute value
    band: int

    @property
    def top(self) -> float:
        return float(self.values[0]) if len(self.values) > 0 else 0.0

    @property
    def count(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"Spectrum(band={self.band}, top={self.top:.4f}, count={self.count})"


@dataclass
class EigenDecomp:
    """Cached eigendecomposition for one band."""
    eigenvalues: NDArray[np.float32]  # Sorted descending by abs
    eigenvectors: NDArray[np.float32]  # Columns = eigenvectors, matching eigenvalue order
    band: int

    def reconstruct(self) -> NDArray[np.float32]:
        """Reconstruct the matrix: V diag(λ) Vᵀ."""
        return (self.eigenvectors * self.eigenvalues) @ self.eigenvectors.T
