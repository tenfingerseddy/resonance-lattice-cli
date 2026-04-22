# SPDX-License-Identifier: BUSL-1.1
"""Field tensor implementations for the Resonance Lattice.

Five backends with different capacity/performance trade-offs:
  - DenseField:            B×D×D tensor, ~82K sources, O(B·D²) retrieval, ~40 MB (f16)
  - AsymmetricDenseField:  B×(D_key×D_value), key/value separation, O(B·Dk·Dv) retrieval
  - FactoredField:         U·Σ·V^T per band, ~100K sources, O(B·D·K) retrieval, ~20 MB (f16)
  - PQField:               M subspaces × K centroids, ~8M sources, O(B·M·K²) retrieval, ~80 MB (f16)
  - MultiVectorField:      Per-source vector sets, soft-MaxSim retrieval, no info loss
"""

from resonance_lattice.field.asymmetric_dense import (
    AsymmetricDenseField,
    AsymmetricResonanceResult,
)
from resonance_lattice.field.dense import DenseField, ResonanceResult
from resonance_lattice.field.factored import FactoredField
from resonance_lattice.field.multi_vector import MultiVectorField, MultiVectorResonanceResult
from resonance_lattice.field.pq import PQField

__all__ = [
    "AsymmetricDenseField",
    "AsymmetricResonanceResult",
    "DenseField",
    "FactoredField",
    "MultiVectorField",
    "MultiVectorResonanceResult",
    "PQField",
    "ResonanceResult",
]
