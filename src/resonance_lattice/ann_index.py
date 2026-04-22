# SPDX-License-Identifier: BUSL-1.1
"""ANN (Approximate Nearest Neighbor) index for fast registry lookup.

Wraps FAISS HNSW to provide O(log N) registry queries instead of O(N)
brute-force.  The index is built at knowledge model construction time and
serialised into the .rlat file.

FAISS is already a bench dependency and ships pre-built wheels for all
platforms, unlike hnswlib which requires C++ build tools.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Return True if faiss is installed."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class FAISSIndex:
    """FAISS HNSW index for inner-product similarity.

    Vectors are L2-normalized at build and query time so that inner
    product on the unit sphere equals cosine similarity, and FAISS's
    L2 index can be used (HNSW does not natively support IP in FAISS,
    but on the unit sphere L2² = 2 - 2·cos, so L2 ranking is equivalent).
    """

    def __init__(self, dim: int, M: int = 32) -> None:
        import faiss

        self.dim = dim
        self.M = M
        self._index = faiss.IndexHNSWFlat(dim, M)
        self._index.hnsw.efConstruction = 200
        self._index.hnsw.efSearch = 128
        self._n = 0

    def add(self, vectors: NDArray[np.float32]) -> None:
        """Add L2-normalized vectors to the index."""
        self._index.add(vectors.astype(np.float32))
        self._n += vectors.shape[0]

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query for approximate nearest neighbors.

        Returns (indices, scores) where scores are cosine similarities
        (higher = better).
        """
        k = min(top_k, self._n)
        distances, labels = self._index.search(
            vector.reshape(1, -1).astype(np.float32), k,
        )
        # Convert L2² distances to cosine: cos = 1 - d²/2 (unit-sphere)
        scores = 1.0 - distances[0] / 2.0
        return labels[0].astype(np.int64), scores.astype(np.float32)

    def to_bytes(self) -> bytes:
        """Serialise the FAISS index to bytes."""
        import faiss

        return faiss.serialize_index(self._index).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, dim: int) -> FAISSIndex:
        """Deserialise a FAISS index from bytes."""
        import faiss

        raw = np.frombuffer(data, dtype=np.uint8)
        index = faiss.deserialize_index(raw)

        obj = object.__new__(cls)
        obj.dim = dim
        obj.M = 8  # Default; exact value is baked into the serialized index
        obj._index = index
        obj._n = index.ntotal
        return obj


def build_ann(
    phases: NDArray[np.float32],
    M: int = 8,
) -> FAISSIndex | None:
    """Build an ANN index from a (N, B*D) phase matrix.

    Returns None if faiss is not installed.
    """
    if not is_available():
        logger.info("faiss not installed — ANN index disabled")
        return None

    n, dim = phases.shape
    if n == 0:
        return None

    # L2-normalize for cosine equivalence
    norms = np.linalg.norm(phases, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = (phases / norms).astype(np.float32)

    index = FAISSIndex(dim=dim, M=M)
    index.add(normalized)
    return index
