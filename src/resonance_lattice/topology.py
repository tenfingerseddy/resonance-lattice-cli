# SPDX-License-Identifier: BUSL-1.1
"""Topological Knowledge Invariants: persistent homology of fields.

Inspired by topological data analysis — the SHAPE of data reveals structure
that statistics miss. Holes, loops, voids.

Computes persistent homology of the field's eigenvalue filtration:
    β₀ = connected components → knowledge CLUSTERS
    β₁ = 1-cycles → CIRCULAR REASONING (tautological loops)
    β₂ = 2-voids → KNOWLEDGE GAPS (conspicuous absences)

    Persistence = death - birth
    Long-lived features = deep truths. Short-lived = noise.

Implementation uses a pure-numpy Vietoris-Rips approach on the
top eigenvectors, no external TDA library required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .field.dense import DenseField


@dataclass
class TopologicalFeature:
    """A single topological feature in the persistence diagram."""
    dimension: int  # 0=cluster, 1=loop
    birth: float  # Eigenvalue threshold where it appears
    death: float  # Eigenvalue threshold where it merges/dies
    persistence: float  # death - birth (larger = more stable)


@dataclass
class FieldTopology:
    """Complete topological analysis of a field band."""
    features: list[TopologicalFeature]
    knowledge_clusters: int  # Persistent H_0 components
    circular_patterns: int  # Persistent H_1 cycles
    total_persistence: float  # Sum of all persistence values
    robustness_score: float  # Fraction of energy in persistent features (0-1)
    eigenvalue_spectrum: NDArray[np.float32]  # Sorted eigenvalues
    band: int


class TopologicalAnalyzer:
    """Topological analysis of knowledge field structure.

    Uses eigenvalue filtration + distance-based persistence to identify:
    - Knowledge clusters (connected components that persist)
    - Circular reasoning (loops in the eigenvector similarity graph)
    - Knowledge gaps (voids — regions conspicuously absent)
    - Robust vs fragile patterns (persistent vs ephemeral features)
    """

    @staticmethod
    def analyze(
        field: DenseField,
        band: int = 0,
        max_eigenvectors: int = 100,
        persistence_threshold: float = 0.01,
    ) -> FieldTopology:
        """Compute topological invariants of the field.

        Steps:
        1. Eigendecompose the field band
        2. Build distance matrix between top eigenvectors
        3. Compute H_0 persistence (cluster merging) via single-linkage
        4. Estimate H_1 (loops) from cycle detection in the distance graph

        Args:
            field: The dense field.
            band: Which band to analyse.
            max_eigenvectors: Number of top eigenvectors to include.
            persistence_threshold: Minimum persistence to count as a real feature.

        Returns:
            FieldTopology with features, cluster count, and robustness.
        """
        F_b = field.F[band]
        F_sym = (F_b + F_b.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        # Sort descending by absolute eigenvalue
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top-K
        K = min(max_eigenvectors, len(eigenvalues))
        top_eigs = eigenvalues[:K]
        top_vecs = eigenvectors[:, :K]

        # Build distance matrix: D_ij = 1 - |v_i · v_j|
        similarity = np.abs(top_vecs.T @ top_vecs)  # (K, K)
        np.fill_diagonal(similarity, 1.0)
        distance = 1.0 - similarity

        # H_0 persistence via single-linkage clustering
        h0_features = TopologicalAnalyzer._compute_h0_persistence(
            distance, top_eigs, persistence_threshold,
        )

        # H_1 estimation via triangle detection
        h1_features = TopologicalAnalyzer._estimate_h1(
            distance, top_eigs, persistence_threshold,
        )

        all_features = h0_features + h1_features

        # Robustness: fraction of total eigenvalue energy in persistent components
        persistent_indices = set()
        for f in h0_features:
            if f.persistence > persistence_threshold:
                # Find eigenvectors active at this feature's birth threshold
                active = np.abs(top_eigs) >= f.birth
                persistent_indices.update(np.where(active)[0])

        total_energy = float(np.sum(np.abs(top_eigs))) + 1e-12
        persistent_energy = float(np.sum(np.abs(top_eigs[list(persistent_indices)]))) if persistent_indices else 0.0
        robustness = persistent_energy / total_energy

        n_clusters = sum(1 for f in h0_features if f.persistence > persistence_threshold)
        n_circular = sum(1 for f in h1_features if f.persistence > persistence_threshold)
        total_persistence = sum(f.persistence for f in all_features)

        return FieldTopology(
            features=all_features,
            knowledge_clusters=n_clusters,
            circular_patterns=n_circular,
            total_persistence=total_persistence,
            robustness_score=float(robustness),
            eigenvalue_spectrum=top_eigs.astype(np.float32),
            band=band,
        )

    @staticmethod
    def _compute_h0_persistence(
        distance: NDArray,
        eigenvalues: NDArray,
        threshold: float,
    ) -> list[TopologicalFeature]:
        """Compute H_0 (connected component) persistence via single-linkage.

        Each eigenvector starts as its own component (born at its eigenvalue).
        Components merge when the distance between them falls below the
        current threshold. The merged component "dies" at the merge threshold.
        """
        K = len(eigenvalues)
        if K < 2:
            return []

        # Each component born at its eigenvalue magnitude
        birth = np.abs(eigenvalues)

        # Single-linkage: merge closest pairs
        # Use a simple union-find approach
        parent = list(range(K))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Get all pairwise distances, sorted ascending
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((distance[i, j], i, j))
        edges.sort()

        features = []
        for d, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                # Merge: the younger component (lower birth value) dies
                birth_ri = birth[ri]
                birth_rj = birth[rj]

                if birth_ri < birth_rj:
                    dying, surviving = ri, rj
                else:
                    dying, surviving = rj, ri

                death_value = float(birth[dying])  # Dies at merge time
                birth_value = float(birth[surviving])

                persistence = abs(birth_value - death_value)
                if persistence > threshold:
                    features.append(TopologicalFeature(
                        dimension=0,
                        birth=birth_value,
                        death=death_value,
                        persistence=persistence,
                    ))

                parent[dying] = surviving

        return features

    @staticmethod
    def _estimate_h1(
        distance: NDArray,
        eigenvalues: NDArray,
        threshold: float,
    ) -> list[TopologicalFeature]:
        """Estimate H_1 (cycle) features from near-triangles in the distance graph.

        A 1-cycle (loop) forms when three eigenvectors are pairwise close
        but not all merged into one component. This indicates circular
        reasoning or tautological knowledge structures.
        """
        K = len(eigenvalues)
        if K < 3:
            return []

        features = []
        # Check all triangles (i, j, k) for near-equilateral configurations
        # These indicate circular knowledge patterns
        max_check = min(K, 50)  # Limit computation for large K

        for i in range(max_check):
            for j in range(i + 1, max_check):
                for k in range(j + 1, max_check):
                    d_ij = distance[i, j]
                    d_jk = distance[j, k]
                    d_ik = distance[i, k]

                    # Triangle inequality violation ratio
                    # A cycle exists when all three distances are similar
                    # and relatively small
                    max_d = max(d_ij, d_jk, d_ik)
                    min_d = min(d_ij, d_jk, d_ik)

                    if max_d < 0.5 and min_d > 0.05:  # Close but not identical
                        equilateral_ratio = min_d / (max_d + 1e-12)
                        if equilateral_ratio > 0.5:  # Near-equilateral
                            birth = float(max(np.abs(eigenvalues[i]),
                                            np.abs(eigenvalues[j]),
                                            np.abs(eigenvalues[k])))
                            death = birth * (1 - equilateral_ratio)
                            persistence = birth - death

                            if persistence > threshold:
                                features.append(TopologicalFeature(
                                    dimension=1,
                                    birth=birth,
                                    death=death,
                                    persistence=persistence,
                                ))

        return features

    @staticmethod
    def robustness_filter(
        field: DenseField,
        band: int = 0,
        min_persistence: float = 0.1,
    ) -> DenseField:
        """Remove ephemeral features, keeping only robust knowledge.

        Eigenvalues below the persistence threshold are suppressed,
        keeping only the topologically stable knowledge structures.

        Args:
            field: The field to filter (NOT modified).
            band: Which band to filter.
            min_persistence: Minimum eigenvalue persistence to keep.

        Returns:
            New DenseField with only robust features.
        """
        filtered = DenseField(bands=field.bands, dim=field.dim)
        filtered.F = field.F.copy()
        filtered._source_count = field.source_count

        F_b = filtered.F[band]
        F_sym = (F_b + F_b.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(F_sym)

        # Keep eigenvalues that are "persistent" (above threshold relative to max)
        max_eig = np.max(np.abs(eigenvalues)) + 1e-12
        keep_mask = np.abs(eigenvalues) / max_eig > min_persistence
        eigenvalues[~keep_mask] = 0.0

        filtered.F[band] = (eigenvectors * eigenvalues) @ eigenvectors.T

        return filtered

    @staticmethod
    def find_knowledge_gaps(
        field: DenseField,
        reference_field: DenseField,
        band: int = 0,
        top_k: int = 10,
    ) -> list[tuple[float, NDArray[np.float32]]]:
        """Find directions strong in reference but weak in field.

        These are "knowledge gaps" — concepts the reference knows about
        but the field doesn't.

        Args:
            field: The field to check for gaps.
            reference_field: The reference field (what knowledge should exist).
            band: Which band to compare.
            top_k: Number of gap directions to return.

        Returns:
            List of (gap_magnitude, direction_vector) tuples.
        """
        # Eigendecompose reference
        F_ref = reference_field.F[band]
        F_ref_sym = (F_ref + F_ref.T) / 2.0
        ref_eigs, ref_vecs = np.linalg.eigh(F_ref_sym)

        # For each reference eigenvector, measure its energy in the field
        gaps = []
        for i in range(len(ref_eigs) - 1, max(len(ref_eigs) - top_k * 2, -1), -1):
            v = ref_vecs[:, i]
            ref_energy = abs(ref_eigs[i])
            field_energy = abs(float(v @ field.F[band] @ v))

            gap = ref_energy - field_energy
            if gap > 0:
                gaps.append((float(gap), v.astype(np.float32)))

        gaps.sort(key=lambda x: x[0], reverse=True)
        return gaps[:top_k]
