# SPDX-License-Identifier: BUSL-1.1
"""Resonance Networks: graphs of interconnected knowledge fields.

Individual .rlat knowledge models become nodes in a network. Resonance channels
connect fields, enabling cross-domain reasoning.

A query enters one field, resonates, and PROPAGATES through channels
to related fields — like neural impulses traversing synapses.

Network G = (V, E) where V = {F₁, F₂, ...}, E = {W_ij}
W_ij ∈ ℝ^(D×D) = resonance channel (learned linear map)

Channel learning via Hebbian rule: "fields that resonate together wire together"
W_ij += η · r_i · r_jᵀ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..field.dense import DenseField, ResonanceResult


@dataclass
class NetworkNode:
    """A field node in the resonance network."""
    name: str
    field: DenseField


@dataclass
class NetworkChannel:
    """A resonance channel connecting two fields."""
    source: str  # Node name
    target: str  # Node name
    W: NDArray[np.float32]  # (D, D) channel matrix
    strength: float  # Scalar strength (Frobenius norm of W)


@dataclass
class PropagationResult:
    """Result of resonance propagation through a network."""
    entry_resonance: ResonanceResult  # Resonance at entry node
    propagated: list[tuple[str, NDArray[np.float32]]]  # [(node_name, resonance_vector)]
    fused: NDArray[np.float32]  # (D,) weighted fusion of all node resonances
    hops: int
    nodes_visited: list[str]


@dataclass
class CascadeHop:
    """Result from a single hop in a routed cascade."""
    node: str                                  # which cartridge this hop hit
    resonance_vectors: NDArray[np.float32]     # (B, D) resonance at this hop
    energy: float                              # total resonance energy


@dataclass
class RoutedCascadeResult:
    """Result of cascading resonance along a user-specified route."""
    hops: list[CascadeHop]                     # per-hop results with provenance
    fused: NDArray[np.float32]                 # (D,) weighted fusion of all hops
    route: list[str]                           # actual nodes visited (may be shorter than requested)
    total_energy: float                        # sum of all hop energies


class ResonanceNetwork:
    """A graph of interconnected knowledge fields.

    Each node is a DenseField (a knowledge model).
    Each edge is a resonance channel — a learned linear map that
    translates resonance from one field's coordinate space to another's.

    Supports:
    - propagate(query, entry, hops) — multi-field cascading resonance
    - learn_channel(node_a, node_b, co_queries) — Hebbian channel learning
    - add_node / remove_node / connect / disconnect
    """

    def __init__(self):
        self._nodes: dict[str, NetworkNode] = {}
        self._channels: dict[tuple[str, str], NetworkChannel] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    def add_node(self, name: str, field: DenseField) -> None:
        """Add a field node to the network."""
        self._nodes[name] = NetworkNode(name=name, field=field)

    def remove_node(self, name: str) -> bool:
        """Remove a node and all its channels."""
        if name not in self._nodes:
            return False

        # Remove all channels involving this node
        to_remove = [
            key for key in self._channels
            if name in key
        ]
        for key in to_remove:
            del self._channels[key]

        del self._nodes[name]
        return True

    def connect(
        self,
        source: str,
        target: str,
        channel: NDArray[np.float32] | None = None,
    ) -> None:
        """Connect two nodes with a resonance channel.

        Args:
            source: Source node name.
            target: Target node name.
            channel: (D, D) channel matrix. If None, initializes as scaled identity.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not in network")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not in network")

        D = self._nodes[source].field.dim
        if channel is None:
            # Default: scaled identity (passes resonance through with mild attenuation)
            channel = 0.1 * np.eye(D, dtype=np.float32)

        self._channels[(source, target)] = NetworkChannel(
            source=source,
            target=target,
            W=channel.astype(np.float32),
            strength=float(np.linalg.norm(channel, "fro")),
        )

    def connect_bidirectional(
        self,
        node_a: str,
        node_b: str,
        channel: NDArray[np.float32] | None = None,
    ) -> None:
        """Connect two nodes bidirectionally."""
        self.connect(node_a, node_b, channel)
        self.connect(node_b, node_a, channel.T if channel is not None else None)

    def disconnect(self, source: str, target: str) -> bool:
        """Remove a channel between two nodes."""
        key = (source, target)
        if key in self._channels:
            del self._channels[key]
            return True
        return False

    def propagate(
        self,
        query_phase: NDArray[np.float32],
        entry_node: str,
        hops: int = 1,
        band: int = 0,
        decay: float = 0.5,
    ) -> PropagationResult:
        """Propagate resonance through the network.

        Starting at entry_node, resonate the query, then propagate
        the resonance through channels to neighboring nodes.

        Args:
            query_phase: Shape (B, D) — the query.
            entry_node: Name of the starting node.
            hops: Number of propagation hops.
            band: Which band to propagate on.
            decay: Attenuation per hop (0-1). 0.5 = halve each hop.

        Returns:
            PropagationResult with per-node resonances and fused output.
        """
        if entry_node not in self._nodes:
            raise ValueError(f"Entry node '{entry_node}' not in network")

        D = self._nodes[entry_node].field.dim

        # Entry resonance
        entry_field = self._nodes[entry_node].field
        entry_resonance = entry_field.resonate(query_phase)
        current_vector = entry_resonance.resonance_vectors[band].copy()

        propagated = [(entry_node, current_vector.copy())]
        visited = [entry_node]

        # BFS propagation
        frontier = [(entry_node, current_vector, 0)]

        while frontier:
            node_name, vector, depth = frontier.pop(0)

            if depth >= hops:
                continue

            # Find all outgoing channels from this node
            for (src, tgt), channel in self._channels.items():
                if src == node_name and tgt not in visited:
                    # Propagate through channel
                    transmitted = channel.W @ vector
                    transmitted *= decay

                    # Resonate in target field
                    target_field = self._nodes[tgt].field
                    target_resonance = target_field.F[band] @ transmitted

                    propagated.append((tgt, target_resonance.astype(np.float32)))
                    visited.append(tgt)
                    frontier.append((tgt, target_resonance, depth + 1))

        # Fuse all propagated resonances
        fused = np.zeros(D, dtype=np.float32)
        for i, (name, vec) in enumerate(propagated):
            weight = decay ** i
            fused += weight * vec

        return PropagationResult(
            entry_resonance=entry_resonance,
            propagated=propagated,
            fused=fused,
            hops=hops,
            nodes_visited=visited,
        )

    def learn_channel(
        self,
        source: str,
        target: str,
        co_queries: list[NDArray[np.float32]],
        band: int = 0,
        lr: float = 0.01,
    ) -> float:
        """Learn a channel via Hebbian rule from co-activated queries.

        "Fields that resonate together wire together."

        W_ij += η · Σ_queries (r_i · r_jᵀ)

        Args:
            source: Source node name.
            target: Target node name.
            co_queries: List of (B, D) queries that should activate BOTH fields.
            band: Which band to learn on.
            lr: Learning rate.

        Returns:
            Channel strength after learning.
        """
        key = (source, target)
        if key not in self._channels:
            self.connect(source, target)

        channel = self._channels[key]
        source_field = self._nodes[source].field
        target_field = self._nodes[target].field

        for q in co_queries:
            r_source = source_field.F[band] @ q[band]
            r_target = target_field.F[band] @ q[band]

            # Hebbian update: W += η · r_source · r_targetᵀ
            channel.W += lr * np.outer(r_source, r_target).astype(np.float32)

        channel.strength = float(np.linalg.norm(channel.W, "fro"))
        return channel.strength

    def cascade_route(
        self,
        query_phase: NDArray[np.float32],
        route: list[str],
        alpha: float = 0.5,
        threshold: float = 0.0,
        band_weights: NDArray[np.float32] | None = None,
    ) -> RoutedCascadeResult:
        """Multi-hop search across a user-specified route of knowledge models.

        "Start in code, follow links into docs, then into incidents."

        Unlike propagate() which does BFS, this follows an explicit path.
        Resonance from each hop feeds through the channel into the next field.
        All bands are used (not just one).

        Args:
            query_phase: Shape (B, D) query.
            route: Ordered list of node names to traverse.
                   Auto-connects adjacent nodes if no channel exists.
            alpha: Decay per hop (0-1). Controls how much earlier hops
                   contribute to the final result.
            threshold: Minimum resonance energy to continue propagating.
                       0.0 = always continue.
            band_weights: Optional per-band weights for resonance fusion.

        Returns:
            RoutedCascadeResult with per-hop resonance and provenance.
        """
        if len(route) < 1:
            raise ValueError("Route must have at least one node")

        for name in route:
            if name not in self._nodes:
                raise ValueError(f"Node '{name}' not in network")

        # Auto-connect adjacent pairs if needed
        for i in range(len(route) - 1):
            src, tgt = route[i], route[i + 1]
            if (src, tgt) not in self._channels:
                self.connect(src, tgt)

        B = self._nodes[route[0]].field.bands
        D = self._nodes[route[0]].field.dim

        # Hop 0: resonate at entry
        entry_field = self._nodes[route[0]].field
        entry_resonance = entry_field.resonate(query_phase, band_weights=band_weights)
        current_vectors = entry_resonance.resonance_vectors.copy()  # (B, D)

        hop_results = [CascadeHop(
            node=route[0],
            resonance_vectors=current_vectors.copy(),
            energy=float(np.sum(entry_resonance.band_energies)),
        )]

        # Subsequent hops
        for i in range(1, len(route)):
            src, tgt = route[i - 1], route[i]
            channel = self._channels[(src, tgt)]
            target_field = self._nodes[tgt].field

            # Propagate through channel (all bands)
            propagated = np.zeros((B, D), dtype=np.float32)
            for b in range(B):
                transmitted = channel.W @ current_vectors[b]
                propagated[b] = target_field.F[b] @ transmitted

            # Resonate in target
            hop_energy = float(np.sum([
                np.linalg.norm(propagated[b]) for b in range(B)
            ]))

            # Gate: stop if energy too low
            if threshold > 0 and hop_energy < threshold:
                break

            hop_results.append(CascadeHop(
                node=tgt,
                resonance_vectors=propagated.copy(),
                energy=hop_energy,
            ))
            current_vectors = propagated

        # Fuse all hops with decay weighting
        fused = np.zeros(D, dtype=np.float32)
        for i, hop in enumerate(hop_results):
            weight = alpha ** i
            # Fuse across bands
            for b in range(B):
                bw = band_weights[b] if band_weights is not None else 1.0 / B
                fused += weight * bw * hop.resonance_vectors[b]

        return RoutedCascadeResult(
            hops=hop_results,
            fused=fused,
            route=route[:len(hop_results)],
            total_energy=sum(h.energy for h in hop_results),
        )

    def get_neighbors(self, node_name: str) -> list[str]:
        """Get all nodes connected to the given node (outgoing channels)."""
        return [
            tgt for (src, tgt) in self._channels
            if src == node_name
        ]

    def info(self) -> dict:
        """Network summary."""
        return {
            "nodes": list(self._nodes.keys()),
            "channels": [(src, tgt) for src, tgt in self._channels],
            "node_count": self.node_count,
            "channel_count": self.channel_count,
        }
