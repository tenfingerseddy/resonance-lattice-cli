# SPDX-License-Identifier: BUSL-1.1
"""Retention policies for layered memory tiers.

Each tier has a RetentionPolicy (TTL, capacity, decay parameters).
``enforce()`` walks the registry and evicts sources that exceed TTL
or capacity, using the existing ``Lattice.remove()`` code path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from resonance_lattice.lattice import Lattice


@dataclass
class RetentionPolicy:
    """Retention rules for a single memory tier."""

    ttl_seconds: float | None = None  # None = no TTL
    capacity: int | None = None  # None = unbounded
    decay_lambda: float = 0.0  # exponential decay rate per second
    hit_boost: float = 1.2  # salience multiplier on retrieval hit

    # Tier-specific defaults
    WORKING = None  # forward-declared below
    EPISODIC = None
    SEMANTIC = None


# Tier presets
RetentionPolicy.WORKING = RetentionPolicy(
    ttl_seconds=86400,  # 24h
    capacity=200,
    decay_lambda=1.0 / 86400,  # linear-ish over 24h
    hit_boost=1.2,
)
RetentionPolicy.EPISODIC = RetentionPolicy(
    ttl_seconds=90 * 86400,  # 90 days
    capacity=5000,
    decay_lambda=1.0 / (30 * 86400),  # exp decay with τ ≈ 30 days
    hit_boost=1.2,
)
RetentionPolicy.SEMANTIC = RetentionPolicy(
    ttl_seconds=None,  # no TTL
    capacity=None,  # unbounded
    decay_lambda=1.0 / (365 * 86400),  # very slow decay
    hit_boost=1.1,
)


def apply_time_decay(lattice: Lattice, policy: RetentionPolicy, now: float | None = None) -> int:
    """Apply exponential salience decay to all sources in a lattice.

    Returns the number of sources whose salience was updated.
    """
    if policy.decay_lambda <= 0:
        return 0

    if now is None:
        now = time.time()

    updated = 0
    for source_id, entry in list(lattice.registry._source_index.items()):
        if entry.last_accessed <= 0:
            continue
        dt = now - entry.last_accessed
        if dt <= 0:
            continue
        new_salience = entry.salience * math.exp(-policy.decay_lambda * dt)
        new_salience = max(new_salience, 1e-6)
        if abs(new_salience - entry.salience) > 1e-6:
            lattice.reweight(source_id, new_salience)
            updated += 1

    return updated


def enforce(lattice: Lattice, policy: RetentionPolicy, now: float | None = None) -> list[str]:
    """Enforce TTL and capacity limits on a lattice.

    Evicts expired sources first, then lowest-salience sources if over capacity.
    Uses existing ``Lattice.remove()`` — no new field math.

    Returns:
        List of removed source_ids.
    """
    if now is None:
        now = time.time()

    removed: list[str] = []

    # Phase 1: TTL eviction
    if policy.ttl_seconds is not None:
        cutoff = now - policy.ttl_seconds
        for source_id, entry in list(lattice.registry._source_index.items()):
            ts = entry.last_accessed if entry.last_accessed > 0 else 0.0
            if ts > 0 and ts < cutoff:
                lattice.remove(source_id)
                removed.append(source_id)

    # Phase 2: capacity eviction (evict lowest-salience first)
    if policy.capacity is not None:
        remaining = list(lattice.registry._source_index.items())
        if len(remaining) > policy.capacity:
            remaining.sort(key=lambda x: x[1].salience)
            excess = len(remaining) - policy.capacity
            for source_id, _entry in remaining[:excess]:
                lattice.remove(source_id)
                removed.append(source_id)

    return removed
