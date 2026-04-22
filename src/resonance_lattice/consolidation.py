# SPDX-License-Identifier: BUSL-1.1
"""Consolidation — promotes sources between memory tiers.

working → episodic: on session expiry or working overflow.
episodic → semantic: when the same content recurs ≥ N times.

Uses existing Lattice.superpose/remove and algebra.novelty — no new math.
"""

from __future__ import annotations

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.lattice import Lattice


def consolidate_working_to_episodic(
    working: Lattice,
    episodic: Lattice,
    session_id: str | None = None,
) -> list[str]:
    """Move sources from working to episodic tier.

    If ``session_id`` is given, only moves sources from that session.
    Otherwise moves all working sources.

    Each source is re-superposed into episodic with its existing salience,
    then removed from working. Order-independent (merge commutativity).

    Returns:
        List of source_ids that were promoted.
    """
    promoted: list[str] = []

    candidates = list(working.registry._source_index.items())
    for source_id, entry in candidates:
        if session_id is not None:
            content = working.store.retrieve(source_id)
            if content is None:
                continue
            meta = content.metadata or {}
            if meta.get("session_id") != session_id:
                continue

        # Read content before removing from working
        content = working.store.retrieve(source_id)
        vectors = entry.phase_vectors.copy()
        salience = entry.salience

        # Superpose into episodic
        episodic.superpose(
            phase_spectrum=vectors,
            salience=salience,
            source_id=source_id,
            content=content,
        )

        # Remove from working
        working.remove(source_id)
        promoted.append(source_id)

    return promoted


def promote_to_semantic(
    episodic: Lattice,
    semantic: Lattice,
    recurrence_threshold: int = 3,
    novelty_threshold: float = 0.3,
    cold_start_seed: int = 0,
) -> list[str]:
    """Promote episodic sources to semantic when they recur frequently.

    A source is promoted if:
      1. Its access_count >= recurrence_threshold, OR
      2. Its novelty score against the semantic field is low (< novelty_threshold),
         meaning episodic has converged on this knowledge.

    Condition 2 detects when multiple episodic sources reinforce the same
    concept — the semantic tier should absorb it.

    Cold-start seed: when ``cold_start_seed > 0`` AND semantic is empty, the
    top-K episodic sources by salience are promoted regardless of
    access_count / novelty. This unblocks the chicken-and-egg case where a
    fresh bulk ingest leaves access_count=0 on every source and semantic has
    nothing for novelty to compare against — without a seed path, neither
    criterion ever fires and the semantic tier stays empty forever.
    Opt-in (default 0), one-shot (only runs while semantic is empty).

    Returns:
        List of source_ids promoted to semantic.
    """
    promoted: list[str] = []
    candidates = list(episodic.registry._source_index.items())

    # Cold-start seed: promote top-K by salience when semantic is empty.
    # Only fires on the first call; subsequent calls see a non-empty semantic
    # and fall through to the normal criteria.
    seed_ids: set[str] = set()
    if cold_start_seed > 0 and semantic.source_count == 0 and candidates:
        ranked = sorted(candidates, key=lambda kv: kv[1].salience, reverse=True)
        seed_ids = {sid for sid, _ in ranked[:cold_start_seed]}

    for source_id, entry in candidates:
        should_promote = source_id in seed_ids

        # Criterion 1: frequently accessed
        if not should_promote and entry.access_count >= recurrence_threshold:
            should_promote = True

        # Criterion 2: low novelty against semantic (already-known knowledge)
        if not should_promote and semantic.source_count > 0:
            score = FieldAlgebra.novelty(
                semantic.field,
                entry.phase_vectors,
                entry.salience,
            )
            if score.score < novelty_threshold:
                should_promote = True

        if not should_promote:
            continue

        content = episodic.store.retrieve(source_id)
        vectors = entry.phase_vectors.copy()
        salience = entry.salience

        semantic.superpose(
            phase_spectrum=vectors,
            salience=salience,
            source_id=source_id,
            content=content,
        )

        episodic.remove(source_id)
        promoted.append(source_id)

    return promoted
