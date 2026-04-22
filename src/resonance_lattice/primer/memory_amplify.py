# SPDX-License-Identifier: BUSL-1.1
"""Amplify code-primer ranking using recent memory signals.

The central insight: files and topics the user has been discussing in
recent memory sessions are the ones they need primer coverage for. A
retrieval-only ranking treats `backend.py` (removed two days ago) and
`algebra.py` (load-bearing, discussed every session) as equally
prominent if they score similarly on the dense field. Memory knows
which one the user actually cares about right now.

This module reads the episodic (and optionally working) tier of a
LayeredMemory, extracts time-decayed signals — which source_files
have been discussed, what keyword topics have been recurring — and
re-scores a list of MaterialisedResult objects. The amplification
is additive and capped so memory cannot fully override retrieval.

Failure modes all degrade gracefully: no memory → no boost, empty
tier → no boost, encoder mismatch → no boost with stderr warning.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resonance_lattice.lattice import MaterialisedResult
    from resonance_lattice.layered_memory import LayeredMemory


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on",
    "at", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "we", "i", "you",
    "what", "which", "who", "where", "when", "why", "how",
    "not", "no", "do", "does", "did", "have", "has", "had",
    "can", "could", "should", "would", "will", "may", "might",
    "with", "from", "into", "onto", "about", "over", "under",
    "all", "any", "some", "each", "every", "more", "most", "less",
    "just", "also", "very", "so", "too", "only", "really",
}
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_TS_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2})):?(\d{2})?")


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse a timestamp string into a UTC datetime, lenient on format."""
    if not raw:
        return None
    raw = raw.strip().strip('"').strip("'")
    # Fast path — ISO 8601 with optional Z
    try:
        cleaned = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        pass
    # Fallback — extract date/time tokens
    m = _TS_RE.search(raw)
    if not m:
        return None
    y, mo, d, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    try:
        return datetime(
            int(y), int(mo), int(d),
            int(hh or 0), int(mm or 0), tzinfo=UTC,
        )
    except ValueError:
        return None


def _decay(dt: datetime, now: datetime, half_life_days: float) -> float:
    """Exponential decay with configurable half-life. Never returns 0 or <0."""
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return math.pow(0.5, age_days / max(1e-6, half_life_days))


def _tokenize(text: str) -> list[str]:
    return [
        t.lower() for t in _WORD_RE.findall(text or "")
        if len(t) > 3 and t.lower() not in _STOPWORDS
    ]


def _collect_memory_signals(
    memory: LayeredMemory,
    *,
    half_life_days: float,
    tier_names: tuple[str, ...] = ("episodic", "working"),
    max_entries: int = 400,
) -> tuple[dict[str, float], Counter, int]:
    """Extract (file_weights, keyword_weights, entries_seen) from memory.

    Each memory entry contributes a weight = exp_decay(age, half_life)
    to (a) the source_file it mentions (if any) and (b) each salient
    keyword in its summary / full_text.
    """
    from resonance_lattice.memory_primer import _extract_timestamp

    now = datetime.now(UTC)
    file_weights: dict[str, float] = {}
    keyword_weights: Counter = Counter()
    entries_seen = 0

    for tier_name in tier_names:
        tier = memory.tiers.get(tier_name) if memory.tiers else None
        if tier is None or tier.source_count == 0:
            continue
        # Iterate entries via the store. Memory tiers are small (hundreds
        # of entries, not hundreds of thousands) so a full sweep is fine.
        entries = []
        try:
            ids = list(tier.store.all_ids())
        except AttributeError:
            continue
        for sid in ids:
            try:
                content = tier.store.retrieve(sid)
            except Exception:
                continue
            if content is not None:
                entries.append(content)

        for content in entries:
            if content is None:
                continue
            raw_ts = _extract_timestamp(content)
            dt = _parse_timestamp(raw_ts)
            if dt is None:
                continue
            entries_seen += 1
            w = _decay(dt, now, half_life_days)
            meta = content.metadata or {}
            sf = meta.get("source_file") or meta.get("path") or ""
            if sf:
                key = Path(sf).name.lower()
                file_weights[key] = file_weights.get(key, 0.0) + w
            # Keyword extraction from summary + first 400 chars of full_text
            text = (content.summary or "") + "\n" + (content.full_text or "")[:400]
            for tok in _tokenize(text):
                keyword_weights[tok] += w
            if entries_seen >= max_entries:
                break
        if entries_seen >= max_entries:
            break

    return file_weights, keyword_weights, entries_seen


def amplify_by_memory(
    results: list[MaterialisedResult],
    memory: LayeredMemory | None,
    *,
    code_encoder_fp: str = "",
    half_life_days: float = 14.0,
    max_boost: float = 0.15,
    top_keywords: int = 40,
) -> tuple[list[MaterialisedResult], dict[str, Any]]:
    """Re-score results by how strongly they resonate with recent memory.

    Returns (reranked_results, diagnostics). Never raises — any failure
    returns the input list unchanged with a diagnostic explaining why.
    """
    diag: dict[str, Any] = {"applied": False, "reason": "", "entries_seen": 0}
    if not results or memory is None:
        diag["reason"] = "no-memory" if memory is None else "no-results"
        return results, diag

    # Encoder fingerprint gate — only applies when the caller provided one.
    # We compare against ANY tier encoder (they share a root encoder).
    if code_encoder_fp:
        try:
            mem_enc = memory.encoder
            if mem_enc is not None:
                from resonance_lattice.cli import _encoder_fingerprint
                mem_fp = _encoder_fingerprint(mem_enc)
                if mem_fp and mem_fp != code_encoder_fp:
                    diag["reason"] = f"encoder-mismatch ({mem_fp} vs {code_encoder_fp})"
                    print(
                        f"memory-amplify: skipped — encoder fingerprint "
                        f"mismatch ({mem_fp} vs {code_encoder_fp})",
                        file=sys.stderr,
                    )
                    return results, diag
        except Exception:
            pass  # fail open

    try:
        file_weights, keyword_weights, entries_seen = _collect_memory_signals(
            memory, half_life_days=half_life_days,
        )
    except Exception as exc:
        diag["reason"] = f"collect-failed: {type(exc).__name__}: {exc}"
        print(f"memory-amplify: skipped — {diag['reason']}", file=sys.stderr)
        return results, diag

    diag["entries_seen"] = entries_seen
    if entries_seen == 0 or (not file_weights and not keyword_weights):
        diag["reason"] = "no-timestamped-entries"
        return results, diag

    # Normalize each signal to [0, 1] so amplification is bounded.
    max_file_w = max(file_weights.values()) if file_weights else 1.0
    top_kw = dict(keyword_weights.most_common(top_keywords))
    max_kw_w = max(top_kw.values()) if top_kw else 1.0

    boosted = 0
    rescored = []
    for r in results:
        base = r.score
        if base <= 0:
            rescored.append(r)
            continue
        meta = (r.content.metadata or {}) if r.content else {}
        sf = meta.get("source_file", "") or meta.get("path", "")
        fname = Path(sf).name.lower() if sf else ""

        file_component = file_weights.get(fname, 0.0) / max_file_w if fname else 0.0
        text = (r.content.full_text or r.content.summary or "") if r.content else ""
        r_toks = set(_tokenize(text[:1200]))
        if r_toks and top_kw:
            hit = sum(top_kw[t] for t in r_toks if t in top_kw)
            kw_component = hit / (max_kw_w * max(6, len(r_toks) / 4))
            kw_component = min(1.0, kw_component)
        else:
            kw_component = 0.0

        # Blend: file match weighs heavier than keyword overlap
        amp_raw = 0.6 * file_component + 0.4 * kw_component
        amp = max_boost * amp_raw
        # Cap at 20% of base so retrieval still dominates
        amp = min(amp, 0.2 * base)
        if amp > 0:
            boosted += 1
            r.score = base + amp
        rescored.append(r)

    rescored.sort(key=lambda r: -r.score)
    diag["applied"] = True
    diag["boosted"] = boosted
    diag["files_with_memory"] = len(file_weights)
    diag["top_keywords"] = len(top_kw)
    return rescored, diag
