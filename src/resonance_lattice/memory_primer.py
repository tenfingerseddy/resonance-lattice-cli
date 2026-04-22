# SPDX-License-Identifier: BUSL-1.1
"""Memory Primer — generate a conversation-memory primer for CLAUDE.md.

Complements the code primer (``rlat summary``) with a second document
that surfaces what the project *knows* from conversations: settled axioms,
decision arcs (with reversal detection), active threads, and recent sessions.

The two primers together provide complete project orientation:
  code primer  -> what the codebase IS
  memory primer -> how we got here and what's hot right now

Cross-primer novelty filtering ensures zero redundancy between them.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.field.dense import DenseField
from resonance_lattice.lattice import Lattice, MaterialisedResult
from resonance_lattice.layered_memory import LayeredMemory
from resonance_lattice.materialiser import _estimate_tokens, _truncate_to_tokens
from resonance_lattice.registry import RegistryEntry
from resonance_lattice.store import SourceContent

# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ConfidenceSignal:
    """Confidence annotation for a memory entry or section."""
    reinforcement: float         # normalized access_count (0-1)
    recency: float               # exp decay, 30-day half-life
    contradiction_penalty: float  # 0.3 if superseded, else 0.0
    composite: float             # weighted combination minus penalty
    label: str                   # "high" | "medium" | "low" | "reversed"


@dataclass
class TrailEntry:
    """A single step in a decision trail."""
    timestamp: str    # ISO 8601
    text: str         # passage summary
    session_id: str
    tier: str
    reversed: bool    # True if contradicted by a later entry in the same trail
    source_id: str = ""


@dataclass
class DecisionTrail:
    """Reconstructed decision arc across conversations."""
    topic: str                     # auto-extracted from highest-scoring entry
    entries: list[TrailEntry] = field(default_factory=list)
    confidence: ConfidenceSignal | None = None
    has_reversal: bool = False     # True if any entry was contradicted


@dataclass
class MemoryPrimerResult:
    """Output of the memory primer generator."""
    markdown: str
    total_tokens: int
    section_tokens: dict[str, int]
    passages_used: int
    tiers_queried: list[str]
    novelty_filtered: int          # passages suppressed by cross-primer filter
    contradictions_found: int      # decision reversals detected


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

MEMORY_BOOTSTRAP_QUERIES = [
    "What are the key decisions and design choices made for this project?",
    "What problems were encountered and how were they resolved?",
    "What is currently being worked on and what are the immediate priorities?",
    "What conventions, preferences, or rules has the team established?",
    "What was discussed or accomplished in recent conversations?",
]

# Per-section tier weight biases
SECTION_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "axioms": {"semantic": 0.7, "episodic": 0.2, "working": 0.1},
    "decisions": {"semantic": 0.4, "episodic": 0.5, "working": 0.1},
    "active": {"semantic": 0.1, "episodic": 0.6, "working": 0.3},
    "recent": {"semantic": 0.05, "episodic": 0.15, "working": 0.8},
}

# Budget allocation as fraction of total
SECTION_BUDGET_FRACTIONS: dict[str, float] = {
    "axioms": 0.20,
    "decisions": 0.30,
    "active": 0.25,
    "recent": 0.25,
}

# Queries mapped to sections for retrieval bias
QUERY_SECTION_MAP: list[tuple[str, str]] = [
    (MEMORY_BOOTSTRAP_QUERIES[0], "axioms"),     # decisions & choices
    (MEMORY_BOOTSTRAP_QUERIES[1], "decisions"),   # problems & resolutions
    (MEMORY_BOOTSTRAP_QUERIES[2], "active"),      # current work
    (MEMORY_BOOTSTRAP_QUERIES[3], "axioms"),      # conventions & rules
    (MEMORY_BOOTSTRAP_QUERIES[4], "recent"),      # recent conversations
]

# Contradiction detection thresholds
CONTRADICTION_TOPIC_SIM = 0.7    # cosine above this = same topic
CONTRADICTION_TEXT_DIV = 0.5     # keyword divergence above this = different conclusion
DECISION_CLUSTER_THRESHOLD = 0.6  # cosine for topic clustering

# ═══════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════


def _compute_confidence(
    entry: RegistryEntry,
    tier_name: str,
    now: float,
    reversed: bool = False,
) -> ConfidenceSignal:
    """Derive a confidence signal from a registry entry's lifecycle stats."""
    reinforcement = min(1.0, entry.access_count / 5)

    age_seconds = now - entry.last_accessed if entry.last_accessed > 0 else 1e9
    age_days = age_seconds / 86400
    recency = math.exp(-age_days / 30)  # 30-day half-life

    penalty = 0.3 if reversed else 0.0
    composite = 0.6 * reinforcement + 0.4 * recency - penalty

    if reversed:
        label = "reversed"
    elif composite > 0.7:
        label = "high"
    elif composite > 0.4:
        label = "medium"
    else:
        label = "low"

    return ConfidenceSignal(
        reinforcement=reinforcement,
        recency=recency,
        contradiction_penalty=penalty,
        composite=composite,
        label=label,
    )


# ═══════════════════════════════════════════════════════════════════
# Text extraction helpers
# ═══════════════════════════════════════════════════════════════════


def _extract_summary(content: SourceContent | None, max_chars: int = 300) -> str:
    """Extract a clean summary from a SourceContent object."""
    if content is None:
        return ""
    text = content.summary if (content.summary and content.summary != content.full_text) else content.full_text
    if not text:
        return ""
    # Strip YAML frontmatter (memory entries may have metadata headers)
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            text = text[end + 3:].strip()
    # Strip conversation speaker prefixes injected by ``chunk_claude_transcript``.
    # The chunker writes ``**Human:**`` with the colon *inside* the asterisks,
    # so the regex must tolerate both placements. Without this, claim
    # extraction treats "**Human:**" as part of the sentence and claims come
    # out as raw dialog chunks.
    text = re.sub(
        r"\*\*\s*(Human|Assistant|User|System)\s*:?\s*\*\*\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?mi)^\s*(Human|Assistant|User)\s*:\s*", "", text)
    # Strip markdown headers
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,4}\s+\S", stripped):
            continue
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    if len(text) > max_chars:
        text = _truncate_to_tokens(text, max_chars // 4)
    return text


# ── Claim extraction ────────────────────────────────────────────────
#
# Passages from the episodic tier are raw conversation chunks (``**Human:**``
# / ``**Assistant:**`` prefixes stripped above). Dumping them verbatim into a
# primer produces decontextualized dialog fragments rather than the compact
# "axiom / decision / active thread" summaries the primer format implies.
# ``_extract_claim`` scores each sentence and returns the strongest
# claim-shaped line, falling back to the first reasonable sentence if no
# sentence scores positively. See DOGFOOD_FINDINGS.md #314 for the failure
# mode this fixes.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(\[`])|\n{2,}")

_DECISION_MARKERS = re.compile(
    r"\b(chose|chosen|decided|pivot(?:ed)?|reject(?:ed)?|flip(?:ped)?|land(?:ed)?|"
    r"closed?|rule[sd]?|broke|works|doesn'?t work|baseline|winner|result|conclusion|"
    r"decision|confirmed|ship(?:ped|s|ping)?|merge[ds]?|deprecated?|removed?)\b",
    re.IGNORECASE,
)
_FACT_MARKERS = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|v\d+(?:\.\d+){0,2}|\d+\.\d+%|\d+\.\d+|R@\d+|MRR|nDCG|"
    r"0-for-\d+|BGE-large|E5-large|bge-|BEIR|LongMemEval|FiQA|SciFact|NFCorpus|"
    r"[a-f0-9]{7,12}|commit\s+`?[a-f0-9]{6,}|issue\s+#\d+|#\d+)\b",
    re.IGNORECASE,
)
_PROJECT_TERMS = re.compile(
    r"\b(rlat|lattice|knowledge model|primer|field|encoder|LocalStore|ExternalStore|BundledStore|RemoteStore|LosslessStore|store_mode|"
    r"memory[- ]primer|three-layer|retrieval|recall|ablate|rerank(?:er)?|chunker|"
    r"MCP|hook|telemetry|dogfood)\b",
    re.IGNORECASE,
)
_NOISE_STARTERS = re.compile(
    r"^\s*(Now|Let me|Here(?:'s| is)|I(?:'ll| will| am| should|'ve|'m)\b|"
    r"OK\b|Okay\b|Good\b|Great\b|Perfect\b|Excellent\b|"
    r"Also[,:]|So\b|Well\b|Wait\b|Hmm\b|Actually\b|Let's\b|Lets\b)",
    re.IGNORECASE,
)
# Request / directive patterns that look claim-like to the token scorer but
# are actually Kane asking for something. An axiom is what we *decided*,
# not what we asked about. Penalise these heavily.
_REQUEST_PATTERNS = re.compile(
    r"^\s*(I want|I need|Read(?:\s)|Have we|Can (?:you|we)|Could (?:you|we)|"
    r"Should (?:we|I)|Please|Do (?:you|we)|Is (?:it|there)|Will (?:you|it)|"
    r"Tell me|Show me|Give me|Check|Run|Build|Update|Fix|Push|Commit|"
    r"Lets?\s+kick|What(?:'s| is)|Where(?:'s| is)|Why (?:don|can|do))\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence candidates, resilient to bullets and fences."""
    # Drop code fences and bullet markers — they confuse sentence splitting
    # and rarely contain the claim we want.
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _score_sentence(sentence: str, min_tokens: int = 4, max_tokens: int = 40) -> int:
    """Return a claim-ness score. Higher = more claim-shaped."""
    tokens = sentence.split()
    n = len(tokens)
    if n < min_tokens or n > max_tokens:
        return -100
    if _NOISE_STARTERS.match(sentence):
        return -50
    if _REQUEST_PATTERNS.match(sentence):
        return -30
    if sentence.endswith("?"):
        return -20
    score = 0
    if _DECISION_MARKERS.search(sentence):
        score += 3
    if _FACT_MARKERS.search(sentence):
        score += 2
    if _PROJECT_TERMS.search(sentence):
        score += 1
    # Penalise sentences dominated by paths / URLs — telltale signs of
    # "here's the file I want you to read" rather than "here's what we
    # decided". Windows drive letters and absolute paths are the common case.
    if re.search(r"[A-Z]:\\|/[A-Za-z]+/[A-Za-z]+/|https?://", sentence):
        score -= 2
    # Prefer sentences that are not predominantly code / paths
    ratio_alpha = sum(1 for c in sentence if c.isalpha()) / max(len(sentence), 1)
    if ratio_alpha < 0.4:
        score -= 2
    return score


def _extract_claim(text: str, max_tokens: int = 40, min_score: int = 1) -> str | None:
    """Pick the strongest claim-shaped sentence, or None if nothing scores.

    Returns None when no sentence reaches ``min_score`` — on this corpus the
    fallback-to-first-reasonable-sentence path produced work-notes like
    "Now I have full understanding of both issues" that look like claims
    but carry no decision content. Better to drop the passage entirely than
    to poison the primer with dialog fragments.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return None
    scored = [(_score_sentence(s, max_tokens=max_tokens), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: (-x[0], x[1]))  # highest score; break ties by earliest position
    best_score, _, best = scored[0]
    if best_score < min_score:
        return None
    return best.strip(" -–—:;.")


def _normalise_for_dedup(text: str) -> str:
    """Lowercase + keep alphanum tokens only, for Jaccard dedup across claims."""
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", text.lower()))))


def _is_near_duplicate(candidate: str, seen: list[str], threshold: float = 0.7) -> bool:
    """True if candidate's keyword set is Jaccard-similar to any seen claim."""
    cand_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    if not cand_tokens:
        return False
    for prior in seen:
        prior_tokens = set(re.findall(r"[a-z0-9]+", prior.lower()))
        if not prior_tokens:
            continue
        overlap = cand_tokens & prior_tokens
        union = cand_tokens | prior_tokens
        if len(overlap) / len(union) >= threshold:
            return True
    return False


def _extract_timestamp(content: SourceContent | None) -> str:
    """Extract timestamp from memory metadata or frontmatter."""
    if content is None:
        return ""
    meta = content.metadata or {}
    ts = meta.get("timestamp", "")
    if ts:
        return ts
    # Try parsing from frontmatter in full_text
    text = content.full_text or ""
    if text.startswith("---"):
        match = re.search(r"timestamp:\s*(.+)", text[:500])
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return ""


def _extract_session_id(content: SourceContent | None) -> str:
    """Extract session_id from memory metadata."""
    if content is None:
        return ""
    meta = content.metadata or {}
    return meta.get("session_id", "")


def _extract_keywords(text: str) -> set[str]:
    """Extract a set of lowered content words for divergence detection."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    # Filter out very common words
    stops = {"the", "and", "for", "that", "this", "with", "from", "are", "was",
             "have", "has", "had", "not", "but", "can", "will", "been", "being",
             "its", "they", "their", "what", "when", "how", "which", "would",
             "could", "should", "about", "into", "than", "then", "also", "just"}
    return {w for w in words if w not in stops}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ═══════════════════════════════════════════════════════════════════
# Contradiction detection
# ═══════════════════════════════════════════════════════════════════


def _detect_contradictions(
    cluster_entries: list[tuple[TrailEntry, NDArray]],
) -> list[TrailEntry]:
    """Within a topic cluster, mark older entries as reversed when contradicted.

    Two entries contradict if they share the same topic (high phase cosine)
    but reach different conclusions (low keyword overlap).

    The most recent entry always wins.

    Args:
        cluster_entries: List of (TrailEntry, fused_phase_vector) sorted by timestamp asc.

    Returns:
        Updated TrailEntry list with ``reversed`` flags set.
    """
    entries = [e for e, _ in cluster_entries]
    vectors = [v for _, v in cluster_entries]

    if len(entries) < 2:
        return entries

    # Compare each pair; later entry can reverse an earlier one
    for i in range(len(entries)):
        if entries[i].reversed:
            continue
        for j in range(i + 1, len(entries)):
            vi = vectors[i]
            vj = vectors[j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i < 1e-8 or norm_j < 1e-8:
                continue

            content_sim = float(np.dot(vi, vj) / (norm_i * norm_j))
            if content_sim < CONTRADICTION_TOPIC_SIM:
                continue

            keywords_i = _extract_keywords(entries[i].text)
            keywords_j = _extract_keywords(entries[j].text)
            text_divergence = 1.0 - _jaccard(keywords_i, keywords_j)

            if text_divergence > CONTRADICTION_TEXT_DIV:
                # Same topic, different conclusion — later entry wins
                entries[i] = TrailEntry(
                    timestamp=entries[i].timestamp,
                    text=entries[i].text,
                    session_id=entries[i].session_id,
                    tier=entries[i].tier,
                    reversed=True,
                    source_id=entries[i].source_id,
                )

    return entries


# ═══════════════════════════════════════════════════════════════════
# Decision trail reconstruction
# ═══════════════════════════════════════════════════════════════════


def _build_decision_trails(
    results: list[MaterialisedResult],
    memory: LayeredMemory,
    max_trails: int = 5,
) -> tuple[list[DecisionTrail], int]:
    """Cluster memory results into decision trails with contradiction detection.

    Returns (trails, contradictions_found).
    """
    now = time.time()

    # Collect entries with their fused phase vectors. ``text`` is the
    # claim-extracted sentence (strongest decision-shaped line in the
    # passage) rather than the raw chunk — otherwise decision trails end
    # up as dialog fragments. Passages with no claim-shaped sentence are
    # dropped entirely; they'd just bloat the trail with work-notes.
    entries_with_vectors: list[tuple[TrailEntry, NDArray]] = []
    for r in results:
        content = r.content
        raw = _extract_summary(content, max_chars=600)
        if not raw or len(raw) < 20:
            continue
        text = _extract_claim(raw, max_tokens=35, min_score=2)
        if not text or len(text) < 20:
            continue

        ts = _extract_timestamp(content)
        session_id = _extract_session_id(content)

        # Get registry entry for phase vectors
        lookup = memory.get_registry_entry(r.source_id)
        if lookup is None:
            continue
        tier_name, reg_entry = lookup
        fused = reg_entry.phase_vectors.mean(axis=0)  # (D,) — fused across bands

        entry = TrailEntry(
            timestamp=ts,
            text=text,
            session_id=session_id,
            tier=tier_name,
            reversed=False,
            source_id=r.source_id,
        )
        entries_with_vectors.append((entry, fused))

    if not entries_with_vectors:
        return [], 0

    # Greedy nearest-neighbor clustering
    clusters: list[list[tuple[TrailEntry, NDArray]]] = []
    assigned = set()

    for i, (entry_i, vec_i) in enumerate(entries_with_vectors):
        if i in assigned:
            continue
        cluster = [(entry_i, vec_i)]
        assigned.add(i)

        for j, (entry_j, vec_j) in enumerate(entries_with_vectors):
            if j in assigned:
                continue
            # Compute centroid of current cluster
            centroid = np.mean([v for _, v in cluster], axis=0)
            norm_c = np.linalg.norm(centroid)
            norm_j = np.linalg.norm(vec_j)
            if norm_c < 1e-8 or norm_j < 1e-8:
                continue

            sim = float(np.dot(centroid, vec_j) / (norm_c * norm_j))
            if sim >= DECISION_CLUSTER_THRESHOLD:
                cluster.append((entry_j, vec_j))
                assigned.add(j)

        if len(cluster) >= 2:
            clusters.append(cluster)

    # Sort each cluster by timestamp, detect contradictions
    trails: list[DecisionTrail] = []
    total_contradictions = 0

    for cluster in clusters:
        # Sort by timestamp (empty timestamps go last)
        cluster.sort(key=lambda x: x[0].timestamp or "9999")

        # Detect contradictions
        resolved_entries = _detect_contradictions(cluster)
        reversals = sum(1 for e in resolved_entries if e.reversed)
        total_contradictions += reversals

        # Topic = first sentence of the most recent non-reversed entry
        current_entries = [e for e in resolved_entries if not e.reversed]
        topic_text = current_entries[-1].text if current_entries else resolved_entries[-1].text
        topic = topic_text.split(".")[0][:80]

        # Confidence from the most recent non-reversed entry
        source_id = current_entries[-1].source_id if current_entries else resolved_entries[-1].source_id
        lookup = memory.get_registry_entry(source_id)
        confidence = None
        if lookup:
            _, reg_entry = lookup
            confidence = _compute_confidence(reg_entry, lookup[0], now)

        trails.append(DecisionTrail(
            topic=topic,
            entries=resolved_entries,
            confidence=confidence,
            has_reversal=reversals > 0,
        ))

    # Rank trails by confidence composite, take top N
    trails.sort(key=lambda t: t.confidence.composite if t.confidence else 0, reverse=True)
    trails = trails[:max_trails]

    return trails, total_contradictions


# ═══════════════════════════════════════════════════════════════════
# Cross-primer novelty filtering
# ═══════════════════════════════════════════════════════════════════


def _filter_by_novelty(
    results: list[MaterialisedResult],
    memory: LayeredMemory,
    code_field: DenseField,
    threshold: float = 0.3,
) -> tuple[list[MaterialisedResult], int]:
    """Remove passages redundant with the code knowledge model.

    Returns (filtered_results, count_removed).
    """
    kept: list[MaterialisedResult] = []
    removed = 0

    for r in results:
        lookup = memory.get_registry_entry(r.source_id)
        if lookup is None:
            kept.append(r)
            continue

        _, reg_entry = lookup
        score = FieldAlgebra.novelty(
            code_field,
            reg_entry.phase_vectors,
            reg_entry.salience,
        )
        if score.score >= threshold:
            kept.append(r)
        else:
            removed += 1

    return kept, removed


# ═══════════════════════════════════════════════════════════════════
# Budget rebalancing
# ═══════════════════════════════════════════════════════════════════


def _rebalance_budgets(
    budgets: dict[str, int],
    section_has_data: dict[str, bool],
) -> dict[str, int]:
    """Donate budgets from empty sections to populated ones."""
    donated = 0
    recipients = []

    for section, has_data in section_has_data.items():
        if not has_data and section in budgets:
            donated += budgets[section]
            budgets[section] = 0
        elif has_data:
            recipients.append(section)

    if donated > 0 and recipients:
        per_recipient = donated // len(recipients)
        for section in recipients:
            budgets[section] += per_recipient

    return budgets


# ═══════════════════════════════════════════════════════════════════
# Section formatters
# ═══════════════════════════════════════════════════════════════════


def _filter_reversed_from_axioms(
    results: list[MaterialisedResult],
    trails: list[DecisionTrail],
) -> list[MaterialisedResult]:
    """Remove any entry marked reversed in a decision trail from axiom candidates."""
    reversed_ids = set()
    for trail in trails:
        for entry in trail.entries:
            if entry.reversed:
                reversed_ids.add(entry.source_id)

    return [r for r in results if r.source_id not in reversed_ids]


def _format_section_axioms(
    results: list[MaterialisedResult],
    budget: int,
    memory: LayeredMemory,
    trails: list[DecisionTrail],
    seen_claims: list[str] | None = None,
) -> tuple[str, int]:
    """Render the Project Axioms section.

    Each axiom is the strongest claim-shaped sentence extracted from the
    underlying passage (see ``_extract_claim``), not the raw chunk. Claims
    are deduplicated against ``seen_claims`` so later sections don't repeat
    the same axiom.
    """
    if not results or budget <= 0:
        return "", 0

    # Remove reversed entries
    results = _filter_reversed_from_axioms(results, trails)
    if not results:
        return "", 0

    now = time.time()
    lines: list[str] = []
    tokens_used = 0
    local_seen: list[str] = [] if seen_claims is None else seen_claims

    for r in results:
        raw = _extract_summary(r.content, max_chars=600)
        if not raw:
            continue
        claim = _extract_claim(raw, max_tokens=40)
        if not claim:
            continue
        if _is_near_duplicate(claim, local_seen):
            continue

        est = _estimate_tokens(claim) + 12  # overhead for bullet + badge
        if tokens_used + est > budget:
            break

        # Build confidence badge (only show when reinforced or low/high).
        lookup = memory.get_registry_entry(r.source_id)
        badge = ""
        if lookup:
            _, reg_entry = lookup
            conf = _compute_confidence(reg_entry, lookup[0], now)
            access = reg_entry.access_count
            if access >= 2 or conf.label in ("high", "low"):
                badge = f" *({conf.label}, reinforced {access}x)*"

        lines.append(f"- {claim}{badge}")
        local_seen.append(claim)
        tokens_used += est

    return "\n".join(lines), tokens_used


def _format_section_decisions(
    trails: list[DecisionTrail],
    budget: int,
    max_entries_per_trail: int = 4,
    skip_all_reversed: bool = True,
) -> tuple[str, int]:
    """Render the Decision Trail section with reversal markers.

    Caps each trail to ``max_entries_per_trail`` entries so a single cluster
    doesn't consume the whole section with decades of dialog history. Trails
    where every entry is reversed are skipped outright — they're pure noise
    from noisy clustering, not a useful decision lineage.
    """
    if not trails or budget <= 0:
        return "", 0

    lines: list[str] = []
    tokens_used = 0

    for trail in trails:
        entries = trail.entries
        if skip_all_reversed and entries and all(e.reversed for e in entries):
            continue
        if not entries:
            continue

        # Keep the most recent live entry + up to (max-1) most recent reversed.
        live = [e for e in entries if not e.reversed]
        rev = [e for e in entries if e.reversed]
        # Sort by timestamp descending; keep the freshest entries.
        live.sort(key=lambda e: e.timestamp or "", reverse=True)
        rev.sort(key=lambda e: e.timestamp or "", reverse=True)
        kept = live[: max_entries_per_trail]
        if len(kept) < max_entries_per_trail:
            kept.extend(rev[: max_entries_per_trail - len(kept)])
        # Render chronologically (oldest → newest) so the arrow chain reads forward
        kept.sort(key=lambda e: e.timestamp or "")

        conf_badge = f" [{trail.confidence.label} confidence]" if trail.confidence else ""
        header = f"**{trail.topic}**{conf_badge}"
        header_tokens = _estimate_tokens(header)
        if tokens_used + header_tokens > budget:
            break

        trail_lines: list[str] = [header]
        trail_tokens = header_tokens
        for i, entry in enumerate(kept):
            date = entry.timestamp[:10] if entry.timestamp else "unknown"
            text = entry.text
            if len(text) > 120:
                text = _truncate_to_tokens(text, 30)
            arrow = "-> " if i > 0 else ""
            if entry.reversed:
                line = f"  {arrow}~~{date} {text}~~ [reversed]"
            else:
                line = f"  {arrow}{date} {text}"
            est = _estimate_tokens(line)
            if tokens_used + trail_tokens + est > budget:
                break
            trail_lines.append(line)
            trail_tokens += est

        lines.extend(trail_lines)
        lines.append("")  # blank line between trails
        tokens_used += trail_tokens

    return "\n".join(lines).rstrip(), tokens_used


def _format_section_active(
    results: list[MaterialisedResult],
    budget: int,
    trail_source_ids: set[str],
    seen_claims: list[str] | None = None,
) -> tuple[str, int]:
    """Render the Active Context section grouped by session.

    Each entry becomes a claim-extracted one-liner (see ``_extract_claim``).
    Sessions with no surviving claim are dropped entirely — a session
    header with zero content is just noise.
    """
    if not results or budget <= 0:
        return "", 0

    # Filter out entries already shown in Decision Trail
    results = [r for r in results if r.source_id not in trail_source_ids]
    if not results:
        return "", 0

    # Group by session_id
    sessions: dict[str, list[MaterialisedResult]] = defaultdict(list)
    for r in results:
        sid = _extract_session_id(r.content) or "unknown"
        sessions[sid].append(r)

    # Sort sessions by most recent entry
    def _session_recency(items: list[MaterialisedResult]) -> str:
        timestamps = [_extract_timestamp(r.content) for r in items]
        timestamps = [t for t in timestamps if t]
        return max(timestamps) if timestamps else ""

    sorted_sessions = sorted(sessions.items(), key=lambda x: _session_recency(x[1]), reverse=True)

    local_seen: list[str] = [] if seen_claims is None else seen_claims
    blocks: list[str] = []
    tokens_used = 0

    for session_id, items in sorted_sessions:
        ts = _extract_timestamp(items[0].content)
        date = ts[:10] if ts else "unknown"
        header = f"**Session {date}** (episodic)"
        header_tokens = _estimate_tokens(header)

        session_lines: list[str] = []
        session_tokens = 0
        for r in items:
            raw = _extract_summary(r.content, max_chars=400)
            if not raw:
                continue
            claim = _extract_claim(raw, max_tokens=35)
            if not claim:
                continue
            if _is_near_duplicate(claim, local_seen):
                continue
            line = f"- {claim}"
            est = _estimate_tokens(line)
            if tokens_used + header_tokens + session_tokens + est > budget:
                break
            session_lines.append(line)
            local_seen.append(claim)
            session_tokens += est

        if not session_lines:
            continue  # skip empty session blocks
        if tokens_used + header_tokens + session_tokens > budget:
            break
        blocks.append(header)
        blocks.extend(session_lines)
        blocks.append("")
        tokens_used += header_tokens + session_tokens

    return "\n".join(blocks).rstrip(), tokens_used


def _format_section_recent(
    results: list[MaterialisedResult],
    budget: int,
    seen_claims: list[str] | None = None,
) -> tuple[str, int]:
    """Render the Recent Sessions section as claim-extracted chronological summaries."""
    if not results or budget <= 0:
        return "", 0

    # Group by session_id
    sessions: dict[str, list[MaterialisedResult]] = defaultdict(list)
    for r in results:
        sid = _extract_session_id(r.content) or "unknown"
        sessions[sid].append(r)

    # Sort sessions by most recent timestamp
    def _session_time(items: list[MaterialisedResult]) -> str:
        timestamps = [_extract_timestamp(r.content) for r in items]
        timestamps = [t for t in timestamps if t]
        return max(timestamps) if timestamps else ""

    sorted_sessions = sorted(sessions.items(), key=lambda x: _session_time(x[1]), reverse=True)

    local_seen: list[str] = [] if seen_claims is None else seen_claims
    blocks: list[str] = []
    tokens_used = 0

    for session_id, items in sorted_sessions:
        ts = _session_time(items)
        date = ts[:10] if ts else "unknown"

        session_lines: list[str] = []
        session_tokens = 0
        for r in items:
            raw = _extract_summary(r.content, max_chars=300)
            if not raw:
                continue
            claim = _extract_claim(raw, max_tokens=30)
            if not claim:
                continue
            if _is_near_duplicate(claim, local_seen):
                continue
            line = f"- {claim}"
            est = _estimate_tokens(line)
            if tokens_used + session_tokens + est > budget:
                break
            session_lines.append(line)
            local_seen.append(claim)
            session_tokens += est

        if not session_lines:
            continue
        header = f"**{date}** ({len(session_lines)} claim{'s' if len(session_lines) != 1 else ''})"
        header_tokens = _estimate_tokens(header)
        if tokens_used + header_tokens + session_tokens > budget:
            break
        blocks.append(header)
        blocks.extend(session_lines)
        blocks.append("")
        tokens_used += header_tokens + session_tokens

    return "\n".join(blocks).rstrip(), tokens_used


# ═══════════════════════════════════════════════════════════════════
# Main primer generator
# ═══════════════════════════════════════════════════════════════════


def generate_memory_primer(
    memory: LayeredMemory,
    code_cartridge: Lattice | None = None,
    budget: int = 2500,
    novelty_threshold: float = 0.3,
) -> MemoryPrimerResult:
    """Generate a conversation-memory primer document.

    Pipeline:
      1. Bootstrap queries with tier-biased weights
      2. Section assignment by query type + tier origin
      3. Cross-primer novelty filter (if code_cartridge provided)
      4. Decision trail reconstruction with contradiction detection
      5. Budget allocation, rebalancing, and rendering

    Args:
        memory: Open LayeredMemory instance with encoder.
        code_cartridge: Optional code knowledge model for cross-primer novelty filter.
        budget: Target token budget.
        novelty_threshold: Passages with novelty below this vs code knowledge model
            are suppressed (0.0 = suppress nothing, 1.0 = suppress everything).

    Returns:
        MemoryPrimerResult with rendered markdown and statistics.
    """
    if memory.total_sources == 0:
        return MemoryPrimerResult(
            markdown=_render_empty_primer(),
            total_tokens=_estimate_tokens(_render_empty_primer()),
            section_tokens={},
            passages_used=0,
            tiers_queried=[],
            novelty_filtered=0,
            contradictions_found=0,
        )

    # ── Phase 1: Bootstrap queries with tier-biased weights ──────────────
    section_results: dict[str, list[MaterialisedResult]] = {
        "axioms": [],
        "decisions": [],
        "active": [],
        "recent": [],
    }

    seen_ids: dict[str, float] = {}  # source_id -> best score (dedup)

    for query, section in QUERY_SECTION_MAP:
        tier_weights = SECTION_TIER_WEIGHTS[section]
        result = memory.recall_text(
            query,
            tier_weights=tier_weights,
            top_k=20,
        )
        for r in result.results:
            prev_score = seen_ids.get(r.source_id, -1)
            if r.score > prev_score:
                seen_ids[r.source_id] = r.score
                # Remove from previous section if it was there
                for sec_list in section_results.values():
                    sec_list[:] = [x for x in sec_list if x.source_id != r.source_id]
                section_results[section].append(r)

    # ── Phase 2a: Cross-cartridge compose boost (decisions only) ─────────
    # When a code cartridge is provided, run the decisions bootstrap query
    # through a composed field (memory.semantic + code_cartridge) — decisions
    # that resonate with present code rise; decisions about deprecated or
    # removed files sink. Additive to the memory-only ranking so tier-weighted
    # recency is preserved.
    if code_cartridge is not None:
        try:
            from resonance_lattice.composition import ComposedCartridge
            sem_tier = memory.tiers.get("semantic") if memory.tiers else None
            # Encoder compatibility: ComposedCartridge.merge will warn/fail
            # if fingerprints disagree, so gate on identity first.
            mem_enc = sem_tier.encoder if sem_tier else None
            code_enc = code_cartridge.encoder
            if sem_tier is not None and mem_enc is not None and code_enc is not None:
                from resonance_lattice.cli import _encoder_fingerprint
                if _encoder_fingerprint(mem_enc) == _encoder_fingerprint(code_enc):
                    composed = ComposedCartridge.merge(
                        {"memory": sem_tier, "code": code_cartridge},
                        weights={"memory": 0.7, "code": 0.3},
                    )
                    # Query that targets decision / tradeoff material
                    decision_query = (
                        MEMORY_BOOTSTRAP_QUERIES[1]  # the decisions bootstrap
                    )
                    # ComposedCartridge exposes .resonate_text via its internal
                    # _merged_lattice — fall back gracefully if that shape ever changes.
                    composed_results = None
                    for attr in ("resonate_text", "search_text"):
                        fn = getattr(composed, attr, None)
                        if callable(fn):
                            composed_results = fn(decision_query, top_k=20)
                            break
                    if composed_results is not None:
                        results_iter = getattr(composed_results, "results", composed_results)
                        boost_scores = {r.source_id: r.score for r in results_iter}
                        decisions = section_results.get("decisions", [])
                        for r in decisions:
                            boost = boost_scores.get(r.source_id)
                            if boost is not None and r.score > 0:
                                # Additive boost capped at 15% of base
                                r.score += min(0.15 * r.score, 0.3 * boost)
                        decisions.sort(key=lambda r: -r.score)
                        section_results["decisions"] = decisions
        except Exception as exc:
            import sys as _sys
            print(
                f"memory-primer: cross-cartridge compose skipped "
                f"({type(exc).__name__}: {exc})",
                file=_sys.stderr,
            )

    # ── Phase 2b: Cross-primer novelty filter ────────────────────────────
    total_novelty_filtered = 0
    if code_cartridge is not None:
        for section_name in section_results:
            filtered, removed = _filter_by_novelty(
                section_results[section_name],
                memory,
                code_cartridge.field,
                novelty_threshold,
            )
            section_results[section_name] = filtered
            total_novelty_filtered += removed

    # ── Phase 3: Decision trail reconstruction ───────────────────────────
    decision_inputs = section_results["decisions"] + section_results["axioms"]
    trails, contradictions = _build_decision_trails(decision_inputs, memory)

    # ── Phase 4: Budget allocation and rebalancing ───────────────────────
    section_has_data = {
        "axioms": len(section_results["axioms"]) > 0,
        "decisions": len(trails) > 0,
        "active": len(section_results["active"]) > 0,
        "recent": len(section_results["recent"]) > 0,
    }

    budgets = {k: int(budget * v) for k, v in SECTION_BUDGET_FRACTIONS.items()}
    budgets = _rebalance_budgets(budgets, section_has_data)

    # Collect source_ids used in trails (for dedup in Active Context)
    trail_source_ids: set[str] = set()
    for trail in trails:
        for entry in trail.entries:
            trail_source_ids.add(entry.source_id)

    # ── Phase 5: Render sections ─────────────────────────────────────────
    section_tokens: dict[str, int] = {}

    # Shared dedup list — a claim shown in Axioms should not reappear in
    # Active Context or Recent. Decision Trails already carry their own
    # extracted text and are tracked via ``trail_source_ids`` above.
    seen_claims: list[str] = []
    # Seed with trail entry texts so sections don't re-surface them.
    for trail in trails:
        for entry in trail.entries:
            if entry.text:
                seen_claims.append(entry.text)

    axioms_md, axioms_tok = _format_section_axioms(
        section_results["axioms"], budgets["axioms"], memory, trails,
        seen_claims=seen_claims,
    )
    section_tokens["axioms"] = axioms_tok

    decisions_md, decisions_tok = _format_section_decisions(trails, budgets["decisions"])
    section_tokens["decisions"] = decisions_tok

    active_md, active_tok = _format_section_active(
        section_results["active"], budgets["active"], trail_source_ids,
        seen_claims=seen_claims,
    )
    section_tokens["active"] = active_tok

    recent_md, recent_tok = _format_section_recent(
        section_results["recent"], budgets["recent"],
        seen_claims=seen_claims,
    )
    section_tokens["recent"] = recent_tok

    # ── Assemble document ────────────────────────────────────────────────
    total_tokens = sum(section_tokens.values())
    passages_used = sum(len(v) for v in section_results.values())
    tiers_queried = [t for t in ("working", "episodic", "semantic")
                     if t in memory.tiers and memory.tiers[t].source_count > 0]

    # Tier counts for header
    tier_counts = {t: memory.tiers[t].source_count
                   for t in ("working", "episodic", "semantic")
                   if t in memory.tiers}
    tier_summary = " | ".join(f"{t}: {c}" for t, c in tier_counts.items() if c > 0)

    header_lines = [
        "# Conversation Memory",
        "",
        "<!-- Auto-generated by `rlat memory primer` -->",
        f"<!-- {memory.total_sources} memories across {len(tiers_queried)} tiers | {tier_summary} -->",
    ]
    if total_novelty_filtered > 0:
        header_lines.append(
            f"<!-- {total_novelty_filtered} passages filtered by cross-primer novelty "
            f"(threshold: {novelty_threshold}) -->"
        )
    header_lines.append("")

    body_parts: list[str] = []

    if axioms_md:
        body_parts.append(f"## Project Axioms\n\n{axioms_md}")

    if decisions_md:
        body_parts.append(f"## Decision Trail\n\n{decisions_md}")

    if active_md:
        body_parts.append(f"## Active Context\n\n{active_md}")

    if recent_md:
        body_parts.append(f"## Recent Sessions\n\n{recent_md}")

    footer = [
        "",
        "---",
        "",
        "For deeper memory recall:",
        "```",
        'rlat memory recall ./memory/ "your question here"',
        "```",
    ]

    markdown = "\n".join(header_lines + ["\n\n".join(body_parts)] + footer)

    return MemoryPrimerResult(
        markdown=markdown,
        total_tokens=total_tokens,
        section_tokens=section_tokens,
        passages_used=passages_used,
        tiers_queried=tiers_queried,
        novelty_filtered=total_novelty_filtered,
        contradictions_found=contradictions,
    )


def _render_empty_primer() -> str:
    """Minimal primer when memory is empty."""
    return (
        "# Conversation Memory\n"
        "\n"
        "<!-- Auto-generated by `rlat memory primer` -->\n"
        "\n"
        "No conversation history yet.\n"
    )
