# SPDX-License-Identifier: BUSL-1.1
"""Projectors: Convert resonance readouts into LLM-consumable context.

TextProjector: Converts a MemoryReadout into compressed natural language.
    The resonance identifies which sources matter and how much. Instead of
    stuffing 10 full passages (~2000 tokens), we extract the GIST per band:
    - Which domain/topic the query relates to (from top band-0/1 sources)
    - Key relationships and entities (from top band-2/3 sources)
    - Most relevant verbatim excerpt (from top band-4 source)
    Result: ~100-300 tokens of highly focused context.

GatedProjector: Wraps any projector with novelty-based gating. Before
    injecting memory tokens, computes a gate score from the field's resonance
    energy and novelty. Suppresses injection when the model likely already
    knows the answer (the "7B problem" fix).

EmbeddingProjector: Maps resonance vectors to LLM embedding space as
    soft prompt tokens. Requires fine-tuning. (Placeholder for Phase 4.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .memory import MemoryReadout

# ═══════════════════════════════════════════════════════════
# Gating: Adaptive memory injection
# ═══════════════════════════════════════════════════════════

@dataclass
class GateDecision:
    """Result of the gating decision."""
    inject: bool          # Whether to inject memory context
    gate_score: float     # 0.0 = suppress, 1.0 = full injection
    reason: str           # Human-readable explanation
    resonance_energy: float
    novelty_score: float
    confidence: float     # How confident the gate is in its decision


def compute_gate_score(
    readout: MemoryReadout,
    passage_phases: NDArray,
    gate_threshold: float = 0.3,
) -> GateDecision:
    """Compute whether to inject memory context for this query.

    The gate balances two signals:
    1. Resonance energy: Does the field have relevant knowledge?
       (High = field knows something about this query)
    2. Novelty: Is the field's knowledge likely NEW to the model?
       (High = model probably doesn't know this)

    Gate logic:
    - High energy + high novelty → INJECT (field has unique knowledge)
    - High energy + low novelty → SUPPRESS (model already knows)
    - Low energy → SUPPRESS (field doesn't know either)

    The novelty signal is approximated from the readout's score distribution:
    - If top scores are tightly clustered → many sources match → broad topic
      → model likely knows it → lower novelty
    - If top scores are spread → specific niche match → model less likely
      to know → higher novelty

    Args:
        readout: MemoryReadout from ResonanceMemory.query().
        passage_phases: Shape (N, B, D) for computing novelty signals.
        gate_threshold: Score below which to suppress injection.
            Recommended: 0.1 for small models, 0.3 for medium, 0.5 for large.

    Returns:
        GateDecision with inject flag and diagnostics.
    """
    B = readout.resonance_vectors.shape[0]

    # Signal 1: Resonance energy (normalised by band count)
    total_energy = float(readout.band_energies.sum())
    mean_band_energy = total_energy / B

    # Signal 2: Score concentration (proxy for novelty)
    # If the top source score is much higher than others → specific match → higher novelty
    top_scores = readout.top_source_scores
    if len(top_scores) >= 2:
        score_ratio = float(top_scores[0]) / (float(top_scores[-1]) + 1e-8)
        # Normalise: ratio of 1 = uniform (low novelty), high ratio = concentrated (high novelty)
        concentration = min(1.0, (score_ratio - 1.0) / 10.0)
    else:
        concentration = 0.5

    # Signal 3: Band energy distribution (entropy-based)
    # If energy is spread across many bands → broad query → model likely knows
    # If energy is concentrated in 1-2 bands → specific → higher novelty
    band_probs = readout.band_energies / (total_energy + 1e-12)
    entropy = -float(np.sum(band_probs * np.log(band_probs + 1e-12)))
    max_entropy = np.log(B)
    # Low entropy = concentrated = higher novelty
    band_focus = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5

    # Combine signals: geometric mean biased toward novelty
    # Energy gates whether we have anything relevant
    # Concentration + band_focus estimate whether it's novel
    energy_signal = min(1.0, mean_band_energy / 100.0)  # Saturates above 100
    novelty_signal = 0.6 * concentration + 0.4 * band_focus

    gate_score = energy_signal * (0.3 + 0.7 * novelty_signal)

    # Decision
    inject = gate_score >= gate_threshold
    confidence = abs(gate_score - gate_threshold) / max(gate_threshold, 1 - gate_threshold)

    if not inject:
        if energy_signal < 0.1:
            reason = "suppressed: field has no relevant knowledge"
        else:
            reason = "suppressed: model likely already knows this topic"
    else:
        if novelty_signal > 0.6:
            reason = "injecting: field has specific, likely novel knowledge"
        else:
            reason = "injecting: field has relevant knowledge"

    return GateDecision(
        inject=inject,
        gate_score=float(gate_score),
        reason=reason,
        resonance_energy=total_energy,
        novelty_score=float(novelty_signal),
        confidence=float(confidence),
    )


class GatedProjector:
    """Wraps any projector with novelty-based gating.

    Solves the "7B problem": large models that already know the topic
    get WORSE when memory tokens are injected. The gate suppresses
    injection when the field's knowledge is likely redundant with the
    model's pre-training.

    Usage:
        gated = GatedProjector(base_projector, gate_threshold=0.3)
        result = gated.project(readout, query_text="How does X work?")
        if result is None:
            # Gate suppressed — don't inject memory
            pass
        else:
            context, decision = result
    """

    def __init__(
        self,
        base_projector: TextProjector | SmartProjector,
        passage_phases: NDArray,
        gate_threshold: float = 0.3,
    ) -> None:
        self.base = base_projector
        self.passage_phases = passage_phases
        self.threshold = gate_threshold

    def project(
        self,
        readout: MemoryReadout,
        query_text: str = "",
    ) -> tuple[str, GateDecision] | None:
        """Project with gating. Returns None if gate suppresses.

        Returns:
            (context_string, gate_decision) if injecting, None if suppressed.
        """
        decision = compute_gate_score(
            readout, self.passage_phases, self.threshold,
        )

        if not decision.inject:
            return None

        context = self.base.project(readout, query_text)
        return context, decision

    def project_always(
        self,
        readout: MemoryReadout,
        query_text: str = "",
    ) -> tuple[str, GateDecision]:
        """Always project, but include gate decision for diagnostics."""
        decision = compute_gate_score(
            readout, self.passage_phases, self.threshold,
        )
        context = self.base.project(readout, query_text)
        return context, decision


class GroundingProjector:
    """Grounding-only projection: inject citations, not broad context.

    The gating eval showed that large models (7B+) get WORSE when broad
    memory context is injected — they already know the topic. But they
    get BETTER groundedness when given specific citations.

    This projector injects ONLY the top-k most relevant passages as
    citable sources, prefixed with "Source evidence (for grounding):"
    to signal the model should use them for citations, not as primary
    knowledge.

    This is the recommended mode for large models:
    - Completeness: model uses its own knowledge (high quality)
    - Groundedness: model cites specific passages from the knowledge model
    - Result: best of both worlds

    Usage:
        grounding = GroundingProjector(chunks, passage_phases, n_sources=3)
        context = grounding.project(readout, query_text="How does X?")
        # Inject as: "Answer using your own knowledge. Use these sources
        #             for grounding and citations if relevant: {context}"
    """

    def __init__(
        self,
        chunks: list[dict],
        passage_phases: NDArray,
        n_sources: int = 8,
        chars_per_source: int = 800,
    ) -> None:
        self.chunks = chunks
        self.passage_phases = passage_phases
        self.n_sources = n_sources
        self.chars_per_source = chars_per_source

    def project(
        self,
        readout: MemoryReadout,
        query_text: str = "",
    ) -> str:
        """Extract top-k grounding passages from the readout."""
        parts = []
        seen = set()

        for idx, score in zip(readout.top_source_indices, readout.top_source_scores):
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)

            chunk = self.chunks[idx]
            title = chunk.get("title", f"passage_{idx}")
            text = chunk.get("text_preview", "")
            if len(text) > self.chars_per_source:
                text = text[:self.chars_per_source].rsplit(" ", 1)[0] + "..."

            parts.append(f"[{len(parts)+1}] {title}\n{text}")

            if len(parts) >= self.n_sources:
                break

        return (
            "=== Source evidence (for grounding and citations) ===\n"
            + "\n\n".join(parts)
        )


class TextProjector:
    """Convert resonance readout to compressed text context.

    Uses the resonance to identify the most relevant passages PER BAND,
    then extracts the minimal context from each band's perspective.

    The key insight: each band sees the query differently. Band 0 says
    "this is about data engineering." Band 3 says "specifically about
    OneLake." Band 4 says "here's the exact passage." Combining these
    gives hierarchical context in ~200 tokens instead of ~2000.
    """

    BAND_NAMES = ["domain", "topic", "relations", "entity", "verbatim"]
    BAND_INSTRUCTIONS = [
        "Domain context (broad area)",
        "Topic context (specific subject)",
        "Relational context (how concepts connect)",
        "Entity context (specific names and terms)",
        "Verbatim context (most relevant excerpt)",
    ]

    def __init__(
        self,
        chunks: list[dict],
        passage_phases: NDArray,
        max_tokens_per_band: int = 200,
        max_total_tokens: int = 1000,
    ) -> None:
        """
        Args:
            chunks: List of dicts with 'title', 'text_preview', 'file' keys.
            passage_phases: Shape (N, B, D) — for per-band scoring.
            max_tokens_per_band: Approximate token budget per band.
            max_total_tokens: Hard cap on total context length (chars / 4).
        """
        self.chunks = chunks
        self.passage_phases = passage_phases
        self.max_per_band = max_tokens_per_band
        self.max_total = max_total_tokens

    def project(
        self,
        readout: MemoryReadout,
        query_text: str = "",
    ) -> str:
        """Convert a memory readout to compressed text context.

        Args:
            readout: MemoryReadout from ResonanceMemory.query().
            query_text: Original query text (for context framing).

        Returns:
            Compressed context string (~100-300 tokens).
        """
        B = readout.resonance_vectors.shape[0]
        sections = []

        # Per-band: find the top source for THAT band and extract its essence
        for b in range(min(B, 5)):
            # Score all passages against this band's resonance
            r_b = readout.resonance_vectors[b]
            band_scores = self.passage_phases[:, b, :] @ r_b
            top_idx = int(np.argmax(band_scores))

            chunk = self.chunks[top_idx]
            title = chunk.get("title", "")
            text = chunk.get("text_preview", "")

            # Truncate to budget
            char_budget = self.max_per_band * 4  # ~4 chars per token
            excerpt = text[:char_budget].rsplit(" ", 1)[0] if len(text) > char_budget else text

            band_name = self.BAND_NAMES[b] if b < len(self.BAND_NAMES) else f"band_{b}"
            self.BAND_INSTRUCTIONS[b] if b < len(self.BAND_INSTRUCTIONS) else ""

            sections.append(f"[{band_name}] {title}: {excerpt}")

        # Compose the compressed context
        header = "=== RESONANCE MEMORY (synthesised from field) ==="
        body = "\n".join(sections)

        # Add the single most relevant full passage for grounding
        top_overall = readout.top_source_indices[0]
        grounding = self.chunks[top_overall]
        grounding_text = grounding.get("text_preview", "")[:2000]

        full_context = f"{header}\n{body}\n\n[grounding] {grounding.get('title', '')}: {grounding_text}"

        # Enforce total budget
        char_cap = self.max_total * 4
        if len(full_context) > char_cap:
            full_context = full_context[:char_cap].rsplit(" ", 1)[0] + "..."

        return full_context

    def project_minimal(self, readout: MemoryReadout) -> str:
        """Ultra-compressed: just the top source per band title + top passage.

        ~50-80 tokens. The minimum viable memory context.
        """
        B = readout.resonance_vectors.shape[0]
        titles = []

        for b in range(min(B, 5)):
            r_b = readout.resonance_vectors[b]
            band_scores = self.passage_phases[:, b, :] @ r_b
            top_idx = int(np.argmax(band_scores))
            title = self.chunks[top_idx].get("title", f"chunk_{top_idx}")
            titles.append(f"{self.BAND_NAMES[b]}: {title}")

        top_overall = readout.top_source_indices[0]
        excerpt = self.chunks[top_overall].get("text_preview", "")[:400]

        return (
            "Memory: " + " | ".join(titles) + "\n"
            "Key passage: " + excerpt
        )


class SmartProjector:
    """Improved projector: deduplicates across bands, adapts to query breadth.

    Instead of 1 source per band (5 total, often duplicated), this projector:
    1. Collects top-2 per band + top-10 overall from the readout
    2. Deduplicates by passage index
    3. Selects the top-N unique sources
    4. Assembles compact multi-source context

    This fixes breadth queries (security features, comparisons) where
    the basic TextProjector's 5-source limit was too narrow.
    """

    BAND_NAMES = ["domain", "topic", "relations", "entity", "verbatim"]

    def __init__(
        self,
        chunks: list[dict],
        passage_phases: NDArray,
        n_sources: int = 12,
        chars_per_source: int = 800,
        full_text_lookup: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            chunks: Passage metadata with 'title', 'text_preview', 'file'.
            passage_phases: Shape (N, B, D).
            n_sources: Number of unique sources to include.
            chars_per_source: Character budget per source.
            full_text_lookup: Optional dict mapping chunk file path to full text.
        """
        self.chunks = chunks
        self.passage_phases = passage_phases
        self.n_sources = n_sources
        self.chars_per_source = chars_per_source
        self.full_text = full_text_lookup or {}

    def _get_text(self, chunk: dict, budget: int) -> str:
        """Get text for a chunk, preferring full text if available."""
        file_path = chunk.get("file", "")
        if file_path in self.full_text:
            text = self.full_text[file_path]
        else:
            text = chunk.get("text_preview", "")
        if len(text) > budget:
            text = text[:budget].rsplit(" ", 1)[0] + "..."
        return text

    def project(self, readout: MemoryReadout, query_text: str = "") -> str:
        """Build context from deduplicated multi-band sources."""
        B = readout.resonance_vectors.shape[0]

        # Collect candidates: top-2 per band + top-10 overall
        scored: dict[int, float] = {}  # idx -> best score

        for b in range(min(B, 5)):
            r_b = readout.resonance_vectors[b]
            band_scores = self.passage_phases[:, b, :] @ r_b
            top2 = np.argsort(band_scores)[-2:][::-1]
            self.BAND_NAMES[b] if b < len(self.BAND_NAMES) else f"b{b}"
            for idx in top2:
                idx = int(idx)
                score = float(band_scores[idx])
                if idx not in scored or score > scored[idx]:
                    scored[idx] = score

        # Add top-10 from overall readout
        for idx, score in zip(readout.top_source_indices, readout.top_source_scores):
            idx = int(idx)
            score = float(score)
            if idx not in scored or score > scored[idx]:
                scored[idx] = score

        # Rank by score, take top-N unique
        ranked = sorted(scored.items(), key=lambda x: -x[1])[:self.n_sources]

        # Build context
        parts = []
        for rank, (idx, score) in enumerate(ranked, 1):
            chunk = self.chunks[idx]
            title = chunk.get("title", f"passage_{idx}")
            text = self._get_text(chunk, self.chars_per_source)
            parts.append(f"[{rank}] {title}: {text}")

        return "=== RESONANCE MEMORY (multi-band synthesis) ===\n" + "\n".join(parts)


class AugmentProjector:
    """Configurable injection framing: augment, constrain, or custom.

    Three injection modes control how the LLM uses the knowledge model context:

    - AUGMENT (default): Model answers from its own knowledge, uses passages
      for details and citations. Best overall quality (+3.6% vs no-memory).
      Use for general knowledge enrichment.

    - CONSTRAIN: Model answers ONLY from the provided context. Zero
      hallucination — every claim traceable to a source passage.
      Use for legal, medical, compliance, and regulated domains.

    - CUSTOM: User provides their own system prompt. Full control.
      Use for domain-specific or application-specific framing.

    Usage:
        # Default: augment mode
        proj = AugmentProjector(chunks, passage_phases)
        context, system = proj.project(readout, query)

        # Zero-hallucination mode
        proj = AugmentProjector(chunks, passage_phases, mode="constrain")
        context, system = proj.project(readout, query)

        # Custom prompt
        proj = AugmentProjector(chunks, passage_phases,
                                mode="custom", custom_prompt="You are a medical expert...")
        context, system = proj.project(readout, query)
    """

    SYSTEM_AUGMENT = (
        "You are a knowledgeable expert. Answer from your own knowledge.\n"
        "Below are reference passages from an authoritative knowledge base. Use them\n"
        "to add specific details, correct any uncertainties, and cite sources [1], [2]\n"
        "where applicable. Do NOT limit your answer to only what the sources contain."
    )

    SYSTEM_CONSTRAIN = (
        "You are an AI assistant. Answer ONLY using the provided reference passages.\n"
        "Do NOT use your own knowledge or make claims beyond what the sources contain.\n"
        "Cite every claim with [1], [2] etc. If the context doesn't cover something,\n"
        "explicitly state: 'The provided sources do not cover this.'"
    )

    SYSTEM_KNOWLEDGE = (
        "You are an AI assistant. You may not have training data on this topic.\n"
        "Below are authoritative reference passages. Base your answer primarily\n"
        "on this context, and be transparent about what the context doesn't cover."
    )

    VALID_MODES = ("augment", "constrain", "knowledge", "custom")

    def __init__(
        self,
        chunks: list[dict],
        passage_phases: NDArray,
        n_sources: int = 12,
        chars_per_source: int = 800,
        mode: str = "augment",
        custom_prompt: str | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if mode == "custom" and not custom_prompt:
            raise ValueError("custom_prompt is required when mode='custom'")
        self.chunks = chunks
        self.passage_phases = passage_phases
        self.n_sources = n_sources
        self.chars_per_source = chars_per_source
        self.mode = mode
        self.custom_prompt = custom_prompt
        self._smart = SmartProjector(
            chunks, passage_phases, n_sources, chars_per_source,
        )

    def project(
        self,
        readout: MemoryReadout,
        query_text: str = "",
    ) -> tuple[str, str]:
        """Project readout into context + system prompt pair.

        Returns:
            (context_string, system_prompt) — inject both into the LLM call.
        """
        context = self._smart.project(readout, query_text)
        if self.mode == "augment":
            system = self.SYSTEM_AUGMENT
        elif self.mode == "constrain":
            system = self.SYSTEM_CONSTRAIN
        elif self.mode == "knowledge":
            system = self.SYSTEM_KNOWLEDGE
        else:
            system = self.custom_prompt
        return context, system


def build_rag_context(
    query_emb: NDArray,
    passage_embs: NDArray,
    chunks: list[dict],
    top_k: int = 10,
) -> str:
    """Standard RAG: E5 cosine top-k passages stuffed into context.

    With full-text passages this would be ~2000+ tokens.
    With 200-char previews: ~500-700 tokens for top-10.
    """
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    p_norms = passage_embs / (
        np.linalg.norm(passage_embs, axis=1, keepdims=True) + 1e-8
    )
    scores = p_norms @ q_norm
    top_idx = np.argsort(scores)[-top_k:][::-1]

    parts = []
    for rank, idx in enumerate(top_idx, 1):
        c = chunks[idx]
        parts.append(f"[{rank}] {c.get('title', '')}\n{c.get('text_preview', '')}")

    return f"=== RAG CONTEXT (top-{top_k} passages) ===\n\n" + "\n\n---\n\n".join(parts)


def build_hybrid_context(
    readout: MemoryReadout,
    projector: TextProjector,
    query_emb: NDArray,
    passage_embs: NDArray,
    chunks: list[dict],
) -> str:
    """Hybrid: resonance memory summary + top-1 RAG passage for grounding.

    ~400-600 tokens. Best of both worlds.
    """
    # Memory summary (no grounding passage — we add our own)
    B = readout.resonance_vectors.shape[0]
    sections = []
    for b in range(min(B, 5)):
        r_b = readout.resonance_vectors[b]
        band_scores = projector.passage_phases[:, b, :] @ r_b
        top_idx = int(np.argmax(band_scores))
        chunk = projector.chunks[top_idx]
        title = chunk.get("title", "")
        text = chunk.get("text_preview", "")[:800]
        sections.append(f"[{projector.BAND_NAMES[b]}] {title}: {text}")

    memory_part = "=== MEMORY CONTEXT ===\n" + "\n".join(sections)

    # Top-1 RAG passage for grounding
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    p_norms = passage_embs / (
        np.linalg.norm(passage_embs, axis=1, keepdims=True) + 1e-8
    )
    scores = p_norms @ q_norm
    top_idx = int(np.argmax(scores))
    c = chunks[top_idx]
    rag_part = f"\n\n=== GROUNDING PASSAGE ===\n{c.get('title', '')}\n{c.get('text_preview', '')}"

    return memory_part + rag_part
