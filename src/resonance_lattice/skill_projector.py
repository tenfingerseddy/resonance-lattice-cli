# SPDX-License-Identifier: BUSL-1.1
"""SkillProjector: four-tier adaptive context injection for knowledge model-backed skills.

Tier 1: Static SKILL.md header (always loaded)
Tier 2: Foundational queries from frontmatter (same every trigger)
Tier 3: User query resonated against knowledge models (unique per request)
Tier 4: LLM-derived queries (implicit needs the user didn't express)

See docs/SKILL_INTEGRATION.md for the architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resonance_lattice.skill import SkillConfig
    from resonance_lattice.skill_runtime import SkillRuntime

logger = logging.getLogger(__name__)


@dataclass
class CartridgeFreshness:
    """Freshness status for a single knowledge model."""
    name: str
    age_hours: float
    status: str              # fresh | consider rebuilding | stale | unknown

    def label(self) -> str:
        if self.status == "fresh":
            return f"{self.name}: fresh ({self.age_hours:.0f}h)"
        if self.status == "stale":
            return f"{self.name}: STALE ({self.age_hours:.0f}h)"
        if self.status == "unknown":
            return f"{self.name}: unknown"
        return f"{self.name}: {self.status} ({self.age_hours:.0f}h)"


@dataclass
class SkillInjection:
    """Complete injection result from the four-tier pipeline."""

    header: str                           # Tier 1: static SKILL.md content
    body: str                             # Tiers 2+3+4: dynamic passages
    mode: str                             # augment | constrain | knowledge
    total_tokens: int                     # estimated token count (header + body)
    tier_tokens: dict[str, int]           # tokens per tier
    queries_used: list[str]               # all queries that produced results
    cartridge_hits: dict[str, int]        # cartridge -> passage count
    gated: bool                           # True if dynamic body was suppressed
    coverage_confidence: float            # 0-1, from enriched result
    freshness: list[CartridgeFreshness] = field(default_factory=list)


# ── Token helpers ────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    return truncated.rsplit(" ", 1)[0] + "..."


# ── Dedup ────────────────────────────────────────────────────────────

def _dedup_passages(passages: list[dict]) -> list[dict]:
    """Remove duplicate passages by source_id, keeping highest score."""
    seen: dict[str, dict] = {}
    for p in passages:
        sid = p.get("source_id", "")
        if sid not in seen or p.get("score", 0) > seen[sid].get("score", 0):
            seen[sid] = p
    return sorted(seen.values(), key=lambda x: -x.get("score", 0))


# ── Passage formatting ───────────────────────────────────────────────

_MODE_PREAMBLES = {
    "augment": (
        "Use your own knowledge, but add detail and citations from these sources.\n"
    ),
    "knowledge": (
        "Base your answer primarily on the following context.\n"
    ),
    "constrain": (
        "Answer ONLY from the following sources. "
        "If the answer isn't here, say so.\n"
    ),
}


def _format_passage(p: dict) -> str:
    """Format a single passage for injection."""
    content = p.get("content")
    if content is None:
        return ""
    text = ""
    if hasattr(content, "full_text"):
        text = content.full_text or content.summary or ""
    elif isinstance(content, dict):
        text = content.get("full_text", "") or content.get("summary", "")
    if not text:
        return ""

    source_file = ""
    if hasattr(content, "metadata") and content.metadata:
        source_file = content.metadata.get("source_file", "")
    elif isinstance(content, dict):
        source_file = content.get("metadata", {}).get("source_file", "")

    cartridge = p.get("knowledge model", "")
    score = p.get("score", 0)

    header = f"[{score:.2f}]"
    if cartridge:
        header += f" ({cartridge})"
    if source_file:
        header += f" {source_file}"

    return f"- {header}\n  {text.strip()}"


# ── SkillProjector ───────────────────────────────────────────────────

class SkillProjector:
    """Four-tier adaptive context injection for knowledge model-backed skills.

    Usage:
        from resonance_lattice.skill_runtime import SkillRuntime
        rt = SkillRuntime(skills_root, project_root)
        projector = SkillProjector(rt)
        injection = projector.project(skill_config, "user's question")
    """

    def __init__(self, runtime: SkillRuntime) -> None:
        self._rt = runtime

    def project(
        self,
        skill: SkillConfig,
        user_query: str,
        derived_queries: list[str] | None = None,
    ) -> SkillInjection:
        """Generate four-tier adaptive context for a skill triggered by a query.

        Args:
            skill: Parsed skill configuration.
            user_query: The user's actual request.
            derived_queries: Pre-computed Tier 4 queries (if None, Tier 4 is skipped
                             here -- caller is responsible for LLM derivation).
        """
        from resonance_lattice.skill import extract_skill_header

        # ── Tier 1: Static header ────────────────────────────────────
        header = extract_skill_header(skill.skill_dir)
        header_tokens = _estimate_tokens(header)

        # ── Resolve cartridges ───────────────────────────────────────
        paths = self._rt.resolve_cartridges(skill)
        if not paths:
            return SkillInjection(
                header=header, body="", mode=skill.cartridge_mode,
                total_tokens=header_tokens, tier_tokens={"t1": header_tokens},
                queries_used=[], cartridge_hits={},
                gated=False, coverage_confidence=0.0,
                freshness=[],
            )

        # ── Freshness check ─────────────────────────────────────────
        freshness = self._check_freshness(paths)

        budget = skill.cartridge_budget
        has_tier4 = bool(derived_queries)

        # Budget splits
        if has_tier4:
            t2_budget = int(budget * 0.40)
            t3_budget = int(budget * 0.30)
            t4_budget = budget - t2_budget - t3_budget
        else:
            t2_budget = int(budget * 0.40) if skill.cartridge_queries else 0
            t3_budget = budget - t2_budget
            t4_budget = 0

        all_passages: list[dict] = []
        queries_used: list[str] = []
        cartridge_hits: dict[str, int] = {}
        tier_tokens: dict[str, int] = {"t1": header_tokens}
        coverage_confidence = 0.0

        # ── Tier 2: Foundational queries ─────────────────────────────
        t2_passages = []
        for q in skill.cartridge_queries:
            results = self._rt.search_cartridges(paths, q, top_k=3)
            for r in results:
                r["tier"] = "t2"
                t2_passages.append(r)
                cart = r.get("knowledge model", "?")
                cartridge_hits[cart] = cartridge_hits.get(cart, 0) + 1
            queries_used.append(q)

        t2_passages = _dedup_passages(t2_passages)
        t2_text, t2_tokens = self._budget_passages(t2_passages, t2_budget)
        tier_tokens["t2"] = t2_tokens
        all_passages.extend(t2_passages)

        # ── Tier 3: User query ───────────────────────────────────────
        t3_results = self._rt.search_cartridges(paths, user_query, top_k=5)
        for r in t3_results:
            r["tier"] = "t3"
            cart = r.get("knowledge model", "?")
            cartridge_hits[cart] = cartridge_hits.get(cart, 0) + 1
        queries_used.append(user_query)

        # Extract coverage confidence from enriched result if available
        if t3_results:
            # Use top score as proxy for confidence
            top_score = max(r.get("score", 0) for r in t3_results)
            coverage_confidence = min(1.0, top_score)

        t3_deduped = _dedup_passages(t3_results)
        # Remove passages already in Tier 2
        t2_ids = {p.get("source_id") for p in t2_passages}
        t3_deduped = [p for p in t3_deduped if p.get("source_id") not in t2_ids]
        t3_text, t3_tokens = self._budget_passages(t3_deduped, t3_budget)
        tier_tokens["t3"] = t3_tokens
        all_passages.extend(t3_deduped)

        # ── Tier 4: LLM-derived queries ──────────────────────────────
        t4_text = ""
        t4_tokens = 0
        if derived_queries:
            t4_passages = []
            existing_ids = {p.get("source_id") for p in all_passages}
            for q in derived_queries:
                results = self._rt.search_cartridges(paths, q, top_k=3)
                for r in results:
                    if r.get("source_id") not in existing_ids:
                        r["tier"] = "t4"
                        t4_passages.append(r)
                        existing_ids.add(r.get("source_id"))
                        cart = r.get("knowledge model", "?")
                        cartridge_hits[cart] = cartridge_hits.get(cart, 0) + 1
                queries_used.append(q)

            t4_deduped = _dedup_passages(t4_passages)
            t4_text, t4_tokens = self._budget_passages(t4_deduped, t4_budget)
            tier_tokens["t4"] = t4_tokens

        # ── Mode-aware gating ────────────────────────────────────────
        total_dynamic_tokens = t2_tokens + t3_tokens + t4_tokens
        gated = False

        if skill.cartridge_mode == "constrain":
            # Never gate constrain mode -- the whole point is to constrain to sources
            gated = False
        elif skill.cartridge_mode == "knowledge":
            # Soft gate: suppress only if very low energy
            if coverage_confidence < 0.10 and total_dynamic_tokens > 0:
                gated = True
                logger.info("Knowledge mode: gating (confidence %.2f < 0.10)", coverage_confidence)
        else:
            # Augment: full gate
            if coverage_confidence < 0.25 and total_dynamic_tokens > 0:
                gated = True
                logger.info("Augment mode: gating (confidence %.2f < 0.25)", coverage_confidence)

        # ── Assemble body ────────────────────────────────────────────
        if gated:
            body = ""
            tier_tokens = {"t1": header_tokens, "t2": 0, "t3": 0, "t4": 0}
            total_dynamic_tokens = 0
        else:
            parts = []
            preamble = _MODE_PREAMBLES.get(skill.cartridge_mode, "")
            if preamble:
                parts.append(preamble)

            stale = [f for f in freshness if f.status == "stale"]
            if stale:
                warning = "**Freshness warning:** " + ", ".join(f.label() for f in stale)
                parts.append(warning)

            if t2_text:
                parts.append("## Foundational Context\n" + t2_text)
            if t3_text:
                parts.append("## Query-Specific Context\n" + t3_text)
            if t4_text:
                parts.append("## Additional Context\n" + t4_text)

            body = "\n\n".join(parts)

        return SkillInjection(
            header=header,
            body=body,
            mode=skill.cartridge_mode,
            total_tokens=header_tokens + total_dynamic_tokens,
            tier_tokens=tier_tokens,
            queries_used=queries_used,
            cartridge_hits=cartridge_hits,
            gated=gated,
            coverage_confidence=coverage_confidence,
            freshness=freshness,
        )

    @staticmethod
    def _check_freshness(paths: list[Path]) -> list[CartridgeFreshness]:
        """Check freshness of each knowledge model by file mtime."""
        from datetime import datetime

        now = datetime.now(UTC)
        results = []
        for p in paths:
            if not p.exists():
                results.append(CartridgeFreshness(p.stem, 0, "unknown"))
                continue
            mtime = p.stat().st_mtime
            built = datetime.fromtimestamp(mtime, tz=UTC)
            age_hours = (now - built).total_seconds() / 3600
            if age_hours < 24:
                status = "fresh"
            elif age_hours < 72:
                status = "consider rebuilding"
            else:
                status = "stale"
            results.append(CartridgeFreshness(p.stem, age_hours, status))
        return results

    def _budget_passages(
        self,
        passages: list[dict],
        budget_tokens: int,
    ) -> tuple[str, int]:
        """Format passages within a token budget. Returns (text, tokens_used)."""
        if not passages or budget_tokens <= 0:
            return "", 0

        lines = []
        used = 0
        for p in passages:
            formatted = _format_passage(p)
            if not formatted:
                continue
            tokens = _estimate_tokens(formatted)
            if used + tokens > budget_tokens:
                # Truncate this passage to fit
                remaining = budget_tokens - used
                if remaining < 50:
                    break
                formatted = _truncate_to_tokens(formatted, remaining)
                tokens = _estimate_tokens(formatted)
            lines.append(formatted)
            used += tokens
            if used >= budget_tokens:
                break

        return "\n".join(lines), used
