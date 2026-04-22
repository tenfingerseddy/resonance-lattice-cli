# SPDX-License-Identifier: BUSL-1.1
"""Reader protocol + base interface (C1 of the v1.0.0 semantic-layer plan).

This module defines the contract between retrieval output and
synthesized answer output. Concrete readers — local OpenVINO (C2),
Anthropic / OpenAI APIs (C3) — implement `Reader.answer`; everything
above them (`rlat ask` CLI in C4, the HTTP server, MCP integration)
consumes only this abstract surface.

Design contract:

1. **Grounding is load-bearing.** A reader MUST only cite Evidence
   that was passed to it. Fabricating citations (pointing at source
   files that weren't in the evidence set, or offsets outside a span)
   is a correctness bug. If the retrieved evidence doesn't support an
   answer, the reader should say so in `Answer.text` rather than guess.

2. **Readers are replaceable.** The Reader ABC is narrow on purpose —
   `answer(query, evidence) -> Answer`. No hidden state, no coupling
   to Lattice / Store / Registry types. Implementations can be local,
   remote, mocked for tests, or scripted for deterministic runs.

3. **Context-pack is stable and public.** `build_context_pack` is the
   default prompt format; `rlat ask --reader context` returns it
   directly. Readers that need a different format build their own
   internally, but the default exists so callers can bypass the LLM
   step entirely (cheap, deterministic, auditable).

4. **Minimal types.** Evidence, Citation, and Answer are deliberately
   small. Richer shapes (structured citation bundles with token ranges,
   per-span confidence) are C5's job — this module only defines the
   skeleton they'll extend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    """A piece of retrieved evidence ready to feed into a reader.

    The source-of-truth input shape for the reader protocol. Callers
    adapt their richer retrieval result (`MaterialisedResult`, search
    payload dict, composition hit) into this minimal form so readers
    don't depend on the retrieval layer's internals.

    Attributes:
        source_file: Path to the source file the evidence came from.
            Used for citation back-reference; format is caller's choice
            (relative or absolute).
        char_offset: Where the evidence text starts in the source file.
            Used for citations and for the context pack's anchor.
        text: The evidence text itself — what the reader will read.
        score: Optional retrieval score. Higher = more relevant.
            Readers may use this to weight / order evidence internally;
            not required for correctness.
        heading: Optional structural context (section heading, function
            name, etc.). Helps the reader place the evidence without
            having to re-derive it.
    """
    source_file: str
    char_offset: int
    text: str
    score: float = 0.0
    heading: str = ""


@dataclass(frozen=True)
class Citation:
    """Anchored pointer from a synthesized answer back to evidence.

    Citations MUST reference Evidence that was in the reader's input.
    The pair `(source_file, char_offset)` should match an Evidence the
    reader saw; `char_length` lets UIs highlight the specific span
    being cited within the evidence's text.

    Attributes:
        source_file: Source file the citation points at.
        char_offset: Start of the cited span (file-level, not relative
            to the evidence's offset).
        char_length: Length of the cited span in chars.
        quote: Optional verbatim text of the cited span. Carrying the
            quote alongside the offsets lets downstream tools render
            the citation without re-reading the source file.
    """
    source_file: str
    char_offset: int
    char_length: int
    quote: str = ""


@dataclass(frozen=True)
class Answer:
    """A reader's synthesized response to a query.

    `text` is the natural-language answer. `citations` link claims in
    `text` back to source evidence in reading order. `model` identifies
    the reader that produced the answer (e.g. `"openvino-qwen-2.5-3b"`,
    `"anthropic-claude-opus-4-7"`) so logs and benchmarks can track
    which system said what.

    Empty citations are permitted: if evidence was insufficient, a
    reader can return an honest `"I don't have enough information to
    answer"` with `citations=[]` rather than fabricate support.
    """
    query: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    # How many Evidence items were actually used to ground the answer.
    # Useful for benchmarks (did the reader engage the evidence?) and
    # for UI "N sources" badges. Not always the same as len(citations)
    # because one source can be cited multiple times.
    evidence_used: int = 0


class Reader(ABC):
    """Protocol for turning retrieved evidence into a grounded answer.

    Implementations live in sibling modules:

      - `LocalReader` (C2): OpenVINO-backed, on-device inference.
      - `APIReader`   (C3): Anthropic / OpenAI / OpenAI-compatible HTTP.

    Subclasses set `name` so logs and `Answer.model` can identify which
    reader produced a given answer without reflection.
    """

    # Human-readable identifier. Concrete readers override.
    name: str = "reader"

    @abstractmethod
    def answer(
        self, query: str, evidence: Sequence[Evidence],
    ) -> Answer:
        """Synthesize a grounded answer to `query` from `evidence`.

        Implementations MUST:
          * Only cite Evidence passed in — no fabricated citations.
          * State insufficiency honestly when evidence doesn't support
            an answer; do not hallucinate facts.
          * Fill `Answer.model` with a stable identifier.

        Implementations MAY:
          * Return `Answer.citations = []` when no grounded claim
            could be supported.
          * Be stateless or carry config via constructor args.
          * Apply internal reranking / truncation of `evidence` as long
            as the final citations reference what the answer actually
            used.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any acquired resources (model handles, HTTP sessions).

        Default no-op. Subclasses with long-lived resources override.
        The ABC offers this so callers can use `with contextlib.closing(
        reader): ...` uniformly regardless of the backend.
        """
        return None


def build_context_pack(
    query: str, evidence: Sequence[Evidence],
) -> str:
    """Format evidence into a prompt-ready block for any reader.

    Produces a stable, labelled format that downstream consumers can
    reference by index:

        # Query
        <query>

        # Evidence

        ## [1] <source_file> (offset 1234, heading: "Section A")
        <text>

        ## [2] <source_file> (offset 5678)
        <text>

    Readers that want a different structure build their own internally.
    This default exists so `rlat ask --reader context` can return a
    useful artefact without invoking an LLM — cheap, deterministic,
    and inspectable.

    Args:
        query: The user's query. Included verbatim so downstream tools
            that only see the pack can recover intent.
        evidence: Retrieved evidence in presentation order. The index
            labels `[1]`, `[2]` follow this ordering.

    Returns:
        A newline-joined string. Safe for LLM prompts, CLI display,
        and log files.
    """
    lines: list[str] = []
    lines.append("# Query")
    lines.append(query.strip() or "(empty)")
    lines.append("")
    lines.append("# Evidence")

    if not evidence:
        lines.append("")
        lines.append("(no evidence)")
        return "\n".join(lines)

    for i, ev in enumerate(evidence, start=1):
        lines.append("")
        header = f"## [{i}] {ev.source_file or '(unknown)'}"
        bits: list[str] = []
        if ev.char_offset or ev.char_offset == 0:
            # Show offset even when 0 so older cartridges without A3
            # offsets don't look like "no anchor"; 0 is a valid anchor.
            bits.append(f"offset {ev.char_offset}")
        if ev.heading:
            bits.append(f'heading: "{ev.heading}"')
        if bits:
            header += " (" + ", ".join(bits) + ")"
        lines.append(header)
        # Evidence text can be multi-line; include verbatim. Readers
        # that want to trim long evidence do so before calling this.
        lines.append(ev.text.rstrip("\n"))

    return "\n".join(lines)
