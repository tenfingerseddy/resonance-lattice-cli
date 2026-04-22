# SPDX-License-Identifier: BUSL-1.1
"""Citation bundle format (C5 of the v1.0.0 semantic-layer plan).

C1's `Citation` is a minimal pointer: source_file + char_offset +
char_length + optional quote. Enough for the reader to produce, not
always enough for a UI or an auditor. This module wraps a reader's
`Answer.citations` in a `CitationBundle` — citations enriched with
line numbers resolved from the source file, plus per-citation
verification that the quote actually appears at the claimed offset.

The verification step is what turns "grounded by prompt" into
"grounded in verifiable bytes": a user (or a downstream tool) can
read the bundle's `verified` flag and know whether the reader
fabricated an offset even within an existing file.

Design contract:

1. **Pure enrichment.** `build_bundle` takes an `Answer` and returns
   a richer shape without mutating the Answer. The reader's output
   is the source of truth; this module adds view-only metadata.

2. **Fail-soft verification.** Missing source files, unreadable
   bytes, or encoding issues mark citations `verified=False` with a
   diagnostic — they don't raise. A partial bundle is more useful
   than no bundle when only some sources are reachable.

3. **JSON-round-trip.** Bundles serialise cleanly via
   `dataclasses.asdict` so `rlat ask --format json` (C4) and future
   HTTP/MCP consumers don't need a custom encoder.

4. **Line numbers are 1-indexed, columns 0-indexed.** Matches editor
   convention so UI tools don't have to translate.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from resonance_lattice.reader.base import Answer, Citation, Evidence


@dataclass(frozen=True)
class EnrichedCitation:
    """A `Citation` enriched with UI-oriented metadata.

    Attributes carried from the base Citation:
        source_file, char_offset, char_length, quote.

    Added:
        line_number: 1-indexed line the span starts on, or None when
            the source file couldn't be read.
        column: 0-indexed column offset within that line, or None.
        verified: True iff the quote (when present) was found at the
            claimed offset in the source. False for any mismatch OR
            when verification couldn't run (see diagnostic).
        diagnostic: Short reason when verified=False. "" when
            verified=True or when verification didn't apply (no quote
            provided to check).
    """
    source_file: str
    char_offset: int
    char_length: int
    quote: str = ""
    line_number: int | None = None
    column: int | None = None
    verified: bool = False
    diagnostic: str = ""


@dataclass(frozen=True)
class CitationBundle:
    """A verifiable bundle for a reader's `Answer`.

    Carries the answer text alongside its enriched citations so
    downstream consumers have a single artefact to inspect, save, or
    ship to a UI without needing the original Answer object.
    """
    query: str
    answer_text: str
    citations: list[EnrichedCitation] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    # Rolled-up verification stats for quick scan without iterating.
    verified_count: int = 0
    total_count: int = 0


def build_bundle(
    answer: Answer,
    evidence: Sequence[Evidence] | None = None,
    *,
    source_root: str | Path | None = None,
    verify: bool = True,
) -> CitationBundle:
    """Wrap an `Answer` as a `CitationBundle` with enriched citations.

    Args:
        answer: The reader's Answer. `citations`, `text`, `model`, and
            `latency_ms` are copied into the bundle.
        evidence: Optional original evidence list. When provided and a
            Citation lacks a `quote`, we fall back to the text of the
            evidence entry at a matching offset — keeps the bundle
            useful even when the reader only emitted offsets.
        source_root: Directory relative-paths resolve against. When
            None, only absolute paths verify; relative citations get
            verified=False with a diagnostic.
        verify: When True (default), read each referenced source file
            and check that the quote matches at the offset. Disable
            for cheap bundle construction when the caller doesn't
            need verification (e.g. upstream already verified).

    Returns:
        A CitationBundle. Always non-None; citations with
        unreadable sources are included with verified=False.
    """
    root = Path(source_root).resolve() if source_root else None

    # Evidence lookup by (source_file, char_offset) for quote fallback.
    ev_by_key: dict[tuple[str, int], Evidence] = {}
    if evidence:
        for ev in evidence:
            ev_by_key[(ev.source_file, ev.char_offset)] = ev

    # Cache file contents during a single bundle build so N citations
    # over the same file only read it once.
    file_cache: dict[Path, str | None] = {}

    enriched: list[EnrichedCitation] = []
    verified_count = 0
    for c in answer.citations:
        quote = c.quote or _fallback_quote(c, ev_by_key)
        resolved_path = _resolve_citation_path(c.source_file, root)

        line_number: int | None = None
        column: int | None = None
        verified = False
        diagnostic = ""

        if resolved_path is None:
            diagnostic = "source file not resolvable"
        else:
            text = _read_file_cached(resolved_path, file_cache)
            if text is None:
                diagnostic = "source file unreadable"
            else:
                line_number, column = _offset_to_line_col(text, c.char_offset)
                if verify:
                    if not quote:
                        # No quote to check — don't falsely claim verified,
                        # but don't falsely claim tampering either.
                        diagnostic = "no quote provided"
                    elif _quote_matches(text, c.char_offset, quote):
                        verified = True
                    else:
                        diagnostic = "quote does not match source at offset"

        if verified:
            verified_count += 1

        enriched.append(EnrichedCitation(
            source_file=c.source_file,
            char_offset=c.char_offset,
            char_length=c.char_length,
            quote=quote,
            line_number=line_number,
            column=column,
            verified=verified,
            diagnostic=diagnostic,
        ))

    return CitationBundle(
        query=answer.query,
        answer_text=answer.text,
        citations=enriched,
        model=answer.model,
        latency_ms=answer.latency_ms,
        verified_count=verified_count,
        total_count=len(enriched),
    )


def bundle_to_dict(bundle: CitationBundle) -> dict:
    """JSON-ready representation of a bundle.

    Just `dataclasses.asdict` under the hood, exposed as a named entry
    point so callers don't need to import `dataclasses` themselves.
    """
    return dataclasses.asdict(bundle)


# ─────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────


def _resolve_citation_path(source_file: str, root: Path | None) -> Path | None:
    """Best-effort file location. Accepts mixed separators since A2
    normalises manifests to forward slashes regardless of build OS."""
    if not source_file:
        return None
    p = Path(source_file)
    if p.is_absolute():
        return p if p.exists() else None
    if root is None:
        return None
    cand = (root / source_file)
    if cand.exists():
        return cand.resolve()
    # Separator normalisation fallback (Windows ↔ POSIX cartridges).
    norm = source_file.replace("\\", "/")
    cand = (root / Path(*norm.split("/")))
    return cand.resolve() if cand.exists() else None


def _read_file_cached(
    path: Path, cache: dict[Path, str | None],
) -> str | None:
    """Read file text once per bundle build. Returns None on any read error."""
    if path in cache:
        return cache[path]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = None
    cache[path] = text
    return text


def _offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    """Convert a character offset to (1-indexed line, 0-indexed column).

    Clamped to the string's bounds so a pathological offset beyond EOF
    returns the last line's coordinates rather than raising.
    """
    if offset <= 0:
        return 1, 0
    if offset > len(text):
        offset = len(text)
    # Walk newlines up to the offset. `count("\n", 0, offset)` is fast
    # and gives 0-indexed line count; add 1 for 1-indexed.
    line_idx = text.count("\n", 0, offset)
    # Column = chars since the last newline before the offset.
    last_nl = text.rfind("\n", 0, offset)
    column = offset - (last_nl + 1) if last_nl >= 0 else offset
    return line_idx + 1, column


def _quote_matches(text: str, offset: int, quote: str) -> bool:
    """True when `quote` appears in `text` starting at `offset`.

    Lenient comparison: trailing whitespace-only differences are
    allowed (readers sometimes trim; drift-detection here shouldn't
    false-positive on newline handling). Leading characters must match
    exactly — that's the anchor.
    """
    if not quote:
        return False
    if offset < 0 or offset > len(text):
        return False
    end = offset + len(quote)
    window = text[offset:end]
    if window == quote:
        return True
    # Allow trailing whitespace difference (rstrip both sides).
    return window.rstrip() == quote.rstrip()


def _fallback_quote(
    citation: Citation,
    ev_by_key: dict[tuple[str, int], Evidence],
) -> str:
    """Recover a quote from the original evidence when the Citation
    didn't carry one. Keeps bundles useful even for readers that emit
    bare offsets."""
    ev = ev_by_key.get((citation.source_file, citation.char_offset))
    if ev is None:
        return ""
    # Trim to the citation's declared length when known, else use the
    # evidence's full text — the bundle's offsets still point at the
    # exact span so UI rendering has all the info it needs.
    if citation.char_length > 0:
        return ev.text[: citation.char_length]
    return ev.text
