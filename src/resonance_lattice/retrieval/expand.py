# SPDX-License-Identifier: BUSL-1.1
"""Adaptive context expansion (B1) + query-time re-chunking (B2) of the
v1.0.0 semantic-layer plan.

The field narrows a query to chunk-sized hits. A chunk is often too small
to stand alone as evidence — a function body without its surrounding
class, a paragraph without its section heading, a sentence that
references an antecedent. `expand_chunk` grows a chunk to the nearest
*natural* boundary so the reader (LLM or human) sees enough context to
resolve those references.

`rechunk` (B2) complements that: where `expand_chunk` takes a `mode`
describing *how far to grow*, `rechunk` takes a `granularity` describing
*what unit of evidence to return* — sentence, passage, or section. The
field retrieval unit stays the same; the response unit is picked at
query time.

Design contract:

1. **Never mutate the input.** The chunk that came from the field is
   what drift detection (A4) fingerprints. Expansion / re-chunking is a
   presentation layer — it returns a new string alongside offsets that
   describe where the original chunk sits within the result.

2. **Source-agnostic core, format-aware boundaries.** The public
   `expand_chunk` function is the single entry point for expansion; it
   dispatches on file extension to pick the right boundary logic
   (markdown headings, Python AST, generic paragraph). Unknown
   extensions fall back to the generic paragraph walker, which works on
   anything text-shaped. `rechunk` shares that dispatch implicitly — its
   `section` granularity delegates to `expand_chunk(mode="natural")`.

3. **Mode hierarchy (expand_chunk).** `off` returns the chunk untouched
   — cheap opt-out. `natural` grows to the smallest enclosing structural
   unit: section / function / paragraph + surrounding. `max` grows
   further to the top-level unit (whole section / outer function /
   broader neighbourhood).

4. **Granularity hierarchy (rechunk).** `sentence` narrows down to the
   sentence at the chunk anchor (below chunk size). `passage` returns
   the single paragraph containing the chunk (roughly chunk-sized, no
   surround). `section` returns the enclosing structural unit (above
   chunk size; reuses `expand_chunk(mode="natural")`).

5. **Bounded output.** `max_chars` caps the result regardless of mode /
   granularity, so a query against a 20 MB markdown file doesn't blow
   the context window.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Public modes. "off" short-circuits. "natural" is the tuned default.
# "max" grows to the largest sensible enclosing unit.
Mode = Literal["off", "natural", "max"]

# Public granularities for `rechunk`. Ordered narrow → wide.
# Chosen to match common IR vocabulary so callers can reason about
# response size without reading the dispatch table.
Granularity = Literal["sentence", "passage", "section"]

# Cap on expanded text size. Chosen to be generous enough to hold a
# whole function or section while staying under a typical LLM context
# window when multiple chunks are combined. Callers can override.
DEFAULT_MAX_CHARS = 8_000


# Markdown heading pattern — same shape the chunker uses so the two
# agree on where section boundaries fall.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ExpandedChunk:
    """Result of expansion.

    Attributes:
        text: The expanded text — what the reader/renderer should consume.
            Equal to the original chunk when `mode="off"` or when no
            expansion opportunity was found.
        char_offset: Offset of `text` in the source file. Always ≤ the
            original chunk's offset.
        original_offset: Where the original chunk sat in the source. The
            offset inside `text` is `original_offset - char_offset`.
        original_length: Character length of the original chunk. Useful
            for highlighting the originally-retrieved span inside the
            expanded context (UI callers often want this).
        expansion_kind: A string tag describing which expansion /
            re-chunk rule fired. One of:
              - "none"       — no expansion applied
              - "section"    — markdown heading-to-heading
              - "function"   — Python AST enclosing def / class
              - "paragraph"  — generic paragraph + surround (expand_chunk)
              - "passage"    — single paragraph at anchor (rechunk)
              - "sentence"   — single sentence at anchor (rechunk)
    """

    text: str
    char_offset: int
    original_offset: int
    original_length: int
    expansion_kind: str


def expand_chunk(
    chunk_text: str,
    char_offset: int,
    source_text: str,
    mode: Mode = "natural",
    source_file: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ExpandedChunk:
    """Grow `chunk_text` to a natural boundary in `source_text`.

    Args:
        chunk_text: The chunk the field returned. Expected to be a
            contiguous substring of `source_text` at `char_offset`, but
            we guard against prepended headings (markdown sub-chunks)
            and similar rewrites.
        char_offset: Where the chunk starts in `source_text`. Coming
            from Chunk.char_offset (A3). Zero is a valid value.
        source_text: Full contents of the source file.
        mode: "off" | "natural" | "max". See module docstring.
        source_file: Optional file path, used only for extension-based
            language dispatch. Empty string → generic text path.
        max_chars: Hard cap on the expanded text length. Expansion stops
            growing once adding the next unit would exceed this.

    Returns:
        ExpandedChunk. `text == chunk_text` for mode="off" or when no
        boundary was found — callers can check `expansion_kind == "none"`.
    """
    # Short-circuit when the caller opts out or the inputs are trivial.
    original_length = len(chunk_text)
    if mode == "off" or not source_text or not chunk_text:
        return ExpandedChunk(
            text=chunk_text,
            char_offset=char_offset,
            original_offset=char_offset,
            original_length=original_length,
            expansion_kind="none",
        )

    ext = Path(source_file).suffix.lower() if source_file else ""

    # Locate the chunk's true span inside source_text. char_offset is
    # authoritative when valid, but some chunkers prepend a heading
    # (markdown sub-chunks) so the chunk_text itself no longer appears
    # verbatim at char_offset. Keep char_offset as the anchor; the
    # expansion only needs the offset, not the text match.
    chunk_start = max(0, min(char_offset, len(source_text)))
    # Use min(original_length, remaining source) rather than len(chunk_text)
    # — safer when the chunk had a prepended heading that isn't in source.
    chunk_end = min(len(source_text), chunk_start + original_length)

    if ext in {".md", ".markdown", ".mdx"}:
        return _expand_markdown(
            chunk_text, chunk_start, chunk_end, source_text,
            mode=mode, max_chars=max_chars,
        )
    if ext == ".py":
        return _expand_python(
            chunk_text, chunk_start, chunk_end, source_text,
            mode=mode, max_chars=max_chars,
        )

    return _expand_generic(
        chunk_text, chunk_start, chunk_end, source_text,
        mode=mode, max_chars=max_chars,
    )


# ─────────────────────────────────────────────────────────────────────
# Markdown expansion
# ─────────────────────────────────────────────────────────────────────


def _expand_markdown(
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    source_text: str,
    *,
    mode: Mode,
    max_chars: int,
) -> ExpandedChunk:
    """Grow to the enclosing section (heading-to-next-heading).

    `natural`: containing section — section heading through the line
    before the next heading at the same or shallower depth.
    `max`: same plus following sibling sections until max_chars runs out.
    """
    # Collect heading positions and their depths so we can find the
    # enclosing section without rescanning per mode.
    headings: list[tuple[int, int]] = []  # (position, depth)
    for m in _HEADING_RE.finditer(source_text):
        headings.append((m.start(), len(m.group(1))))

    if not headings:
        # No headings — fall back to generic paragraph expansion.
        return _expand_generic(
            chunk_text, chunk_start, chunk_end, source_text,
            mode=mode, max_chars=max_chars,
        )

    # Nearest heading *at or before* chunk_start is the section head.
    enclosing_idx = -1
    for i, (pos, _depth) in enumerate(headings):
        if pos <= chunk_start:
            enclosing_idx = i
        else:
            break

    if enclosing_idx < 0:
        # Chunk sits above the first heading — preamble. Grow forward
        # up to the first heading.
        start = 0
        end = headings[0][0]
    else:
        section_start, section_depth = headings[enclosing_idx]
        start = section_start
        # Section ends at the next heading of equal or shallower depth,
        # or EOF. This treats "## A / ### A.1 / ## B" correctly — the
        # section containing A.1 is A, not the file.
        end = len(source_text)
        for pos, depth in headings[enclosing_idx + 1:]:
            if depth <= section_depth:
                end = pos
                break

    # For mode="max", keep absorbing sibling sections forward until we
    # hit the size cap. This is the "give me as much context as you can
    # fit" escape hatch for callers who have a big context window.
    if mode == "max":
        remaining = max_chars - (end - start)
        # Resume from the heading that terminated our section.
        cursor = end
        for pos, _depth in headings:
            if pos < cursor:
                continue
            # Look ahead one more section.
            next_end = len(source_text)
            # Find where this sibling section ends.
            sibling_idx = next(
                (j for j, (p, _) in enumerate(headings) if p == pos), None
            )
            if sibling_idx is not None and sibling_idx + 1 < len(headings):
                next_end = headings[sibling_idx + 1][0]
            added = next_end - pos
            if added > remaining:
                break
            remaining -= added
            cursor = next_end
        end = cursor

    # Apply the size cap — prefer to preserve content before the chunk
    # (the heading context) over content after, since the heading is
    # usually what disambiguates a chunk.
    if end - start > max_chars:
        # Keep the heading and as much of the chunk as fits.
        end = start + max_chars

    text = source_text[start:end].rstrip("\n")
    if not text or start == chunk_start and end == chunk_end:
        return _as_original(chunk_text, chunk_start)

    return ExpandedChunk(
        text=text,
        char_offset=start,
        original_offset=chunk_start,
        original_length=chunk_end - chunk_start,
        expansion_kind="section",
    )


# ─────────────────────────────────────────────────────────────────────
# Python expansion
# ─────────────────────────────────────────────────────────────────────


def _expand_python(
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    source_text: str,
    *,
    mode: Mode,
    max_chars: int,
) -> ExpandedChunk:
    """Grow to the enclosing function / class via AST.

    natural: smallest enclosing def / async def / class.
    max: outermost top-level def / async def / class.
    Unparseable files (syntax errors) fall through to generic expansion.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return _expand_generic(
            chunk_text, chunk_start, chunk_end, source_text,
            mode=mode, max_chars=max_chars,
        )

    # Precompute line-start char offsets so we can convert the chunk
    # position to a line number quickly.
    line_starts = [0]
    for line in source_text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def line_of(offset: int) -> int:
        # 1-indexed to match ast.lineno
        for i in range(len(line_starts) - 1, -1, -1):
            if line_starts[i] <= offset:
                return i + 1
        return 1

    chunk_line = line_of(chunk_start)

    # Walk the AST looking for function/class nodes that contain chunk_line.
    # Track all of them so we can pick the innermost (natural) or outermost
    # (max) later.
    enclosing: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (
                    child.lineno <= chunk_line
                    and (child.end_lineno or child.lineno) >= chunk_line
                ):
                    enclosing.append(child)
                    visit(child)
            else:
                visit(child)

    visit(tree)

    if not enclosing:
        return _expand_generic(
            chunk_text, chunk_start, chunk_end, source_text,
            mode=mode, max_chars=max_chars,
        )

    # natural: innermost containing scope (last visited = deepest).
    # max: outermost top-level (first in the list).
    target = enclosing[-1] if mode == "natural" else enclosing[0]
    start = line_starts[max(0, target.lineno - 1)]
    end_line = target.end_lineno or target.lineno
    end = line_starts[min(len(line_starts) - 1, end_line)]

    if end - start > max_chars:
        end = start + max_chars

    text = source_text[start:end].rstrip("\n")
    if not text:
        return _as_original(chunk_text, chunk_start)

    return ExpandedChunk(
        text=text,
        char_offset=start,
        original_offset=chunk_start,
        original_length=chunk_end - chunk_start,
        expansion_kind="function",
    )


# ─────────────────────────────────────────────────────────────────────
# Generic (paragraph-based) expansion
# ─────────────────────────────────────────────────────────────────────


def _expand_generic(
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    source_text: str,
    *,
    mode: Mode,
    max_chars: int,
) -> ExpandedChunk:
    """Grow by absorbing surrounding paragraphs.

    natural: one paragraph before + one after.
    max: keep adding paragraphs on both sides until max_chars runs out.

    Implementation: enumerate paragraph spans once, locate the chunk's
    paragraph by index, then widen the index range by the requested
    amount. Simpler and less error-prone than walking newline boundaries
    directly (which has fiddly edge cases at file ends and multi-newline
    sequences).
    """
    before_paragraphs = 1 if mode == "natural" else 10
    after_paragraphs = 1 if mode == "natural" else 10

    spans = _paragraph_spans(source_text)
    if not spans:
        return _as_original(chunk_text, chunk_start)

    # Find which paragraph the chunk belongs to. Use the chunk's start
    # offset as the anchor — even if chunk_text crosses a paragraph
    # boundary (rare but possible), we expand relative to where it began.
    chunk_para_idx = None
    for i, (ps, pe) in enumerate(spans):
        if ps <= chunk_start < pe:
            chunk_para_idx = i
            break
        # Chunk start landing exactly on the boundary of the next
        # paragraph also counts as "in" that paragraph.
        if chunk_start == pe and i + 1 < len(spans):
            chunk_para_idx = i + 1
            break
    if chunk_para_idx is None:
        # Chunk start beyond any paragraph span — nothing to expand.
        return _as_original(chunk_text, chunk_start)

    start_idx = max(0, chunk_para_idx - before_paragraphs)
    end_idx = min(len(spans) - 1, chunk_para_idx + after_paragraphs)
    start = spans[start_idx][0]
    end = spans[end_idx][1]

    # Apply size cap, keeping the chunk centred when we have to trim.
    if end - start > max_chars:
        overshoot = (end - start) - max_chars
        trim_before = min(overshoot // 2, chunk_start - start)
        trim_after = overshoot - trim_before
        start += trim_before
        end -= trim_after

    if start >= chunk_start and end <= chunk_end:
        return _as_original(chunk_text, chunk_start)

    text = source_text[start:end].rstrip("\n")
    if not text:
        return _as_original(chunk_text, chunk_start)

    return ExpandedChunk(
        text=text,
        char_offset=start,
        original_offset=chunk_start,
        original_length=chunk_end - chunk_start,
        expansion_kind="paragraph",
    )


def _paragraph_spans(source: str) -> list[tuple[int, int]]:
    """Return (start, end) char ranges for every paragraph in `source`,
    where paragraphs are separated by one or more blank lines (`\\n\\n`).

    The last paragraph has no trailing "\\n\\n" so we emit it explicitly.
    Extra newlines at a boundary are collapsed — they don't create empty
    paragraphs.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(source)
    while start < n:
        end = source.find("\n\n", start)
        if end < 0:
            spans.append((start, n))
            break
        if end > start:
            spans.append((start, end))
        start = end + 2
        while start < n and source[start] == "\n":
            start += 1
    return spans


# ─────────────────────────────────────────────────────────────────────
# B2 — query-time re-chunking
# ─────────────────────────────────────────────────────────────────────


# Sentence end: common prose punctuation followed by whitespace, newline,
# or end-of-string. Kept deliberately simple — `rechunk` is for narrowing
# *within* a chunk we already retrieved, so an edge-case miss just means
# the chunk is served closer to as-is. No Unicode punctuation, no
# abbreviation heuristics; those are cost not yet justified by evidence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])(?:\s+|\n|$)")


def rechunk(
    chunk_text: str,
    char_offset: int,
    source_text: str,
    granularity: Granularity = "passage",
    source_file: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ExpandedChunk:
    """Serve a field-retrieved chunk at the requested granularity.

    Orthogonal to `expand_chunk`: this picks a response *size* rather
    than a boundary *distance*. Use when the caller wants a consistent
    unit of evidence (e.g. one sentence for a tooltip, one passage for
    a snippet, one section for a reader context pack) regardless of
    how the chunker happened to cut the corpus.

    Args:
        chunk_text: The chunk the field returned.
        char_offset: Where the chunk starts in `source_text` (from
            `Chunk.char_offset`). Zero is valid.
        source_text: Full contents of the source file.
        granularity: "sentence" | "passage" | "section". See module
            docstring for the hierarchy.
        source_file: Optional file path, used for extension dispatch
            when `granularity="section"` (which delegates to
            `expand_chunk`).
        max_chars: Hard cap on returned text length. A single sentence
            or passage exceeding the cap is trimmed around the anchor.

    Returns:
        ExpandedChunk tagged with the granularity that fired. Pass-
        through (`expansion_kind="none"`) when inputs are empty or no
        boundary could be located.
    """
    original_length = len(chunk_text)
    if not source_text or not chunk_text:
        return _as_original(chunk_text, char_offset)

    chunk_start = max(0, min(char_offset, len(source_text)))
    chunk_end = min(len(source_text), chunk_start + original_length)

    if granularity == "section":
        # Section granularity is exactly B1's "natural" expansion — the
        # enclosing structural unit. Delegating keeps a single boundary
        # implementation instead of two that can drift apart.
        return expand_chunk(
            chunk_text,
            char_offset,
            source_text,
            mode="natural",
            source_file=source_file,
            max_chars=max_chars,
        )

    if granularity == "sentence":
        return _narrow_to_sentence(
            chunk_text, chunk_start, chunk_end, source_text, max_chars,
        )

    return _serve_passage(
        chunk_text, chunk_start, chunk_end, source_text, max_chars,
    )


def _sentence_spans(source: str) -> list[tuple[int, int]]:
    """Return (start, end) char ranges for sentences in `source`.

    A sentence ends at `[.!?]` followed by whitespace, newline, or EOF.
    Text with no terminator (e.g. a bare identifier) becomes a single
    span spanning the whole input — callers can still anchor into it.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(source):
        end = m.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(source):
        spans.append((start, len(source)))
    return spans


def _narrow_to_sentence(
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    source_text: str,
    max_chars: int,
) -> ExpandedChunk:
    """Return the sentence containing the chunk anchor.

    Anchors on `chunk_start` rather than the chunk's midpoint: the start
    is where the field's attention landed, and it round-trips cleanly
    when the chunk already *is* a single sentence.
    """
    spans = _sentence_spans(source_text)
    if not spans:
        return _as_original(chunk_text, chunk_start)

    target: tuple[int, int] | None = None
    for s, e in spans:
        if s <= chunk_start < e:
            target = (s, e)
            break
    if target is None:
        return _as_original(chunk_text, chunk_start)

    start, end = target
    # Trim leading whitespace/newlines so the sentence reads cleanly
    # when it follows another sentence's terminator.
    while start < end and source_text[start] in " \t\n\r":
        start += 1

    if end - start > max_chars:
        # Oversized sentence — keep the anchor visible rather than
        # blindly truncating the tail.
        if chunk_start - start > max_chars // 2:
            start = max(start, chunk_start - max_chars // 2)
        end = start + max_chars

    text = source_text[start:end].rstrip()
    if not text:
        return _as_original(chunk_text, chunk_start)

    return ExpandedChunk(
        text=text,
        char_offset=start,
        original_offset=chunk_start,
        original_length=chunk_end - chunk_start,
        expansion_kind="sentence",
    )


def _serve_passage(
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    source_text: str,
    max_chars: int,
) -> ExpandedChunk:
    """Return the single paragraph containing the chunk.

    Distinct from `_expand_generic(mode="natural")`, which adds one
    paragraph before and after. Passage granularity deliberately stops
    at paragraph boundaries: callers who ask for a passage want one.
    """
    spans = _paragraph_spans(source_text)
    if not spans:
        return _as_original(chunk_text, chunk_start)

    para: tuple[int, int] | None = None
    for s, e in spans:
        if s <= chunk_start < e:
            para = (s, e)
            break
    if para is None:
        return _as_original(chunk_text, chunk_start)

    start, end = para
    if end - start > max_chars:
        overshoot = (end - start) - max_chars
        trim_before = min(overshoot // 2, chunk_start - start)
        trim_after = overshoot - trim_before
        start += trim_before
        end -= trim_after

    text = source_text[start:end].rstrip("\n")
    if not text:
        return _as_original(chunk_text, chunk_start)

    return ExpandedChunk(
        text=text,
        char_offset=start,
        original_offset=chunk_start,
        original_length=chunk_end - chunk_start,
        expansion_kind="passage",
    )


def _as_original(chunk_text: str, char_offset: int) -> ExpandedChunk:
    """Helper: wrap the original chunk as an ExpandedChunk with kind='none'.
    Used when expansion decided there was nothing to add."""
    return ExpandedChunk(
        text=chunk_text,
        char_offset=char_offset,
        original_offset=char_offset,
        original_length=len(chunk_text),
        expansion_kind="none",
    )
