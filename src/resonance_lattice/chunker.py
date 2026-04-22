# SPDX-License-Identifier: BUSL-1.1
"""Content-aware chunking for Resonance Lattice ingestion.

Replaces naive text.split("\\n\\n") with structure-aware chunking that
respects document boundaries (markdown headings, code blocks, tables).
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Length of the hex-encoded sha256 prefix we use as a per-chunk content
# fingerprint (see Chunk.content_hash). 32 hex chars = 128 bits, which is
# collision-safe for any realistic corpus while keeping the per-chunk
# storage overhead in the manifest to ~35 bytes including the JSON key.
CONTENT_HASH_HEX_LEN = 32


@dataclass
class Chunk:
    """A single chunk with metadata.

    `char_offset` records where this chunk starts in the original source
    text. Leave at 0 when the chunker cannot determine a meaningful
    position (synthetic chunks, composite sources, etc.).

    `content_hash` is derived lazily from `text` — it's a fingerprint of
    what we indexed, used downstream for drift detection in external-store
    knowledge models (A4+).
    """

    text: str
    source_file: str = ""
    heading: str = ""
    chunk_type: str = "doc"  # "doc", "source", "table", "code_block", "conversation"
    char_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """sha256 fingerprint of the chunk text, truncated to
        CONTENT_HASH_HEX_LEN hex chars. Deterministic across runs for the
        same text. Recompute is cheap (sha256 is ~2 GB/s) so we don't
        cache — keeps the dataclass pickle/JSON-serialisable as-is."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:CONTENT_HASH_HEX_LEN]


def contextualize_chunk(
    chunk: Chunk,
    total_chunks: int = 1,
    chunk_index: int = 0,
    enabled: bool | None = None,
) -> str:
    """Prepend structural context to a chunk's text before encoding.

    Contextual chunking (per Anthropic's Contextual Retrieval paper) prepends
    file path, section heading, and chunk position so the encoder captures
    WHERE the chunk came from, not just what it says. A chunk that says
    "This feature supports..." embeds differently when the encoder sees
    "[auth.md > JWT Tokens]" vs "[database.md > Connection Pooling]".

    Returns the contextualized text string. The original chunk.text is not
    modified — call this at encoding time, not at chunking time, so the
    stored full_text remains clean for display.

    Corpus-aware default (enabled=None): skip prefix for conversation chunks,
    whose source stems are hash-like session IDs (e.g. "sharegpt_yywfIrx_0")
    that add noise rather than semantic context. LongMemEval v6 benchmark
    showed raw chunking beats contextual on conversation data (R@1 0.701 vs
    0.639). Explicit True/False overrides the corpus-aware decision.
    """
    if enabled is False:
        return chunk.text
    if enabled is None and chunk.chunk_type == "conversation":
        return chunk.text

    parts = []

    # File stem (not full path — avoids leaking local paths into embeddings)
    if chunk.source_file:
        stem = Path(chunk.source_file).stem
        parts.append(stem)

    # Section heading
    if chunk.heading:
        parts.append(chunk.heading)

    # Chunk position (only if multi-chunk document)
    if total_chunks > 1:
        parts.append(f"chunk {chunk_index + 1}/{total_chunks}")

    if not parts:
        return chunk.text

    context = " > ".join(parts)
    return f"{context}: {chunk.text}"


# ── Markdown chunker ──────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_IMAGE_TAG_RE = re.compile(r":::image.*?:::", re.DOTALL)
_INCLUDE_RE = re.compile(r"\[!INCLUDE\s+\[.*?\]\(.*?\)\]")


def _normalise_newlines(text: str) -> str:
    """Normalize CRLF/CR newlines to LF for chunking and regex cleanup."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_noise(text: str) -> str:
    """Remove YAML frontmatter, image tags, and include directives."""
    text = _normalise_newlines(text)
    # Strip YAML frontmatter
    text = _FRONTMATTER_RE.sub("", text)
    # Strip :::image ... ::: blocks
    text = _IMAGE_TAG_RE.sub("", text)
    # Strip [!INCLUDE ...] directives
    text = _INCLUDE_RE.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_inside_fence(text: str, pos: int) -> bool:
    """Check if a position in text is inside a fenced code block."""
    fences_before = len(_FENCE_RE.findall(text[:pos]))
    return fences_before % 2 == 1


def _trailing_overlap(text: str, overlap_chars: int) -> str:
    """Return the trailing ~overlap_chars of text, snapped to a paragraph
    boundary inside that window when possible, else a word boundary."""
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if overlap_chars > 0 else ""
    tail = text[-overlap_chars:]
    para_boundary = tail.find("\n\n")
    if para_boundary >= 0:
        return tail[para_boundary + 2:]
    space = tail.find(" ")
    if 0 <= space < overlap_chars // 2:
        return tail[space + 1:]
    return tail


def _split_respecting_blocks(
    text: str, max_chars: int, overlap_chars: int = 0,
) -> list[str]:
    """Split text on paragraph boundaries, never inside code blocks or tables.

    When `overlap_chars > 0`, each new chunk is seeded with the trailing
    ~overlap_chars of the previous chunk (paragraph-boundary-aligned when
    possible) so adjacent chunks share context at their boundary.
    """
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            parts.append(current)
            if overlap_chars > 0:
                seed = _trailing_overlap(current, overlap_chars)
                current = f"{seed}\n\n{paragraph}".strip() if seed else paragraph
            else:
                current = paragraph

    if current:
        parts.append(current)

    # If we still have oversized parts (no paragraph boundaries), hard-split
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            # Split at word or char boundary
            while len(part) > max_chars:
                split_at = part[:max_chars].rfind(" ")
                if split_at <= 0:
                    split_at = max_chars
                final.append(part[:split_at])
                part = part[split_at:].lstrip()
            if part:
                final.append(part)

    return final


def _merge_small_markdown_sections(
    sections: list[tuple[str, str, int]],
    min_chars: int,
) -> list[tuple[str, str, int]]:
    """Merge undersized markdown sections into adjacent sections.

    When small sections merge into their predecessor the predecessor's
    `start_offset` is preserved — the new merged chunk effectively starts
    where the first contributing section started. A pending prefix that
    gets merged forward keeps its own start offset.
    """
    merged: list[tuple[str, str, int]] = []
    pending_prefix = ""
    pending_offset = 0

    for idx, (heading_text, body, start_offset) in enumerate(sections):
        body = body.strip()
        if pending_prefix:
            body = f"{pending_prefix}\n\n{body}".strip()
            effective_offset = pending_offset
            pending_prefix = ""
        else:
            effective_offset = start_offset

        is_last = idx == len(sections) - 1
        if len(body) >= min_chars or (not merged and is_last):
            merged.append((heading_text, body, effective_offset))
            continue

        if merged:
            prev_heading, prev_body, prev_offset = merged[-1]
            # Appending a tail into the previous section — previous start
            # offset is still correct.
            merged[-1] = (prev_heading, f"{prev_body}\n\n{body}".strip(), prev_offset)
            continue

        pending_prefix = body
        pending_offset = effective_offset

    if pending_prefix:
        if merged:
            prev_heading, prev_body, prev_offset = merged[-1]
            merged[-1] = (prev_heading, f"{prev_body}\n\n{pending_prefix}".strip(), prev_offset)
        else:
            merged.append(("", pending_prefix, pending_offset))

    return merged


def chunk_markdown(
    text: str,
    source_file: str = "",
    max_chars: int = 1200,
    min_chars: int = 150,
    overlap_chars: int = 0,
) -> list[Chunk]:
    """Chunk markdown text on heading boundaries.

    Strategy:
      1. Strip frontmatter, image tags, and include directives
      2. Find all heading positions
      3. Split into sections at heading boundaries
      4. Sections under max_chars become one chunk with heading prepended
      5. Oversized sections split on paragraph boundaries (with optional overlap)
      6. Code blocks and tables are never split mid-block
    """
    if min_chars >= max_chars:
        raise ValueError(
            f"min_chars ({min_chars}) must be less than max_chars ({max_chars})"
        )

    text = _strip_noise(text)

    if not text.strip():
        return []

    # Find all headings and their positions
    headings: list[tuple[int, int, str]] = []  # (pos, level, text)
    for m in _HEADING_RE.finditer(text):
        if not _is_inside_fence(text, m.start()):
            headings.append((m.start(), len(m.group(1)), m.group(2).strip()))

    if not headings:
        # No headings — fall back to paragraph chunking
        return chunk_text(text, source_file, max_chars, min_chars, overlap_chars=overlap_chars)

    # Split into sections at heading boundaries. Track the start offset
    # of each section in the original `text` so downstream consumers can
    # seek into the source file for evidence retrieval.
    sections: list[tuple[str, str, int]] = []  # (heading_text, section_body, start_offset)
    for i, (pos, _level, heading_text) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = text[pos:end].strip()
        sections.append((heading_text, body, pos))

    # Content before the first heading
    preamble = text[: headings[0][0]].strip()
    if preamble:
        sections.insert(0, ("", preamble, 0))

    sections = _merge_small_markdown_sections(sections, min_chars)

    chunks: list[Chunk] = []
    for heading_text, body, section_offset in sections:
        if len(body) < min_chars and len(sections) > 1:
            continue

        if len(body) <= max_chars:
            chunks.append(Chunk(
                text=body,
                source_file=source_file,
                heading=heading_text,
                chunk_type="doc",
                char_offset=section_offset,
            ))
        else:
            # Split oversized section on paragraph boundaries. We need
            # each sub-chunk's offset within the original text; walk the
            # section body and use `str.find` from the previous end so
            # duplicate paragraphs resolve to the correct occurrence.
            sub_parts = _split_respecting_blocks(body, max_chars, overlap_chars=overlap_chars)
            cursor_in_body = 0
            for part in sub_parts:
                stripped = part.strip()
                if len(stripped) < min_chars:
                    continue
                # Locate where `stripped` sits inside `body` (starting at
                # the cursor so duplicate paragraphs advance monotonically).
                found = body.find(stripped, cursor_in_body)
                if found < 0:
                    # `_split_respecting_blocks` can emit joined text that
                    # doesn't appear verbatim in body; fall back to the
                    # section start in that rare case.
                    part_offset = section_offset
                else:
                    part_offset = section_offset + found
                    cursor_in_body = found + len(stripped)
                # Prepend heading to sub-chunks for context. This mutates
                # `stripped` but leaves char_offset pointing at the
                # original bytes (without the prepended heading) — that's
                # what drift detection wants.
                if heading_text and not stripped.startswith("#"):
                    stripped = f"## {heading_text}\n\n{stripped}"
                chunks.append(Chunk(
                    text=stripped,
                    source_file=source_file,
                    heading=heading_text,
                    chunk_type="doc",
                    char_offset=part_offset,
                ))

    return chunks


# ── Python source chunker ─────────────────────────────────────────────


def chunk_python(
    text: str,
    source_file: str = "",
    max_chars: int = 1000,
    min_chars: int = 80,
) -> list[Chunk]:
    """Chunk Python source using AST boundaries (function/class).

    Falls back to paragraph chunking on syntax errors.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(text, source_file, max_chars, min_chars)

    lines = text.splitlines(keepends=True)
    # Precompute the cumulative char offset of each line start, so we can
    # convert AST line numbers to char positions in the original `text`
    # without re-scanning for every node.
    line_starts: list[int] = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    def _line_to_offset(line_0: int) -> int:
        """0-indexed line number → char offset in `text`. Out-of-range
        lines clamp to the end so we never return a negative offset."""
        if line_0 < 0:
            return 0
        if line_0 >= len(line_starts):
            return line_starts[-1]
        return line_starts[line_0]

    chunks: list[Chunk] = []
    covered: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno if node.end_lineno else start + 1
        segment = "".join(lines[start:end]).strip()
        covered.update(range(start, end))

        if len(segment) < min_chars:
            continue

        node_offset = _line_to_offset(start)

        if len(segment) <= max_chars:
            chunks.append(Chunk(
                text=segment,
                source_file=source_file,
                heading=node.name,
                chunk_type="source",
                char_offset=node_offset,
            ))
        else:
            # Class too large — try per-method
            if isinstance(node, ast.ClassDef):
                class_chunks = _split_class(node, lines, source_file, max_chars, min_chars,
                                            line_starts=line_starts)
                chunks.extend(class_chunks)
            else:
                # Large function — keep as-is (truncation handled downstream)
                chunks.append(Chunk(
                    text=segment,
                    source_file=source_file,
                    heading=node.name,
                    chunk_type="source",
                    char_offset=node_offset,
                ))

    # Module-level code not covered by any function/class. Its offset is
    # the first uncovered line start — the remaining content is noncontiguous
    # in general (scattered imports, decorators, docstrings) so the offset
    # points at the first fragment, which is the most useful anchor.
    module_lines = []
    first_module_line: int | None = None
    for i, line in enumerate(lines):
        if i not in covered:
            if first_module_line is None and line.strip():
                first_module_line = i
            module_lines.append(line)

    module_text = "".join(module_lines).strip()
    module_offset = _line_to_offset(first_module_line) if first_module_line is not None else 0
    if module_text and len(module_text) >= min_chars:
        # Split module-level code if too large
        for part in _split_respecting_blocks(module_text, max_chars):
            part = part.strip()
            if len(part) >= min_chars:
                chunks.append(Chunk(
                    text=part,
                    source_file=source_file,
                    heading="module-level",
                    chunk_type="source",
                    char_offset=module_offset,
                ))

    return chunks


def _split_class(
    node: ast.ClassDef,
    lines: list[str],
    source_file: str,
    max_chars: int,
    min_chars: int,
    line_starts: list[int] | None = None,
) -> list[Chunk]:
    """Split a large class into per-method chunks with class name prefix.

    char_offset on each method chunk points at the method's first line in
    the source file (ignoring the synthetic `class X:\\n    ...\\n\\n`
    prefix added to the chunk text for encoder context). If `line_starts`
    isn't provided we recompute it from `lines` — the caller in
    `chunk_python` hands it in to avoid redundant work.
    """
    chunks: list[Chunk] = []
    class_name = node.name

    if line_starts is None:
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))

    def _offset(line_0: int) -> int:
        if line_0 < 0:
            return 0
        if line_0 >= len(line_starts):
            return line_starts[-1]
        return line_starts[line_0]

    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = child.lineno - 1
        end = child.end_lineno if child.end_lineno else start + 1
        segment = "".join(lines[start:end]).strip()
        if len(segment) < min_chars:
            continue
        chunks.append(Chunk(
            text=f"class {class_name}:\n    ...\n\n{segment}",
            source_file=source_file,
            heading=f"{class_name}.{child.name}",
            chunk_type="source",
            char_offset=_offset(start),
        ))

    # If no methods found, return the whole class as one chunk
    if not chunks:
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else start + 1
        segment = "".join(lines[start:end]).strip()
        if len(segment) >= min_chars:
            chunks.append(Chunk(
                text=segment,
                source_file=source_file,
                heading=class_name,
                chunk_type="source",
                char_offset=_offset(start),
            ))

    return chunks


# ── Tree-sitter chunker (multi-language AST) ─────────────────────────

_TREESITTER_AVAILABLE: bool | None = None


def _has_treesitter() -> bool:
    """Lazy check for tree-sitter availability."""
    global _TREESITTER_AVAILABLE
    if _TREESITTER_AVAILABLE is None:
        try:
            from tree_sitter import Language, Parser  # noqa: F401
            _TREESITTER_AVAILABLE = True
        except ImportError:
            _TREESITTER_AVAILABLE = False
    return _TREESITTER_AVAILABLE


# Language → (module, language_func, top-level node types to extract)
_TS_LANG_CONFIG: dict[str, tuple[str, str, list[str]]] = {
    ".go": (
        "tree_sitter_go", "language",
        ["function_declaration", "method_declaration", "type_declaration"],
    ),
    ".rs": (
        "tree_sitter_rust", "language",
        ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
    ),
    ".ts": (
        "tree_sitter_typescript", "language_typescript",
        ["function_declaration", "class_declaration", "interface_declaration",
         "type_alias_declaration", "export_statement"],
    ),
    ".tsx": (
        "tree_sitter_typescript", "language_tsx",
        ["function_declaration", "class_declaration", "interface_declaration",
         "type_alias_declaration", "export_statement"],
    ),
    ".js": (
        "tree_sitter_typescript", "language_typescript",
        ["function_declaration", "class_declaration", "export_statement"],
    ),
    ".jsx": (
        "tree_sitter_typescript", "language_tsx",
        ["function_declaration", "class_declaration", "export_statement"],
    ),
    ".java": (
        "tree_sitter_java", "language",
        ["class_declaration", "interface_declaration", "enum_declaration",
         "method_declaration"],
    ),
}


def _get_node_name(node) -> str:
    """Extract the name of an AST node across languages."""
    # Direct name field (Go func, Rust fn/struct, Java class, TS function)
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    # For export_statement, look for the inner declaration
    if node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration", "class_declaration",
                "interface_declaration", "type_alias_declaration",
                "lexical_declaration",
            ):
                inner_name = child.child_by_field_name("name")
                if inner_name:
                    return inner_name.text.decode("utf-8", errors="replace")
    # For impl blocks (Rust), extract the type name
    if node.type == "impl_item":
        for child in node.children:
            if child.type == "type_identifier":
                return f"impl {child.text.decode('utf-8', errors='replace')}"
    return ""


def _get_preceding_doc_comment(source_bytes: bytes, node) -> str:
    """Extract doc comment text immediately before a node."""
    start_byte = node.start_byte
    if start_byte == 0:
        return ""
    # Look at the ~500 bytes before the node for comments
    search_start = max(0, start_byte - 500)
    preceding = source_bytes[search_start:start_byte].decode("utf-8", errors="replace")
    lines = preceding.rstrip().split("\n")

    doc_lines = []
    # Walk backward from the last line collecting doc comments
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("///"):
            doc_lines.insert(0, stripped[3:].strip())
        elif stripped.startswith("//"):
            doc_lines.insert(0, stripped[2:].strip())
        elif stripped.startswith("*") and not stripped.startswith("*/"):
            doc_lines.insert(0, stripped.lstrip("* ").strip())
        elif stripped.startswith("/**"):
            doc_lines.insert(0, stripped[3:].strip())
            break
        elif stripped == "" and doc_lines:
            break  # gap between comment and node
        else:
            break  # non-comment line

    return " ".join(l for l in doc_lines if l)[:200]


def chunk_treesitter(
    text: str,
    source_file: str = "",
    max_chars: int = 1000,
    min_chars: int = 80,
) -> list[Chunk]:
    """Chunk source code using tree-sitter AST boundaries.

    Falls back to an empty list on parse failure (caller should fall back
    to chunk_text).
    """
    if not _has_treesitter():
        return []

    import importlib

    from tree_sitter import Language, Parser

    ext = Path(source_file).suffix.lower() if source_file else ""
    lang_config = _TS_LANG_CONFIG.get(ext)
    if not lang_config:
        return []

    module_name, lang_func_name, target_types = lang_config

    try:
        mod = importlib.import_module(module_name)
        lang_func = getattr(mod, lang_func_name)
        language = Language(lang_func())
        parser = Parser(language)
    except Exception:
        return []

    source_bytes = text.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    root = tree.root_node
    chunks: list[Chunk] = []
    covered: set[int] = set()  # byte ranges covered
    filename = Path(source_file).name if source_file else ""

    def _extract_node(node, depth: int = 0):
        """Recursively extract top-level declarations."""
        if depth > 2:
            return
        if node.type in target_types:
            start = node.start_byte
            end = node.end_byte
            segment = source_bytes[start:end].decode("utf-8", errors="replace").strip()

            if len(segment) < min_chars:
                return

            name = _get_node_name(node)

            # Prepend file context
            prefix = f"// file: {filename}\n" if filename else ""

            if len(segment) <= max_chars:
                chunks.append(Chunk(
                    text=prefix + segment,
                    source_file=source_file,
                    heading=name,
                    chunk_type="source",
                ))
                covered.update(range(start, end))
            else:
                # Large node — try splitting children (e.g. methods in a class/impl)
                # For classes, methods live inside body nodes (class_body,
                # declaration_list, block) rather than as direct children.
                _BODY_TYPES = {
                    "class_body", "declaration_list", "block",
                    "statement_block", "interface_body", "enum_body",
                    "impl_item",
                }
                candidates = list(node.children)
                # Recurse one level into body nodes to find methods
                for child in node.children:
                    if child.type in _BODY_TYPES:
                        candidates.extend(child.children)

                child_chunks = []
                for child in candidates:
                    if child.type in target_types or child.type in (
                        "function_item", "method_declaration",
                        "function_declaration", "method_definition",
                    ):
                        c_start = child.start_byte
                        c_end = child.end_byte
                        c_seg = source_bytes[c_start:c_end].decode(
                            "utf-8", errors="replace"
                        ).strip()
                        if len(c_seg) >= min_chars:
                            c_name = _get_node_name(child)
                            child_heading = f"{name}.{c_name}" if name and c_name else (c_name or name)
                            child_chunks.append(Chunk(
                                text=prefix + c_seg,
                                source_file=source_file,
                                heading=child_heading,
                                chunk_type="source",
                            ))
                            covered.update(range(c_start, c_end))

                if child_chunks:
                    chunks.extend(child_chunks)
                else:
                    # Can't split — keep the whole node truncated
                    chunks.append(Chunk(
                        text=prefix + segment[:max_chars],
                        source_file=source_file,
                        heading=name,
                        chunk_type="source",
                    ))
                    covered.update(range(start, end))
        else:
            # Recurse into children (e.g. export_statement wrapping a function)
            for child in node.children:
                _extract_node(child, depth + 1)

    for child in root.children:
        _extract_node(child)

    # Module-level code not covered by any extracted node
    uncovered_bytes = bytearray()
    for i in range(len(source_bytes)):
        if i not in covered:
            uncovered_bytes.append(source_bytes[i])

    module_text = bytes(uncovered_bytes).decode("utf-8", errors="replace").strip()
    if module_text and len(module_text) >= min_chars:
        prefix = f"// file: {filename}\n" if filename else ""
        for part in _split_respecting_blocks(module_text, max_chars):
            part = part.strip()
            if len(part) >= min_chars:
                chunks.append(Chunk(
                    text=prefix + part,
                    source_file=source_file,
                    heading="module-level",
                    chunk_type="source",
                ))

    return chunks


# ── Plain text chunker (improved fallback) ────────────────────────────


def chunk_text(
    text: str,
    source_file: str = "",
    max_chars: int = 800,
    min_chars: int = 80,
    overlap_chars: int = 0,
) -> list[Chunk]:
    """Improved paragraph-based chunking with merging and splitting.

    char_offset is propagated by walking `text` with `str.find` — each
    emitted chunk's offset is the position of its text in the original
    input. `find` is anchored at the previous match end so duplicate
    paragraphs resolve to the correct occurrence.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge short consecutive paragraphs
    merged: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = para
    if current:
        merged.append(current)

    # Split oversized chunks and filter by min_chars
    chunks: list[Chunk] = []
    cursor = 0

    def _locate(substr: str) -> int:
        """Position of `substr` in `text`, starting from cursor. Returns
        cursor (not -1) on miss so the chunk still gets a monotonic
        best-effort offset."""
        nonlocal cursor
        # Match the first significant prefix of the candidate — substr may
        # have been rewritten (stripped, overlap-seeded) and not appear
        # verbatim in `text`, so anchor on a leading slice that's likely
        # present.
        anchor = substr[:40] if len(substr) > 40 else substr
        found = text.find(anchor, cursor)
        if found < 0:
            return cursor
        cursor = found + len(anchor)
        return found

    for block in merged:
        if len(block) < min_chars:
            continue
        if len(block) <= max_chars:
            offset = _locate(block)
            chunks.append(Chunk(
                text=block,
                source_file=source_file,
                chunk_type="doc",
                char_offset=offset,
            ))
        else:
            # Split on sentence boundaries within the block
            for part in _split_respecting_blocks(block, max_chars, overlap_chars=overlap_chars):
                part = part.strip()
                if len(part) >= min_chars:
                    offset = _locate(part)
                    chunks.append(Chunk(
                        text=part,
                        source_file=source_file,
                        chunk_type="doc",
                        char_offset=offset,
                    ))

    return chunks


# ── Conversation chunker ─────────────────────────────────────────────

# Turn boundary patterns (heading-style and bold-style)
_TURN_HEADING_RE = re.compile(
    r"^#{1,3}\s+(Human|User|Assistant|Claude|System)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_TURN_BOLD_RE = re.compile(
    r"^\*\*(Human|User|Assistant|Claude|System)\s*:\*\*",
    re.MULTILINE | re.IGNORECASE,
)
# Timestamp patterns in frontmatter or inline
_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?)"
)

# Roles that count as "human" vs "assistant"
_HUMAN_ROLES = {"human", "user"}
_ASSISTANT_ROLES = {"assistant", "claude"}
_SYSTEM_ROLES = {"system"}


def _normalise_role(raw: str) -> str:
    """Normalise a speaker label to human/assistant/system."""
    lower = raw.strip().lower()
    if lower in _HUMAN_ROLES:
        return "human"
    if lower in _ASSISTANT_ROLES:
        return "assistant"
    if lower in _SYSTEM_ROLES:
        return "system"
    return lower


def _extract_frontmatter_timestamp(text: str) -> str:
    """Extract a timestamp from YAML frontmatter if present."""
    fm = _FRONTMATTER_RE.match(text)
    if fm:
        m = _TIMESTAMP_RE.search(fm.group(0))
        if m:
            return m.group(1)
    return ""


def _detect_conversation(text: str) -> bool:
    """Detect if text is a conversation based on turn markers in the first ~2000 chars."""
    sample = text[:2000]
    heading_matches = _TURN_HEADING_RE.findall(sample)
    bold_matches = _TURN_BOLD_RE.findall(sample)
    # Need at least 2 turn markers to be confident
    return len(heading_matches) + len(bold_matches) >= 2


def _parse_turns(text: str) -> list[tuple[str, str]]:
    """Parse conversation text into (role, content) turns.

    Supports heading-style (## Human) and bold-style (**Human:**) markers.
    """
    # Find all turn boundaries with their positions
    boundaries: list[tuple[int, str]] = []

    for m in _TURN_HEADING_RE.finditer(text):
        if not _is_inside_fence(text, m.start()):
            boundaries.append((m.start(), m.group(1)))

    for m in _TURN_BOLD_RE.finditer(text):
        if not _is_inside_fence(text, m.start()):
            boundaries.append((m.start(), m.group(1)))

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    if not boundaries:
        return []

    turns: list[tuple[str, str]] = []
    for i, (pos, role) in enumerate(boundaries):
        # Content starts after the marker line
        marker_end = text.index("\n", pos) + 1 if "\n" in text[pos:] else len(text)
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content = text[marker_end:end].strip()
        if content:
            turns.append((_normalise_role(role), content))

    return turns


def chunk_conversation(
    text: str,
    source_file: str = "",
    max_chars: int = 2400,
    min_chars: int = 100,
    session_id: str = "",
    timestamp: str = "",
) -> list[Chunk]:
    """Chunk conversation markdown into Q+A pairs.

    Strategy:
      1. Parse turn boundaries (heading or bold markers)
      2. Group into Q+A pairs (human + following assistant response)
      3. If a pair exceeds max_chars, split assistant response at paragraph boundaries
      4. Each chunk gets conversation metadata (speaker, turn_index, session_id)
    """
    text = _normalise_newlines(text)

    # Extract timestamp from frontmatter if not provided
    if not timestamp:
        timestamp = _extract_frontmatter_timestamp(text)

    # Strip frontmatter for parsing but keep conversation content
    text_body = _FRONTMATTER_RE.sub("", text).strip()

    if not text_body:
        return []

    turns = _parse_turns(text_body)

    if not turns:
        # Not a conversation — fall back to markdown chunker
        return chunk_markdown(text_body, source_file, max_chars=max_chars, min_chars=min_chars)

    # Derive session_id from filename if not provided
    if not session_id and source_file:
        session_id = Path(source_file).stem

    # Group turns into Q+A pairs
    chunks: list[Chunk] = []
    turn_idx = 0
    i = 0

    while i < len(turns):
        role, content = turns[i]

        if role == "human" and i + 1 < len(turns) and turns[i + 1][0] == "assistant":
            # Q+A pair: keep together
            q_content = content
            a_role, a_content = turns[i + 1]
            pair_text = f"**Human:**\n{q_content}\n\n**Assistant:**\n{a_content}"

            pair_meta = {
                "speaker": "qa_pair",
                "turn_index": turn_idx,
                "session_id": session_id,
                "timestamp": timestamp,
                "chunk_type": "conversation",
            }

            if len(pair_text) <= max_chars:
                chunks.append(Chunk(
                    text=pair_text,
                    source_file=source_file,
                    heading=_first_sentences(q_content, 60).rstrip(".!? "),
                    chunk_type="conversation",
                    metadata=pair_meta,
                ))
            else:
                # Split: keep human question as context prefix, split assistant response
                q_prefix = f"**Human:**\n{q_content}\n\n**Assistant:**\n"
                available = max_chars - len(q_prefix)
                if available < min_chars:
                    # Question itself is huge — chunk separately
                    chunks.append(Chunk(
                        text=f"**Human:**\n{q_content}",
                        source_file=source_file,
                        heading=_first_sentences(q_content, 60).rstrip(".!? "),
                        chunk_type="conversation",
                        metadata={**pair_meta, "speaker": "human"},
                    ))
                    for part in _split_respecting_blocks(a_content, max_chars - 20):
                        part = part.strip()
                        if len(part) >= min_chars:
                            chunks.append(Chunk(
                                text=f"**Assistant:**\n{part}",
                                source_file=source_file,
                                heading=_first_sentences(q_content, 60).rstrip(".!? "),
                                chunk_type="conversation",
                                metadata={**pair_meta, "speaker": "assistant"},
                            ))
                else:
                    parts = _split_respecting_blocks(a_content, available)
                    for j, part in enumerate(parts):
                        part = part.strip()
                        if len(part) < min_chars and len(parts) > 1:
                            continue
                        chunk_text_val = q_prefix + part if j == 0 else f"**Assistant:** (continued)\n{part}"
                        chunks.append(Chunk(
                            text=chunk_text_val,
                            source_file=source_file,
                            heading=_first_sentences(q_content, 60).rstrip(".!? "),
                            chunk_type="conversation",
                            metadata={**pair_meta, "speaker": "qa_pair" if j == 0 else "assistant"},
                        ))

            turn_idx += 1
            i += 2
        else:
            # Standalone turn (system message, or unpaired human/assistant)
            turn_text = f"**{role.title()}:**\n{content}"
            turn_meta = {
                "speaker": role,
                "turn_index": turn_idx,
                "session_id": session_id,
                "timestamp": timestamp,
                "chunk_type": "conversation",
            }

            if len(turn_text) <= max_chars:
                chunks.append(Chunk(
                    text=turn_text,
                    source_file=source_file,
                    heading=_first_sentences(content, 60).rstrip(".!? "),
                    chunk_type="conversation",
                    metadata=turn_meta,
                ))
            else:
                for part in _split_respecting_blocks(content, max_chars - 20):
                    part = part.strip()
                    if len(part) >= min_chars:
                        chunks.append(Chunk(
                            text=f"**{role.title()}:**\n{part}",
                            source_file=source_file,
                            heading=_first_sentences(content, 60).rstrip(".!? "),
                            chunk_type="conversation",
                            metadata=turn_meta,
                        ))

            turn_idx += 1
            i += 1

    return chunks


# ── Claude Code transcript parser ─────────────────────────────────────

def chunk_claude_transcript(
    jsonl_path: str,
    session_id: str = "",
    max_chars: int = 2400,
    min_chars: int = 100,
) -> list[Chunk]:
    """Parse a Claude Code session transcript (.jsonl) into conversation chunks.

    Claude Code writes session transcripts as JSONL files at
    ``~/.claude/projects/<project>/<session>.jsonl``. Each line is a JSON
    object with ``type`` (user, assistant, tool_use, tool_result) and
    ``content``. This parser filters to user/assistant turns, groups Q+A
    pairs, and produces the same Chunk format as ``chunk_conversation()``.

    Tool calls and results are skipped — they are implementation detail,
    not conversational memory.

    Args:
        jsonl_path: Path to the ``.jsonl`` transcript file.
        session_id: Session identifier. Derived from filename if empty.
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.

    Returns:
        List of Chunks with conversation metadata.
    """
    import json

    path = Path(jsonl_path)
    if not session_id:
        session_id = path.stem

    # Parse JSONL into turns
    turns: list[tuple[str, str, str]] = []  # (role, content, timestamp)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")
            if entry_type not in ("user", "assistant"):
                continue

            # Claude Code transcripts nest content inside `entry["message"]["content"]`.
            # Older export formats put it at `entry["content"]`. Fall back in that order.
            content = entry.get("content", "")
            if not content:
                msg = entry.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
            if not content or not isinstance(content, str):
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)
                if not content:
                    continue

            ts = entry.get("timestamp", "")
            role = "human" if entry_type == "user" else "assistant"
            turns.append((role, content.strip(), ts))

    if not turns:
        return []

    # Use the first timestamp as the session timestamp
    first_ts = next((ts for _, _, ts in turns if ts), "")

    # Group into Q+A pairs using the same logic as chunk_conversation
    chunks: list[Chunk] = []
    turn_idx = 0
    i = 0

    while i < len(turns):
        role, content, ts = turns[i]

        if role == "human" and i + 1 < len(turns) and turns[i + 1][0] == "assistant":
            q_content = content
            _, a_content, _ = turns[i + 1]
            pair_text = f"**Human:**\n{q_content}\n\n**Assistant:**\n{a_content}"

            pair_meta = {
                "speaker": "qa_pair",
                "turn_index": turn_idx,
                "session_id": session_id,
                "timestamp": ts or first_ts,
                "chunk_type": "conversation",
                "source_format": "claude_transcript",
            }

            if len(pair_text) <= max_chars:
                chunks.append(Chunk(
                    text=pair_text,
                    source_file=str(path),
                    heading=_first_sentences(q_content, 60).rstrip(".!? "),
                    chunk_type="conversation",
                    metadata=pair_meta,
                ))
            else:
                q_prefix = f"**Human:**\n{q_content}\n\n**Assistant:**\n"
                available = max_chars - len(q_prefix)
                if available < min_chars:
                    chunks.append(Chunk(
                        text=f"**Human:**\n{q_content}",
                        source_file=str(path),
                        heading=_first_sentences(q_content, 60).rstrip(".!? "),
                        chunk_type="conversation",
                        metadata={**pair_meta, "speaker": "human"},
                    ))
                    for part in _split_respecting_blocks(a_content, max_chars - 20):
                        part = part.strip()
                        if len(part) >= min_chars:
                            chunks.append(Chunk(
                                text=f"**Assistant:**\n{part}",
                                source_file=str(path),
                                heading=_first_sentences(q_content, 60).rstrip(".!? "),
                                chunk_type="conversation",
                                metadata={**pair_meta, "speaker": "assistant"},
                            ))
                else:
                    parts = _split_respecting_blocks(a_content, available)
                    for j, part in enumerate(parts):
                        part = part.strip()
                        if len(part) < min_chars and len(parts) > 1:
                            continue
                        chunk_text = q_prefix + part if j == 0 else f"**Assistant:** (continued)\n{part}"
                        chunks.append(Chunk(
                            text=chunk_text,
                            source_file=str(path),
                            heading=_first_sentences(q_content, 60).rstrip(".!? "),
                            chunk_type="conversation",
                            metadata={**pair_meta, "speaker": "qa_pair" if j == 0 else "assistant"},
                        ))

            turn_idx += 1
            i += 2
        else:
            turn_text = f"**{role.title()}:**\n{content}"
            turn_meta = {
                "speaker": role,
                "turn_index": turn_idx,
                "session_id": session_id,
                "timestamp": ts or first_ts,
                "chunk_type": "conversation",
                "source_format": "claude_transcript",
            }

            if len(turn_text) <= max_chars:
                chunks.append(Chunk(
                    text=turn_text,
                    source_file=str(path),
                    heading=_first_sentences(content, 60).rstrip(".!? "),
                    chunk_type="conversation",
                    metadata=turn_meta,
                ))
            else:
                for part in _split_respecting_blocks(content, max_chars - 20):
                    part = part.strip()
                    if len(part) >= min_chars:
                        chunks.append(Chunk(
                            text=f"**{role.title()}:**\n{part}",
                            source_file=str(path),
                            heading=_first_sentences(content, 60).rstrip(".!? "),
                            chunk_type="conversation",
                            metadata=turn_meta,
                        ))

            turn_idx += 1
            i += 1

    return chunks


# ── Auto-dispatch ─────────────────────────────────────────────────────

_MARKDOWN_EXTS = {".md", ".rst", ".mdx"}
_PYTHON_EXTS = {".py", ".pyi"}
_CODE_EXTS = {".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".cs", ".rb"}


def auto_chunk(
    text: str,
    source_file: str = "",
    max_chars: int = 1200,
    min_chars: int = 150,
    format_override: str = "",
    overlap_chars: int = 0,
) -> list[Chunk]:
    """Auto-detect file type and use the appropriate chunker.

    Args:
        format_override: Force a specific chunker. "conversation" uses the
            conversation chunker regardless of file extension.
        overlap_chars: When > 0, oversized sections split with a sliding
            window that seeds each new chunk with the trailing ~overlap_chars
            of the previous chunk. Only honored by the markdown and plain-text
            paths; other chunkers (conversation, python, tree-sitter) ignore it.
    """
    if min_chars >= max_chars:
        raise ValueError(
            f"min_chars ({min_chars}) must be less than max_chars ({max_chars})"
        )

    # Conversation chunker historically clamped max_chars up to 2400 for
    # docs/RAG workloads. Callers tuning conversation-memory geometry
    # (e.g. LongMemEval @ 800-char chunks) should be able to specify a
    # smaller max. Clamp only when caller used the legacy default (1200);
    # explicit smaller values are honored.
    def _conv_max(m):
        return m if m != 1200 else 2400

    if format_override == "conversation":
        return chunk_conversation(text, source_file, max_chars=_conv_max(max_chars), min_chars=min_chars)

    if format_override == "claude_transcript":
        return chunk_claude_transcript(source_file, max_chars=_conv_max(max_chars), min_chars=min_chars)

    ext = Path(source_file).suffix.lower() if source_file else ""

    if ext == ".jsonl":
        return chunk_claude_transcript(source_file, max_chars=_conv_max(max_chars), min_chars=min_chars)

    if ext in _MARKDOWN_EXTS:
        # Auto-detect conversation content within markdown files
        if _detect_conversation(text):
            return chunk_conversation(text, source_file, max_chars=_conv_max(max_chars), min_chars=min_chars)
        return chunk_markdown(text, source_file, max_chars, min_chars, overlap_chars=overlap_chars)
    elif ext in _PYTHON_EXTS:
        return chunk_python(text, source_file, max_chars, min_chars)
    elif ext in _CODE_EXTS:
        # Prefer tree-sitter AST chunking when available
        if _has_treesitter() and ext in _TS_LANG_CONFIG:
            ts_chunks = chunk_treesitter(text, source_file, max_chars, min_chars)
            if ts_chunks:
                return ts_chunks
        # Fallback: text chunker with source type and file context
        chunks = chunk_text(text, source_file, max_chars, min_chars, overlap_chars=overlap_chars)
        filename = Path(source_file).name if source_file else ""
        for c in chunks:
            c.chunk_type = "source"
            if filename:
                c.text = f"// file: {filename}\n{c.text}"
        return chunks
    else:
        return chunk_text(text, source_file, max_chars, min_chars, overlap_chars=overlap_chars)


# ── Extractive summary generation ────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"[.!?]\s+|\n")
_SIGNATURE_RE = re.compile(
    r"^((?:async\s+)?(?:def|class)\s+\w+[^:]*:)", re.MULTILINE,
)
_DOCSTRING_RE = re.compile(r'\s*"""(.*?)"""', re.DOTALL)

# Multi-language signature patterns
_GO_SIGNATURE_RE = re.compile(
    r"^(func\s+(?:\([^)]+\)\s+)?\w+\([^)]*\)(?:\s+\([^)]+\)|\s+\S+)?)",
    re.MULTILINE,
)
_RUST_SIGNATURE_RE = re.compile(
    r"^((?:pub(?:\(crate\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+\w+[^{]*)",
    re.MULTILINE,
)
_TS_SIGNATURE_RE = re.compile(
    r"^((?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const)\s+\w+[^{;]*)",
    re.MULTILINE,
)
_JAVA_SIGNATURE_RE = re.compile(
    r"^((?:@\w+\s+)*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|enum|[\w<>\[\]]+)\s+\w+[^{;]*)",
    re.MULTILINE,
)
# Doc comment patterns (Go //, Rust ///, TS/Java /** */)
_LINE_DOC_RE = re.compile(r"^\s*///?\s*(.*)", re.MULTILINE)
_BLOCK_DOC_RE = re.compile(r"/\*\*\s*(.*?)(?:\*/|\n\s*\*\s)", re.DOTALL)

_LANG_SIGNATURES: dict[str, re.Pattern] = {
    ".go": _GO_SIGNATURE_RE,
    ".rs": _RUST_SIGNATURE_RE,
    ".ts": _TS_SIGNATURE_RE,
    ".tsx": _TS_SIGNATURE_RE,
    ".js": _TS_SIGNATURE_RE,
    ".jsx": _TS_SIGNATURE_RE,
    ".java": _JAVA_SIGNATURE_RE,
    ".cs": _JAVA_SIGNATURE_RE,
}


def _first_sentences(text: str, max_chars: int) -> str:
    """Extract the first 1-2 sentences up to *max_chars*."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try to break at a sentence boundary
    end = 0
    for m in _SENTENCE_END_RE.finditer(text):
        if m.end() > max_chars:
            break
        end = m.end()
    if end > max_chars * 0.3:
        return text[:end].strip()
    # No good boundary — truncate at word boundary
    trunc = text[:max_chars]
    last_space = trunc.rfind(" ")
    if last_space > max_chars * 0.5:
        return trunc[:last_space].rstrip() + "..."
    return trunc.rstrip() + "..."


def generate_summary(chunk: Chunk, max_chars: int = 200) -> str:
    """Create a short extractive summary from a chunk.

    For doc chunks: heading + first 1-2 sentences.
    For source chunks: def/class signature + first docstring line.
    Fallback: first *max_chars* chars truncated at sentence boundary.
    """
    text = chunk.text.strip()
    if not text:
        return ""

    if chunk.chunk_type == "source":
        # Try Python signature first, then language-specific
        sig_match = _SIGNATURE_RE.search(text)
        if not sig_match:
            ext = Path(chunk.source_file).suffix.lower() if chunk.source_file else ""
            lang_re = _LANG_SIGNATURES.get(ext)
            if lang_re:
                sig_match = lang_re.search(text)

        if sig_match:
            sig = sig_match.group(1).strip()
            # Try to grab a doc comment/docstring near the signature
            after_sig = text[sig_match.end():]
            before_sig = text[:sig_match.start()]
            doc_line = ""
            # Python docstrings
            doc_match = _DOCSTRING_RE.search(after_sig[:300])
            if doc_match:
                doc_line = doc_match.group(1).strip().split("\n")[0].strip()
            # Line doc comments (Go //, Rust ///) — check before the signature
            if not doc_line and before_sig:
                for m in _LINE_DOC_RE.finditer(before_sig[-200:]):
                    candidate = m.group(1).strip()
                    if candidate and len(candidate) > 5:
                        doc_line = candidate
                        break
            # Block doc comments (/** ... */)
            if not doc_line:
                block_match = _BLOCK_DOC_RE.search(before_sig[-300:] + after_sig[:300])
                if block_match:
                    doc_line = block_match.group(1).strip().split("\n")[0].strip()
            if doc_line:
                combined = f"{sig}  // {doc_line}"
                if len(combined) <= max_chars:
                    return combined
                return combined[:max_chars - 3].rstrip() + "..."
            # Signature only
            if len(sig) <= max_chars:
                return sig
            return sig[:max_chars - 3].rstrip() + "..."

    # Doc or fallback: use heading + first sentences
    if chunk.heading:
        prefix = f"{chunk.heading}: "
        remaining = max_chars - len(prefix)
        if remaining > 40:
            # Strip the heading line from the body if it starts with one
            body = text
            lines = body.split("\n", 1)
            if lines and lines[0].lstrip().startswith("#"):
                body = lines[1].strip() if len(lines) > 1 else ""
            if body:
                return prefix + _first_sentences(body, remaining)
            return chunk.heading[:max_chars]

    return _first_sentences(text, max_chars)
