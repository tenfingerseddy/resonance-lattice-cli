# SPDX-License-Identifier: BUSL-1.1
"""MCP (Model Context Protocol) server for Resonance Lattice.

Provides a persistent, stdio-transport server that keeps the lattice
and encoder warm in memory.  13 tools exposed:

  Search & context:
    ``rlat_search``        — enriched query with cascade/contradictions/subgraph
    ``rlat_resonate``      — LLM-optimised context injection
    ``rlat_compose_search`` — multi-knowledge model composition search

  Knowledge Model management:
    ``rlat_switch``        — switch primary knowledge model at runtime
    ``rlat_info``          — knowledge model metadata
    ``rlat_discover``      — list knowledge models and skill-backed knowledge models
    ``rlat_freshness``     — freshness check with optional incremental sync

  Diagnostics:
    ``rlat_profile``       — semantic profile (energy, rank, entropy)
    ``rlat_compare``       — compare two knowledge models
    ``rlat_locate``        — query positioning (coverage, gaps)
    ``rlat_xray``          — corpus health diagnostics

  Skill integration:
    ``rlat_skill_route``   — route query to relevant skills
    ``rlat_skill_inject``  — four-tier adaptive context injection

Usage from Claude Code settings::

    "mcpServers": {
        "rlat": {
            "command": "rlat",
            "args": ["mcp", "project.rlat"]
        }
    }
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import OrderedDict
from contextvars import ContextVar
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from resonance_lattice.lattice import BAND_NAMES, Lattice

if TYPE_CHECKING:
    from resonance_lattice.algebra import ContradictionResult
    from resonance_lattice.discover import Manifest
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.skill_runtime import SkillRuntime

logger = logging.getLogger(__name__)

# ── Telemetry: per-call metadata for dogfooding hooks ────────────────
# Handlers may enrich _CALL_META with tool-specific fields (timings_ms,
# result_count, coverage_label). _attach_meta appends the merged dict to
# the last TextContent as a fenced HTML-comment marker. Additive only —
# humans reading the text ignore the marker; PostToolUse hooks parse it.
_CALL_META: ContextVar[dict | None] = ContextVar("_CALL_META", default=None)
_META_MARKER_PREFIX = "<!-- rlat-meta: "
_META_MARKER_SUFFIX = " -->"


def _meta_enrich(**fields) -> None:
    meta = _CALL_META.get()
    if meta is not None:
        meta.update(fields)


def _attach_meta(content: list[TextContent], meta: dict) -> list[TextContent]:
    if not content:
        return content
    last = content[-1]
    if not isinstance(last, TextContent):
        return content
    marker = f"\n\n{_META_MARKER_PREFIX}{json.dumps(meta, separators=(',', ':'), default=str)}{_META_MARKER_SUFFIX}"
    content[-1] = TextContent(type="text", text=last.text + marker)
    return content


def parse_meta(text: str) -> dict | None:
    """Extract the rlat-meta JSON block appended by call_tool. Returns None if absent."""
    idx = text.rfind(_META_MARKER_PREFIX)
    if idx < 0:
        return None
    end = text.find(_META_MARKER_SUFFIX, idx)
    if end < 0:
        return None
    payload = text[idx + len(_META_MARKER_PREFIX):end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None

# ── Server singleton ─────────────────────────────────────────────────

_lattice: Lattice | None = None
_cartridge_path: str = ""
_cartridge_cache: OrderedDict[str, Lattice] = OrderedDict()  # LRU cache
_CACHE_MAX = 8  # max cached cartridges (raised for skill integration)
_manifest: Manifest | None = None  # loaded from .rlat/manifest.json
_project_root: Path | None = None  # inferred from cartridge path
_skills_root: Path | None = None  # .claude/skills/ if it exists
_skill_runtime: SkillRuntime | None = None  # lazy init on first skill tool call


def _get_or_load_cartridge(path: str) -> Lattice:
    """Load and cache a knowledge model with LRU eviction."""
    key = str(path)
    if key in _cartridge_cache:
        _cartridge_cache.move_to_end(key)
        return _cartridge_cache[key]
    if key == _cartridge_path and _lattice is not None:
        return _lattice
    logger.info("Loading additional knowledge model: %s", path)
    lattice = Lattice.load(path)
    # Evict LRU if cache full
    if len(_cartridge_cache) >= _CACHE_MAX:
        _cartridge_cache.popitem(last=False)
    _cartridge_cache[key] = lattice
    return lattice


def _resolve_cartridge(name_or_path: str | None) -> Lattice:
    """Resolve a knowledge model by manifest name or file path, using cache."""
    if name_or_path is None:
        return _get_lattice()
    if _manifest:
        entry = _manifest.find(name_or_path)
        if entry:
            return _get_or_load_cartridge(entry.path)
    return _get_or_load_cartridge(name_or_path)


def _get_skill_runtime():
    """Lazily initialise the SkillRuntime singleton."""
    global _skill_runtime
    if _skill_runtime is not None:
        return _skill_runtime
    if _skills_root is None or not _skills_root.is_dir():
        raise RuntimeError(
            "No skills directory found. "
            "Expected .claude/skills/ under the project root."
        )
    from resonance_lattice.skill_runtime import SkillRuntime
    _skill_runtime = SkillRuntime(_skills_root, _project_root or _skills_root.parent.parent)
    # Share the already-warm encoder to avoid loading ~500 MB twice
    if _lattice is not None and _lattice.encoder is not None:
        _skill_runtime._encoder = _lattice.encoder
    _skill_runtime.discover()
    return _skill_runtime

app = Server("rlat")


_source_root: str | Path | None = None  # stored for deferred load
_onnx_dir: str | None = None  # stored for deferred ONNX attach


def _get_lattice() -> Lattice:
    """Return the primary lattice, loading it lazily on first access."""
    if _lattice is None:
        if not _cartridge_path:
            raise RuntimeError("No knowledge model configured — call load_cartridge() first")
        _do_load()
    return _lattice


def _do_load() -> None:
    """Actually load the knowledge model + encoder. Called once on first tool use."""
    global _lattice, _manifest, _project_root, _skills_root
    import time

    start = time.perf_counter()
    logger.info("Loading knowledge model (deferred): %s", _cartridge_path)
    _lattice = Lattice.load(_cartridge_path, source_root=_source_root)
    elapsed = time.perf_counter() - start
    logger.info(
        "Knowledge Model loaded in %.1fs: %d sources, %d bands, dim=%d",
        elapsed,
        _lattice.source_count,
        _lattice.config.bands,
        _lattice.config.dim,
    )

    # Attach ONNX backbone if available (faster inference)
    if _onnx_dir:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            if _lattice.encoder is not None:
                attach_onnx_backbone(_lattice.encoder, _onnx_dir)
                logger.info("ONNX backbone attached: %s", _onnx_dir)
        except Exception as exc:
            logger.debug("ONNX attach failed: %s", exc)

    # Infer project root by walking up from cartridge path
    cart = Path(_cartridge_path).resolve()
    for parent in [cart.parent, *cart.parents]:
        if any((parent / marker).exists() for marker in (".git", "pyproject.toml", ".claude")):
            _project_root = parent
            break
    else:
        _project_root = cart.parent.parent  # fallback: assume .rlat/foo.rlat layout

    # Detect skills directory for lazy SkillRuntime init
    skills_candidate = _project_root / ".claude" / "skills"
    if skills_candidate.is_dir():
        _skills_root = skills_candidate
        logger.info("Skills directory found: %s", _skills_root)

    # Load manifest if available (for discovery and auto-routing)
    try:
        from resonance_lattice.discover import Manifest
        manifest_path = Path(_cartridge_path).parent / "manifest.json"
        if manifest_path.exists():
            _manifest = Manifest.load(manifest_path)
            logger.info(
                "Manifest loaded: %d knowledge model(s) discoverable",
                len(_manifest.cartridges),
            )
    except Exception as exc:
        logger.debug("No manifest loaded: %s", exc)


def load_cartridge(
    path: str | Path,
    source_root: str | Path | None = None,
    onnx_dir: str | None = None,
) -> None:
    """Configure which knowledge model to serve. Loading is deferred to first tool call.

    This returns instantly so the MCP handshake completes without delay.
    The encoder + knowledge model are loaded lazily on the first actual tool use.
    """
    global _cartridge_path, _source_root, _onnx_dir

    # Configure logging to stderr so MCP clients can see diagnostics
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _cartridge_path = str(path)
    _source_root = source_root
    _onnx_dir = onnx_dir
    logger.info("Knowledge Model configured (deferred load): %s", path)


# ── Tool definitions ─────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── 1. Search (enriched) ────────────────────────────────────
        Tool(
            name="rlat_search",
            description=(
                "Search the knowledge model. Returns ranked passages with "
                "per-band scores, coverage analysis, and optional cascade/"
                "contradiction detection. Use for understanding what the "
                "knowledge model knows about a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results (1-20)",
                    },
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                    "enable_cascade": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include multi-hop related topics",
                    },
                    "enable_contradictions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Detect contradictions between results",
                    },
                    "enable_subgraph": {
                        "type": "boolean",
                        "default": False,
                        "description": "Expand results with spectral neighbours",
                    },
                    "enable_lexical": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include keyword/phrase matching",
                    },
                    "enable_cross_encoder": {
                        "type": "boolean",
                        "default": False,
                        "description": "Cross-encoder reranking (slower, higher quality)",
                    },
                    "asymmetric": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use asymmetric query encoding",
                    },
                    "session": {
                        "type": "string",
                        "description": "Filter by conversation session ID (memory knowledge models)",
                    },
                    "after": {
                        "type": "string",
                        "description": "Filter results after ISO timestamp (e.g. '2024-01-01')",
                    },
                    "before": {
                        "type": "string",
                        "description": "Filter results before ISO timestamp",
                    },
                    "speaker": {
                        "type": "string",
                        "enum": ["human", "assistant", "system", "qa_pair"],
                        "description": "Filter by speaker role (conversation memory)",
                    },
                    "recency_weight": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Blend recency into ranking (0.0=off, 0.3=moderate). Requires timestamp metadata.",
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 2. Resonate (LLM-ready context) ─────────────────────────
        Tool(
            name="rlat_resonate",
            description=(
                "Retrieve passages for LLM context injection. Returns clean, "
                "concise text optimised for adding to a prompt. Use this when "
                "you need knowledge to answer a question, NOT for inspecting "
                "search quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you need to know",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Passages to return (1-30)",
                    },
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["augment", "knowledge", "constrain"],
                        "default": "knowledge",
                        "description": "Injection mode preamble",
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 3. Switch primary cartridge ──────────────────────────────
        Tool(
            name="rlat_switch",
            description=(
                "Switch the primary knowledge model at runtime. The old primary is "
                "cached so switching back is instant."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name (from rlat_discover) or file path",
                    },
                },
                "required": ["knowledge model"],
            },
        ),
        # ── 4. Locate (query positioning) ────────────────────────────
        Tool(
            name="rlat_locate",
            description=(
                "Locate a query within the field's knowledge landscape. Returns "
                "coverage classification (strong/partial/edge/gap), per-band "
                "energy, anti-resonance ratio, and expansion hints. Use BEFORE "
                "searching to decide if the knowledge model can answer a question."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to locate in the field",
                    },
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 5. X-ray (corpus diagnostics) ────────────────────────────
        Tool(
            name="rlat_xray",
            description=(
                "Corpus diagnostic X-ray. Reports per-band health "
                "(rich/adequate/thin/noisy), spectral properties, SNR, "
                "saturation, and actionable diagnostics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                    "deep": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include community detection and topology (slower, 30s+)",
                    },
                },
            },
        ),
        # ── 6. Skill route ───────────────────────────────────────────
        Tool(
            name="rlat_skill_route",
            description=(
                "Route a query to the most relevant skills by resonance energy. "
                "Returns ranked skills with coverage classification. Use to "
                "discover which skills have knowledge about a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to route",
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of skills to return (1-10)",
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 7. Skill inject ──────────────────────────────────────────
        Tool(
            name="rlat_skill_inject",
            description=(
                "Generate four-tier adaptive context injection for a named "
                "skill. Tiers: static header, foundational queries, user query "
                "resonation, and LLM-derived queries. Returns structured context "
                "with mode-aware gating."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill name (from rlat_skill_route or rlat_discover)",
                    },
                    "user_query": {
                        "type": "string",
                        "description": "The user's actual request",
                    },
                    "derived_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tier 4 LLM-derived queries (optional)",
                    },
                },
                "required": ["skill_name", "user_query"],
            },
        ),
        # ── 15-17. Memory tools ──────────────────────────────────────
        Tool(
            name="rlat_memory_recall",
            description=(
                "Search the agent's memory knowledge model for relevant prior knowledge. "
                "Returns passages from previous sessions with a confidence assessment. "
                "The memory knowledge model is at .rlat/memory.rlat."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to recall from memory",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of memories to return (1-20)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rlat_memory_save",
            description=(
                "Save key facts from this session to the agent's memory knowledge model. "
                "Content is encoded and added to .rlat/memory.rlat. "
                "Reports novelty — how much of the content was new information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Facts or context to remember",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier for filtering (optional)",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="rlat_memory_forget",
            description=(
                "Provably remove a topic or session from the agent's memory. "
                "Uses algebraic removal with a residual certificate (<3e-8). "
                "The information is mathematically gone, not just hidden."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to forget (searches memory, removes matching sources)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Forget all memories from a specific session",
                    },
                },
            },
        ),
        # ── 18. Knowledge negotiation ────────────────────────────────
        Tool(
            name="rlat_negotiate",
            description=(
                "Analyze the relationship between two bodies of knowledge. "
                "Shows shared ground (intersection), unique contributions (novelty), "
                "and disagreements (contradictions). Use when comparing teams, "
                "repos, or knowledge domains."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cartridge_a": {
                        "type": "string",
                        "description": "First knowledge model path",
                    },
                    "cartridge_b": {
                        "type": "string",
                        "description": "Second knowledge model path",
                    },
                    "query": {
                        "type": "string",
                        "description": "Focus the analysis on a specific topic (optional)",
                    },
                },
                "required": ["cartridge_a", "cartridge_b"],
            },
        ),
        # ── 14. Health check ─────────────────────────────────────────
        Tool(
            name="rlat_health",
            description=(
                "Check the health of a knowledge model. Reports signal quality, "
                "per-band health, saturation, and capacity. Optionally compare "
                "against a baseline to detect contradictions, coverage regressions, "
                "and semantic drift."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model to check; omit for primary",
                    },
                    "baseline": {
                        "type": "string",
                        "description": (
                            "Compare against this knowledge model for drift and "
                            "contradiction detection (optional)"
                        ),
                    },
                },
            },
        ),
        # ── 8. Info (unchanged) ──────────────────────────────────────
        Tool(
            name="rlat_info",
            description="Get metadata about the loaded knowledge model",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # ── 9. Profile (+ cartridge param) ───────────────────────────
        Tool(
            name="rlat_profile",
            description=(
                "Semantic profile of the knowledge model: per-band energy, "
                "effective rank, spectral entropy, and topic communities"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                },
            },
        ),
        # ── 10. Compare (+ cartridge param) ──────────────────────────
        Tool(
            name="rlat_compare",
            description="Compare a knowledge model against another knowledge model",
            inputSchema={
                "type": "object",
                "properties": {
                    "cartridge_b": {
                        "type": "string",
                        "description": "Path to the knowledge model to compare against",
                    },
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model to compare FROM; omit for primary",
                    },
                },
                "required": ["cartridge_b"],
            },
        ),
        # ── 11. Compose search (unchanged) ───────────────────────────
        Tool(
            name="rlat_compose_search",
            description=(
                "Search across multiple composed knowledge models. "
                "Supports merging, projection (through), diff, topic boost/suppress, "
                "and per-knowledge model injection modes (augment/constrain). "
                "The 'through' parameter accepts a .rlat knowledge model, a .rlens lens file, "
                "or comma-separated topics for an ad-hoc lens (e.g. 'security,auth,encryption')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "with_cartridges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional .rlat knowledge model paths to compose with the loaded knowledge model",
                    },
                    "through": {
                        "type": "string",
                        "description": (
                            "Semantic perspective for search. Accepts: "
                            "(1) .rlat knowledge model path, "
                            "(2) .rlens lens file path, "
                            "(3) comma-separated topics for ad-hoc lens (e.g. 'security,auth,encryption')"
                        ),
                    },
                    "invert": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "When using a lens (through), show what the lens HIDES "
                            "instead of what it reveals — finds blind spots"
                        ),
                    },
                    "diff_against": {
                        "type": "string",
                        "description": "Show what's new vs this baseline knowledge model",
                    },
                    "boost": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to boost (amplify during search)",
                    },
                    "suppress": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to suppress (attenuate during search)",
                    },
                    "tune": {
                        "type": "string",
                        "enum": ["focus", "explore", "denoise"],
                        "description": (
                            "Tune retrieval mode for the task. "
                            "'focus' = precision for specific questions (e.g. 'what's the exact syntax for X?'). "
                            "'explore' = breadth for research queries (e.g. 'what are the trade-offs of X?'). "
                            "'denoise' = clean mode for noisy corpora with lots of boilerplate. "
                            "Pick the mode that matches your question type."
                        ),
                    },
                    "sharpen": {
                        "type": "number",
                        "description": (
                            "Sharpen the corpus for more precise retrieval (0.5-2.0 typical). "
                            "Higher = crisper, more discriminative results. "
                            "Use when getting too many fuzzy/tangential matches."
                        ),
                    },
                    "soften": {
                        "type": "number",
                        "description": (
                            "Soften the corpus for broader exploration (0.5-1.5 typical). "
                            "Higher = flatter spectrum, surfaces buried topics. "
                            "Use when results feel too narrow or dominated by one theme."
                        ),
                    },
                    "contrast": {
                        "type": "string",
                        "description": (
                            "Path to a background knowledge model for asymmetric contrast. "
                            "Returns results that are distinctive to THIS knowledge model vs the background. "
                            "Use for 'what does my docs know that the generic docs don't?' queries."
                        ),
                    },
                    "injection_modes": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-knowledge model injection modes: {name: 'augment'|'constrain'|'knowledge'}",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results (1-20)",
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 12. Discover (+ skill listing) ───────────────────────────
        Tool(
            name="rlat_discover",
            description=(
                "List all available knowledge models and skills in this "
                "project. Returns knowledge model names, source counts, domains, "
                "freshness, and skill-backed knowledge models. "
                "Use this FIRST to find out what knowledge is available."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # ── C7: rlat_ask (grounded synthesis with citations) ────────
        Tool(
            name="rlat_ask",
            description=(
                "Ask a question and get a grounded answer with citations. "
                "Runs the three-layer semantic-layer flow: field retrieval "
                "-> adaptive context expansion -> reader synthesis. Use "
                "this when you need a direct answer rather than ranked "
                "passages. mode='context' returns a pre-LLM evidence pack "
                "(cheap, deterministic). mode='llm' synthesizes via the "
                "configured reader backend (local OpenVINO, Anthropic API, "
                "or OpenAI-compatible)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question",
                    },
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name or path; omit for primary",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["context", "llm"],
                        "default": "context",
                        "description": (
                            "context = return evidence pack only (no LLM). "
                            "llm = synthesize answer with citations."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Evidence items to retrieve (1-20)",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "local", "anthropic", "openai"],
                        "default": "auto",
                        "description": (
                            "LLM backend for mode=llm. 'auto' prefers local "
                            "(if installed + model set), then anthropic "
                            "(CLAUDE_API), then openai (OPENAI_API_KEY)."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Model id. HF id for local (e.g. "
                            "'Qwen/Qwen2.5-3B-Instruct'); provider model for "
                            "anthropic/openai. Defaults to claude-opus-4-7 / "
                            "gpt-4o-mini if omitted."
                        ),
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 1024,
                        "description": "Generation cap for mode=llm",
                    },
                    "temperature": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Sampling temperature (0.0-2.0)",
                    },
                    "expand": {
                        "type": "string",
                        "enum": ["off", "natural", "max"],
                        "default": "natural",
                        "description": "Context expansion on retrieved evidence",
                    },
                    "source_root": {
                        "type": "string",
                        "description": (
                            "Source directory for external-mode knowledge models. "
                            "Required for expand to fire when the knowledge model "
                            "lacks a source-root hint."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        # ── 13. Freshness (+ sync) ──────────────────────────────────
        Tool(
            name="rlat_freshness",
            description=(
                "Check if a knowledge model's knowledge is fresh or stale. "
                "Reports build age, files changed since build, and rebuild "
                "recommendation. Optionally triggers incremental sync."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge model": {
                        "type": "string",
                        "description": "Knowledge Model name (from rlat_discover) or path",
                    },
                    "source_dir": {
                        "type": "string",
                        "description": "Source directory to check for changed files",
                    },
                    "sync": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true and knowledge model is stale, run incremental sync",
                    },
                },
            },
        ),
    ]


_TOOL_DISPATCH: dict[str, callable] = {}  # populated after handler definitions


_META_REDACT_KEYS = frozenset({
    "query", "user_query", "content", "text", "prompt", "system_prompt", "input",
})


def _sanitize_arg_values(arguments: dict) -> dict:
    """Capture argument values the LLM chose, so rollups can see feature uptake
    (which `mode`, which `enable_*` flags, `top_k`, etc.) — without logging raw
    query text. Raw queries are hashed by the PostToolUse hook, not stored here.
    """
    if not isinstance(arguments, dict):
        return {}
    out: dict = {}
    for key, value in arguments.items():
        if key in _META_REDACT_KEYS:
            out[key] = {"_redacted": True, "chars": len(value) if isinstance(value, str) else None}
            continue
        if isinstance(value, (bool, int, float)) or value is None:
            out[key] = value
        elif isinstance(value, str):
            out[key] = value if len(value) <= 64 else value[:64] + "…"
        elif isinstance(value, (list, tuple)):
            out[key] = [v for v in value if isinstance(v, (bool, int, float, str)) and (not isinstance(v, str) or len(v) <= 64)][:10]
        elif isinstance(value, dict):
            out[key] = {"_keys": sorted(value.keys())[:10]}
        else:
            out[key] = {"_type": type(value).__name__}
    return out


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    meta: dict = {
        "tool": name,
        "arg_keys": sorted(arguments.keys()) if isinstance(arguments, dict) else [],
        "arg_values": _sanitize_arg_values(arguments),
    }
    token = _CALL_META.set(meta)
    t0 = time.perf_counter()
    is_error = False
    try:
        handler = _TOOL_DISPATCH.get(name)
        if handler is None:
            is_error = True
            result = [TextContent(type="text", text=f"Unknown tool: {name}")]
        else:
            result = handler(arguments)
    except Exception as exc:
        logger.exception("Tool call failed: %s", name)
        is_error = True
        result = [TextContent(type="text", text=f"Error: {exc}")]
    finally:
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        meta["knowledge model"] = Path(_cartridge_path).name if _cartridge_path else ""
        meta["is_error"] = is_error
        if isinstance(result, list):
            meta["text_chars"] = sum(len(c.text) for c in result if isinstance(c, TextContent))
        _CALL_META.reset(token)
    return _attach_meta(result, meta)


# ── Knowledge Assessment helper ──────────────────────────────────────

def _build_knowledge_assessment(lattice: Lattice, query: str) -> list[str]:
    """Build a plain-language Knowledge Assessment block for search results.

    Composes QueryLocator.locate() and FieldConsciousness.introspect()
    to give Claude structured cues about confidence, gaps, and query type.
    Returns lines to prepend to the search response, or empty list on failure.
    """
    from resonance_lattice.field.dense import DenseField

    if not isinstance(lattice.field, DenseField) or lattice.encoder is None:
        return []

    try:
        from resonance_lattice.confidence import FieldConsciousness
        from resonance_lattice.locate import QueryLocator

        phase = lattice.encoder.encode_query(query)

        # Query positioning: coverage, band focus, gaps, expansion
        location = QueryLocator.locate(
            field=lattice.field,
            query_phase=phase.vectors,
            query_text=query,
            registry=lattice.registry,
            store=lattice.store,
        )

        # Introspection: confidence, contradiction risk, depth
        consciousness = FieldConsciousness(lattice.field)
        introspection = consciousness.introspect(phase.vectors)

        # ── Translate to plain language ──────────────────────────────

        lines = ["Knowledge Assessment:"]

        # Coverage label (from locate)
        lines.append(f"  Coverage: {location.coverage_label} ({introspection.confidence:.0%} confidence)")

        # Query type from band focus
        band_type_map = {
            "broad_semantic": "broad conceptual",
            "topic": "conceptual",
            "relations": "structural (how things connect)",
            "entity": "specific lookup",
            "verbatim": "exact-match",
        }
        query_type = band_type_map.get(location.band_focus, location.band_focus)
        lines.append(f"  Query type: {query_type} ({location.band_focus} band {location.band_focus_pct:.0f}%)")

        # Gaps (from anti-resonance)
        if location.anti_resonance_ratio > 0.3:
            weakest_band = location.band_names[
                int(__import__("numpy").argmin(location.band_energies))
            ]
            gap_msg = f"  Gaps: {location.anti_resonance_ratio:.0%} of query falls outside coverage"
            gap_msg += f" (weakest: {weakest_band} band)"
            lines.append(gap_msg)
        else:
            lines.append("  Gaps: none detected")

        # Contradiction risk (from consciousness)
        if introspection.contradiction_risk > 0.5:
            lines.append(f"  Contradiction risk: high ({introspection.contradiction_risk:.0%})")
        elif introspection.contradiction_risk > 0.2:
            lines.append(f"  Contradiction risk: moderate ({introspection.contradiction_risk:.0%})")
        else:
            lines.append("  Contradiction risk: low")

        # Expansion hint
        if location.expansion_hint and location.anti_resonance_ratio > 0.15:
            lines.append(f"  Suggestion: nearby richer topic — \"{location.expansion_hint}\"")

        return lines
    except Exception as exc:
        logger.debug("Knowledge assessment skipped: %s", exc)
        return []


# ── Handlers ─────────────────────────────────────────────────────────

def _handle_search(arguments: dict) -> list[TextContent]:
    lattice = _resolve_cartridge(arguments.get("knowledge model"))

    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(type="text", text="Error: 'query' argument is required")]

    top_k = min(max(arguments.get("top_k", 5), 1), 20)

    result = lattice.enriched_query(
        query,
        top_k=top_k,
        enable_cascade=arguments.get("enable_cascade", False),
        enable_contradictions=arguments.get("enable_contradictions", False),
        enable_subgraph=arguments.get("enable_subgraph", False),
        enable_lexical=arguments.get("enable_lexical", True),
        enable_cross_encoder=arguments.get("enable_cross_encoder", False),
        asymmetric=arguments.get("asymmetric", False),
    )

    # Apply conversation memory filters if any are set
    session_f = arguments.get("session")
    after_f = arguments.get("after")
    before_f = arguments.get("before")
    speaker_f = arguments.get("speaker")
    recency_w = arguments.get("recency_weight", 0.0)

    if any([session_f, after_f, before_f, speaker_f]):
        filtered = []
        for r in result.results:
            meta = r.content.metadata if r.content and r.content.metadata else {}
            if session_f and meta.get("session_id") != session_f:
                continue
            if speaker_f and meta.get("speaker") != speaker_f:
                continue
            if after_f and meta.get("timestamp", "") and meta["timestamp"] < after_f:
                continue
            if before_f and meta.get("timestamp", "") and meta["timestamp"] > before_f:
                continue
            filtered.append(r)
        result.results = filtered

    if recency_w and recency_w > 0:
        import math
        from datetime import datetime
        now = datetime.now(UTC)
        for r in result.results:
            meta = r.content.metadata if r.content and r.content.metadata else {}
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age_days = (now - t).total_seconds() / 86400
                    recency_score = math.exp(-age_days / 30)
                    r.score = (1 - recency_w) * r.score + recency_w * recency_score
                except (ValueError, TypeError):
                    pass
        result.results.sort(key=lambda r: r.score, reverse=True)

    lines: list[str] = []

    # ── Knowledge Assessment (locate + introspect) ──────────────────
    assessment = _build_knowledge_assessment(lattice, query)
    if assessment:
        lines.extend(assessment)
        lines.append("")

    # Coverage header
    cov = result.coverage
    lines.append(f"Coverage (confidence: {cov.confidence:.0%})")
    for name, energy in zip(cov.band_names, cov.band_energies):
        lines.append(f"  {name:<12} {float(energy):.3f}")
    if cov.gaps:
        lines.append(f"  Gaps: {', '.join(cov.gaps)}")
    lines.append("")

    # Results with band scores
    for i, r in enumerate(result.results[:top_k], 1):
        score = f"{r.score:.3f}"
        source = r.source_id

        # Source file path if available
        source_file = ""
        if r.content and r.content.metadata:
            source_file = r.content.metadata.get("source_file", "")

        # Band scores
        band_str = ""
        if r.band_scores is not None:
            band_str = " bands=[" + ", ".join(f"{s:.2f}" for s in r.band_scores) + "]"

        # Provenance tag
        prov = f" ({r.provenance})" if r.provenance != "dense" else ""

        # Content
        text = ""
        if r.content:
            text = r.content.summary or r.content.full_text[:300]

        header = f"{i}. [{score}]{band_str}{prov} {source}"
        if source_file and source_file != source:
            header += f" ({source_file})"
        lines.append(header)
        lines.append(f"   {text}")

    # Cascade: related topics
    if result.related:
        lines.append("")
        lines.append("Related Topics:")
        for rt in result.related:
            lines.append(f"  [{rt.score:.3f}] {rt.source_id} (hop {rt.hop})")

    # Contradictions
    if result.contradictions:
        lines.append("")
        lines.append("Contradictions:")
        for cp in result.contradictions:
            lines.append(f"  {cp.source_a} vs {cp.source_b} (interference={cp.interference:.3f})")

    timing = result.timings_ms
    footer = f"\n---\n{len(result.results)} results in {timing.get('total', 0):.0f}ms"

    _meta_enrich(
        result_count=len(result.results),
        timings_ms={k: round(float(v), 2) for k, v in timing.items()},
        coverage_confidence=round(float(result.coverage.confidence), 4),
        coverage_gaps=list(result.coverage.gaps or []),
        related_count=len(result.related or []),
        contradictions_count=len(result.contradictions or []),
    )
    return [TextContent(type="text", text="\n".join(lines) + footer)]


def _handle_info() -> list[TextContent]:
    lattice = _get_lattice()
    info = (
        f"Cartridge: {_cartridge_path}\n"
        f"Sources: {lattice.source_count}\n"
        f"Bands: {lattice.config.bands}\n"
        f"Dim: {lattice.config.dim}\n"
        f"Field: {lattice.field_type.value}\n"
        f"ANN index: {'yes' if lattice.registry.has_ann else 'no'}"
    )
    return [TextContent(type="text", text=info)]


def _handle_profile(arguments: dict) -> list[TextContent]:
    from resonance_lattice.calculus import FieldCalculus
    from resonance_lattice.field.dense import DenseField

    lattice = _resolve_cartridge(arguments.get("knowledge model"))

    if not isinstance(lattice.field, DenseField):
        return [TextContent(
            type="text",
            text=f"Profile requires DenseField backend (got {lattice.field_type.value})",
        )]

    info = lattice.info()
    B = lattice.config.bands
    band_names = list(BAND_NAMES[:B])
    if B > len(BAND_NAMES):
        band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

    lines: list[str] = [
        f"Semantic Profile: {_cartridge_path}",
        f"  Sources: {info['source_count']} | Field: {info['field_size_mb']:.1f} MB | SNR: {info['snr']:.1f}",
        "",
    ]

    for b in range(B):
        fc = FieldCalculus.field_confidence(lattice.field, band=b, top_k=10)
        topo = lattice.eigendecompose(band=b, top_k=5)
        eigs = [round(float(v), 4) for v in topo["eigenvalues"][:5]]
        lines.append(f"  Band {b} ({band_names[b]}):")
        lines.append(f"    Energy:           {float(info['field_energy'][b]):.4f}")
        lines.append(f"    Effective rank:   {fc.effective_rank:.1f}")
        lines.append(f"    Spectral entropy: {fc.spectral_entropy:.4f}")
        lines.append(f"    Condition number: {fc.condition_number:.1f}")
        lines.append(f"    Top eigenvalues:  {eigs}")
        lines.append("")

    # Topic communities
    try:
        communities = lattice.detect_communities(n_communities=8, band=0)
        if communities and communities.get("communities"):
            lines.append(f"  Topic Communities ({communities['n_communities']} detected):")
            lines.append("")
            for c in communities["communities"]:
                pct = c["fraction"] * 100
                coh = c["coherence"]
                reps = c["representatives"][:3]
                lines.append(
                    f"    Community {c['rank']:2d}: {c['size']:5d} sources "
                    f"({pct:4.1f}%)  coherence={coh:.2f}"
                )
                lines.append(f"      Representatives: {', '.join(reps)}")
    except Exception:
        logger.debug("Community detection failed", exc_info=True)

    return [TextContent(type="text", text="\n".join(lines))]


def _handle_compare(arguments: dict) -> list[TextContent]:
    cartridge_b = arguments.get("cartridge_b")
    if not cartridge_b or not isinstance(cartridge_b, str):
        return [TextContent(
            type="text",
            text="Error: 'cartridge_b' argument is required",
        )]

    path_b = Path(cartridge_b)
    if not path_b.exists():
        return [TextContent(type="text", text=f"Error: file not found: {cartridge_b}")]

    lattice_a = _resolve_cartridge(arguments.get("knowledge model"))
    lattice_b = _get_or_load_cartridge(str(path_b))

    comparison = Lattice.compare(lattice_a, lattice_b)

    lines: list[str] = [
        f"Comparing: {_cartridge_path} ({comparison['a_sources']} sources) "
        f"vs {cartridge_b} ({comparison['b_sources']} sources)",
        "",
    ]
    for line in comparison["summary_lines"]:
        lines.append(line)
    lines.append("")
    lines.append("Per-band detail:")
    for bp in comparison["per_band"]:
        lines.append(
            f"  {bp['name']}: overlap={bp['overlap']:.0%}  "
            f"A={bp['energy_a']:.2f}  B={bp['energy_b']:.2f}  "
            f"A-B={bp['diff_a_minus_b']:.2f}  B-A={bp['diff_b_minus_a']:.2f}"
        )

    # ── Contradiction detection ──────────────────────────────────────
    contradiction_lines = _detect_contradictions(lattice_a, lattice_b, cartridge_b)
    if contradiction_lines:
        lines.append("")
        lines.extend(contradiction_lines)

    return [TextContent(type="text", text="\n".join(lines))]


def _detect_contradictions(
    lattice_a: Lattice,
    lattice_b: Lattice,
    label_b: str,
) -> list[str]:
    """Detect and surface contradictions between two lattices.

    Returns formatted lines to append to the compare output,
    or empty list if contradiction detection is unavailable.
    """
    from resonance_lattice.field.dense import DenseField

    if not isinstance(lattice_a.field, DenseField) or not isinstance(lattice_b.field, DenseField):
        return []
    if lattice_a.field.bands != lattice_b.field.bands or lattice_a.field.dim != lattice_b.field.dim:
        return []

    try:
        from resonance_lattice.algebra import FieldAlgebra

        result = FieldAlgebra.contradict(lattice_a.field, lattice_b.field)

        lines: list[str] = []

        if result.contradiction_ratio < 0.01:
            lines.append("Contradictions: none detected")
            return lines

        ratio_label = "low"
        if result.contradiction_ratio > 0.15:
            ratio_label = "high"
        elif result.contradiction_ratio > 0.05:
            ratio_label = "moderate"

        lines.append(
            f"Contradictions: {ratio_label} "
            f"({result.contradiction_ratio:.1%} contradiction ratio)"
        )

        # Per-band contradiction breakdown
        for b, energy in enumerate(result.per_band_contradiction):
            if energy > 0.01:
                name = BAND_NAMES[b] if b < len(BAND_NAMES) else f"band_{b}"
                lines.append(f"  {name}: contradiction energy {energy:.3f}")

        # Try to surface specific contradicting passages by resonating
        # the contradiction field against source phase vectors
        try:
            _surface_contradiction_passages(
                lines, result, lattice_a, lattice_b, label_b,
            )
        except Exception:
            pass  # passage surfacing is best-effort

        return lines
    except Exception as exc:
        logger.debug("Contradiction detection skipped: %s", exc)
        return []


def _surface_contradiction_passages(
    lines: list[str],
    result: ContradictionResult,
    lattice_a: Lattice,
    lattice_b: Lattice,
    label_b: str,
) -> None:
    """Best-effort: find specific passages that contradict each other.

    Resonates each source's phase vectors through the contradiction field
    to find the sources most involved in disagreements.
    """
    import numpy as np

    cfield = result.contradiction_field
    max_pairs = 3

    if lattice_a.registry is None or lattice_b.registry is None:
        return
    if lattice_a.store is None or lattice_b.store is None:
        return

    def _top_contradicting_sources(registry, n: int = 5):
        """Score each source by its energy in the contradiction field."""
        scored = []
        for source_id, entry in registry._source_index.items():
            res = cfield.resonate(entry.phase_vectors)
            energy = float(np.sum(res.band_energies))
            if energy > 1e-6:
                scored.append((source_id, energy))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    top_a = _top_contradicting_sources(lattice_a.registry)
    top_b = _top_contradicting_sources(lattice_b.registry)

    if not top_a or not top_b:
        return

    lines.append("  Contradicting passages:")
    found = 0
    for source_id_a, _energy_a in top_a:
        content_a = lattice_a.store.retrieve(source_id_a)
        if not content_a:
            continue
        text_a = (content_a.summary or (content_a.full_text or "")[:120]).strip()
        if not text_a:
            continue

        for source_id_b, _energy_b in top_b:
            content_b = lattice_b.store.retrieve(source_id_b)
            if not content_b:
                continue
            text_b = (content_b.summary or (content_b.full_text or "")[:120]).strip()
            if not text_b:
                continue

            found += 1
            file_a = (content_a.metadata or {}).get("source_file", source_id_a)
            file_b = (content_b.metadata or {}).get("source_file", source_id_b)
            lines.append(f"    {found}. \"{text_a}\" ({file_a})")
            lines.append(f"       vs \"{text_b}\" ({file_b})")
            if found >= max_pairs:
                return
            break  # one match per source_a, then move to next


def _handle_discover() -> list[TextContent]:
    """Handle rlat_discover: list all available knowledge models and skills."""
    from resonance_lattice.discover import check_freshness, scan_cartridges

    lines = []

    if _manifest and _manifest.cartridges:
        lines.append(f"Project: {_manifest.project_name or 'unknown'}")
        lines.append(f"Available cartridges: {len(_manifest.cartridges)}")
        lines.append("")

        for c in _manifest.cartridges:
            loaded = "(loaded)" if c.path == _cartridge_path or c.path in _cartridge_cache else ""
            freshness = check_freshness(c)
            status = "STALE" if freshness.stale else "fresh"
            lines.append(f"  {c.name} {loaded} [{status}]")
            lines.append(f"    Path: {c.path}")
            lines.append(f"    Sources: {c.sources}")
            lines.append(f"    Domain: {c.domain or c.corpus_type or 'general'}")
            lines.append(f"    Built: {c.built_at} ({freshness.age_hours:.0f}h ago)")
            if c.primer_path:
                lines.append(f"    Primer: {c.primer_path}")
            lines.append("")
    else:
        # Fallback: scan for .rlat files
        cart_dir = Path(_cartridge_path).parent if _cartridge_path else Path(".rlat")
        found = scan_cartridges(cart_dir)
        if found:
            lines.append(f"No manifest found. Discovered {len(found)} .rlat file(s):")
            for p in found:
                loaded = "(loaded)" if str(p) == _cartridge_path else ""
                lines.append(f"  {p.stem} {loaded}: {p}")
        else:
            lines.append("No knowledge models found. Run `rlat init-project` to create one.")
        lines.append("")
        lines.append("Tip: Run `rlat init-project --auto-integrate` to generate a manifest.")

    # Skill-backed cartridges
    if _skills_root and _skills_root.is_dir():
        try:
            rt = _get_skill_runtime()
            cart_skills = rt.cartridge_skills()
            if cart_skills:
                lines.append("")
                lines.append(f"Skill-backed cartridges ({len(cart_skills)}):")
                for skill in cart_skills:
                    mode = getattr(skill, "cartridge_mode", "augment")
                    paths = [str(p.name) for p in rt.resolve_cartridges(skill)]
                    lines.append(f"  {skill.name}: {', '.join(paths)} [{mode}]")
        except Exception as exc:
            logger.debug("Skill discovery failed: %s", exc)

    return [TextContent(type="text", text="\n".join(lines))]


def _handle_freshness(arguments: dict) -> list[TextContent]:
    """Handle rlat_freshness: check knowledge model freshness, optionally sync."""
    from resonance_lattice.discover import check_freshness

    cartridge_name = arguments.get("knowledge model", "")
    source_dir = arguments.get("source_dir")
    do_sync = arguments.get("sync", False)

    if _manifest:
        if cartridge_name:
            entry = _manifest.find(cartridge_name)
            if entry is None:
                entry = next((c for c in _manifest.cartridges if c.path == cartridge_name), None)
            if entry is None:
                return [TextContent(type="text", text=f"Unknown cartridge: {cartridge_name}. Available: {', '.join(c.name for c in _manifest.cartridges)}")]
        elif _manifest.cartridges:
            entry = _manifest.cartridges[0]
        else:
            entry = None

        if entry:
            report = check_freshness(entry, source_dir)
            lines = [
                f"Cartridge: {report.cartridge_name}",
                f"Built: {report.built_at}",
                f"Age: {report.age_hours:.1f} hours",
                f"Status: {'STALE' if report.stale else 'FRESH'}",
            ]
            if report.files_changed is not None:
                lines.append(f"Files changed since build: {report.files_changed}")
            lines.append(f"Recommendation: {report.recommendation}")

            # Incremental sync if requested and stale
            if do_sync and report.stale:
                sync_result = _do_incremental_sync(entry, source_dir)
                lines.append("")
                lines.append(sync_result)

            return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text="No manifest available. Run `rlat init-project --auto-integrate` first.")]


def _do_incremental_sync(entry, source_dir: str | None) -> str:
    """Run incremental sync on a knowledge model. Returns status string."""
    global _lattice, _cartridge_path
    import time

    from resonance_lattice.chunker import auto_chunk

    cart_path = Path(entry.path)
    if not cart_path.exists():
        return f"Sync failed: cartridge not found at {entry.path}"

    lattice = _get_or_load_cartridge(str(cart_path))

    # Determine source directory
    src_dir = Path(source_dir) if source_dir else None
    if src_dir is None and _project_root:
        # Try common source directories
        for candidate in ["src", "docs", "."]:
            d = _project_root / candidate
            if d.is_dir():
                src_dir = d
                break
    if src_dir is None:
        return "Sync failed: could not determine source directory"

    # Collect files
    from resonance_lattice.cli import _collect_files
    files = _collect_files([src_dir])
    if not files:
        return f"Sync: no ingestable files found in {src_dir}"

    # Load or create manifest
    from resonance_lattice.cli import _canonical_path, _load_manifest, _save_manifest
    manifest = _load_manifest(lattice)
    encoder_fp = ""
    if lattice.encoder:
        from resonance_lattice.cli import _encoder_fingerprint
        encoder_fp = _encoder_fingerprint(lattice.encoder)

    start = time.perf_counter()
    added = updated = skipped = 0

    for f in files:
        key = _canonical_path(f)
        content_hash = manifest.hash_file(f)
        if not manifest.needs_update(key, content_hash, encoder_fp):
            skipped += 1
            continue

        # Remove old chunks if updating
        old_ids = manifest.remove_file(key)
        for cid in old_ids:
            try:
                lattice.remove(cid)
            except Exception:
                pass

        # Ingest new content
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        chunks = auto_chunk(text, source_file=str(f))
        chunk_ids = []
        for chunk in chunks:
            sid = lattice.superpose_text(
                chunk.text,
                source_id=chunk.source_id,
                metadata={
                    "source_file": str(f),
                    "heading": chunk.heading or "",
                    "chunk_type": chunk.chunk_type or "doc",
                },
            )
            chunk_ids.append(sid)

        manifest.record(key, content_hash, chunk_ids, encoder_fp)
        if old_ids:
            updated += 1
        else:
            added += 1

    _save_manifest(lattice, manifest)
    lattice.save(str(cart_path))
    elapsed = time.perf_counter() - start

    # Reload if this was the primary cartridge
    if str(cart_path) == _cartridge_path:
        _lattice = Lattice.load(cart_path)
        # Invalidate cache entry
        _cartridge_cache.pop(str(cart_path), None)

    return (
        f"Sync complete: {added} added, {updated} updated, {skipped} unchanged "
        f"({elapsed:.1f}s)"
    )


def _apply_lens_to_lattice(lattice: Lattice, rlens_path: str, invert: bool = False) -> bool:
    """Apply a saved .rlens file to a lattice's field (non-destructive).

    Creates a new DenseField and replaces lattice.field temporarily.
    The original field is NOT modified — the lens creates a new field object.
    Returns True on success.
    """
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lens import Lens as LensBase

    if not isinstance(lattice.field, DenseField):
        return False
    try:
        lens = LensBase.load(rlens_path)
        # Lens.apply/invert returns a NEW DenseField, not mutated
        if invert:
            lattice.field = lens.invert(lattice.field)
        else:
            lattice.field = lens.apply(lattice.field)
        return True
    except Exception as exc:
        logger.debug("Lens application failed: %s", exc)
        return False


def _apply_topic_lens_to_lattice(
    lattice: Lattice, topics: list[str], invert: bool = False,
) -> bool:
    """Build an ad-hoc lens from topic strings and apply (non-destructive)."""
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lens import LensBuilder

    if not isinstance(lattice.field, DenseField) or lattice.encoder is None:
        return False
    try:
        phases = [lattice.encoder.encode_passage(topic).vectors for topic in topics]
        lens = LensBuilder.from_exemplars(
            name=",".join(topics[:3]),
            phase_vectors_list=phases,
        )
        # Returns a NEW DenseField, not mutated
        if invert:
            lattice.field = lens.invert(lattice.field)
        else:
            lattice.field = lens.apply(lattice.field)
        return True
    except Exception as exc:
        logger.debug("Topic lens failed: %s", exc)
        return False


def _handle_compose_search(arguments: dict) -> list[TextContent]:
    """Handle rlat_compose_search: composed multi-knowledge model search."""
    from resonance_lattice.composition import ComposedCartridge

    lattice = _get_lattice()
    query = arguments["query"]
    top_k = min(max(arguments.get("top_k", 5), 1), 20)

    with_cartridges = arguments.get("with_cartridges", [])
    through = arguments.get("through")
    diff_against = arguments.get("diff_against")
    boost_topics = arguments.get("boost", [])
    suppress_topics = arguments.get("suppress", [])
    injection_modes = arguments.get("injection_modes", {})
    invert = arguments.get("invert", False)

    primary_name = Path(_cartridge_path).stem

    # Build composition
    # Track if we need to restore the field after lens application
    _original_field = None

    if through:
        # Check if it's a .rlens file, comma-separated topics, or a cartridge
        through_str = str(through)
        lens_applied = False

        if through_str.endswith(".rlens") and Path(through_str).exists():
            # Save original field so we can restore it after search
            _original_field = lattice.field
            lens_applied = _apply_lens_to_lattice(
                lattice, through_str, invert=invert,
            )
        elif "," in through_str and not Path(through_str).exists():
            # Comma-separated topics — build ad-hoc lens
            topics = [t.strip() for t in through_str.split(",") if t.strip()]
            _original_field = lattice.field
            lens_applied = _apply_topic_lens_to_lattice(
                lattice, topics, invert=invert,
            )

        if lens_applied:
            # Lens was applied directly to the field; use simple merge composition
            composed = ComposedCartridge.merge({primary_name: lattice})
        else:
            # Fall through to cartridge-based projection (existing behavior)
            if _original_field is not None:
                lattice.field = _original_field  # restore on failure
                _original_field = None
            lens_lattice = _get_or_load_cartridge(through)
            lens_name = Path(through).stem
            composed = ComposedCartridge.project(
                source={primary_name: lattice},
                lens={lens_name: lens_lattice},
            )
    elif diff_against:
        baseline = _get_or_load_cartridge(diff_against)
        baseline_name = Path(diff_against).stem
        composed = ComposedCartridge.diff(
            newer={primary_name: lattice},
            older={baseline_name: baseline},
        )
    elif with_cartridges:
        constituents = {primary_name: lattice}
        for path in with_cartridges:
            extra = _get_or_load_cartridge(path)
            constituents[Path(path).stem] = extra
        composed = ComposedCartridge.merge(constituents)
    else:
        composed = ComposedCartridge.merge({primary_name: lattice})

    # Apply topic sculpting
    if boost_topics or suppress_topics:
        composed = composed.sculpt_topics(
            boost_topics=boost_topics,
            suppress_topics=suppress_topics,
        )

    # Apply EML corpus transforms
    tune_preset = arguments.get("tune")
    sharpen_strength = arguments.get("sharpen")
    soften_strength = arguments.get("soften")
    contrast_path = arguments.get("contrast")

    if any(x is not None for x in (tune_preset, sharpen_strength, soften_strength, contrast_path)):
        from resonance_lattice.compiler import (
            CompilationContext as _CC_ctx,
        )
        from resonance_lattice.compiler import (
            EmlContrast as _EmlContrast,
        )
        from resonance_lattice.compiler import (
            EmlSharpen as _EmlSharpen,
        )
        from resonance_lattice.compiler import (
            EmlSoften as _EmlSoften,
        )
        from resonance_lattice.compiler import (
            EmlTune as _EmlTune,
        )
        from resonance_lattice.composition.composed import ComposedCartridge as _CC_cls
        eml_ctx = _CC_ctx()
        transformed_field = composed.composed_field

        if contrast_path:
            bg_lattice = _get_or_load_cartridge(contrast_path)
            transformed_field = _EmlContrast(bg_lattice.field).apply(transformed_field, eml_ctx)
        if tune_preset:
            transformed_field = _EmlTune(tune_preset).apply(transformed_field, eml_ctx)
        if sharpen_strength is not None:
            transformed_field = _EmlSharpen(float(sharpen_strength)).apply(transformed_field, eml_ctx)
        if soften_strength is not None:
            transformed_field = _EmlSoften(float(soften_strength)).apply(transformed_field, eml_ctx)

        composed = _CC_cls(composed._constituents, transformed_field, composed._composition_type)

    # Set injection modes
    if injection_modes:
        composed.set_injection_modes(injection_modes)

    # Search
    try:
        results = composed.search(query, top_k=top_k)
    finally:
        # Restore original field if a lens was applied to the cached lattice
        if _original_field is not None:
            lattice.field = _original_field

    # Format output
    lines = []
    info = composed.info()
    composition_desc = info.composition_type
    if through and (str(through).endswith(".rlens") or "," in str(through)):
        lens_label = str(through)
        if invert:
            lens_label += " (INVERTED — showing what the lens hides)"
        composition_desc = f"lens: {lens_label}"

    lines.append(
        f"Composed search ({composition_desc}): "
        f"{', '.join(info.constituent_names)} "
        f"({info.total_sources} sources)"
    )
    if boost_topics:
        lines.append(f"Boosted: {', '.join(boost_topics)}")
    if suppress_topics:
        lines.append(f"Suppressed: {', '.join(suppress_topics)}")
    lines.append("")

    for i, r in enumerate(results, 1):
        mode = composed.get_injection_mode(r.cartridge)
        source_file = ""
        excerpt = ""
        if r.content:
            source_file = (r.content.metadata or {}).get("source_file", "")
            excerpt = r.content.full_text[:300] if r.content.full_text else ""

        lines.append(f"{i}. [{r.score:.3f}] [{r.cartridge}:{mode}] {r.source_id}")
        if source_file:
            lines.append(f"   {source_file}")
        if excerpt:
            lines.append(f"   {excerpt}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


# ── New handlers ─────────────────────────────────────────────────────


def _handle_resonate(arguments: dict) -> list[TextContent]:
    """Handle rlat_resonate: LLM-optimised context injection."""
    lattice = _resolve_cartridge(arguments.get("knowledge model"))

    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(type="text", text="Error: 'query' argument is required")]

    top_k = min(max(arguments.get("top_k", 10), 1), 30)
    mode = arguments.get("mode", "knowledge")

    result = lattice.resonate_text(query=query, top_k=top_k)

    # Mode preamble
    from resonance_lattice.projector import AugmentProjector
    preambles = {
        "augment": AugmentProjector.SYSTEM_AUGMENT,
        "constrain": AugmentProjector.SYSTEM_CONSTRAIN,
        "knowledge": AugmentProjector.SYSTEM_KNOWLEDGE,
    }
    preamble = preambles.get(mode, preambles["knowledge"])

    lines: list[str] = [f"[Mode: {mode}]", preamble, ""]

    kept = 0
    for r in result.results[:top_k]:
        if r.content:
            text = r.content.full_text or r.content.summary or ""
            if text:
                lines.append(f"- [{r.score:.2f}] {text}")
                kept += 1

    _meta_enrich(
        result_count=kept,
        mode=mode,
        top_k=top_k,
        timings_ms={k: round(float(v), 2) for k, v in (getattr(result, "timings_ms", {}) or {}).items()},
    )
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_switch(arguments: dict) -> list[TextContent]:
    """Handle rlat_switch: switch primary knowledge model at runtime."""
    global _lattice, _cartridge_path

    cart_ref = arguments.get("knowledge model")
    if not cart_ref or not isinstance(cart_ref, str):
        return [TextContent(type="text", text="Error: 'knowledge model' argument is required")]

    # Move current primary into cache before replacing
    if _lattice is not None and _cartridge_path:
        key = _cartridge_path
        if key not in _cartridge_cache:
            if len(_cartridge_cache) >= _CACHE_MAX:
                _cartridge_cache.popitem(last=False)
            _cartridge_cache[key] = _lattice

    # Resolve the new cartridge
    new_path: str | None = None
    if _manifest:
        entry = _manifest.find(cart_ref)
        if entry:
            new_path = entry.path
    if new_path is None:
        new_path = cart_ref

    # Check cache first
    if new_path in _cartridge_cache:
        _cartridge_cache.move_to_end(new_path)
        _lattice = _cartridge_cache.pop(new_path)
    else:
        p = Path(new_path)
        if not p.exists():
            return [TextContent(type="text", text=f"Error: cartridge not found: {cart_ref}")]
        _lattice = Lattice.load(p)

    _cartridge_path = new_path
    info = _lattice.info()
    return [TextContent(
        type="text",
        text=(
            f"Switched to: {Path(_cartridge_path).stem}\n"
            f"  Path: {_cartridge_path}\n"
            f"  Sources: {info['source_count']}\n"
            f"  Bands: {info['bands']} x {info['dim']}d\n"
            f"  Domain: {info.get('domain', 'general')}"
        ),
    )]


def _handle_ask(arguments: dict) -> list[TextContent]:
    """Handle rlat_ask: grounded synthesis with citations (C7).

    Runs the three-layer flow: search → expand → reader. In context
    mode, returns the prompt pack directly. In llm mode, synthesizes
    via the configured reader backend.

    Reuses the same `_results_to_evidence`, `_build_reader`, and
    `_select_reader_backend` helpers as the `rlat ask` CLI so the
    behaviour is identical across transports.
    """
    from argparse import Namespace

    from resonance_lattice.cli import (
        _build_reader,
        _results_to_evidence,
        _select_reader_backend,
    )
    from resonance_lattice.reader import build_bundle, build_context_pack

    lattice = _resolve_cartridge(arguments.get("knowledge model"))

    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(
            type="text",
            text="Error: 'query' argument is required",
        )]

    mode = arguments.get("mode", "context")
    if mode not in ("context", "llm"):
        return [TextContent(
            type="text",
            text=f"Error: 'mode' must be 'context' or 'llm', got {mode!r}",
        )]

    top_k = min(max(int(arguments.get("top_k", 5) or 5), 1), 20)

    # Field + materialise. Simpler config than rlat_search — the
    # reader flow doesn't need cascade / contradictions / subgraph.
    enriched = lattice.enriched_query(
        query,
        top_k=top_k,
        enable_cascade=False,
        enable_contradictions=False,
        enable_subgraph=False,
    )
    payload = enriched.to_dict()
    evidence = _results_to_evidence(payload.get("results") or [])

    if mode == "context":
        pack = build_context_pack(query, evidence)
        return [TextContent(type="text", text=pack)]

    # mode == "llm": synthesize and return an answer + citations.
    # Map MCP arguments to an argparse-shaped namespace so we can
    # hand them to _build_reader without duplicating dispatch logic.
    args_ns = Namespace(
        reader_backend=arguments.get("backend", "auto"),
        reader_model=arguments.get("model"),
        max_tokens=int(arguments.get("max_tokens", 1024) or 1024),
        temperature=float(arguments.get("temperature", 0.3) or 0.3),
    )

    # Validate backend selection up front so the error message is
    # precise if nothing is available.
    try:
        _select_reader_backend(args_ns.reader_backend, args_ns.reader_model)
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {e}")]

    try:
        reader = _build_reader(args_ns)
    except Exception as e:
        return [TextContent(type="text", text=f"Error building reader: {e}")]

    try:
        answer = reader.answer(query, evidence)
    except Exception as e:
        try:
            reader.close()
        except Exception:
            pass
        return [TextContent(type="text", text=f"Error during synthesis: {e}")]

    try:
        reader.close()
    except Exception:
        pass

    # Enrich citations with line numbers / verification so the MCP
    # client (e.g. Claude Code) can render footnotes with editor-
    # ready anchors.
    bundle = build_bundle(
        answer,
        evidence=evidence,
        source_root=arguments.get("source_root"),
        verify=bool(arguments.get("source_root")),
    )

    # Format: answer text + structured citations footer. Keep the
    # output compact — the caller is an LLM that will re-render it.
    lines: list[str] = [answer.text.rstrip()]
    if bundle.citations:
        lines.append("")
        lines.append(
            f"Sources ({bundle.verified_count}/{bundle.total_count} verified):"
        )
        for i, c in enumerate(bundle.citations, start=1):
            loc = c.source_file or "(unknown)"
            if c.line_number is not None:
                loc = f"{loc}:{c.line_number}"
            elif c.char_offset:
                loc = f"{loc} (offset {c.char_offset})"
            marker = "✓" if c.verified else "○"
            lines.append(f"  [{i}] {marker} {loc}")
    lines.append("")
    lines.append(
        f"[model: {answer.model}, "
        f"latency: {answer.latency_ms:.0f}ms, "
        f"evidence: {answer.evidence_used}, "
        f"citations: {len(bundle.citations)}]"
    )

    return [TextContent(type="text", text="\n".join(lines))]


def _handle_locate(arguments: dict) -> list[TextContent]:
    """Handle rlat_locate: query positioning in the field."""
    from resonance_lattice.locate import QueryLocator

    lattice = _resolve_cartridge(arguments.get("knowledge model"))
    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(type="text", text="Error: 'query' argument is required")]

    if lattice.encoder is None:
        return [TextContent(type="text", text="Error: encoder not available for locate")]

    phase = lattice.encoder.encode_query(query)
    location = QueryLocator.locate(
        field=lattice.field,
        query_phase=phase.vectors,
        query_text=query,
        registry=lattice.registry,
        store=lattice.store,
    )

    return [TextContent(type="text", text=location.to_prompt())]


def _handle_xray(arguments: dict) -> list[TextContent]:
    """Handle rlat_xray: corpus diagnostics."""
    from resonance_lattice.xray import FieldXRay

    lattice = _resolve_cartridge(arguments.get("knowledge model"))
    deep = arguments.get("deep", False)

    if deep:
        result = FieldXRay.deep(
            lattice.field,
            lattice.source_count,
            lattice_path=_cartridge_path,
            lattice=lattice,
        )
    else:
        result = FieldXRay.quick(
            lattice.field,
            lattice.source_count,
            lattice_path=_cartridge_path,
        )

    return [TextContent(type="text", text=result.to_text())]


def _get_memory_path() -> Path:
    """Resolve the standard memory knowledge model path."""
    if _project_root:
        return _project_root / ".rlat" / "memory.rlat"
    return Path(".rlat/memory.rlat")


def _handle_memory_recall(arguments: dict) -> list[TextContent]:
    """Handle rlat_memory_recall: search agent memory with confidence."""
    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(type="text", text="Error: 'query' argument is required")]

    mem_path = _get_memory_path()
    if not mem_path.exists():
        return [TextContent(type="text", text="No memory knowledge model found. Use rlat_memory_save to create one.")]

    top_k = min(max(arguments.get("top_k", 5), 1), 20)

    try:
        memory_lattice = _get_or_load_cartridge(str(mem_path))
        result = memory_lattice.enriched_query(query, top_k=top_k)

        lines: list[str] = []

        # Build assessment using confidence estimation
        assessment = _build_knowledge_assessment(memory_lattice, query)
        if assessment:
            # Replace "Knowledge Assessment" with "Memory Assessment"
            assessment[0] = "Memory Assessment:"
            lines.extend(assessment)
            lines.append("")

        # Memory count
        lines.append(f"Memory cartridge: {memory_lattice.source_count} memories")
        lines.append("")

        # Results
        if not result.results:
            lines.append("No relevant memories found.")
        else:
            lines.append("Memories:")
            for i, r in enumerate(result.results[:top_k], 1):
                text = ""
                session = ""
                if r.content:
                    text = r.content.summary or (r.content.full_text or "")[:200]
                    meta = r.content.metadata or {}
                    session = meta.get("session_id", "")
                session_tag = f" [session: {session}]" if session else ""
                lines.append(f"  {i}. [{r.score:.3f}]{session_tag} {text}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error recalling memory: {exc}")]


def _handle_memory_save(arguments: dict) -> list[TextContent]:
    """Handle rlat_memory_save: add content to memory knowledge model."""
    content = arguments.get("content")
    if not content or not isinstance(content, str):
        return [TextContent(type="text", text="Error: 'content' argument is required")]

    session_id = arguments.get("session_id", "")
    mem_path = _get_memory_path()

    try:
        import tempfile
        from datetime import datetime

        # Get encoder from primary lattice
        primary = _get_lattice()
        if primary.encoder is None:
            return [TextContent(type="text", text="Error: encoder not available")]

        # Write content to a temp file so we can use Lattice.build / add
        timestamp = datetime.now(UTC).isoformat()
        metadata_header = f"---\nsession_id: {session_id}\ntimestamp: {timestamp}\n---\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(metadata_header + content)
            tmp_path = f.name

        mem_path.parent.mkdir(parents=True, exist_ok=True)

        if mem_path.exists():
            # Add to existing memory
            memory_lattice = _get_or_load_cartridge(str(mem_path))

            # Compute novelty before adding
            from resonance_lattice.algebra import FieldAlgebra
            from resonance_lattice.field.dense import DenseField

            phase = primary.encoder.encode_passage(content)
            novelty_score = 0.0
            if isinstance(memory_lattice.field, DenseField):
                try:
                    novelty = FieldAlgebra.novelty(
                        memory_lattice.field, [phase.vectors],
                    )
                    novelty_score = novelty.aggregate_novelty
                except Exception:
                    novelty_score = 1.0  # assume novel on error

            # Add via Lattice.add
            memory_lattice.add_files([tmp_path])
            memory_lattice.save(str(mem_path))

            # Invalidate cache
            key = str(mem_path)
            if key in _cartridge_cache:
                del _cartridge_cache[key]

            source_count = memory_lattice.source_count
        else:
            # Build new memory cartridge
            new_lattice = Lattice.build(
                paths=[tmp_path],
                encoder=primary.encoder,
            )
            new_lattice.save(str(mem_path))
            novelty_score = 1.0
            source_count = new_lattice.source_count

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return [TextContent(type="text", text=(
            f"Saved to memory ({source_count} memories).\n"
            f"Novelty: {novelty_score:.0%} new information."
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error saving memory: {exc}")]


def _handle_memory_forget(arguments: dict) -> list[TextContent]:
    """Handle rlat_memory_forget: provably remove a topic or session."""
    topic = arguments.get("topic")
    session_id = arguments.get("session_id")

    if not topic and not session_id:
        return [TextContent(type="text", text="Error: provide 'topic' or 'session_id' to forget")]

    mem_path = _get_memory_path()
    if not mem_path.exists():
        return [TextContent(type="text", text="No memory knowledge model found.")]

    try:
        from resonance_lattice.algebra import FieldAlgebra
        from resonance_lattice.field.dense import DenseField

        memory_lattice = _get_or_load_cartridge(str(mem_path))
        if not isinstance(memory_lattice.field, DenseField):
            return [TextContent(type="text", text="Error: memory requires DenseField")]

        # Find sources to forget
        sources_to_forget = []

        if session_id and memory_lattice.store:
            # Find all sources from this session
            for sid in list(memory_lattice.registry._source_index.keys()):
                content = memory_lattice.store.retrieve(sid)
                if content and content.metadata and content.metadata.get("session_id") == session_id:
                    sources_to_forget.append(sid)

        elif topic and memory_lattice.encoder:
            # Search for matching sources and forget the top matches
            result = memory_lattice.enriched_query(topic, top_k=10)
            for r in result.results:
                if r.score > 0.3:  # only forget reasonably relevant matches
                    sources_to_forget.append(r.source_id)

        if not sources_to_forget:
            return [TextContent(type="text", text="No memories found matching the request.")]

        # Forget each source
        total_residual = 0.0
        for sid in sources_to_forget:
            entry = memory_lattice.registry._source_index.get(sid)
            if entry is not None:
                certificate = FieldAlgebra.forget(
                    memory_lattice.field, [entry.phase_vectors],
                )
                total_residual = max(total_residual, certificate.residual_ratio)
                # Remove from registry and store
                memory_lattice.registry.unregister(sid)
                if memory_lattice.store:
                    memory_lattice.store.remove(sid)

        # Save updated memory
        memory_lattice.save(str(mem_path))

        # Invalidate cache
        key = str(mem_path)
        if key in _cartridge_cache:
            del _cartridge_cache[key]

        label = f"topic \"{topic}\"" if topic else f"session \"{session_id}\""
        return [TextContent(type="text", text=(
            f"Removed {len(sources_to_forget)} memories about {label}.\n"
            f"Removal certificate: residual < {total_residual:.1e} (algebraically exact).\n"
            f"Memory cartridge: {memory_lattice.source_count} memories remaining."
        ))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error forgetting: {exc}")]


def _handle_negotiate(arguments: dict) -> list[TextContent]:
    """Handle rlat_negotiate: knowledge relationship analysis."""
    cartridge_a_path = arguments.get("cartridge_a")
    cartridge_b_path = arguments.get("cartridge_b")

    if not cartridge_a_path or not cartridge_b_path:
        return [TextContent(type="text", text="Error: both cartridge_a and cartridge_b are required")]

    for p in [cartridge_a_path, cartridge_b_path]:
        if not Path(p).exists():
            return [TextContent(type="text", text=f"Error: file not found: {p}")]

    try:
        import numpy as np

        from resonance_lattice.algebra import FieldAlgebra
        from resonance_lattice.field.dense import DenseField

        lattice_a = _get_or_load_cartridge(cartridge_a_path)
        lattice_b = _get_or_load_cartridge(cartridge_b_path)

        if not isinstance(lattice_a.field, DenseField) or not isinstance(lattice_b.field, DenseField):
            return [TextContent(type="text", text="Error: negotiate requires DenseField knowledge models")]

        if lattice_a.field.bands != lattice_b.field.bands or lattice_a.field.dim != lattice_b.field.dim:
            return [TextContent(type="text", text="Error: knowledge models must have matching dimensions")]

        name_a = Path(cartridge_a_path).stem
        name_b = Path(cartridge_b_path).stem

        lines = [f"Knowledge Negotiation: {name_a} vs {name_b}", ""]

        # ── Intersection (shared ground) ──
        intersection = FieldAlgebra.intersect(lattice_a.field, lattice_b.field)
        lines.append(f"Shared ground: {intersection.overlap_fraction:.0%} overlap")

        # Find what's strongest in the shared field
        if intersection.intersection_field and lattice_a.registry and lattice_a.store:
            shared_topics = _find_top_topics(
                intersection.intersection_field, lattice_a, max_topics=3,
            )
            if shared_topics:
                lines.append(f"  Strongest agreement: {', '.join(shared_topics)}")
        lines.append("")

        # ── Novelty (unique contributions) ──
        # What A has that B doesn't
        novelty_a_scores = []
        if lattice_a.registry:
            for sid, entry in list(lattice_a.registry._source_index.items())[:100]:
                n = FieldAlgebra.novelty(lattice_b.field, entry.phase_vectors)
                novelty_a_scores.append(n.score)
        avg_novelty_a = float(np.mean(novelty_a_scores)) if novelty_a_scores else 0

        # What B has that A doesn't
        novelty_b_scores = []
        if lattice_b.registry:
            for sid, entry in list(lattice_b.registry._source_index.items())[:100]:
                n = FieldAlgebra.novelty(lattice_a.field, entry.phase_vectors)
                novelty_b_scores.append(n.score)
        avg_novelty_b = float(np.mean(novelty_b_scores)) if novelty_b_scores else 0

        lines.append(f"Unique to {name_a}: {avg_novelty_a:.0%} novelty")
        if lattice_a.registry and lattice_b.store:
            # Find what A has that's most novel to B
            unique_a = _find_most_novel(lattice_a, lattice_b.field, max_topics=3)
            if unique_a:
                lines.append(f"  {', '.join(unique_a)}")

        lines.append(f"Unique to {name_b}: {avg_novelty_b:.0%} novelty")
        if lattice_b.registry and lattice_a.store:
            unique_b = _find_most_novel(lattice_b, lattice_a.field, max_topics=3)
            if unique_b:
                lines.append(f"  {', '.join(unique_b)}")
        lines.append("")

        # ── Contradictions ──
        contradiction = FieldAlgebra.contradict(lattice_a.field, lattice_b.field)
        if contradiction.contradiction_ratio < 0.01:
            lines.append("Disagreements: none detected")
        else:
            ratio_label = "high" if contradiction.contradiction_ratio > 0.15 else "moderate" if contradiction.contradiction_ratio > 0.05 else "low"
            lines.append(
                f"Disagreements: {ratio_label} ({contradiction.contradiction_ratio:.1%} contradiction ratio)"
            )
            # Surface specific passages
            contradiction_lines = []
            _surface_contradiction_passages(
                contradiction_lines, contradiction,
                lattice_a, lattice_b, name_b,
            )
            if contradiction_lines:
                lines.extend(contradiction_lines)
        lines.append("")

        # ── Recommendation ──
        lines.append("Recommendation:")
        if contradiction.contradiction_ratio > 0.05:
            lines.append("  Resolve disagreements before merging.")
        if avg_novelty_b > avg_novelty_a + 0.1:
            lines.append(f"  {name_b} has significantly more unique knowledge — consider cross-pollinating.")
        elif avg_novelty_a > avg_novelty_b + 0.1:
            lines.append(f"  {name_a} has significantly more unique knowledge — consider cross-pollinating.")
        if intersection.overlap_fraction > 0.7:
            lines.append("  High overlap — merging adds limited new knowledge.")
        elif intersection.overlap_fraction < 0.3:
            lines.append("  Low overlap — merging (--with) would create broad coverage.")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as exc:
        logger.exception("Negotiate failed")
        return [TextContent(type="text", text=f"Error: {exc}")]


def _find_top_topics(
    field: DenseField, lattice: Lattice, max_topics: int = 3,
) -> list[str]:
    """Find the top-resonating topics in a field using a lattice's registry/store."""
    import numpy as np

    topics = []
    if lattice.registry is None or lattice.store is None:
        return topics

    scored = []
    for sid, entry in lattice.registry._source_index.items():
        res = field.resonate(entry.phase_vectors)
        energy = float(np.sum(res.band_energies))
        scored.append((sid, energy))
    scored.sort(key=lambda x: -x[1])

    for sid, _ in scored[:max_topics * 2]:
        content = lattice.store.retrieve(sid)
        if content:
            text = (content.summary or (content.full_text or "")[:60]).strip()
            if text and len(text) > 5:
                topics.append(text[:60])
                if len(topics) >= max_topics:
                    break
    return topics


def _find_most_novel(
    source_lattice: Lattice, target_field: DenseField, max_topics: int = 3,
) -> list[str]:
    """Find sources in source_lattice that are most novel to target_field."""
    from resonance_lattice.algebra import FieldAlgebra

    scored = []
    if source_lattice.registry is None or source_lattice.store is None:
        return []

    for sid, entry in list(source_lattice.registry._source_index.items())[:100]:
        n = FieldAlgebra.novelty(target_field, entry.phase_vectors)
        scored.append((sid, n.score))
    scored.sort(key=lambda x: -x[1])

    topics = []
    for sid, _ in scored[:max_topics * 2]:
        content = source_lattice.store.retrieve(sid)
        if content:
            text = (content.summary or (content.full_text or "")[:60]).strip()
            if text and len(text) > 5:
                topics.append(text[:60])
                if len(topics) >= max_topics:
                    break
    return topics


def _handle_health(arguments: dict) -> list[TextContent]:
    """Handle rlat_health: composed knowledge health check."""
    from resonance_lattice.health import HealthCheck

    lattice = _resolve_cartridge(arguments.get("knowledge model"))

    baseline = None
    baseline_path = arguments.get("baseline")
    if baseline_path:
        bp = Path(baseline_path)
        if not bp.exists():
            return [TextContent(type="text", text=f"Error: baseline not found: {baseline_path}")]
        baseline = _get_or_load_cartridge(str(bp))

    report = HealthCheck.run(
        lattice=lattice,
        baseline=baseline,
        lattice_path=_cartridge_path,
    )

    return [TextContent(type="text", text=report.to_text())]


def _handle_skill_route(arguments: dict) -> list[TextContent]:
    """Handle rlat_skill_route: route a query to the most relevant skills."""
    rt = _get_skill_runtime()

    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return [TextContent(type="text", text="Error: 'query' argument is required")]

    top_n = min(max(arguments.get("top_n", 3), 1), 10)
    ranked = rt.route(query, top_n=top_n)

    if not ranked:
        return [TextContent(type="text", text="No knowledge model-backed skills found.")]

    lines = ["Skill routing results:", ""]
    for i, match in enumerate(ranked, 1):
        lines.append(
            f"  {i}. {match.name} [{match.coverage}] "
            f"energy={match.energy:.1f} mode={match.mode}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_skill_inject(arguments: dict) -> list[TextContent]:
    """Handle rlat_skill_inject: four-tier adaptive context injection."""
    from resonance_lattice.skill_projector import SkillProjector

    rt = _get_skill_runtime()
    skill_name = arguments.get("skill_name")
    if not skill_name:
        return [TextContent(type="text", text="Error: 'skill_name' argument is required")]

    user_query = arguments.get("user_query")
    if not user_query:
        return [TextContent(type="text", text="Error: 'user_query' argument is required")]

    skill = rt.get_skill(skill_name)
    if skill is None:
        available = [s.name for s in rt.discover()]
        return [TextContent(
            type="text",
            text=f"Error: skill '{skill_name}' not found. Available: {', '.join(available) or 'none'}",
        )]

    derived = arguments.get("derived_queries")
    projector = SkillProjector(rt)
    injection = projector.project(skill, user_query, derived_queries=derived)

    lines = [
        f"Mode: {injection.mode}",
        f"Confidence: {injection.coverage_confidence:.0%}",
        f"Tokens: {injection.total_tokens:,} "
        f"(t1:{injection.tier_tokens.get('t1', 0)}, "
        f"t2:{injection.tier_tokens.get('t2', 0)}, "
        f"t3:{injection.tier_tokens.get('t3', 0)}, "
        f"t4:{injection.tier_tokens.get('t4', 0)})",
    ]
    if injection.gated:
        lines.append("(Dynamic body gated — low coverage)")
    lines.append("")
    if injection.header:
        lines.append(injection.header)
        lines.append("")
    if injection.body:
        lines.append(injection.body)

    return [TextContent(type="text", text="\n".join(lines))]


# ── Dispatch table ───────────────────────────────────────────────────

_TOOL_DISPATCH.update({
    "rlat_search": _handle_search,
    "rlat_resonate": _handle_resonate,
    "rlat_ask": _handle_ask,
    "rlat_switch": _handle_switch,
    "rlat_locate": _handle_locate,
    "rlat_xray": _handle_xray,
    "rlat_health": _handle_health,
    "rlat_memory_recall": _handle_memory_recall,
    "rlat_memory_save": _handle_memory_save,
    "rlat_memory_forget": _handle_memory_forget,
    "rlat_negotiate": _handle_negotiate,
    "rlat_skill_route": _handle_skill_route,
    "rlat_skill_inject": _handle_skill_inject,
    "rlat_info": lambda _: _handle_info(),
    "rlat_profile": _handle_profile,
    "rlat_compare": _handle_compare,
    "rlat_compose_search": _handle_compose_search,
    "rlat_discover": lambda _: _handle_discover(),
    "rlat_freshness": _handle_freshness,
})


# ── Entry point ──────────────────────────────────────────────────────

async def run_server() -> None:
    """Run the MCP server on stdio.

    The knowledge model is loaded lazily on first tool call, so this starts
    instantly and the MCP handshake completes without delay.
    """
    if not _cartridge_path:
        raise RuntimeError("No knowledge model configured — call load_cartridge() first")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
