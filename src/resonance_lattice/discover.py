# SPDX-License-Identifier: BUSL-1.1
"""Knowledge Model Discovery: manifest, auto-routing, and integration helpers.

This module solves the "installed and forgotten" problem. It provides:

1. Knowledge Model manifest — machine-readable index of all .rlat files
2. Auto-integration — wire knowledge models into .mcp.json, CLAUDE.md, etc.
3. Freshness tracking — know when a knowledge model is stale
4. Auto-routing — given a query, pick the best knowledge model(s)

The manifest lives at `.rlat/manifest.json` and is regenerated on
build, sync, and init-project. AI assistants read it to discover
what knowledge is available without globbing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from resonance_lattice.field.dense import DenseField


@dataclass
class CartridgeEntry:
    """A single knowledge model in the manifest."""
    path: str                    # relative path from project root
    name: str                    # human-readable name (stem)
    sources: int                 # number of indexed chunks
    bands: int                   # field bands
    dim: int                     # field dimension
    domain: str                  # auto-detected or user-specified domain description
    built_at: str                # ISO 8601 timestamp
    primer_path: str | None = None  # path to summary/primer file
    encoder: str = ""            # encoder name
    field_type: str = "dense"    # dense, factored, pq
    corpus_type: str = ""        # code, docs, mixed


@dataclass
class Manifest:
    """Index of all knowledge models in a project."""
    version: int = 1
    project_name: str = ""
    cartridges: list[CartridgeEntry] = field(default_factory=list)
    updated_at: str = ""
    memory_primer_path: str | None = None  # path to memory primer file

    def find(self, name: str) -> CartridgeEntry | None:
        """Find a knowledge model by name."""
        for c in self.cartridges:
            if c.name == name:
                return c
        return None

    def add_or_update(self, entry: CartridgeEntry) -> None:
        """Add or update a knowledge model entry."""
        for i, c in enumerate(self.cartridges):
            if c.name == entry.name:
                self.cartridges[i] = entry
                return
        self.cartridges.append(entry)
        self.updated_at = datetime.now(UTC).isoformat()

    def remove(self, name: str) -> bool:
        """Remove a knowledge model by name."""
        before = len(self.cartridges)
        self.cartridges = [c for c in self.cartridges if c.name != name]
        return len(self.cartridges) < before

    def to_json(self) -> str:
        """Serialise to JSON."""
        data = {
            "version": self.version,
            "project_name": self.project_name,
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
            "knowledge models": [asdict(c) for c in self.cartridges],
        }
        if self.memory_primer_path:
            data["memory_primer_path"] = self.memory_primer_path
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Deserialise from JSON. Ignores unknown fields for forward-compatibility."""
        import dataclasses
        data = json.loads(text)
        known_fields = {f.name for f in dataclasses.fields(CartridgeEntry)}
        entries = [
            CartridgeEntry(**{k: v for k, v in c.items() if k in known_fields})
            for c in data.get("knowledge models", [])
        ]
        return cls(
            version=data.get("version", 1),
            project_name=data.get("project_name", ""),
            cartridges=entries,
            updated_at=data.get("updated_at", ""),
            memory_primer_path=data.get("memory_primer_path"),
        )

    def save(self, path: str | Path) -> None:
        """Write manifest to file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load manifest from file, or return empty if not found."""
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_json(p.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════
# Manifest generation
# ═══════════════════════════════════════════════════════════

def _detect_corpus_type(source_files: list[str]) -> str:
    """Guess corpus type from file extensions."""
    code_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".cs", ".rb"}
    doc_exts = {".md", ".txt", ".rst", ".adoc", ".html", ".xml"}
    code_count = sum(1 for f in source_files if Path(f).suffix.lower() in code_exts)
    doc_count = sum(1 for f in source_files if Path(f).suffix.lower() in doc_exts)
    if code_count > doc_count * 2:
        return "code"
    elif doc_count > code_count * 2:
        return "docs"
    return "mixed"


def _detect_domain(source_files: list[str], max_files: int = 50) -> str:
    """Auto-detect domain from directory names and file paths."""
    dirs = set()
    for f in source_files[:max_files]:
        parts = Path(f).parts
        for p in parts[:-1]:  # exclude filename
            if p not in (".", "..", "src", "lib", "docs", "test", "tests"):
                dirs.add(p.lower())
    if not dirs:
        return "general"
    return ", ".join(sorted(dirs)[:5])


def generate_manifest_entry(
    cartridge_path: str | Path,
    source_files: list[str] | None = None,
    primer_path: str | None = None,
    source_count: int | None = None,
    bands: int | None = None,
    dim: int | None = None,
) -> CartridgeEntry:
    """Generate a manifest entry for a knowledge model.

    If source_count/bands/dim are not provided, attempts to load the
    knowledge model header. Falls back gracefully if the knowledge model can't be read.
    """
    path = Path(cartridge_path)

    if source_count is None or bands is None or dim is None:
        try:
            from resonance_lattice.lattice import Lattice
            lattice = Lattice.load(path, restore_encoder=False)
            source_count = source_count or lattice.source_count
            bands = bands or lattice.config.bands
            dim = dim or lattice.config.dim
        except Exception:
            source_count = source_count or 0
            bands = bands or 5
            dim = dim or 2048

    corpus_type = _detect_corpus_type(source_files) if source_files else ""
    domain = _detect_domain(source_files) if source_files else ""

    return CartridgeEntry(
        path=str(path),
        name=path.stem,
        sources=source_count,
        bands=bands,
        dim=dim,
        domain=domain,
        built_at=datetime.now(UTC).isoformat(),
        primer_path=primer_path,
        encoder="",
        field_type="dense",
        corpus_type=corpus_type,
    )


def update_manifest(
    manifest_path: str | Path,
    cartridge_path: str | Path,
    source_files: list[str] | None = None,
    primer_path: str | None = None,
    project_name: str = "",
) -> Manifest:
    """Update (or create) the manifest with a knowledge model entry."""
    manifest = Manifest.load(manifest_path)
    if project_name:
        manifest.project_name = project_name

    entry = generate_manifest_entry(cartridge_path, source_files, primer_path)
    manifest.add_or_update(entry)
    manifest.save(manifest_path)
    return manifest


def scan_cartridges(directory: str | Path = ".rlat") -> list[Path]:
    """Scan a directory for .rlat files."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.rlat"))


# ═══════════════════════════════════════════════════════════
# Auto-integration
# ═══════════════════════════════════════════════════════════

def update_mcp_json(
    mcp_path: str | Path = ".mcp.json",
    manifest_path: str | Path = ".rlat/manifest.json",
) -> bool:
    """Update .mcp.json to load the primary knowledge model via MCP server.

    If multiple knowledge models exist, uses the largest one as primary.
    The MCP server's rlat_discover tool exposes all knowledge models.
    """
    manifest = Manifest.load(manifest_path)
    if not manifest.cartridges:
        return False

    # Pick primary cartridge (most sources)
    primary = max(manifest.cartridges, key=lambda c: c.sources)

    mcp_path = Path(mcp_path)
    if mcp_path.exists():
        existing = json.loads(mcp_path.read_text(encoding="utf-8"))
    else:
        existing = {}

    servers = existing.setdefault("mcpServers", {})
    if "rlat" in servers:
        # Preserve existing fields (cwd, env, etc.) — only update args
        servers["rlat"]["args"] = ["mcp", primary.path]
    else:
        servers["rlat"] = {
            "command": "rlat",
            "args": ["mcp", primary.path],
        }

    mcp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return True


def generate_claude_md_section(
    manifest_path: str | Path = ".rlat/manifest.json",
) -> str:
    """Generate a CLAUDE.md section describing available knowledge models.

    Returns markdown text to inject into CLAUDE.md.
    """
    manifest = Manifest.load(manifest_path)
    if not manifest.cartridges:
        return ""

    lines = [
        "",
        "## Available Knowledge Models",
        "",
        "This project has semantic knowledge models indexed by Resonance Lattice.",
        "Use these for conceptual questions, architecture understanding, and cross-file synthesis.",
        "",
        "| Knowledge Model | Sources | Domain | Use for |",
        "|-----------|---------|--------|---------|",
    ]

    for c in manifest.cartridges:
        use_for = {
            "code": "implementation questions, API details, function behavior",
            "docs": "architecture, design decisions, workflows, getting started",
            "mixed": "general project questions, cross-domain synthesis",
        }.get(c.corpus_type, "semantic search over indexed content")
        lines.append(f"| `{c.name}` | {c.sources} | {c.domain or c.corpus_type} | {use_for} |")

    lines.append("")
    lines.append("### How to query")
    lines.append("")

    if len(manifest.cartridges) == 1:
        c = manifest.cartridges[0]
        lines.append("```bash")
        lines.append(f'rlat search {c.path} "your question here"')
        lines.append("```")
    else:
        lines.append("**Single knowledge model:**")
        lines.append("```bash")
        for c in manifest.cartridges:
            lines.append(f'rlat search {c.path} "question about {c.corpus_type or c.name}"')
        lines.append("```")
        lines.append("")
        lines.append("**Composed search (multiple knowledge models):**")
        lines.append("```bash")
        primary = manifest.cartridges[0]
        others = manifest.cartridges[1:]
        with_flags = " ".join(f"--with {c.path}" for c in others)
        lines.append(f'rlat search {primary.path} "question" {with_flags}')
        lines.append("```")

    lines.append("")
    lines.append("**When to use rlat vs grep:**")
    lines.append("- Conceptual/architecture questions -> `rlat search`")
    lines.append("- Exact symbol or string lookup -> `Grep`")
    lines.append("- Multi-file synthesis or design rationale -> `rlat search`")
    lines.append("")

    # Reference primers if available
    primers = [c for c in manifest.cartridges if c.primer_path]
    if primers:
        for c in primers:
            lines.append(f"@{c.primer_path}")
    if manifest.memory_primer_path:
        lines.append(f"@{manifest.memory_primer_path}")
    if primers or manifest.memory_primer_path:
        lines.append("")

    return "\n".join(lines)


def inject_claude_md(
    claude_md_path: str | Path = "CLAUDE.md",
    manifest_path: str | Path = ".rlat/manifest.json",
) -> bool:
    """Inject or update the knowledge model section in CLAUDE.md.

    Looks for existing section markers and replaces, or appends if not found.
    """
    section = generate_claude_md_section(manifest_path)
    if not section:
        return False

    claude_md = Path(claude_md_path)
    start_marker = "<!-- rlat:knowledge models:start -->"
    end_marker = "<!-- rlat:knowledge models:end -->"

    wrapped = f"{start_marker}\n{section}\n{end_marker}"

    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
        if start_marker in content and end_marker in content:
            # Replace existing section
            import re
            pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
            content = re.sub(pattern, wrapped, content, flags=re.DOTALL)
        else:
            # Append
            content = content.rstrip() + "\n\n" + wrapped + "\n"
    else:
        content = wrapped + "\n"

    claude_md.write_text(content, encoding="utf-8")
    return True


# ═══════════════════════════════════════════════════════════
# Auto-routing
# ═══════════════════════════════════════════════════════════

def auto_route_query(
    query_phase: NDArray,
    cartridge_fields: dict[str, DenseField],
    top_n: int = 2,
) -> list[tuple[str, float]]:
    """Given a query, rank knowledge models by resonance energy.

    Returns top_n (name, energy) pairs, best first.
    Used by the MCP server to route queries to the right knowledge model(s).
    """
    scores = []
    for name, dense_field in cartridge_fields.items():
        resonance = dense_field.resonate(query_phase)
        energy = float(np.sum(resonance.band_energies))
        scores.append((name, energy))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


# ═══════════════════════════════════════════════════════════
# Freshness
# ═══════════════════════════════════════════════════════════

@dataclass
class FreshnessReport:
    """Report on whether a knowledge model's primer is fresh."""
    cartridge_name: str
    built_at: str
    age_hours: float
    stale: bool               # True if built_at > 72 hours ago
    files_changed: int | None  # number of source files modified since build (if detectable)
    recommendation: str       # "fresh", "consider rebuilding", "stale — rebuild recommended"


def check_freshness(
    entry: CartridgeEntry,
    source_dir: str | Path | None = None,
) -> FreshnessReport:
    """Check if a knowledge model is fresh relative to its source files."""
    try:
        built = datetime.fromisoformat(entry.built_at)
        age_hours = (datetime.now(UTC) - built).total_seconds() / 3600
    except (ValueError, TypeError):
        age_hours = 999.0

    files_changed = None
    if source_dir:
        try:
            built_ts = datetime.fromisoformat(entry.built_at).timestamp()
            changed = 0
            for f in Path(source_dir).rglob("*"):
                if f.is_file() and f.stat().st_mtime > built_ts:
                    changed += 1
            files_changed = changed
        except Exception:
            pass

    if age_hours < 24 and (files_changed is None or files_changed == 0):
        stale = False
        rec = "fresh"
    elif age_hours < 72 and (files_changed is None or files_changed < 10):
        stale = False
        rec = "consider rebuilding" if files_changed else "fresh"
    else:
        stale = True
        rec = f"stale -- rebuild recommended ({age_hours:.0f}h old"
        if files_changed is not None:
            rec += f", {files_changed} files changed"
        rec += ")"

    return FreshnessReport(
        cartridge_name=entry.name,
        built_at=entry.built_at,
        age_hours=age_hours,
        stale=stale,
        files_changed=files_changed,
        recommendation=rec,
    )
