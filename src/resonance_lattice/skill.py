# SPDX-License-Identifier: BUSL-1.1
"""Skill Integration: parse skill frontmatter and orchestrate knowledge model operations.

Skills are Claude Code's context injection mechanism.  This module lets skills
declare .rlat knowledge models in their YAML frontmatter so that Resonance Lattice
can provide adaptive, query-aware context instead of static document loading.

See docs/SKILL_INTEGRATION.md for the full architecture.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml  # type: ignore[import-untyped]
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ── Frontmatter parsing ─────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


@dataclass
class SkillConfig:
    """Parsed skill configuration from SKILL.md frontmatter."""

    name: str
    skill_dir: Path
    description: str = ""

    # Cartridge integration (all optional)
    cartridges: list[str] = field(default_factory=list)
    cartridge_sources: list[str] = field(default_factory=list)
    cartridge_queries: list[str] = field(default_factory=list)
    cartridge_mode: str = "augment"
    cartridge_budget: int = 2000
    cartridge_rebuild: str = "none"
    cartridge_derive: bool = True
    cartridge_derive_count: int = 3

    @property
    def has_cartridges(self) -> bool:
        """True if this skill has any knowledge model integration."""
        return bool(self.cartridges or self.cartridge_sources)

    def resolve_cartridge_paths(self, project_root: Path | None = None) -> list[Path]:
        """Resolve knowledge model paths to absolute paths.

        Paths are resolved relative to the skill directory first,
        then relative to the project root if provided.
        """
        resolved = []
        for p in self.cartridges:
            path = Path(p)
            if path.is_absolute():
                resolved.append(path)
            elif (self.skill_dir / path).exists():
                resolved.append((self.skill_dir / path).resolve())
            elif project_root and (project_root / path).exists():
                resolved.append((project_root / path).resolve())
            else:
                # Keep as-is; caller will handle missing files
                resolved.append(self.skill_dir / path)
        return resolved

    def resolve_source_paths(self) -> list[Path]:
        """Resolve knowledge model-sources to absolute paths relative to skill dir."""
        if not self.cartridge_sources:
            # Default: references/ if it exists
            refs = self.skill_dir / "references"
            return [refs] if refs.is_dir() else []
        resolved = []
        for p in self.cartridge_sources:
            path = Path(p)
            if path.is_absolute():
                resolved.append(path)
            else:
                resolved.append((self.skill_dir / p).resolve())
        return resolved

    def local_cartridge_path(self) -> Path:
        """Path where the skill-local knowledge model should be stored."""
        return self.skill_dir / "knowledge model" / f"{self.name}.rlat"

    def local_primer_path(self) -> Path:
        """Path where the skill-local primer should be stored."""
        return self.skill_dir / "knowledge model" / "primer.md"


def _parse_yaml_value(value: str) -> Any:
    """Minimal YAML value parser — handles strings, ints, bools."""
    value = value.strip()
    if not value:
        return ""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_yaml_block(text: str) -> dict[str, Any]:
    """Minimal YAML frontmatter parser.

    Handles: key: value, key: >- (folded scalar), key: followed by - items.
    Does NOT handle nested objects, anchors, or full YAML spec.
    """
    result: dict[str, Any] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue

        match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", line)
        if not match:
            i += 1
            continue

        key = match.group(1)
        rest = match.group(2).strip()

        # Folded scalar (>- or > or | or |-)
        if rest in (">-", ">", "|", "|-"):
            parts = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                if lines[i].strip():
                    parts.append(lines[i].strip())
                i += 1
            result[key] = " ".join(parts)
            continue

        # List on next lines
        if not rest:
            items = []
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                item_match = re.match(r"^\s+-\s+(.*)", lines[i])
                if item_match:
                    val = item_match.group(1).strip()
                    if (val.startswith('"') and val.endswith('"')) or \
                       (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    items.append(val)
                i += 1
            result[key] = items
            continue

        result[key] = _parse_yaml_value(rest)
        i += 1

    return result


def _parse_frontmatter_text(raw: str) -> dict[str, Any]:
    """Parse YAML frontmatter text, preferring PyYAML when available."""
    if _HAS_YAML:
        try:
            data = _yaml.safe_load(raw)
            if isinstance(data, dict):
                return data
        except _yaml.YAMLError as exc:
            logger.warning("PyYAML failed on frontmatter, falling back: %s", exc)
    return _parse_yaml_block(raw)


def parse_skill_frontmatter(skill_dir: Path) -> SkillConfig | None:
    """Parse SKILL.md frontmatter for knowledge model configuration.

    Uses PyYAML when available, falls back to minimal parser.
    Returns None if SKILL.md doesn't exist or has no valid frontmatter.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    text = skill_md.read_text(encoding="utf-8", errors="replace")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None

    data = _parse_frontmatter_text(match.group(1))

    name = data.get("name", skill_dir.name)
    if not isinstance(name, str):
        name = str(name)

    cartridges_raw = data.get("knowledge models", [])
    if isinstance(cartridges_raw, str):
        cartridges_raw = [cartridges_raw]

    sources_raw = data.get("knowledge model-sources", [])
    if isinstance(sources_raw, str):
        sources_raw = [sources_raw]

    queries_raw = data.get("knowledge model-queries", [])
    if isinstance(queries_raw, str):
        queries_raw = [queries_raw]

    return SkillConfig(
        name=name,
        skill_dir=skill_dir.resolve(),
        description=data.get("description", ""),
        cartridges=cartridges_raw,
        cartridge_sources=sources_raw,
        cartridge_queries=queries_raw,
        cartridge_mode=str(data.get("knowledge model-mode", "augment")),
        cartridge_budget=int(data.get("knowledge model-budget", 2000)),
        cartridge_rebuild=str(data.get("knowledge model-rebuild", "none")),
        cartridge_derive=bool(data.get("knowledge model-derive", True)),
        cartridge_derive_count=int(data.get("knowledge model-derive-count", 3)),
    )


# ── Skill discovery ──────────────────────────────────────────────────

def discover_skills(skills_root: Path) -> list[SkillConfig]:
    """Scan a skills directory for all skills with valid frontmatter."""
    if not skills_root.is_dir():
        return []
    skills = []
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        config = parse_skill_frontmatter(child)
        if config is not None:
            skills.append(config)
    return skills


def discover_cartridge_skills(skills_root: Path) -> list[SkillConfig]:
    """Discover only skills that have knowledge model integration configured."""
    return [s for s in discover_skills(skills_root) if s.has_cartridges]


def find_skill(skills_root: Path, name: str) -> SkillConfig | None:
    """Find a specific skill by name."""
    # Direct directory match
    skill_dir = skills_root / name
    if skill_dir.is_dir():
        return parse_skill_frontmatter(skill_dir)
    # Search all skills
    for skill in discover_skills(skills_root):
        if skill.name == name:
            return skill
    return None


# ── Skill header extraction ──────────────────────────────────────────

def extract_skill_header(skill_dir: Path) -> str:
    """Extract the Tier 1 static content from SKILL.md (everything after frontmatter).

    This is the standing instructions, templates, and workflow steps that
    load on every trigger regardless of the query.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return ""
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    match = _FRONTMATTER_RE.match(text)
    if match:
        return text[match.end():].lstrip("\n")
    return text
