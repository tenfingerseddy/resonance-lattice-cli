# SPDX-License-Identifier: BUSL-1.1
"""Named context files (.rctx): reusable composition configurations.

A .rctx file is a YAML manifest that declares:
- Which knowledge models to compose
- How to compose them (expression or weights)
- Topic boost/suppress
- Per-knowledge model injection modes
- Lens to apply

Example:
    name: team-context
    knowledge models:
      docs: ./docs.rlat
      code: ./code.rlat
      design: ./design-docs.rlat
    weights:
      docs: 0.7
      code: 0.2
      design: 0.1
    suppress:
      - "meeting notes"
      - "draft proposals"
    injection_modes:
      docs: augment
      code: constrain
    lens: sharpen
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

import json


@dataclass
class ContextConfig:
    """Parsed .rctx context configuration."""
    name: str
    cartridges: dict[str, str]             # alias -> path
    weights: dict[str, float] | None = None
    expression: str | None = None           # RQL expression (overrides weights)
    boost: list[str] = field(default_factory=list)
    suppress: list[str] = field(default_factory=list)
    injection_modes: dict[str, str] = field(default_factory=dict)
    lens: str | None = None
    boost_strength: float = 0.5
    suppress_strength: float = 0.3


def load_context(path: str | Path) -> ContextConfig:
    """Load a .rctx context file.

    Supports YAML (if pyyaml installed) or JSON.
    Knowledge Model paths are resolved relative to the .rctx file's directory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Parse YAML or JSON
    if _HAS_YAML:
        data = yaml.safe_load(text)
    else:
        # Fallback: try JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ImportError(
                "pyyaml is required to parse .rctx files in YAML format. "
                "Install with: pip install pyyaml"
            )

    if not isinstance(data, dict):
        raise ValueError(f"Invalid .rctx file: expected a mapping, got {type(data).__name__}")

    # Resolve cartridge paths relative to .rctx directory
    rctx_dir = path.parent
    cartridges = {}
    for alias, cart_path in data.get("knowledge models", {}).items():
        resolved = rctx_dir / cart_path
        cartridges[alias] = str(resolved)

    return ContextConfig(
        name=data.get("name", path.stem),
        cartridges=cartridges,
        weights=data.get("weights"),
        expression=data.get("expression") or data.get("compose"),
        boost=data.get("boost", []),
        suppress=data.get("suppress", []),
        injection_modes=data.get("injection_modes", {}),
        lens=data.get("lens"),
        boost_strength=data.get("boost_strength", 0.5),
        suppress_strength=data.get("suppress_strength", 0.3),
    )


def validate_context(config: ContextConfig) -> list[str]:
    """Validate a context configuration. Returns list of warnings."""
    warnings = []

    if not config.cartridges:
        warnings.append("No knowledge models defined")

    for alias, path in config.cartridges.items():
        if not Path(path).exists():
            warnings.append(f"Cartridge not found: {alias} -> {path}")

    valid_modes = {"augment", "constrain", "knowledge"}
    for alias, mode in config.injection_modes.items():
        if mode not in valid_modes:
            warnings.append(f"Invalid injection mode for {alias}: {mode} (use: {valid_modes})")
        if alias not in config.cartridges:
            warnings.append(f"Injection mode for unknown cartridge: {alias}")

    if config.weights:
        for alias in config.weights:
            if alias not in config.cartridges:
                warnings.append(f"Weight for unknown cartridge: {alias}")

    return warnings
