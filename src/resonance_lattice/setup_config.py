# SPDX-License-Identifier: BUSL-1.1
"""Setup configuration — persistent project-level setup decisions.

Stored as `.rlat/setup.toml`. Read/written by the setup wizard and
replayable via `rlat setup --config ... --non-interactive`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SetupConfig:
    """All decisions captured by the setup wizard."""

    # Source selection
    sources: list[str] = field(default_factory=list)  # empty = auto-detect
    cartridge_path: str = ".rlat/project.rlat"

    # Encoder
    encoder_preset: str = "e5-large-v2"
    onnx: bool = False
    onnx_dir: str = ".rlat/onnx/"

    # Field
    bands: int = 5
    dim: int = 2048
    field_type: str = "dense"
    precision: str = "f32"
    compression: str = "none"

    # Memory
    memory_enabled: bool = True
    memory_root: str = ".rlat/memory"

    # Integration
    mcp: bool = True
    claude_md: bool = True
    summary_path: str = ".claude/resonance-context.md"

    def to_toml(self) -> str:
        """Serialise to TOML string."""
        lines = [
            "[project]",
            f'sources = [{", ".join(repr(s) for s in self.sources)}]',
            f'cartridge_path = "{self.cartridge_path}"',
            "",
            "[encoder]",
            f'preset = "{self.encoder_preset}"',
            f"onnx = {'true' if self.onnx else 'false'}",
            f'onnx_dir = "{self.onnx_dir}"',
            "",
            "[field]",
            f"bands = {self.bands}",
            f"dim = {self.dim}",
            f'field_type = "{self.field_type}"',
            f'precision = "{self.precision}"',
            f'compression = "{self.compression}"',
            "",
            "[memory]",
            f"enabled = {'true' if self.memory_enabled else 'false'}",
            f'root = "{self.memory_root}"',
            "",
            "[integration]",
            f"mcp = {'true' if self.mcp else 'false'}",
            f"claude_md = {'true' if self.claude_md else 'false'}",
            f'summary_path = "{self.summary_path}"',
            "",
        ]
        return "\n".join(lines)

    @classmethod
    def from_toml(cls, text: str) -> SetupConfig:
        """Deserialise from TOML string."""
        data = tomllib.loads(text)

        project = data.get("project", {})
        encoder = data.get("encoder", {})
        field_cfg = data.get("field", {})
        memory = data.get("memory", {})
        integration = data.get("integration", {})

        return cls(
            sources=project.get("sources", []),
            cartridge_path=project.get("cartridge_path", ".rlat/project.rlat"),
            encoder_preset=encoder.get("preset", "e5-large-v2"),
            onnx=encoder.get("onnx", False),
            onnx_dir=encoder.get("onnx_dir", ".rlat/onnx/"),
            bands=field_cfg.get("bands", 5),
            dim=field_cfg.get("dim", 2048),
            field_type=field_cfg.get("field_type", "dense"),
            precision=field_cfg.get("precision", "f32"),
            compression=field_cfg.get("compression", "none"),
            memory_enabled=memory.get("enabled", True),
            memory_root=memory.get("root", ".rlat/memory"),
            mcp=integration.get("mcp", True),
            claude_md=integration.get("claude_md", True),
            summary_path=integration.get("summary_path", ".claude/resonance-context.md"),
        )

    def save(self, path: str | Path) -> None:
        """Write config to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_toml(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SetupConfig | None:
        """Load config from file, or None if not found."""
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_toml(p.read_text(encoding="utf-8"))
