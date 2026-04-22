# SPDX-License-Identifier: BUSL-1.1
"""Setup Wizard — guided project setup for Resonance Lattice.

Walks users through knowledge model building, encoder/field configuration,
skill-knowledge model wiring, layered memory, integration, and optional deps.

Usage::

    rlat setup                                    # interactive
    rlat setup --non-interactive                  # all defaults
    rlat setup --config .rlat/setup.toml --ni     # replay saved config
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from resonance_lattice.setup_config import SetupConfig

SETUP_TOML = ".rlat/setup.toml"


# ═══════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════


def detect_existing_setup() -> dict:
    """Scan for existing rlat artefacts and return their state."""
    state: dict = {}

    setup_toml = Path(SETUP_TOML)
    if setup_toml.exists():
        state["config"] = SetupConfig.load(setup_toml)

    cartridge = Path(".rlat/project.rlat")
    if cartridge.exists():
        state["knowledge model"] = {
            "path": cartridge,
            "size_mb": cartridge.stat().st_size / (1024 * 1024),
        }

    memory_root = Path(".rlat/memory")
    if memory_root.exists():
        state["memory"] = {
            name: (memory_root / f"{name}.rlat").exists()
            for name in ("working", "episodic", "semantic")
        }

    skills_root = Path(".claude/skills")
    if skills_root.is_dir():
        state["skills"] = sorted(
            d.name for d in skills_root.iterdir() if d.is_dir()
        )

    return state


def _scan_directories() -> list[dict]:
    """Scan top-level directories for ingestable files with counts."""
    from resonance_lattice.cli import INGEST_EXTENSIONS, SKIP_DIRS

    cwd = Path(".")
    results = []
    for child in sorted(cwd.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in SKIP_DIRS:
            continue
        count = sum(
            1 for ext in INGEST_EXTENSIONS
            for _ in child.rglob(f"*{ext}")
        )
        if count > 0:
            results.append({"name": str(child), "count": count})
    return results


# ═══════════════════════════════════════════════════════════
# Interactive step functions
# ═══════════════════════════════════════════════════════════


def step_sources(config: SetupConfig, existing: dict) -> None:
    """Step 1: Source selection."""
    from resonance_lattice import wizard_ui as ui

    dirs = _scan_directories()
    if not dirs:
        print("No ingestable directories found. Using current directory.", file=sys.stderr)
        config.sources = ["."]
        return

    # Common auto-detected names get pre-checked
    auto_names = {"docs", "src", "lib"}
    choices = [
        {"name": f"{d['name']} ({d['count']} files)", "checked": Path(d["name"]).name in auto_names}
        for d in dirs
    ]

    selected = ui.checkbox("Source directories to index:", choices)
    # Extract directory names from display strings
    config.sources = [s.split(" (")[0] for s in selected] if selected else ["."]

    config.cartridge_path = ui.text("Knowledge Model output path:", default=config.cartridge_path)


def step_encoder(config: SetupConfig, existing: dict) -> None:
    """Step 2: Encoder and field configuration."""
    from resonance_lattice import wizard_ui as ui
    from resonance_lattice.config import ENCODER_PRESETS

    presets = list(ENCODER_PRESETS.keys())
    preset_display = []
    for name in presets:
        info = ENCODER_PRESETS[name]
        backbone = info["backbone"].split("/")[-1]
        max_len = info.get("max_length", 512)
        preset_display.append(f"{name} ({backbone}, {max_len} tokens)")

    selected = ui.select(
        "Encoder preset:",
        choices=preset_display,
        default=next((d for d in preset_display if d.startswith(config.encoder_preset)), preset_display[0]),
    )
    config.encoder_preset = selected.split(" (")[0]

    # Field config — offer fast path
    customize = not ui.confirm(
        f"Accept default field config? (bands={config.bands}, dim={config.dim}, "
        f"{config.field_type}, {config.precision})",
        default=True,
    )
    if customize:
        raw_bands = ui.text("Bands:", default=str(config.bands))
        config.bands = int(raw_bands) if raw_bands.isdigit() else config.bands
        raw_dim = ui.text("Dimension:", default=str(config.dim))
        config.dim = int(raw_dim) if raw_dim.isdigit() else config.dim
        config.field_type = ui.select(
            "Field type:", choices=["dense", "factored", "pq"], default=config.field_type,
        )
        config.precision = ui.select(
            "Precision:", choices=["f16", "bf16", "f32"], default=config.precision,
        )
        config.compression = ui.select(
            "Compression:", choices=["none", "zstd", "lz4"], default=config.compression,
        )

    # ONNX
    try:
        import onnxruntime  # noqa: F401
        onnx_available = True
    except ImportError:
        onnx_available = False

    if onnx_available:
        config.onnx = ui.confirm("Enable ONNX acceleration? (onnxruntime detected)", default=True)
    else:
        install_onnx = ui.confirm(
            "ONNX acceleration (2-5x faster CPU encoding) — onnxruntime not installed. Install now?",
            default=False,
        )
        if install_onnx:
            config.onnx = True  # will be installed in execute step


def step_skills(config: SetupConfig, existing: dict) -> None:
    """Step 3: Skill-knowledge model configuration."""
    from resonance_lattice import wizard_ui as ui
    from resonance_lattice.skill import discover_skills

    skills_root = Path(".claude/skills")
    if not skills_root.is_dir():
        ui.skip("No .claude/skills/ directory found")
        return

    skills = discover_skills(skills_root)
    if not skills:
        ui.skip("No skills with frontmatter found")
        return

    ui.header("Discovered skills")
    for s in skills:
        if s.cartridges:
            missing = [c for c in s.cartridges if not Path(c).exists()]
            if missing:
                ui.warn(f"{s.name} — cartridges: {', '.join(s.cartridges)} (missing: {', '.join(missing)})")
            else:
                ui.done(f"{s.name} — cartridges: {', '.join(s.cartridges)}")
        elif s.cartridge_sources:
            ui.info(f"{s.name} — has references/, no cartridge built yet")
        else:
            ui.skip(f"{s.name} — no cartridge fields")

    # Offer to build skill cartridges from references/
    buildable = [s for s in skills if s.resolve_source_paths()]
    if buildable:
        choices = [
            {"name": f"{s.name} (from {', '.join(str(p) for p in s.resolve_source_paths())})", "checked": True}
            for s in buildable
        ]
        selected = ui.checkbox("Build knowledge models for skills with reference materials?", choices)
        # Store the skill names to build during execution
        config._skills_to_build = [s.split(" (")[0] for s in selected]  # type: ignore[attr-defined]


def step_memory(config: SetupConfig, existing: dict) -> None:
    """Step 4: Layered memory configuration."""
    from resonance_lattice import wizard_ui as ui

    if "memory" in existing:
        for t, v in existing["memory"].items():
            if v:
                ui.done(f"{t} tier exists")
            else:
                ui.warn(f"{t} tier missing")
        config.memory_enabled = ui.confirm("Keep existing memory?", default=True)
        if config.memory_enabled:
            return  # keep existing, don't reinitialise

    config.memory_enabled = ui.confirm(
        "Enable layered memory for chat history? (working/episodic/semantic tiers)",
        default=True,
    )
    if config.memory_enabled:
        config.memory_root = ui.text("Memory location:", default=config.memory_root)


def step_integration(config: SetupConfig, existing: dict) -> None:
    """Step 5: Integration wiring."""
    from resonance_lattice import wizard_ui as ui

    choices = [
        {"name": "Configure MCP server in .mcp.json", "checked": config.mcp},
        {"name": "Generate context summary", "checked": True},
        {"name": "Inject knowledge model section into CLAUDE.md", "checked": config.claude_md},
    ]
    selected = ui.checkbox("Auto-integrate with development tools?", choices)
    selected_names = set(selected)
    config.mcp = "Configure MCP server in .mcp.json" in selected_names
    config.claude_md = "Inject knowledge model section into CLAUDE.md" in selected_names

    if "Generate context summary" in selected_names:
        config.summary_path = ui.text("Summary path:", default=config.summary_path)


# ═══════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════


def execute(config: SetupConfig, *, quiet: bool = False) -> None:
    """Execute all setup actions based on the config."""
    from resonance_lattice import wizard_ui as ui
    from resonance_lattice.cli import (
        _auto_detect_inputs,
        _collect_files,
        build_project_cartridge,
        generate_project_summary,
        wire_integration,
    )

    total = 5

    # ── Step 1: Build project cartridge ──────────────────────────────
    if not quiet:
        ui.header("Setting up Resonance Lattice")
        ui.step_banner(1, total, "Building project knowledge model...")

    if config.sources:
        input_paths = [Path(s) for s in config.sources]
    else:
        input_paths = _auto_detect_inputs()
        if not quiet:
            ui.info(f"Auto-detected: {', '.join(str(p) for p in input_paths)}")

    files = _collect_files(input_paths)
    if not files:
        if not quiet:
            ui.warn("No files found.")
        sys.exit(1)

    cartridge_path = Path(config.cartridge_path)

    if quiet:
        _lattice, count = build_project_cartridge(
            files, cartridge_path,
            encoder_preset=config.encoder_preset,
            bands=config.bands, dim=config.dim,
            field_type_str=config.field_type,
            precision_str=config.precision,
            compression_str=config.compression,
            onnx_dir=config.onnx_dir if config.onnx else None,
            quiet=True,
        )
    else:
        with ui.Spinner(f"Encoding {len(files)} files into {cartridge_path}"):
            _lattice, count = build_project_cartridge(
                files, cartridge_path,
                encoder_preset=config.encoder_preset,
                bands=config.bands, dim=config.dim,
                field_type_str=config.field_type,
                precision_str=config.precision,
                compression_str=config.compression,
                onnx_dir=config.onnx_dir if config.onnx else None,
                quiet=True,
            )
        ui.done(f"{cartridge_path} ({count} chunks)")

    # ── Step 2: ONNX ─────────────────────────────────────────────────
    if not quiet:
        ui.step_banner(2, total, "Optional dependencies")
    if config.onnx:
        try:
            import onnxruntime  # noqa: F401
            if not quiet:
                ui.done("onnxruntime already installed")
        except ImportError:
            if not quiet:
                with ui.Spinner("Installing onnxruntime"):
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "onnxruntime>=1.17"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
            else:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "onnxruntime>=1.17"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            if not quiet:
                ui.done("onnxruntime installed")
    else:
        if not quiet:
            ui.skip("ONNX acceleration not selected")

    # ── Step 3: Layered memory ───────────────────────────────────────
    if not quiet:
        ui.step_banner(3, total, "Layered memory")
    if config.memory_enabled:
        memory_root = Path(config.memory_root)
        if not (memory_root / "working.rlat").exists():
            from resonance_lattice.layered_memory import LayeredMemory
            LayeredMemory.init(memory_root)
            if not quiet:
                ui.done(f"{memory_root}/ (working, episodic, semantic)")
        else:
            if not quiet:
                ui.done(f"Memory already exists at {memory_root}/")
    else:
        if not quiet:
            ui.skip("Memory disabled")

    # ── Step 4: Skill cartridges ─────────────────────────────────────
    if not quiet:
        ui.step_banner(4, total, "Skill knowledge models")
    skills_to_build: list[str] = getattr(config, "_skills_to_build", [])
    if skills_to_build:
        from resonance_lattice.skill import find_skill
        skills_root = Path(".claude/skills")
        for skill_name in skills_to_build:
            skill = find_skill(skills_root, skill_name)
            if skill is None:
                if not quiet:
                    ui.warn(f"{skill_name}: not found")
                continue
            source_paths = skill.resolve_source_paths()
            if not source_paths:
                if not quiet:
                    ui.skip(f"{skill_name}: no reference materials")
                continue
            skill_files = _collect_files(source_paths)
            if skill_files:
                cart_path = skill.local_cartridge_path()
                cart_path.parent.mkdir(parents=True, exist_ok=True)
                if quiet:
                    _lat, n = build_project_cartridge(
                        skill_files, cart_path,
                        encoder_preset=config.encoder_preset,
                        bands=config.bands, dim=config.dim,
                        field_type_str=config.field_type,
                        precision_str=config.precision,
                        compression_str=config.compression,
                        quiet=True,
                    )
                else:
                    with ui.Spinner(f"Building {skill_name}"):
                        _lat, n = build_project_cartridge(
                            skill_files, cart_path,
                            encoder_preset=config.encoder_preset,
                            bands=config.bands, dim=config.dim,
                            field_type_str=config.field_type,
                            precision_str=config.precision,
                            compression_str=config.compression,
                            quiet=True,
                        )
                    ui.done(f"{skill_name} -> {cart_path} ({n} chunks)")
    else:
        if not quiet:
            ui.skip("No skill knowledge models selected")

    # ── Step 5: Integration wiring ───────────────────────────────────
    if not quiet:
        ui.step_banner(5, total, "Integration wiring")
    summary_path = Path(config.summary_path)
    if quiet:
        generate_project_summary(cartridge_path, summary_path, count, quiet=True)
        wire_integration(
            cartridge_path, files, summary_path,
            update_mcp=config.mcp, update_claude=config.claude_md, quiet=True,
        )
    else:
        with ui.Spinner("Generating context summary"):
            generate_project_summary(cartridge_path, summary_path, count, quiet=True)
        ui.done(f"Summary: {summary_path}")

        wire_integration(
            cartridge_path, files, summary_path,
            update_mcp=config.mcp, update_claude=config.claude_md, quiet=True,
        )
        if config.mcp:
            ui.done("MCP server configured in .mcp.json")
        if config.claude_md:
            ui.done("Knowledge Model section injected into CLAUDE.md")
        ui.done("Manifest updated")

    # Save config
    config.save(SETUP_TOML)
    if not quiet:
        ui.done(f"Configuration saved to {SETUP_TOML}")

    # ── Final summary ────────────────────────────────────────────────
    if not quiet:
        ui.final_summary([
            "Setup complete!",
            "",
            f"  Cartridge:  {config.cartridge_path} ({count} chunks)",
            f"  Memory:     {'enabled' if config.memory_enabled else 'disabled'}",
            f"  MCP:        {'configured' if config.mcp else 'skipped'}",
            f"  Config:     {SETUP_TOML}",
            "",
            "Next steps:",
            f'  rlat search {config.cartridge_path} "your question"',
            "  rlat setup --reconfigure",
        ])


# ═══════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════


def run_wizard(
    *,
    non_interactive: bool = False,
    config_path: str | None = None,
    reconfigure: bool = False,
    # CLI overrides
    encoder: str | None = None,
    no_memory: bool = False,
    precision: str | None = None,
    compression: str | None = None,
) -> None:
    """Run the setup wizard (interactive or non-interactive)."""
    # Load existing config as defaults
    existing = detect_existing_setup()
    config = existing.get("config") or SetupConfig()

    # Load from explicit config file if provided
    if config_path:
        loaded = SetupConfig.load(config_path)
        if loaded:
            config = loaded

    # Apply CLI overrides
    if encoder:
        config.encoder_preset = encoder
    if no_memory:
        config.memory_enabled = False
    if precision:
        config.precision = precision
    if compression:
        config.compression = compression

    if non_interactive:
        # Non-interactive: use config as-is (defaults + overrides)
        execute(config)
        return

    # Interactive flow
    from resonance_lattice import wizard_ui as ui
    ui.header("rlat setup — Resonance Lattice project wizard")

    if existing.get("knowledge model") and not reconfigure:
        cart = existing["knowledge model"]
        ui.info(f"Cartridge: {cart['path']} ({cart['size_mb']:.1f} MB)")
        if "memory" in existing:
            tiers = sum(1 for v in existing["memory"].values() if v)
            ui.info(f"Memory: {tiers}/3 tiers")
        if "skills" in existing:
            ui.info(f"Skills: {len(existing['skills'])} found")

        action = ui.select("What would you like to do?", choices=[
            "Full setup (recommended for first time)",
            "Reconfigure existing setup",
            "Rebuild knowledge models only",
            "Configure memory only",
            "Configure skills only",
        ], default="Reconfigure existing setup")

        if action == "Rebuild knowledge models only":
            execute(config)
            return
        elif action == "Configure memory only":
            step_memory(config, existing)
            execute(config)
            return
        elif action == "Configure skills only":
            step_skills(config, existing)
            execute(config)
            return
        # else: full or reconfigure — fall through to all steps

    # Run all steps
    step_sources(config, existing)
    step_encoder(config, existing)
    step_skills(config, existing)
    step_memory(config, existing)
    step_integration(config, existing)

    # Execute
    execute(config)
