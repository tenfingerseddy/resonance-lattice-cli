---
title: MCP
slug: mcp
description: Use the MCP server to expose knowledge model search, diagnostics, discovery, and skill tools directly inside assistant workflows.
nav_group: Interfaces
nav_order: 10
aliases:
---

# MCP

## What Is The MCP Server?

The MCP server is the assistant-native interface for Resonance Lattice. It runs over stdio and exposes knowledge model tools directly to clients such as Claude Code.

## Why Should I Use It?

Use MCP when you want the knowledge model to behave like a warm tool inside the conversation instead of shelling out to the CLI for each request.

It is the best fit when you want:

- repeated low-latency queries in one session
- native search and diagnostics tools inside the assistant
- discovery and skill-routing support tied to the same knowledge model layer

## How Does It Work?

### Tool Groups

The MCP server exposes **19 tools** (as of v0.11.0) grouped by purpose:

- **Search and context** — `rlat_search`, `rlat_resonate`, `rlat_compose_search`, `rlat_ask`
- **Info and diagnostics** — `rlat_info`, `rlat_profile`, `rlat_compare`, `rlat_locate`, `rlat_xray`, `rlat_health`, `rlat_negotiate`
- **Discovery and freshness** — `rlat_discover`, `rlat_freshness`, `rlat_switch`
- **Skill routing and injection** — `rlat_skill_route`, `rlat_skill_inject`
- **Memory** — `rlat_memory_recall`, `rlat_memory_save`, `rlat_memory_forget`

### Runtime Behavior

The server defers knowledge model loading until the first tool call, then keeps the knowledge model and encoder warm in memory. Additional knowledge models can be cached and switched at runtime.

## How Do I Use It?

### Basic Setup

```json
{
  "mcpServers": {
    "rlat": {
      "command": "rlat",
      "args": ["mcp", ".rlat/project.rlat"]
    }
  }
}
```

### When To Choose MCP Instead Of CLI Or HTTP

- choose MCP for assistant-native conversations
- choose CLI for terminal workflows and scripts
- choose HTTP when another application needs a conventional request/response interface

## Technical Guide

### Current Tool Surface

The shipped MCP server exposes the following tools, each with the canonical `rlat_*` name used in the MCP wire protocol:

| Tool | Purpose |
|------|---------|
| `rlat_search` | Enriched search with per-band scores, coverage, and optional cascade / contradictions / cross-encoder rerank |
| `rlat_resonate` | Clean LLM-ready context (passages formatted for prompt injection; no scoring noise) |
| `rlat_compose_search` | Multi-knowledge-model search (`--with` / `--through` / `--diff` composition) |
| `rlat_ask` | Auto-lens dispatcher — picks the best retrieval mode (search / locate / profile / compare / compose / negotiate) from the question |
| `rlat_info` | Knowledge model summary: size, bands, source count, encoder fingerprint |
| `rlat_profile` | Semantic shape: per-band energy, effective rank, spectral entropy, coverage |
| `rlat_compare` | Side-by-side structural comparison of two knowledge models |
| `rlat_locate` | Query positioning — where a question sits in the knowledge landscape |
| `rlat_xray` | Deep diagnostics — signal/noise, saturation, spectral gap, purity |
| `rlat_health` | Quick health gate; Marchenko–Pastur signal/noise split |
| `rlat_negotiate` | Multi-knowledge-model budget / content negotiation helper |
| `rlat_switch` | Runtime switch to a cached secondary knowledge model |
| `rlat_discover` | Enumerate available knowledge models in the project `.rlat/` directory |
| `rlat_freshness` | Remote-mode drift check against the pinned upstream SHA |
| `rlat_skill_route` | Rank available skills by resonance energy for a given query |
| `rlat_skill_inject` | Four-tier adaptive context injection for a knowledge-model-backed skill |
| `rlat_memory_recall` | Layered-memory recall (working / episodic / semantic) |
| `rlat_memory_save` | Write a memory entry through the LayeredMemory pipeline |
| `rlat_memory_forget` | Exact-subtraction removal of a memory entry with certificate |

See `src/resonance_lattice/mcp_server.py` for the authoritative list and input schemas.

### Operational Notes

- the primary knowledge model is configured at startup
- manifest-backed discovery (`rlat_discover`) improves multi-knowledge-model workflows
- skill runtime is initialized lazily and shares the warm encoder when possible
- storage modes are transparent to the MCP client — a `local` / `bundled` / `remote` knowledge model all present the same tool surface

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
