# Resonance Lattice

**Portable semantic knowledge infrastructure for AI assistants.**

Resonance Lattice applies semantic-model thinking to unstructured knowledge, turning docs and code into portable `.rlat` knowledge cartridges you can search, inspect, compose, diff, and use across assistant workflows.

Build local knowledge cartridges for retrieval, memory management, skill integration, fine-grained context control, context priming, and offline or LLM-free use.

```bash
pip install resonance-lattice
rlat build ./docs ./src -o project.rlat
rlat search project.rlat "how does this project work?" --format text
```

---

## What Is Resonance Lattice?

Resonance Lattice is a local semantic knowledge system for unstructured knowledge and AI assistants. It turns docs, code, and other file collections into portable `.rlat` knowledge cartridges.

A knowledge cartridge is a semantic model packaged as a file. You can search it, inspect it, compose it with other cartridges, diff versions, and use it across assistant workflows without standing up a hosted retrieval stack.

It is not a hosted retrieval service, not a vector database, and not an LLM. It is portable knowledge infrastructure: a queryable layer between your source material and the assistants or tools that use it.

For the fuller product and architecture story, see [Overview](docs/OVERVIEW.md), [Cartridge Architecture](docs/CARTRIDGE_ARCHITECTURE.md), and [Semantic Model](docs/SEMANTIC_MODEL.md).

---

## Why Use Resonance Lattice

Use Resonance Lattice when you want retrieval that behaves like infrastructure: portable, inspectable, composable, and local. It sits between raw source material and the assistants or tools that need grounded access to it.

| If you use | What it is good at | What Resonance Lattice adds |
|------------|--------------------|-----------------------------|
| **grep** | Exact text match | Semantic retrieval, inspection, and reusable cartridges |
| **Standard RAG** | Hosted or index-backed retrieval | Portable file artifacts, local control, and composition |
| **LLM direct** | Reasoning and generation | Grounded evidence from your own sources |
| **Note vaults** | Human-authored organization | Automatic semantic modeling over the files you already have |

### 1. Your knowledge is a file

A cartridge is a single `.rlat` artifact you can move, version, archive, diff, and share like any other file. The knowledge layer is portable, not tied to a service dependency.

### 2. It is composable for fine-grained context control

Cartridges are building blocks, not monoliths. Build per-domain cartridges for docs, code, design, or compliance and combine them per question. Composition happens locally at query time.

See [Context Control](docs/CONTEXT_CONTROL.md) for the deeper composition model, reusable context setups, and cartridge algebra workflows.

| Operation | What it does | Guarantee |
|-----------|--------------|-----------|
| **merge** | Combine cartridges into a unified field | Commutative, associative |
| **diff** | Surface what changed between two snapshots | Directional signed delta |
| **forget** | Remove a knowledge subset cleanly | Algebraically precise |
| **project** | View one cartridge through the lens of another | Semantic projection, not keyword filtering |
| **contradict** | Find where two cartridges disagree | Symmetric divergence detection |

```bash
# Multi-cartridge search
rlat search docs.rlat "auth flow" --with code.rlat

# View code through a compliance lens
rlat search code.rlat "data handling" --through compliance.rlat

# What changed since baseline?
rlat search current.rlat "what changed?" --diff baseline.rlat
```

### 3. It is inspectable

You do not just get top-k results. Resonance Lattice exposes structural diagnostics that standard vector-database workflows do not usually surface directly:

| Command | What it tells you |
|---------|-------------------|
| `rlat xray corpus.rlat` | Corpus-level health: signal quality, saturation, diagnostic flags |
| `rlat locate corpus.rlat "query"` | Where a query sits in the knowledge landscape and what the field does not cover |
| `rlat probe corpus.rlat <recipe>` | Quick insight recipes for novelty, saturation, coverage gaps, and more |

### 4. It is local-first

No hosted service, no API keys, and no LLM required. The first build downloads the default encoder from Hugging Face; after that, building and querying can run fully locally. Pair it with a local LLM such as Ollama or llama.cpp for a fully private stack.

### 5. It is mathematical, not generative

The field has algebraic structure. Adding a source, removing it, merging cartridges, and diffing versions are defined operations with stable behavior. Removing a file returns the field to the same state it had before that file was added. Merge is order-independent. The same inputs produce the same field. No temperature, no sampling, no drift.

---

## Quick Start

Python `>=3.11` is required. The first build downloads the default encoder. Install `onnxruntime` if you want faster local encoding.

```bash
pip install resonance-lattice
pip install onnxruntime  # optional
```

Build a cartridge from your docs and code:

```bash
rlat build ./docs ./src -o project.rlat
```

Search it:

```bash
rlat search project.rlat "how does auth work?" --format text
```

Inspect what the cartridge covers:

```bash
rlat profile project.rlat
```

When you want assistant-native setup, initialize the project and wire in the defaults:

```bash
rlat init-project --auto-integrate
```

This auto-detects common project inputs, builds `.rlat/project.rlat`, generates `.claude/resonance-context.md`, creates `.rlat/manifest.json`, and with `--auto-integrate` updates `.mcp.json` and injects a cartridge section into `CLAUDE.md`.

For the full walkthrough - including profiling, comparison, MCP integration, assistant context files, and HTTP serving - see [Getting Started](docs/GETTING_STARTED.md).

---

## Assistant Integration

These paths are composable rather than exclusive. You might use MCP and CLI side by side, and layer cartridge-backed skills or instruction files on top when you want dynamic, targeted context for different applications of the same skill.

See [CLI](docs/CLI.md), [MCP](docs/MCP.md), [API Reference](docs/API_REFERENCE.md), and [Skills Integration and Architecture](docs/SKILL_INTEGRATION.md) for the detailed interface docs.

### MCP Server

Use the MCP server when your assistant supports MCP and you want cartridge search and diagnostics available as native tools inside the conversation.

| Assistant | Config file |
|-----------|------------|
| Claude Code | `.mcp.json` in project root |
| VS Code / GitHub Copilot | `.vscode/mcp.json` |
| Cursor | `cursor/mcp.json` or Settings > MCP |

```json
{
  "mcpServers": {
    "rlat": {
      "command": "rlat",
      "args": ["mcp", "project.rlat"]
    }
  }
}
```

Once loaded, the assistant can:

- search or generate LLM-ready context from a cartridge
- compose multiple cartridges, diff them, and inspect freshness or discovery state
- run diagnostics such as profile, compare, locate, and xray
- route into cartridge-backed skill workflows when those are configured

Because the server keeps the cartridge and encoder warm in memory, repeated queries avoid the cold-start path.

### CLI

Use the CLI when the assistant has terminal access or when you want zero extra configuration. This works with Claude Code, GitHub Copilot, Cursor, Codex, Cline, Windsurf, and any other tool that can run shell commands.

```bash
rlat search project.rlat "how does auth work?" --format json
rlat resonate project.rlat "what are the design constraints?" --format context
rlat profile project.rlat
```

When you want LLM-ready output, `--mode` controls how the context is framed:

| Mode | What it tells the LLM |
|------|----------------------|
| **augment** | Use your own knowledge, but add detail and citations from these sources |
| **constrain** | Answer ONLY from the provided sources — if it's not covered, say so |
| **knowledge** | Base your answer primarily on this context; be transparent about gaps |
| **custom** | Your own system prompt, your rules |

```bash
rlat resonate project.rlat "how does auth work?" --mode constrain --format context
```

### Skills and Instruction Files

Use cartridge-backed skills and instruction files alongside MCP or CLI when you want the same skill to keep its workflow structure while its knowledge changes with the request. This is the key advantage of the skills system: dynamic, targeted context instead of the same static background on every run.

A skill becomes **cartridge-backed** by adding a few frontmatter fields:

```yaml
---
name: fabric-notebook-ingest
description: Create Fabric notebooks for data ingestion...
cartridges:
  - .rlat/fabric-docs.rlat
  - .rlat/pyspark-docs.rlat
cartridge-queries:
  - "How do you create a notebook in Fabric through the API"
  - "Fabric workspace authentication and authorization patterns"
cartridge-mode: augment
cartridge-budget: 2000
---
```

When a cartridge-backed skill triggers, context can load in four tiers:

| Tier | Source | What it provides |
|------|--------|-----------------|
| **1. Static** | SKILL.md header | Workflow structure, templates, decision trees |
| **2. Foundational** | Skill-authored queries | Baseline knowledge the skill always needs (40% of budget) |
| **3. Specific** | User query | Context unique to this request (30% of budget) |
| **4. Derived** | Caller-supplied queries | Implicit needs the user didn't express (30% of budget) |

Tiers 1-3 are automatic. Tier 4 accepts additional queries from the caller via `--derived`, which lets an orchestrating agent surface knowledge the user did not explicitly ask for. Skills without `cartridge-*` fields work exactly as before. See [docs/SKILL_INTEGRATION.md](docs/SKILL_INTEGRATION.md) for the full architecture.

You can also generate a supplemental context file and reference it from your instruction file:

```bash
rlat summary project.rlat -o .rlat/resonance-context.md
```

| Tool | Instruction file | Reference syntax |
|------|-----------------|------------------|
| Claude Code | `CLAUDE.md` | `@.rlat/resonance-context.md` |
| GitHub Copilot | `.github/copilot-instructions.md` | `@.rlat/resonance-context.md` |
| Cursor | `.cursorrules` | `@.rlat/resonance-context.md` |

### How It Complements Other Layers

| Layer | Role |
|-------|------|
| Instruction files | Rules and project conventions |
| Skills | Executable capabilities and tool-use patterns |
| Memory | Persistent facts across conversations; Resonance Lattice can also back this layer when you index notes, sessions, or other artifacts |
| **Resonance Lattice** | Queryable project knowledge and adaptive context |

Resonance Lattice is complementary to the other layers. It gives them a reusable semantic knowledge layer they can query, inspect, and adapt to the task at hand.

---

## Common Workflows

### Search and inspect a cartridge

Use `search` when you want evidence-backed retrieval, and `profile` when you want to understand the semantic shape of the cartridge. `search` supports `text`, `json`, `context`, and `prompt` formats depending on whether you are reading, scripting, or feeding an assistant.

```bash
rlat search project.rlat "how does auth work?" --format text
rlat search project.rlat "how does auth work?" --format json
rlat profile project.rlat
```

### Compare versions or knowledge domains

Use `compare` when you want overlap and coverage differences, and `diff` when you want a queryable semantic delta.

```bash
rlat compare baseline.rlat current.rlat
rlat diff current.rlat baseline.rlat -o delta.rlat
```

### Update cartridges as sources change

Use `add` when you want to append new sources, and `sync` when you want the cartridge to track file additions, updates, and deletions across source directories.

```bash
rlat add project.rlat ./new_docs
rlat sync project.rlat ./docs ./src
```

### Compose context across cartridges

Use composition when the answer depends on more than one knowledge domain.

```bash
rlat search docs.rlat "auth flow" --with code.rlat
rlat search code.rlat "data handling" --through compliance.rlat
rlat search current.rlat "what changed?" --diff baseline.rlat
```

### Generate and share assistant context

Use `summary` when you want a compact assistant primer, and `export --field-only` when you want to share the semantic model without the embedded evidence text.

```bash
rlat summary project.rlat -o .rlat/resonance-context.md
rlat export project.rlat -o shared.rlat --field-only
```

---

## How It Works

A knowledge cartridge is a semantic model packaged as a file. `rlat build` chunks your sources, encodes them into a shared semantic space, and writes the result to a `.rlat` cartridge that can be queried later.

Each cartridge contains three layers:

| Layer | What it does |
|-------|--------------|
| **Field** | Stores the semantic model of the corpus |
| **Registry** | Maps semantic hits back to ranked sources |
| **Store** | Returns evidence text, passages, and metadata |

At query time, the question is encoded into the same space. The field and registry identify the best matches, and the store returns the evidence needed for reading, scripting, or assistant injection.

The encoder is part of the contract. The cartridge records the encoder fingerprint used at build time, and query commands normally restore it automatically.

You can package cartridges in two ways:

| Store mode | What you get | Best for |
|-----------|---------------|----------|
| **embedded** (default) | Self-contained `.rlat` with field, registry, and evidence text | Portability, sharing, and archiveability |
| **external** | `.rlat` with field and registry only; evidence is read from local source files via `--source-root` | Smaller artifacts or workflows where source text should stay outside the cartridge |

For deeper inspection, Resonance Lattice also exposes `xray`, `locate`, `probe`, and [RQL](docs/RQL_REFERENCE.md). Those are advanced tools for diagnostics and programmable field operations, not required for normal use.

---

## Benchmarks

Resonance Lattice is not "a new embedding model." Its value comes from the portable cartridge, the retrieval pipeline, and the assistant workflow.

- On a 24,635-chunk Microsoft Fabric documentation corpus with 100 evaluation questions, the full `rlat` retrieval pipeline reached **Recall@5 1.00**, **MRR 0.93**, and **0% failed retrieval**.
- On a 2,266-file corpus, `rlat search` returned **24.6x fewer tokens** than a `grep + read top 5 files` workflow while keeping ranked passages and source attribution intact.
- In LLM-grounding evaluation, feeding `rlat` context reduced hallucinations from **78% to 16%** and raised fact recall from **0.27 to 0.91**.
- On five BEIR datasets, the full pipeline beat flat E5 on three datasets and came within 97% on SciFact. Results vary by corpus, which is why fit to your own data matters more than any single leaderboard.

For full methodology and extended results, see [Benchmarks](docs/BENCHMARKS.md), [Benchmark Runbook](docs/RETRIEVAL_BENCHMARK_RUNBOOK.md), and `benchmarks/results/`.

---

## Practical Limits

The `.rlat` cartridge stays stable as the user-facing abstraction, but a few implementation limits are worth knowing.

| What | Detail | What to do |
|------|--------|------------|
| **Initial build is CPU-intensive** | First build encodes every chunk through E5-large-v2 | Incremental sync only re-processes changed files. ONNX runtime (`pip install onnxruntime`) provides 2-5x CPU speedup. CUDA GPU is supported if available. |
| **Default encoder is English-optimised** | Non-English text can retrieve less reliably with the default setup | The backbone is configurable with `--encoder`, but multilingual alternatives should be revalidated on your corpus. |
| **Best benchmark numbers use the full pipeline** | The strongest retrieval numbers come from lexical injection plus reranking on factual and technical corpora | Full pipeline adds latency, but `rlat search` defaults to auto reranking and stays on the dense path when results are already well-separated. |

---

## Status

**Alpha** (`0.9.0`)

See [Status and Boundaries](docs/STATUS_AND_BOUNDARIES.md) for shipped surfaces, experimental areas, and current limits.

## Docs

- [Overview](docs/OVERVIEW.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [CLI Reference](docs/CLI.md)
- [MCP](docs/MCP.md)
- [Context Control](docs/CONTEXT_CONTROL.md)
- [Encoder Guide](docs/ENCODERS.md)
- [RQL Reference](docs/RQL_REFERENCE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [FAQ](docs/FAQ.md)

## License

[Apache 2.0](LICENSE.md).
