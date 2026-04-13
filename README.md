# Resonance Lattice

Portable semantic knowledge infrastructure for AI assistants.

Resonance Lattice turns docs, code, notes, and other file collections into local `.rlat` cartridges you can search, inspect, compare, compose, and inject into assistant workflows.

```bash
pip install resonance-lattice
rlat build ./docs ./src -o project.rlat
rlat search project.rlat "how does this project work?"
```

## What It Is

Resonance Lattice is a local semantic knowledge layer. It is not a hosted retrieval service, not a vector database product, and not an LLM. The core unit is the cartridge: a portable artifact that packages a semantic field, a source registry, and an evidence store.

Read [Overview](docs/OVERVIEW.md), [Cartridge Architecture](docs/CARTRIDGE_ARCHITECTURE.md), and [Semantic Model](docs/SEMANTIC_MODEL.md) for the fuller product and architecture story.

## Why Use It

- package project knowledge as a reusable file
- ground assistants in your own sources
- inspect and compare knowledge states instead of only retrieving top-k text
- compose multiple knowledge domains at query time
- reuse the same cartridge across CLI, MCP, HTTP, and plugin surfaces

See [Overview](docs/OVERVIEW.md) for the product thesis, [Benchmarks](docs/BENCHMARKS.md) for the evidence, [FAQ](docs/FAQ.md) for practical comparisons and adoption questions, and [Context Control](docs/CONTEXT_CONTROL.md) for cartridge composition, projection, forgetting, and reusable context workflows.

## Quick Start

Python `>=3.11` is required.

```bash
pip install resonance-lattice
pip install onnxruntime  # optional CPU acceleration
```

Build a cartridge:

```bash
rlat build ./docs ./src -o project.rlat
```

Search it:

```bash
rlat search project.rlat "how does authentication work?"
```

Inspect it:

```bash
rlat profile project.rlat
```

Set up the assistant-facing defaults:

```bash
rlat init-project --auto-integrate
```

For installation, first build, first search, and next steps, see [Getting Started](docs/GETTING_STARTED.md).

## Choose An Interface

- [CLI](docs/CLI.md): build, query, profile, compare, compose, export
- [MCP](docs/MCP.md): assistant-native tool surface with warm cartridge state
- [HTTP](docs/API_REFERENCE.md): local app and plugin integration
- [Obsidian Plugin](docs/OBSIDIAN_PLUGIN.md): vault build, sync, and semantic search
- [Cloud Platform (Planned)](docs/CLOUD_PLATFORM.md): future hosted build, cartridge, and team workflows

## Documentation

The canonical technical documentation now lives in [`docs/`](docs/), and the website renders from that source of truth.

Start here:

- [Overview](docs/OVERVIEW.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Cartridge Architecture](docs/CARTRIDGE_ARCHITECTURE.md)
- [Semantic Model](docs/SEMANTIC_MODEL.md)

Core interfaces:

- [CLI](docs/CLI.md)
- [MCP](docs/MCP.md)
- [API Reference](docs/API_REFERENCE.md)
- [Obsidian Plugin](docs/OBSIDIAN_PLUGIN.md)
- [Cloud Platform (Planned)](docs/CLOUD_PLATFORM.md)

Retrieval and context:

- [Context Control](docs/CONTEXT_CONTROL.md)
- [Skills Integration and Architecture](docs/SKILL_INTEGRATION.md)
- [Discovery and Auto-Integration](docs/DISCOVERY.md)
- [LLM Memory Architecture](docs/LLM_MEMORY.md)

Deep dives:

- [Encoder Guide](docs/ENCODERS.md)
- [RQL Guide](docs/RQL_GUIDE.md)
- [RQL Reference](docs/RQL_REFERENCE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Benchmark Runbook](docs/RETRIEVAL_BENCHMARK_RUNBOOK.md)

Reference:

- [FAQ](docs/FAQ.md)
- [Glossary](docs/GLOSSARY.md)
- [Status and Boundaries](docs/STATUS_AND_BOUNDARIES.md)

## Benchmark Highlights

- the full retrieval pipeline is benchmarked as a workflow, not just as a raw encoder
- token-efficiency runs compare RL output with whole-file read workflows
- grounding benchmarks measure answer quality with and without RL context
- BEIR and comparative runs help separate production claims from corpus-specific wins

See [Benchmarks](docs/BENCHMARKS.md) and [Benchmark Runbook](docs/RETRIEVAL_BENCHMARK_RUNBOOK.md) for the current benchmark contract.

## Status

Alpha (`0.9.0`).

See [Status and Boundaries](docs/STATUS_AND_BOUNDARIES.md) for shipped surfaces, experimental areas, and current product limits.

## License

[Apache 2.0](LICENSE.md)
