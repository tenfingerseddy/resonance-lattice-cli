---
title: Encoder Guide
slug: encoders
description: Choose, operate, and evaluate encoder presets and backbone configurations for Resonance Lattice.
nav_group: Deep Dives
nav_order: 10
aliases:
---

# Encoder Guide

Resonance Lattice uses a transformer encoder to convert text into semantic vectors during build and query. The encoder path affects retrieval quality, build speed, language coverage, warm-path behavior, and knowledge model compatibility.

## What Do Encoders Do?

The encoder is the front door into the knowledge model. It defines how text is projected into the phase-space representation used by the field, the registry, and the query path.

## Why Should I Care?

Encoder choice is one of the biggest practical quality levers in the system. It changes:

- retrieval quality
- build throughput and warm-path latency
- language and context-window coverage
- compatibility with existing knowledge models

## How Does Encoder Selection Work?

Use named presets when you want a tested configuration. Pass a raw model ID only when you are deliberately exploring beyond the curated presets.

## How Do I Use This Guide?

Use this page to choose a starting encoder, understand the preset catalog, and interpret the benchmark-backed recommendations before changing the production path.

## What Are Encoder Presets?

An encoder preset is a short CLI name that maps to a full encoder configuration: the HuggingFace model ID, query/passage prefixes, pooling strategy, and maximum token length.

```bash
# Use a preset by name
rlat build ./docs ./src -o project.rlat --encoder qwen3-0.6b

# List all available presets
rlat encoders
```

You can also pass any HuggingFace model ID directly:

```bash
rlat build ./docs ./src -o project.rlat --encoder "sentence-transformers/all-MiniLM-L6-v2"
```

Unlisted models use mean pooling and no prefix by default.

## Technical Guide

## Available Presets

### General Purpose

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `e5-large-v2` | intfloat/e5-large-v2 | 335M | 512 | mean | Fast English baseline, proven and no surprises |
| `nomic-v2` | nomic-ai/nomic-embed-text-v2-moe | 305M | 8K | mean | MoE efficiency, longer context than E5 |
| `arctic-embed-2` | Snowflake/snowflake-arctic-embed-l-v2.0 | 335M | 8K | cls | Strong retrieval quality per parameter |
| `nemotron-1b` | nvidia/llama-nemotron-embed-1b-v2 | 1B | 8K | mean | Strong open-weight quality, NVIDIA backed |

### Multilingual

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `bge-m3` | BAAI/bge-m3 | 568M | 8K | cls | Production workhorse, 1000+ languages |
| `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B | 600M | 32K | mean | Long context, multilingual, Apache 2.0 |
| `qwen3-4b` | Qwen/Qwen3-Embedding-4B | 4B | 32K | mean | Serious quality jump over 0.6B |
| `qwen3-8b` | Qwen/Qwen3-Embedding-8B | 8B | 32K | mean | Top of MTEB multilingual leaderboard |

### Code-Strong

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `gte-large` | Alibaba-NLP/gte-large-en-v1.5 | 335M | 8K | mean | Best drop-in E5 replacement, strong on code (47% better MRR) |
| `gte-qwen2-1.5b` | Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1.5B | 8K | mean | Best general quality ≤1.5B, code in training mix |
| `jina-v3` | jinaai/jina-embeddings-v3 | 570M | 8K | mean | Task-specific LoRA adapters, best overall MRR in bake-off |

## Choosing an Encoder

### Decision Guide

**Code-heavy repository** (mostly source files, few docs):
Use `gte-large` or `jina-v3`. GTE-large is the easiest upgrade from E5 (same 335M size, 47% better MRR on code). Jina v3 is the strongest overall in the bake-off (2.5x better code MRR than E5).

**Mixed code and documentation**:
Use `jina-v3` or `gte-large`. These handle both code and prose well. For longer context or multilingual, consider `qwen3-0.6b`.

**Multilingual content**:
Use `bge-m3` (1000+ languages, production-grade) or the `qwen3` series (100+ languages, longer context).

**Fast and lightweight** (laptop, CI, quick iteration):
Use `e5-large-v2` (335M, proven) or `nomic-v2` (305M MoE, 8K context). These load fast and run on any hardware.

**Maximum quality** (server, GPU available):
Use `qwen3-4b` or `qwen3-8b`. These are large and slow to build, but produce the strongest retrieval quality.

### What the Default Gives You

The default encoder is `e5-large-v2`. It is English-optimised, fast, well-tested, and runs on ordinary hardware. The 512-token context limit means long functions or documents are truncated — this is the main reason to consider a larger encoder for code-heavy corpora.

### The Practical Mindset

Do not overthink encoder choice on day one. The product story is:

1. Start with the default
2. Keep the same `.rlat` workflow
3. Upgrade quality, multilinguality, or context length later by switching presets

The knowledge model format and query interface stay the same regardless of which encoder you use.

## How Encoders Bind to Knowledge Models

**One encoder per knowledge model.** The encoder is locked at build time and its fingerprint is stored in the knowledge model manifest.

- `rlat add` and `rlat sync` verify that the current encoder matches the one used to build the knowledge model
- If there is a mismatch, the CLI will refuse to proceed and tell you which encoder the knowledge model expects
- Changing the encoder means rebuilding the knowledge model

This is by design. All vectors in the field must be in the same embedding space for resonance to be meaningful. Mixing encoders would produce a field where similarity scores are unreliable.

## ONNX Acceleration

ONNX Runtime provides 2-5x CPU encoding speedup. To use it:

```bash
pip install onnxruntime

# Build with ONNX acceleration
rlat build ./docs ./src -o project.rlat --onnx ./onnx_backbone
```

The `--onnx` flag points to a directory containing the exported ONNX model. See the [CLI page](/docs/cli) for the full ONNX export workflow.

Not all presets have been tested with ONNX export. The default `e5-large-v2` and standard transformer architectures generally work. Larger models and those requiring `trust_remote_code=True` (like Jina v3 and GTE-Qwen2) may need additional configuration.

## Preset Notes

**GTE-large** is the easiest E5 upgrade: same 335M parameter class, same mean pooling, no special dependencies. In the bake-off it achieved 0.44 MRR (vs E5's 0.30) and matched Jina on documentation queries. Drop-in replacement — rebuild with `--encoder gte-large`.

**GTE-Qwen2** requires last-token pooling for meaningful embeddings. The current encoder pipeline uses mean pooling, which does not produce useful results with this model. This preset is included for forward compatibility — it will work once last-token pooling support is added. Do not use it yet.

**Jina v3** has LoRA task adapters for different retrieval tasks (code, classification, etc.). The preset loads the base model without task routing, which still provides strong general-purpose retrieval. Requires `pip install einops`. Task-specific adapter selection may be supported in future versions.

**Qwen3 models** use a long instruction-format query prefix. This is handled automatically by the preset — you do not need to format your queries differently. Note: in the bake-off, Qwen3-0.6b underperformed on code retrieval, likely because the long prefix overwhelms short code chunks.

## Benchmark Results

Tested on 4 public repositories (httpx/Python, serde/Rust, trpc/TypeScript, echo/Go) with 18 handcrafted evaluation questions spanning code, docs, and mixed categories.

### Overall Ranking

| Encoder | Params | MRR | R@5 | Code MRR | Docs MRR | Mixed MRR |
|---------|--------|-----|-----|----------|----------|-----------|
| **jina-v3** | 570M | **0.48** | **0.43** | **0.61** | 0.60 | **0.48** |
| gte-large | 335M | 0.44 | 0.39 | 0.60 | 0.60 | 0.42 |
| e5-large-v2 | 335M | 0.30 | 0.37 | 0.25 | 0.33 | 0.35 |
| qwen3-0.6b | 600M | 0.14 | 0.10 | 0.07 | 0.50 | 0.18 |
| gte-qwen2-1.5b | 1.5B | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Key Findings

- **Jina v3** is the strongest overall encoder for mixed code+docs retrieval. 60% higher MRR than E5, and 2.5x better on code questions.
- **GTE-large** is the best drop-in E5 replacement: same 335M parameter size, 47% better MRR, strong on code. The easiest upgrade path.
- **E5-large-v2** remains a solid baseline for documentation-only corpora but is weak on code.
- **Qwen3-0.6b** underperformed expectations. The long instruction prefix may hurt retrieval over short code chunks.
- **GTE-Qwen2-1.5b** produced zero results due to mean pooling incompatibility. Needs last-token pooling to function.
- **trpc** (large TypeScript monorepo) was near-zero across all encoders. This suggests the text chunker struggles with large TS codebases, or the evaluation questions need refinement for monorepo structures.

### Recommendation

For mixed code and documentation corpora, use **jina-v3** (requires `pip install einops`). For a zero-dependency drop-in upgrade from E5, use **gte-large** (same 335M size, same mean pooling, 47% better MRR).

Full results: `benchmarks/results/encoder_bakeoff/bakeoff_v2.json`
Analysis: `benchmarks/results/encoder_bakeoff/analysis_bakeoff_v2.md`
