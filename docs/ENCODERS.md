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

For the per-workload decision guide (when to pick BGE vs E5 vs Qwen3-8B), see [Encoder Choice](/docs/encoder-choice). For the full measurement story across the three shipped encoders, see [Honest Claims](/docs/honest-claims).

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

Use this page to understand the preset catalog, the three well-supported encoders, and the current operational details before changing the production path. For the benchmark-backed recommendation, see [Encoder Choice](/docs/encoder-choice).

## What Are Encoder Presets?

An encoder preset is a short CLI name that maps to a full encoder configuration: the HuggingFace model ID, query/passage prefixes, pooling strategy, and maximum token length.

```bash
# Use a preset by name
rlat build ./docs ./src -o project.rlat --encoder qwen3-8b

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

### Default

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `bge-large-en-v1.5` | BAAI/bge-large-en-v1.5 | 335M | 512 | cls | **CLI default since 2026-04-20.** Portable, CPU-friendly, strong on docs / science / code. Wins 4/5 BEIR corpora in the 2026-04-22 5-BEIR rebench. |

### General Purpose

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `e5-large-v2` | intfloat/e5-large-v2 | 335M | 512 | mean | Opt-in alternative; strongest on counter-argument retrieval (ArguAna-like). Legacy default before 2026-04-20. |
| `nomic-v2` | nomic-ai/nomic-embed-text-v2-moe | 305M | 8K | mean | MoE efficiency, longer context than BGE / E5 |
| `arctic-embed-2` | Snowflake/snowflake-arctic-embed-l-v2.0 | 335M | 8K | cls | Strong retrieval quality per parameter |
| `nemotron-1b` | nvidia/llama-nemotron-embed-1b-v2 | 1B | 8K | mean | Strong open-weight quality, NVIDIA backed |

### Multilingual

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `bge-m3` | BAAI/bge-m3 | 568M | 8K | cls | Multilingual workhorse, 1000+ languages |
| `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B | 600M | 32K | last | Long context, multilingual, Apache 2.0 |
| `qwen3-4b` | Qwen/Qwen3-Embedding-4B | 4B | 32K | last | Serious quality jump over 0.6B |
| `qwen3-8b` | Qwen/Qwen3-Embedding-8B | 8B | 32K | last | **Frontier quality (BEIR-5 avg 0.500).** Needs ~16 GB GPU. Top of MTEB multilingual. |

### Code-Strong

| Preset | Backbone | Params | Context | Pooling | Best For |
|--------|----------|--------|---------|---------|----------|
| `gte-large` | Alibaba-NLP/gte-large-en-v1.5 | 335M | 8K | mean | Historic code-strong option. BGE (default) now wins on the 2026-04-22 5-BEIR rebench; keep for corpora where `gte-large` has been specifically validated. |
| `gte-qwen2-1.5b` | Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1.5B | 8K | mean | Best general quality ≤1.5B, code in training mix |
| `jina-v3` | jinaai/jina-embeddings-v3 | 570M | 8K | mean | Task-specific LoRA adapters |

## Choosing an Encoder

The short version: three encoders are well-supported and benchmarked against the canonical 5-BEIR harness. See [Encoder Choice](/docs/encoder-choice) for the per-workload decision guide and the Block D / Block E measurement detail.

### The three well-supported encoders

- **`bge-large-en-v1.5`** (default since 2026-04-20) — portable, CPU-friendly, 2026-04-22 5-BEIR avg **0.445** (provisional, Block D). Wins SciFact, NFCorpus, FiQA, SciDocs vs E5. Loses ArguAna by ~9.7 pts.
- **`e5-large-v2`** — opt-in alternative, 2026-04-22 5-BEIR avg **0.455**. Strongest on counter-argument retrieval (ArguAna-like corpora). Keep as the choice when your workload is debate / counter-claim.
- **`qwen3-8b`** — frontier tier, 2026-04-22 5-BEIR avg **0.500** (Block E, final). Needs ~16 GB GPU. Roughly 1 pt below `text-embedding-3-large` on the same 5-BEIR harness.

### Decision Guide

**Default English corpus** (docs, code, science, mixed):
Use `bge-large-en-v1.5`. This is the CLI default and wins 4/5 of the canonical BEIR corpora.

**Counter-argument / debate retrieval** (ArguAna-like):
Use `e5-large-v2`. Net +1.0 pt on 5-BEIR average vs BGE on this workload class.

**Frontier quality** (server + 16 GB GPU available):
Use `qwen3-8b`. BEIR-5 avg 0.500; frontier-adjacent.

**Multilingual content**:
Use `bge-m3` (1000+ languages, production-grade) or the `qwen3` series (100+ languages, longer context).

**Fast and lightweight** (laptop, CI, quick iteration):
The default `bge-large-en-v1.5` runs on any CPU at 335M and is CPU-friendly. `nomic-v2` (305M MoE, 8K context) is an alternative if you need longer context at that parameter count.

### What the Default Gives You

The default encoder is `BAAI/bge-large-en-v1.5` (flipped from E5-large-v2 on 2026-04-20, commit `3e0642f`). It is English-optimised, CPU-friendly, well-tested, and wins 4 out of 5 BEIR corpora in the 2026-04-22 rebench. The 512-token context limit means long functions or documents are truncated — if that matters for your corpus, consider `qwen3-8b` (32K context) or `bge-m3` (8K context) instead.

### The Practical Mindset

Do not overthink encoder choice on day one. The product story is:

1. Start with the default (`bge-large-en-v1.5`)
2. Keep the same `.rlat` workflow
3. Upgrade quality, multilinguality, or context length later by switching presets

The knowledge model format and query interface stay the same regardless of which encoder you use.

## How Encoders Bind to Knowledge Models

**One encoder per knowledge model.** The encoder is locked at build time and its fingerprint is stored in the knowledge model manifest. Querying auto-restores the same preset (pooling, prefixes, max-length) — you cannot accidentally query a BGE knowledge model with E5.

- `rlat add`, `rlat sync`, and `rlat refresh` verify that the current encoder matches the one used to build the knowledge model
- If there is a mismatch, the CLI will refuse to proceed and tell you which encoder the knowledge model expects
- Changing the encoder means rebuilding the knowledge model

This is by design. All vectors in the field must be in the same embedding space for resonance to be meaningful. Mixing encoders would produce a field where similarity scores are unreliable.

### No Trained Heads

All three well-supported encoders (and every preset listed above) ship pretrained weights with **random projection heads** only. Trained heads were tested across nine experiments and rejected — they broke build/query parity and regressed retrieval quality. If you see references to "checkpoint promotion" in historical benchmarks, those paths are not shipped in production.

## ONNX Acceleration

ONNX Runtime provides 2-5x CPU encoding speedup. To use it:

```bash
pip install onnxruntime

# Build with ONNX acceleration
rlat build ./docs ./src -o project.rlat --onnx ./onnx_backbone
```

The `--onnx` flag points to a directory containing the exported ONNX model. See the [CLI page](/docs/cli) for the full ONNX export workflow.

Not all presets have been tested with ONNX export. Standard transformer architectures (BGE, E5, `gte-large`) generally work. Larger models and those requiring `trust_remote_code=True` (like Jina v3 and GTE-Qwen2) may need additional configuration.

## Preset Notes

**BGE-large-en-v1.5** (default) uses CLS pooling and no query prefix. It is the CPU-friendly portable choice and is the 2026-04-20 default flip.

**E5-large-v2** uses mean pooling and a `query:` / `passage:` prefix convention applied automatically by the preset. E5 remains the `EncoderConfig` dataclass default for legacy-cartridge back-compat, even though the CLI default is now BGE.

**Qwen3 models** (0.6B, 4B, 8B) require **last-token pooling**, not mean. Mean pooling collapses Qwen3 embeddings — on FiQA the broken mean-pool configuration produced 7× worse retrieval; fixing to last-token pooling lifted the BEIR-5 average from a broken 0.250 back to 0.500. They also use a long instruction-format query prefix that the preset applies automatically.

**GTE-large** is a 335M code-strong alternative. In the historic bake-off (see below) it was the strongest drop-in upgrade from E5 for code-heavy corpora. The default BGE now wins on cross-corpus 5-BEIR; `gte-large` remains a valid choice on corpora where it has been specifically validated.

**GTE-Qwen2** requires last-token pooling for meaningful embeddings. The current encoder pipeline supports last-token pooling for Qwen3; `gte-qwen2-1.5b` will work once the same pooling path is wired for it. Do not use it yet.

**Jina v3** has LoRA task adapters for different retrieval tasks. The preset loads the base model without task routing, which still provides strong general-purpose retrieval. Requires `pip install einops`.

## Benchmark Results

The definitive cross-corpus measurement is the **2026-04-22 5-BEIR rebench** — see [Encoder Choice](/docs/encoder-choice) and [Honest Claims](/docs/honest-claims) for the full tables with per-corpus nDCG@10. Summary:

| Encoder | BEIR-5 avg (nDCG@10) | Notes |
|---------|----------------------|-------|
| `qwen3-8b` | **0.500** | Frontier tier, Block E final, 2026-04-22 |
| `e5-large-v2` | 0.455 | Opt-in, legacy default |
| `bge-large-en-v1.5` | **0.445** (provisional, Block D) | CLI default since 2026-04-20. Wins SciFact, NFCorpus, FiQA, SciDocs. Loses ArguAna by ~9.7 pts. |

### Historic Internal Bake-Off

The earlier internal bake-off (reproduced below for continuity) evaluated 4 public repositories with 18 handcrafted questions. It pre-dates the BGE default flip and does not include BGE or the 2026-04-22 5-BEIR corpora. Treat it as historical context; the 5-BEIR rebench is the canonical cross-corpus measurement.

| Encoder | Params | MRR | R@5 | Code MRR | Docs MRR | Mixed MRR |
|---------|--------|-----|-----|----------|----------|-----------|
| **jina-v3** | 570M | **0.48** | **0.43** | **0.61** | 0.60 | **0.48** |
| gte-large | 335M | 0.44 | 0.39 | 0.60 | 0.60 | 0.42 |
| e5-large-v2 | 335M | 0.30 | 0.37 | 0.25 | 0.33 | 0.35 |
| qwen3-0.6b | 600M | 0.14 | 0.10 | 0.07 | 0.50 | 0.18 |
| gte-qwen2-1.5b | 1.5B | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Key Findings

- The **2026-04-22 5-BEIR rebench** is the authoritative cross-corpus measurement. `qwen3-8b` is the frontier tier; `bge-large-en-v1.5` is the portable default; `e5-large-v2` is the counter-argument alternative. See [Encoder Choice](/docs/encoder-choice).
- The **GTE-Qwen2 zero result** in the historic bake-off came from mean-pool incompatibility — last-token pooling is required. The same class of failure is why Qwen3 presets pin `pooling = last`.
- The **trpc near-zero result** in the historic bake-off suggests the text chunker struggles with large TypeScript monorepos, or the evaluation questions need refinement for monorepo structures.

### Recommendation

Start with the CLI default (`bge-large-en-v1.5`). Switch deliberately: `e5-large-v2` for counter-argument workloads, `qwen3-8b` when you have 16 GB GPU headroom and want the frontier-quality tier. See [Encoder Choice](/docs/encoder-choice) for the full decision guide.

Full results: `benchmarks/results/beir/` (5-BEIR), `benchmarks/results/encoder_bakeoff/` (historic)

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
