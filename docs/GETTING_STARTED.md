---
title: Getting Started
slug: getting-started
description: Install Resonance Lattice, build your first knowledge model, and choose the right interface for the job.
nav_group: Foundations
nav_order: 20
aliases:
---

# Getting Started

## What Is This?

This guide gets a new user from install to a working knowledge model. It is the fastest path to understanding the product shape before you go deeper into the architecture or reference docs.

## Why Start Here?

The main mental model of Resonance Lattice becomes clear only after you build and query a knowledge model once. A short first-run loop makes the later architecture and reference material easier to place.

## How Does The First Workflow Work?

The first workflow has four steps:

1. install the package and optional fast path
2. build a knowledge model from docs, code, or both
3. query or profile it
4. choose the interface you want to keep using

## How Do I Use It?

### Install

```bash
pip install resonance-lattice
pip install onnxruntime  # optional CPU acceleration
```

Python `>=3.11` is required. The first build downloads the default encoder (`BAAI/bge-large-en-v1.5`, ~1.3 GB) unless you pass `--encoder` to pick another. Three encoders are well-supported: BGE (default, portable), E5 (opt-in — stronger on counter-argument corpora), Qwen3-8B (opt-in — frontier quality, needs 16 GB GPU). See [Encoder Choice](/docs/encoder-choice) for the decision guide.

### Build Your First Knowledge Model

```bash
rlat build ./docs ./src -o project.rlat
```

This chunks your sources, encodes them, and writes a knowledge model with the semantic field, source registry, and lossless store. The default `--store-mode local` keeps the knowledge model thin and resolves source files from disk at query time. Use `--store-mode bundled` to pack source files inside the `.rlat` for a self-contained artifact, or build from a GitHub URL for `--store-mode remote` with SHA-pinned upstream tracking. See [Storage Modes](/docs/storage-modes) for the full comparison.

### Search, Ask, Profile, And Compare

```bash
rlat search project.rlat "how does authentication work?"
rlat ask project.rlat "how does authentication work?"
rlat profile project.rlat
rlat compare baseline.rlat project.rlat
```

Use `search` for grounded evidence, `ask` when you want the auto-lens dispatcher to pick the best retrieval mode for the question, `profile` for knowledge model shape, and `compare` when you need overlap and drift between snapshots.

### Generate Assistant Primers

```bash
rlat summary project.rlat -o .claude/resonance-context.md
rlat memory primer ./memory/ -o .claude/memory-primer.md
```

The dual primer system produces two complementary context documents: a **code primer** (what the project IS — structure, conventions, patterns) and a **memory primer** (how the work has unfolded — settled decisions, reversals, active threads). They de-duplicate against each other so an assistant does not pay twice for the same fact. Reference both from `CLAUDE.md` or your equivalent system prompt.

### Choose CLI vs MCP vs HTTP

#### CLI

Use the CLI when you want direct terminal workflows, scripts, or local automation.

#### MCP

Use MCP when your assistant supports native tool calling and you want the knowledge model to stay warm in the conversation.

#### HTTP

Use the local HTTP server when another app or plugin needs a simple request/response interface.

## What Should I Do Next?

- Read [Knowledge Model Architecture](/docs/knowledge-model-architecture) to understand the artifact itself.
- Read [Storage Modes](/docs/storage-modes) to pick between `local` (default), `bundled`, and `remote`.
- Read [Encoder Choice](/docs/encoder-choice) before switching off the default BGE encoder.
- Read [CLI](/docs/cli) if terminal workflows are primary.
- Read [MCP](/docs/mcp) if assistant-native integration is primary.
- Read [Semantic Model](/docs/semantic-model) if you want to understand retrieval quality, coverage, and limits.

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
