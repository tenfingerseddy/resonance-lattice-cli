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

This chunks your sources, encodes them, and writes a knowledge model with the semantic field, source registry, and evidence store.

### Search, Profile, And Compare

```bash
rlat search project.rlat "how does authentication work?"
rlat profile project.rlat
rlat compare baseline.rlat project.rlat
```

Use `search` for grounded evidence, `profile` for knowledge model shape, and `compare` when you need overlap and drift between snapshots.

### Choose CLI vs MCP vs HTTP

#### CLI

Use the CLI when you want direct terminal workflows, scripts, or local automation.

#### MCP

Use MCP when your assistant supports native tool calling and you want the knowledge model to stay warm in the conversation.

#### HTTP

Use the local HTTP server when another app or plugin needs a simple request/response interface.

## What Should I Do Next?

- Read [Knowledge Model Architecture](/docs/knowledge-model-architecture) to understand the artifact itself.
- Read [CLI](/docs/cli) if terminal workflows are primary.
- Read [MCP](/docs/mcp) if assistant-native integration is primary.
- Read [Semantic Model](/docs/semantic-model) if you want to understand retrieval quality, coverage, and limits.
