---
title: Getting Started
slug: getting-started
description: Install Resonance Lattice, build your first cartridge, and choose the right interface for the job.
nav_group: Foundations
nav_order: 20
aliases:
---

# Getting Started

## What Is This?

This guide gets a new user from install to a working cartridge. It is the fastest path to understanding the product shape before you go deeper into the architecture or reference docs.

## Why Start Here?

The main mental model of Resonance Lattice becomes clear only after you build and query a cartridge once. A short first-run loop makes the later architecture and reference material easier to place.

## How Does The First Workflow Work?

The first workflow has four steps:

1. install the package and optional fast path
2. build a cartridge from docs, code, or both
3. query or profile it
4. choose the interface you want to keep using

## How Do I Use It?

### Install

```bash
pip install resonance-lattice
pip install onnxruntime  # optional CPU acceleration
```

Python `>=3.11` is required. The first build downloads the default encoder unless you choose another one.

### Build Your First Cartridge

```bash
rlat build ./docs ./src -o project.rlat
```

This chunks your sources, encodes them, and writes a cartridge with the semantic field, source registry, and evidence store.

### Search, Profile, And Compare

```bash
rlat search project.rlat "how does authentication work?"
rlat profile project.rlat
rlat compare baseline.rlat project.rlat
```

Use `search` for grounded evidence, `profile` for cartridge shape, and `compare` when you need overlap and drift between snapshots.

### Choose CLI vs MCP vs HTTP

#### CLI

Use the CLI when you want direct terminal workflows, scripts, or local automation.

#### MCP

Use MCP when your assistant supports native tool calling and you want the cartridge to stay warm in the conversation.

#### HTTP

Use the local HTTP server when another app or plugin needs a simple request/response interface.

## What Should I Do Next?

- Read [Cartridge Architecture](/docs/cartridge-architecture) to understand the artifact itself.
- Read [CLI](/docs/cli) if terminal workflows are primary.
- Read [MCP](/docs/mcp) if assistant-native integration is primary.
- Read [Semantic Model](/docs/semantic-model) if you want to understand retrieval quality, coverage, and limits.
