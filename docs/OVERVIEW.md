---
title: Overview
slug: overview
description: What Resonance Lattice is, why it exists, what is genuinely different about it, and where it fits best today.
nav_group: Foundations
nav_order: 10
aliases:
---

# Overview

## What Is Resonance Lattice?

Resonance Lattice is portable semantic knowledge infrastructure for AI assistants and developer tools. It turns documentation, code, notes, and other file collections into `.rlat` knowledge models that can be searched, profiled, compared, composed, and injected into assistant workflows.

At the core it is a **three-layer semantic router**: the field routes queries to the handful of chunks most likely to be relevant, a lossless store resolves those chunks from the actual source file, and any LLM you like synthesizes the final answer from the served passages. The field is a fast index over topic and entity structure — it is not an embedding store, and your raw corpus remains the source of truth.

The core unit is the knowledge model. A knowledge model packages a semantic field, a source registry, and the lossless store into one local artifact. That makes the knowledge layer portable instead of tying it to a hosted retrieval service, a database deployment, or a single model stack.

The simplest useful mental model is:

- your `README` explains the project to humans
- your `.rlat` knowledge model makes the wider corpus queryable to tools and assistants

Those layers work best together.

## Why Existing Approaches Break Down

Most assistant workflows still rely on one of four weak patterns:

- exact-text search that misses conceptual matches
- hosted RAG systems that are hard to move, inspect, or reuse across tools
- prompt stuffing that burns tokens and hides provenance
- static skill and instruction files that load the same background context every time

Each pattern solves a real problem, but each leaves a gap:

| Approach | What it does well | What breaks down |
|----------|-------------------|------------------|
| `grep` and text search | exact symbols, known strings, quick navigation | no semantic understanding, no ranking by meaning, no reusable context layer |
| standard RAG | retrieval for LLM workflows | often tied to a specific stack, weak inspectability, poor portability |
| vector databases | storage and serving infrastructure | gives you retrieval plumbing, not a portable knowledge artifact |
| static skill docs | clear workflow instructions | loads the same context whether or not the query needs it |

Resonance Lattice exists to make retrieval behave like infrastructure: local, inspectable, composable, and reusable across tools.

## What Is Genuinely New Here?

The main idea is not just "better search." The main idea is that project knowledge can become a portable semantic artifact with structure you can inspect and operate on.

That changes the working model in three ways:

### The knowledge layer is a file

A knowledge model can be versioned, copied, diffed, archived, shared, attached to a branch, and reused in a new tool without recreating a retrieval stack.

### Retrieval is not the only interface

The same knowledge model supports search, assistant context injection, profile, compare, xray, locate, and programmable field operations.

### The assistant does not have to guess from raw files

Instead of shoving whole files into a prompt, the assistant can pull ranked evidence, coverage signals, and related topics from a structured model of the corpus.

## Why This Is Different

### Compared with `grep`

Use `grep` when you know the string. Use Resonance Lattice when you need to understand what the corpus knows about a topic, even when the exact words do not line up.

### Compared with standard RAG

Standard RAG is usually a retrieval step bolted onto an LLM workflow. Resonance Lattice makes the knowledge layer itself portable, inspectable, and reusable whether or not an LLM is involved.

### Compared with vector databases

Vector databases are storage and retrieval infrastructure. Resonance Lattice is a portable knowledge artifact with a query surface, diagnostics, composition operations, and assistant-facing integration paths.

### Compared with static skill or instruction docs

Static docs tell the assistant how to work. Knowledge Model-backed workflows let the knowledge itself adapt to the request instead of loading the same fixed background on every run.

## Groundbreaking Capabilities

The strongest product story is not one magic feature. It is the combination of a few capabilities that usually do not live together in one system.

### Portable knowledge models

Knowledge is packaged as a reusable `.rlat` artifact instead of being trapped inside a hosted system or local database.

### Algebra over knowledge artifacts

Knowledge Models can be merged, differenced, projected, and selectively forgotten. That makes version comparison and multi-domain context control part of the product instead of an afterthought.

### Inspectability

The system does not stop at top-k passages. `profile`, `compare`, `locate`, `xray`, and RQL make it possible to inspect what the model appears to cover and where it is weak.

### Context control and composition

You can combine multiple knowledge models at query time, suppress topics, apply lenses, route through skills, and package reusable context configurations.

### Assistant-native integration surfaces

The same knowledge layer can be used from the CLI, MCP, HTTP, and the Obsidian plugin today. That makes the knowledge model a shared substrate instead of a one-off integration, and it leaves room for future surfaces without changing the underlying knowledge model.

A dual-primer system (`rlat summary` for the code primer + `rlat memory primer` for the memory primer) generates two complementary context documents — what the project *is* and how the work has *unfolded* — with cross-primer deduplication so the assistant never pays for the same fact twice. `rlat ask` auto-selects the best retrieval lens for a given question, so you do not have to choose between `search`, `locate`, `profile`, and `compose` by hand.

## Why It Is Worth Setting Up Even If You Already Have Other Tools

One of the right first questions is whether this solves a real problem you still have after putting time into an Obsidian wiki, a skill system, `CLAUDE.md`, or assistant memory. In many teams, the answer is yes because those tools are valuable but they do not replace a portable, inspectable retrieval layer.

### If you already use an Obsidian LLM wiki

Obsidian is strong as a workspace for note-taking, graph navigation, and manual curation. But as a retrieval layer for assistant grounding, Resonance Lattice is better. The published comparison used a strong Obsidian-style baseline with summaries, aliases, keywords, and more than 11,000 wikilinks, and RL still outperformed it on retrieval quality, ranking, and failed retrieval rate. The practical message is simple: if the job is grounded assistant retrieval over a serious corpus, RL is the stronger system. See [Comparative / Obsidian](/docs/benchmarks#comparative-obsidian) and the longer FAQ answer at [Why not just create an Obsidian vault?](/docs/faq#why-not-just-create-an-obsidian-vault).

### If you already use skills, `CLAUDE.md`, or prompt docs

Keep them. They solve a different layer of the problem. Skills and instruction files tell an assistant how to behave, which workflows to prefer, and which constraints matter. Resonance Lattice gives the assistant dynamic evidence about the project itself so it does not have to carry the same static background context on every run. In practice, the strongest setup is usually human-written instructions plus knowledge-model-backed retrieval, not one replacing the other. See [Does it replace skills, CLAUDE.md, and memory?](/docs/faq#does-it-replace-skills-claudemd-and-memory).

### If you already use assistant memory

Assistant memory is useful for conversational continuity. It is not the same thing as a portable project knowledge artifact. Memory is usually runtime-scoped, product-scoped, and hard to inspect deeply. A knowledge model is explicit, versionable, shareable, diffable, and queryable outside one assistant session. Use memory for continuity; use RL when you need grounded project retrieval that survives tools, sessions, branches, and model changes.

## Why Trust It?

Skepticism is reasonable. The AI tooling market is crowded with thin wrappers, short-lived demos, and products that make sweeping claims without enough evidence. Resonance Lattice has to earn trust in a repo-first way.

The trust case is:

- the core artifact is local and inspectable rather than hidden behind a hosted black box
- the retrieval story is benchmarked across multiple families, not described only with anecdotes
- the repo preserves raw outputs, not just a polished summary page
- the product claims are intentionally scoped, with explicit caveats about what not to overclaim

That does not prove the project is right for every workflow. It does mean the case for adoption is checkable. You can inspect the knowledge model, inspect the retrieved evidence, inspect the benchmark families, and inspect the limits. Start with [Benchmarks](/docs/benchmarks#claims-we-can-defend-today), then read [What Not To Overclaim](/docs/benchmarks#what-not-to-overclaim) and the FAQ entries on [project status](/docs/faq#what-is-the-project-status) and [main limitations](/docs/faq#what-are-the-main-limitations-today).

## Where It Works Best Today

Resonance Lattice is strongest today on:

- technical documentation corpora
- code-and-docs project orientation
- assistant grounding for proprietary or fast-changing project knowledge
- workflows where portability and inspectability matter as much as raw retrieval
- multi-surface use where the same knowledge layer needs to serve CLI, MCP, and app integrations

It is especially strong when the alternative would be reading whole files into an assistant or standing up a heavier RAG stack just to make a project queryable.

## Who It Is For

Use Resonance Lattice when you want:

- project knowledge packaged as a reusable artifact
- grounded assistant context with evidence and provenance
- one knowledge layer shared across multiple tools
- semantic comparison between versions or domains
- retrieval that can be inspected, benchmarked, and composed

## Who It Is Not For

It is not the right primary tool when you only need:

- raw symbol lookup, where `rg` or a code indexer is faster
- a hosted team search product with a full admin and permissions layer
- a general-purpose note-taking UX
- end-user chat without any need for grounded retrieval or project knowledge

## Proof At A Glance

The positioning above only matters if the system performs well enough to justify the setup cost. The current benchmark story supports a measured but strong claim:

- on the internal Fabric retrieval benchmark, the full pipeline reaches `Recall@5 = 1.00`, `MRR = 0.93`, and `0%` failed retrieval
- on token-efficiency testing, `rlat search` returned `24.6x` fewer tokens than a grep-plus-read workflow while preserving ranked passages and source attribution
- in the current grounding evaluation, adding RL context reduced hallucination rate from `0.78` to `0.16` and lifted fact recall from `0.27` to `0.91`
- on the 2026-04-22 5-BEIR rebench, the default encoder (`BAAI/bge-large-en-v1.5`, flipped from E5 on 2026-04-20) wins `4 of 5` corpora versus flat E5; the frontier-tier `qwen3-8b` option reaches a 5-BEIR average of `0.500` (vs BGE `0.445` provisional, E5 `0.455`)

These results do not prove that RL wins every corpus or every retrieval benchmark. They do show that the knowledge-model-plus-pipeline story is strong enough to matter in real assistant workflows, including workflows that already have skills, memory, or an Obsidian-style knowledge layer. For the full evidence page, see [Benchmarks](/docs/benchmarks).

## Where To Go Next

- Read [Benchmarks](/docs/benchmarks) if you want the proof and caveats.
- Read [FAQ](/docs/faq) if you want the longer comparison and practical adoption answers.
- Read [Getting Started](/docs/getting-started) if you want the fastest path to first success.
- Read [CLI](/docs/cli), [MCP](/docs/mcp), or [Context Control](/docs/context-control) if you already know the product shape and want the operational details.

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
