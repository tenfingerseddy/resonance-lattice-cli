---
title: Obsidian Plugin
slug: obsidian-plugin
description: Semantic search across an Obsidian vault using local cartridge build, sync, and search workflows.
nav_group: Interfaces
nav_order: 30
aliases:
---

# Obsidian Plugin

## What Is It?

The Obsidian plugin brings Resonance Lattice into an Obsidian vault. It adds local build, sync, and search workflows so a vault can be queried semantically instead of only by exact text or manual links.

## Why Should I Use It?

Use the plugin when Obsidian is your working knowledge surface but you want the cartridge model underneath it:

- vault-wide semantic search
- incremental rebuild and sync behavior
- local results with source navigation
- reuse of the same cartridge ideas as the CLI and HTTP server

## How Does It Work?

The plugin builds a vault cartridge, talks to the local HTTP server, and renders results inside Obsidian. It also wires settings for encoder choice, top-k, cascade behavior, contradiction detection, and debounce timing.

## How Do I Use It?

### Typical Workflow

1. build a cartridge from the vault
2. start or restart the local server
3. search from the modal or sidebar
4. enable auto-sync if you want the cartridge to track vault changes

### Settings That Matter

- `rlat` binary path
- local port
- top-k
- cascade and contradiction toggles
- encoder and checkpoint overrides
- auto-sync

## Technical Guide

### Runtime Dependencies

The plugin depends on the local `rlat` binary and the local HTTP server. It is not a standalone hosted integration.

### Current Boundaries

- desktop-only
- local-machine oriented
- constrained by the local server and cartridge lifecycle
