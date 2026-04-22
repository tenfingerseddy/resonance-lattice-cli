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

- search and context tools
- diagnostics such as `profile`, `compare`, `locate`, and `xray`
- discovery and freshness tools
- skill routing and injection tools

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

The shipped server exposes search, resonate, compose search, switch, info, discover, freshness, profile, compare, locate, xray, skill route, and skill inject capabilities.

### Operational Notes

- the primary knowledge model is configured at startup
- manifest-backed discovery improves multi-knowledge model workflows
- skill runtime is initialized lazily and shares the warm encoder when possible
