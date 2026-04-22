# Security Policy

## Supported versions

Resonance Lattice is a pre-1.0 project. Only the most recent released version on PyPI receives security fixes.

Once v1.0.0 ships, this table will be updated to reflect the supported range.

| Version | Supported |
|---|---|
| 0.11.x  | current dev |
| < 0.11  | no |

## Reporting a vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

Report privately via one of:

1. **GitHub Security Advisory (preferred):** open a private advisory at <https://github.com/tenfingerseddy/resonance-lattice/security/advisories/new>. This keeps the discussion confidential and produces a CVE when appropriate.
2. **Email:** kanesnyder1@gmail.com, with subject line prefix `[rlat-security]`.

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce (a minimal PoC is ideal)
- The affected version(s) and commit SHA if known
- Your suggested remediation, if any

## Disclosure timeline

- **Acknowledgement:** within 5 business days.
- **Initial assessment:** within 10 business days — severity triage + confirmation.
- **Fix window:** target 90 days from acknowledgement for High / Critical; shorter for actively-exploited issues.
- **Public disclosure:** coordinated with the reporter once a fix is released, or at the 90-day mark, whichever comes first.

Reporters are credited in release notes unless they request anonymity.

## Scope

In scope:

- The `resonance_lattice` Python package (PyPI `resonance-lattice`)
- The `rlat` CLI
- The MCP server (`rlat serve` / stdio MCP endpoint)
- The HTTP `serve` endpoint
- The `rlat pull` remote-knowledge model resolver and its cache
- Bundled, remote, and local knowledge model I/O paths

Out of scope:

- Attacks that require compromising the user's local machine in advance
- DoS via adversarially-crafted knowledge models at build time (knowledge models are a trust boundary — treat third-party knowledge models as you would third-party code)
- Dependency CVEs that do not have a reachable exploit path through `resonance_lattice`
- Rate-limit and quota-exhaustion attacks against third-party services the user integrates with (HuggingFace Hub, GitHub, Anthropic API)

## Trust boundaries

- **Local knowledge models** (built by the user) are trusted.
- **Remote knowledge models** (fetched via `rlat pull` from GitHub / HuggingFace Hub) are SHA-pinned and verified against `knowledge-models/index.yaml`. The pinning is the trust anchor; treat the underlying upstream repos like you would any other third-party dependency.
- **MCP clients** are trusted within a user's IDE. The MCP server does not authenticate callers; do not expose it over the network without a separate auth layer.
- **Queries** entering the system via CLI, MCP, or HTTP are not sanitised — they are treated as search input, not executable code.

## Dependency hygiene

Every release runs `pip-audit` against the locked dependency set in CI. Critical / High findings block the release.
