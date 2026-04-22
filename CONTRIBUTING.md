# Contributing to Resonance Lattice

Thanks for your interest in contributing. This document covers what you need to know to land a change.

## Licensing

Resonance Lattice is licensed under **Business Source License 1.1** (see [LICENSE.md](LICENSE.md)). By contributing, you agree that your contributions are licensed under the same terms.

No CLA. Contributions are accepted under the **Developer Certificate of Origin 1.1** — sign off each commit with `git commit -s`. The signoff line is a statement that you have the right to contribute the change. See <https://developercertificate.org/>.

## Before you start

For anything non-trivial, open an issue first. This prevents wasted work if the direction doesn't align with the roadmap. The board at <https://github.com/users/tenfingerseddy/projects/1> is public and shows current priorities.

Small bug fixes and doc improvements don't need prior discussion — send the PR.

## Development setup

```bash
git clone https://github.com/tenfingerseddy/resonance-lattice.git
cd resonance-lattice
pip install -e ".[dev]"
```

Python 3.11 or 3.12. The test suite assumes `numpy`, `scipy`, `torch`, and `transformers` are importable; the `dev` extra adds pytest, hypothesis, and ruff.

## Running tests

```bash
PYTHONPATH=src pytest tests/ -x --tb=short
```

Full suite is a few hundred tests; most run in under 30 seconds. For a fast smoke:

```bash
PYTHONPATH=src pytest tests/test_cli_loader.py tests/test_breakthroughs.py -x
```

## Coding conventions

- Python-first. NumPy for field operations; PyTorch for the encoder.
- Field operations must preserve algebraic properties (merge commutativity, forget exactness).
- Encoder changes must keep build/query parity and be gated by live `search` benchmarks.
- Every RQL operation returns typed results, not raw `ndarray`.
- Lint with `ruff check src/ tests/` — the ruleset is in [pyproject.toml](pyproject.toml).
- Line length 100; the check is advisory, not CI-blocking.
- **No bare inline `TODO` / `FIXME` / `XXX` / `HACK` comments.** If a follow-up is genuinely needed, file an issue and reference the number: `# TODO(#NNN): short description`. The main branch is kept TODO-free as a hygiene invariant.

## Commit messages

Single subject line ≤ 72 chars, imperative mood. Body wrapped at ~72 chars. The subject should start with a subsystem prefix where it fits (e.g. `encoder:`, `cli:`, `field:`, `docs:`). Link related issues in the body (`Refs #N`, `Closes #N`).

## Pull requests

- Branch from `main`. Rebase onto `main` before requesting review.
- Include tests for behaviour changes. Regressions without a failing test are hard to prevent a second time.
- For performance-sensitive changes, include a benchmark number. Relevant benchmarks live in `benchmarks/`.
- For retrieval-quality changes, attach a BEIR or LongMemEval delta.
- CI must be green. If the test suite can't cover your change, explain in the PR body how you verified it.

## What kind of change is likely to land?

**Likely to land quickly:**

- Bug fixes with a failing test
- Doc improvements
- Performance improvements with a benchmark number
- New knowledge model storage backends that follow the existing `SourceStore` interface

**Likely to need discussion first:**

- Changes to the public API surface (`src/resonance_lattice/__init__.py::__all__`)
- New encoder backbones or projection heads
- New RQL operators
- Changes to knowledge model file format
- New top-level CLI subcommands

**Likely to be deferred:**

- Refactors without a motivating benchmark or bug
- New research modules without a benchmark gate
- Changes that remove user-visible features for internal-cleanliness reasons

## Reporting security issues

Please do **not** use public issues for vulnerabilities. See [SECURITY.md](SECURITY.md) for the private disclosure process.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Project interactions happen under the Contributor Covenant 2.1.

## Questions

Open a GitHub Discussion, or find an existing issue that fits. The project is small — a single issue is usually enough to move forward.
