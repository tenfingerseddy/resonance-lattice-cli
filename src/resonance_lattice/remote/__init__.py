# SPDX-License-Identifier: BUSL-1.1
"""Remote origin adapters for lossless-store knowledge models.

Exposes the ``Fetcher`` protocol that ``RemoteStore`` (Phase 4) uses to
retrieve source bytes from an HTTP origin with SHA-pinned immutable
caching. The first concrete implementation is ``GithubFetcher``; future
fetchers (S3, generic HTTPS, filesystem URLs) plug in behind the same
protocol.

Design notes
  - Stdlib ``urllib`` only — zero new runtime dependencies.
  - Knowledge Models pin to a commit SHA at build time. Load/query paths
    never touch the network; freshness is only checked when the user
    runs ``rlat freshness`` or ``rlat sync``.
  - All cache keys include the pinned SHA, so cached bytes are
    immutable and eviction is safe (LRU by atime, bounded by size).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Fetcher(Protocol):
    """Resolve origin coordinates -> raw file bytes.

    Every fetcher is SHA-pinned: once a commit/object hash is chosen at
    build time, ``fetch(sha, rel_path)`` returns the same bytes forever.
    This immutability is what lets the local disk cache skip
    revalidation on every query.
    """

    def list_files(self, ref: str) -> tuple[str, list[str]]:
        """Resolve ``ref`` (branch/tag/sha) to a pinned SHA + file list.

        Returns ``(commit_sha, sorted_relative_paths)``. Called once at
        build time; the returned SHA becomes the knowledge model's pinned
        origin.
        """
        ...

    def fetch(self, sha: str, rel_path: str) -> bytes:
        """Return raw bytes for ``rel_path`` at the pinned ``sha``.

        Called on query-time cache misses. Implementations should not
        add retry logic beyond a short timeout — upstream cache layers
        handle persistence.
        """
        ...

    def compare(self, base_sha: str, head_sha: str) -> dict:
        """Return the set of files that changed between two SHAs.

        Used by ``rlat sync`` to avoid O(corpus) refetches — only the
        changed files are re-pulled. Return shape:
        ``{"added": [...], "modified": [...], "removed": [...],
        "head_sha": <resolved head>}``.
        """
        ...


from resonance_lattice.remote.cache import DiskCache  # noqa: E402
from resonance_lattice.remote.github import GithubFetcher  # noqa: E402

__all__ = ["Fetcher", "DiskCache", "GithubFetcher"]
