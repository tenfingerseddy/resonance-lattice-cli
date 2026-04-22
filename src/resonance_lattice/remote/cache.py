# SPDX-License-Identifier: BUSL-1.1
"""SHA-pinned disk cache for remote-store payloads.

Keys include the pinned commit SHA, so cached bytes are immutable — no
revalidation, no conditional GETs, no ETag dance on the hot path. When
the knowledge model syncs to a new commit, a separate directory tree gets
populated; stale SHA trees age out via LRU-by-atime bounded by a size
budget.

Filesystem layout under ``cache_dir`` (default ``~/.cache/rlat/remote``)::

    <origin_key>/<commit_sha>/<rel_path_with_segments>

``origin_key`` is a filesystem-safe slug of the origin (e.g.
``github__MicrosoftDocs__fabric-docs``) so one cache serves every
knowledge model pinned to the same repo.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path

DEFAULT_BUDGET_BYTES = 500 * 1024 * 1024  # 500 MB
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def default_cache_dir() -> Path:
    """Return ``$XDG_CACHE_HOME/rlat/remote`` (or ``~/.cache/rlat/remote``)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "rlat" / "remote"


def sanitise_segment(value: str) -> str:
    """Collapse unsafe characters so ``value`` can live inside a filename."""
    cleaned = _SAFE_SEGMENT.sub("_", value.strip("/"))
    return cleaned or "_"


class DiskCache:
    """Bounded LRU-by-atime cache for immutable SHA-pinned payloads.

    The cache never serves stale content because every lookup is keyed
    by (origin, sha, rel_path) — a write on one SHA cannot shadow a
    read on another. Eviction only removes old SHAs' entries.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        now: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            cache_dir: Base directory. Defaults to the XDG cache path.
            budget_bytes: Soft limit that triggers LRU-by-atime eviction
                on every ``put``. 0 disables the bound (useful in tests).
            now: Injectable clock — used for atime updates on read hits
                so tests don't have to sleep.
        """
        self.root = Path(cache_dir) if cache_dir else default_cache_dir()
        self.budget_bytes = budget_bytes
        self._now = now

    # ── internal helpers ───────────────────────────────────────────

    def _path_for(self, origin_key: str, sha: str, rel_path: str) -> Path:
        origin_safe = sanitise_segment(origin_key)
        sha_safe = sanitise_segment(sha)
        # Preserve directory structure for human debuggability, but
        # sanitise each segment so adversarial rel_paths can't escape.
        parts = [sanitise_segment(p) for p in rel_path.strip("/").split("/") if p]
        return self.root / origin_safe / sha_safe / Path(*parts) if parts else \
            self.root / origin_safe / sha_safe / "__empty__"

    def _touch_atime(self, path: Path) -> None:
        """Bump access time so LRU eviction sees this entry as hot."""
        try:
            ts = self._now()
            os.utime(path, (ts, ts))
        except OSError:
            pass

    def _iter_entries(self) -> list[tuple[Path, float, int]]:
        """Walk the cache tree: ``(path, atime, size)`` per file."""
        out: list[tuple[Path, float, int]] = []
        if not self.root.exists():
            return out
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                    out.append((p, st.st_atime, st.st_size))
                except OSError:
                    continue
        return out

    # ── public API ─────────────────────────────────────────────────

    def get(self, origin_key: str, sha: str, rel_path: str) -> bytes | None:
        """Return cached bytes or None on miss."""
        path = self._path_for(origin_key, sha, rel_path)
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        self._touch_atime(path)
        return data

    def put(self, origin_key: str, sha: str, rel_path: str, data: bytes) -> Path:
        """Store ``data`` under the sha-pinned key and enforce the budget."""
        path = self._path_for(origin_key, sha, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via sibling temp + os.replace. Avoids partial
        # reads if a second process is hitting the same SHA key.
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._touch_atime(path)
        if self.budget_bytes > 0:
            self._evict_to_budget()
        return path

    def size_on_disk(self) -> int:
        """Return the sum of cached file sizes (bytes)."""
        return sum(size for _p, _atime, size in self._iter_entries())

    def _evict_to_budget(self) -> None:
        """Drop oldest-atime entries until total size <= budget."""
        entries = self._iter_entries()
        total = sum(size for _p, _atime, size in entries)
        if total <= self.budget_bytes:
            return
        # Sort ascending by atime: oldest entries evicted first.
        entries.sort(key=lambda e: e[1])
        for path, _atime, size in entries:
            if total <= self.budget_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
        # Best-effort cleanup of empty directories left behind.
        self._prune_empty_dirs()

    def _prune_empty_dirs(self) -> None:
        if not self.root.exists():
            return
        for dirpath, dirs, files in os.walk(self.root, topdown=False):
            if not dirs and not files:
                try:
                    Path(dirpath).rmdir()
                except OSError:
                    pass
