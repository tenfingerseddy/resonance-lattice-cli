# SPDX-License-Identifier: BUSL-1.1
"""Self-maintaining primer refresh orchestrator.

Coordinates the two primer generators (`rlat summary` → code primer,
`rlat memory primer` → memory primer) behind a single entry point,
with:

- Lockfile at `.claude/.primer-refresh.lock` so concurrent refresh
  invocations (nightly + SessionStart + manual) serialize safely.
- Stamped header `<!-- generated-at: ISO | git-head: SHA -->` so hooks
  can detect whether a primer was generated for the current HEAD.
- Atomic write-then-rename so a mid-refresh crash never leaves a
  truncated primer on disk.

This is wired to three triggers:
- `rlat primer refresh` (manual / CLI)
- SessionStart hook (async, non-blocking) when staleness + HEAD mismatch
- Nightly dogfood (unconditional at 03:00 local)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

LOCKFILE_NAME = ".primer-refresh.lock"
CODE_PRIMER_DEFAULT = Path(".claude/resonance-context.md")
MEMORY_PRIMER_DEFAULT = Path(".claude/memory-primer.md")
LOCK_TTL_SECONDS = 30 * 60  # a stale lock after 30 min is considered crashed


def _git_head(repo_root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        return ""
    try:
        proc = subprocess.run(
            [git, "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=5, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def stamp_header(body: str, git_head: str) -> str:
    """Prepend a machine-readable stamp so staleness checks can skip stable primers."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = (
        f"<!-- generated-at: {now} -->\n"
        f"<!-- git-head: {git_head or 'unknown'} -->\n"
    )
    if body.startswith("<!--"):
        return stamp + body
    return stamp + body


def _acquire_lock(lock_path: Path) -> bool:
    """Try to acquire the refresh lock. Returns True on success."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age < LOCK_TTL_SECONDS:
                return False
            # Stale lock — reclaim
            try:
                lock_path.unlink()
            except OSError:
                return False
        lock_path.write_text(f"pid={os.getpid()} at={time.time()}\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def read_stamp(primer_path: Path) -> dict:
    """Parse the header stamp out of an existing primer. Returns {}."""
    if not primer_path.is_file():
        return {}
    try:
        head = primer_path.read_text(encoding="utf-8", errors="replace")[:400]
    except OSError:
        return {}
    out: dict = {}
    for line in head.splitlines():
        if line.startswith("<!-- generated-at:"):
            out["generated_at"] = line.split(":", 1)[1].strip().rstrip("-->").strip()
        elif line.startswith("<!-- git-head:"):
            out["git_head"] = line.split(":", 1)[1].strip().rstrip("-->").strip()
    return out


def _run(cmd: Sequence[str], *, cwd: Path) -> tuple[int, str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        list(cmd), cwd=str(cwd),
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def refresh_primers(
    repo_root: Path | str = ".",
    *,
    cartridge: str = "project.rlat",
    memory_root: str = "./memory",
    code_out: Path = CODE_PRIMER_DEFAULT,
    mem_out: Path = MEMORY_PRIMER_DEFAULT,
    source_root: str | None = ".",
    wait_for_lock: bool = False,
) -> dict:
    """Regenerate both primers with stamped headers and atomic writes.

    Returns a status dict:
        {"code": {"ok": bool, "path": str, "bytes": int, "err": str},
         "memory": {...},
         "git_head": "...",
         "locked": bool}
    """
    repo = Path(repo_root).resolve()
    lock_path = repo / ".claude" / LOCKFILE_NAME
    status: dict = {"code": {}, "memory": {}, "git_head": "", "locked": False}

    # Acquire lock; if another run is in flight, exit cleanly unless wait.
    tries = 0
    while not _acquire_lock(lock_path):
        if not wait_for_lock:
            status["locked"] = True
            status["code"] = {"ok": False, "err": "lock held by another refresh"}
            status["memory"] = {"ok": False, "err": "lock held by another refresh"}
            return status
        tries += 1
        if tries > 30:
            status["locked"] = True
            status["code"] = {"ok": False, "err": "lock wait timeout"}
            status["memory"] = {"ok": False, "err": "lock wait timeout"}
            return status
        time.sleep(2)

    try:
        git_head = _git_head(repo)
        status["git_head"] = git_head

        # ── code primer ──
        code_path = repo / code_out
        cart_path = repo / cartridge
        if cart_path.is_file():
            cmd = [sys.executable, "-m", "resonance_lattice.cli",
                   "summary", str(cart_path),
                   "--budget", "4000"]
            if source_root is not None:
                cmd += ["--source-root", source_root]
            rc, out, err = _run(cmd, cwd=repo)
            if rc == 0 and out:
                body = stamp_header(out, git_head)
                try:
                    _atomic_write(code_path, body)
                    status["code"] = {
                        "ok": True, "path": str(code_path),
                        "bytes": len(body), "err": "",
                    }
                except OSError as exc:
                    status["code"] = {"ok": False, "err": f"write failed: {exc}"}
            else:
                status["code"] = {
                    "ok": False,
                    "err": f"rc={rc}: {err.strip()[:300]}",
                }
        else:
            status["code"] = {"ok": False, "err": f"missing cartridge: {cart_path}"}

        # ── memory primer ──
        mem_path = repo / mem_out
        mem_root = repo / memory_root
        if mem_root.is_dir():
            cmd = [sys.executable, "-m", "resonance_lattice.cli",
                   "memory", "primer", str(mem_root),
                   "--budget", "3500"]
            if cart_path.is_file():
                cmd += ["--code-knowledge model", str(cart_path)]
            rc, out, err = _run(cmd, cwd=repo)
            if rc == 0:
                # memory primer writes to its default output file; read and re-stamp.
                if mem_path.is_file():
                    raw = mem_path.read_text(encoding="utf-8", errors="replace")
                    body = stamp_header(raw, git_head)
                    try:
                        _atomic_write(mem_path, body)
                        status["memory"] = {
                            "ok": True, "path": str(mem_path),
                            "bytes": len(body), "err": "",
                        }
                    except OSError as exc:
                        status["memory"] = {"ok": False, "err": f"write failed: {exc}"}
                else:
                    status["memory"] = {
                        "ok": False,
                        "err": "memory primer produced no output file",
                    }
            else:
                status["memory"] = {
                    "ok": False,
                    "err": f"rc={rc}: {err.strip()[:300]}",
                }
        else:
            status["memory"] = {"ok": False, "err": f"missing memory root: {mem_root}"}

        return status
    finally:
        _release_lock(lock_path)


def is_stale(
    primer_path: Path,
    *,
    repo_root: Path,
    stale_hours: float = 72.0,
) -> tuple[bool, str]:
    """Decide whether the primer needs a refresh. Returns (stale, reason)."""
    if not primer_path.is_file():
        return True, "missing"
    try:
        age_hours = (time.time() - primer_path.stat().st_mtime) / 3600.0
    except OSError:
        return True, "stat-failed"
    if age_hours > stale_hours:
        return True, f"age-{age_hours:.0f}h"
    stamp = read_stamp(primer_path)
    head = _git_head(repo_root)
    stamped = (stamp.get("git_head") or "").strip()
    if head and stamped and head != stamped:
        return True, "head-diverged"
    return False, "fresh"
