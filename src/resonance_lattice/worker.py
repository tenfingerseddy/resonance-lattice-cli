# SPDX-License-Identifier: BUSL-1.1
"""Auto-managed local worker lifecycle for warm CLI search.

Provides utilities to detect, probe, spawn, and clean up background
HTTP workers so that repeated ``rlat search`` calls can skip the
cold-start knowledge model/encoder load.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# How long a worker can sit idle before it exits (seconds).
IDLE_TIMEOUT_S = 30 * 60  # 30 minutes

# Max age for a state file before cleanup considers it stale (seconds).
MAX_STATE_AGE_S = 24 * 60 * 60  # 24 hours


# ── Worker key ────────────────────────────────────────────────────────

def worker_key(
    cartridge_path: Path,
    encoder: str | None = None,
    source_root: str | None = None,
) -> str:
    """Compute a deterministic key for a knowledge model + encoder combination.

    Uses the resolved path and file mtime so that knowledge model rebuilds
    automatically invalidate any running worker.  ``source_root``
    is included so that workers started with different external store
    roots are not incorrectly reused.
    """
    resolved = cartridge_path.resolve()
    try:
        mtime_ns = str(resolved.stat().st_mtime_ns)
    except OSError:
        mtime_ns = "0"
    parts = f"{resolved}|{mtime_ns}|{encoder or ''}|{source_root or ''}"
    return hashlib.md5(parts.encode()).hexdigest()[:16]


# ── State directory and files ─────────────────────────────────────────

def state_dir() -> Path:
    """Return (and lazily create) the worker state directory."""
    d = Path(tempfile.gettempdir()) / "rlat-workers"
    d.mkdir(exist_ok=True)
    return d


def state_path(key: str) -> Path:
    return state_dir() / f"{key}.json"


def read_state(key: str) -> dict | None:
    """Read a worker state file.  Returns None if missing, corrupt, or stale."""
    p = state_path(key)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pid = data.get("pid")
        if pid and not is_pid_alive(pid):
            # Process is dead — remove stale state
            remove_state(key)
            return None
        return data
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def write_state(key: str, pid: int, port: int, cartridge: str) -> None:
    """Atomically write a worker state file."""
    p = state_path(key)
    tmp = p.with_suffix(".tmp")
    payload = json.dumps({
        "pid": pid,
        "port": port,
        "knowledge model": cartridge,
        "started": time.time(),
    })
    tmp.write_text(payload, encoding="utf-8")
    os.replace(str(tmp), str(p))


def remove_state(key: str) -> None:
    try:
        state_path(key).unlink()
    except OSError:
        pass


# ── Process utilities ─────────────────────────────────────────────────

def is_pid_alive(pid: int) -> bool:
    """Cross-platform check whether a process is still running."""
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, OverflowError):
            return False


# ── Network utilities ─────────────────────────────────────────────────

def probe(port: int, timeout: float = 1.0) -> bool:
    """Check whether a worker is healthy on 127.0.0.1:port."""
    try:
        req = Request(f"http://127.0.0.1:{port}/health")
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def warm_search(port: int, params: dict, timeout: float = 10.0) -> dict | None:
    """POST a search request to a warm worker.  Returns the JSON dict or None."""
    try:
        body = json.dumps(params).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{port}/search",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ── Spawn ─────────────────────────────────────────────────────────────

def spawn_worker(
    cartridge: str,
    key: str,
    encoder: str | None = None,
    onnx: str | None = None,
) -> None:
    """Launch a background worker process for the given knowledge model.

    The worker is a detached subprocess running ``worker_main.py``.
    It writes its own state file once it has bound a port.
    """
    cmd = [
        sys.executable, "-m", "resonance_lattice.worker_main",
        cartridge, "--key", key,
    ]
    if encoder:
        cmd += ["--encoder", encoder]
    if onnx:
        cmd += ["--onnx", onnx]

    log_path = state_dir() / f"{key}.log"
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    kwargs: dict = {"stdout": log_fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    # Inherit PYTHONPATH so the worker can find resonance_lattice
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    if src_dir not in existing:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
    # Prevent Fortran runtime crash on Windows when detached process
    # receives console close event (forrtl error 200)
    env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    kwargs["env"] = env

    try:
        subprocess.Popen(cmd, **kwargs)
    except Exception:
        log_fh.close()
        raise


# ── Cleanup ───────────────────────────────────────────────────────────

def cleanup_stale() -> None:
    """Remove state files for dead workers or files older than MAX_STATE_AGE_S."""
    try:
        sd = state_dir()
    except OSError:
        return
    now = time.time()
    for f in sd.glob("*.json"):
        try:
            age = now - f.stat().st_mtime
            if age > MAX_STATE_AGE_S:
                f.unlink()
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid and not is_pid_alive(pid):
                f.unlink()
        except Exception:
            pass
