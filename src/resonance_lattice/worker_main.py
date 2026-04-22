# SPDX-License-Identifier: BUSL-1.1
"""Background worker entry point for warm CLI search.

Launched as a detached subprocess by ``worker.spawn_worker``.
Loads a knowledge model once, serves it over HTTP on 127.0.0.1 with an
OS-assigned port, and exits after an idle timeout.

Usage (internal — not a user-facing command):
    python -m resonance_lattice.worker_main corpus.rlat --key <hex> [--encoder ...]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from http.server import HTTPServer
from pathlib import Path

# Idle timeout before the worker exits (seconds).
_IDLE_TIMEOUT = 30 * 60  # 30 minutes


def _find_onnx_dir(cartridge_path: Path) -> str | None:
    """Auto-detect an ONNX backbone directory alongside the knowledge model."""
    candidates = [
        cartridge_path.parent / f"{cartridge_path.stem}_onnx",
        cartridge_path.parent / "onnx_backbone",
    ]
    for c in candidates:
        if c.is_dir() and (c / "model.onnx").exists():
            return str(c)
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rlat background worker")
    p.add_argument("lattice", help="Path to .rlat knowledge model")
    p.add_argument("--key", required=True, help="Worker state key")
    p.add_argument("--encoder", default=None)
    p.add_argument("--onnx", default=None, help="ONNX backbone directory")
    return p.parse_args()


class _IdleWatchdog:
    """Shuts down the server after *timeout* seconds of inactivity."""

    def __init__(self, server: HTTPServer, timeout: float):
        self._server = server
        self._timeout = timeout
        self._last_activity = time.monotonic()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def _watch(self) -> None:
        while True:
            time.sleep(60)  # check every minute
            with self._lock:
                idle = time.monotonic() - self._last_activity
            if idle >= self._timeout:
                self._server.shutdown()
                return


def main() -> None:
    args = _parse_args()

    # Import heavy modules only after fork
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.server import LatticeHandler
    from resonance_lattice.worker import write_state

    lattice_path = Path(args.lattice)

    # Load lattice with encoder (the cold cost, paid once)
    lattice = Lattice.load(lattice_path, restore_encoder=args.encoder is None)

    if args.encoder:
        from resonance_lattice.encoder import Encoder
        if args.encoder == "random":
            lattice.encoder = Encoder.random(bands=lattice.config.bands, dim=lattice.config.dim)
        else:
            lattice.encoder = Encoder.from_backbone(
                model_name=args.encoder,
                bands=lattice.config.bands,
                dim=lattice.config.dim,
            )

    # Attach ONNX backbone for faster inference (2-3x speedup)
    onnx_dir = args.onnx or _find_onnx_dir(lattice_path)
    if onnx_dir and lattice.encoder is not None:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            attach_onnx_backbone(lattice.encoder, onnx_dir)
        except Exception:
            pass  # Fall back to PyTorch

    # Pre-warm: run a dummy query to populate caches (ANN index,
    # BLAS thread pool, SQLite page cache, encoder thread pool).
    if lattice.encoder is not None and lattice.source_count > 0:
        try:
            lattice.resonate(lattice.encoder.encode_query("warmup"), top_k=1)
        except Exception:
            pass  # Warmup is best-effort

    # Bind to OS-assigned port on loopback only
    LatticeHandler.lattice = lattice
    server = HTTPServer(("127.0.0.1", 0), LatticeHandler)
    port = server.server_address[1]

    # Write state so the CLI can discover us
    write_state(args.key, os.getpid(), port, str(lattice_path))

    # Idle watchdog — exit after _IDLE_TIMEOUT seconds of no requests
    watchdog = _IdleWatchdog(server, _IDLE_TIMEOUT)

    # Patch handler to touch the watchdog on every request
    _orig_do_GET = LatticeHandler.do_GET
    _orig_do_POST = LatticeHandler.do_POST

    def _do_GET(self):  # type: ignore[override]
        watchdog.touch()
        return _orig_do_GET(self)

    def _do_POST(self):  # type: ignore[override]
        watchdog.touch()
        return _orig_do_POST(self)

    LatticeHandler.do_GET = _do_GET
    LatticeHandler.do_POST = _do_POST

    watchdog.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        # Clean up our state file
        from resonance_lattice.worker import remove_state
        remove_state(args.key)


if __name__ == "__main__":
    main()
