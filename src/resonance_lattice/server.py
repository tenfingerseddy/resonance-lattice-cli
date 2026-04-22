# SPDX-License-Identifier: BUSL-1.1
"""Lightweight HTTP server for the Resonance Lattice.

Provides a minimal REST API for querying and managing a lattice over HTTP.
Uses stdlib http.server — zero additional dependencies.

Endpoints:
    POST /query       {"text": "...", "top_k": 10}
    POST /search      {"text": "...", "top_k": 10, ...}  — enriched: passages + coverage + cascade + sculpting
    POST /add         {"source_id": "...", "text": "...", "salience": 1.0}
    POST /remove      {"source_id": "..."}
    POST /locate      {"text": "..."}  — query position: band energy, coverage, uncertainty
    GET  /xray        Corpus diagnostics: band health, SNR, saturation
    GET  /info        Returns lattice metadata
    GET  /health      Returns {"status": "ok"}

Usage:
    rlat serve corpus.rlat --port 8080 --encoder intfloat/e5-large-v2
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from resonance_lattice.lattice import Lattice

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MB


class LatticeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Lattice REST API."""

    lattice: Lattice  # Set by serve()

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        if length > MAX_REQUEST_BYTES:
            self.send_error(413, "Request body too large")
            return None
        body = self.rfile.read(length)
        return json.loads(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/info":
            self._send_json(self.lattice.info())
        elif self.path == "/xray":
            self._handle_xray()
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_POST(self) -> None:
        try:
            data = self._read_json()
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        if data is None:
            return  # _read_json already sent an error response (e.g. 413)

        if self.path == "/query":
            self._handle_query(data)
        elif self.path == "/search":
            self._handle_search(data)
        elif self.path == "/add":
            self._handle_add(data)
        elif self.path == "/remove":
            self._handle_remove(data)
        elif self.path == "/locate":
            self._handle_locate(data)
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def _handle_query(self, data: dict[str, Any]) -> None:
        text = data.get("text", "")
        if not text:
            self._send_json({"error": "Missing 'text' field"}, 400)
            return

        top_k = data.get("top_k", 10)
        start = time.perf_counter()
        results = self.lattice.query(text, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._send_json({
            "query": text,
            "latency_ms": round(elapsed_ms, 2),
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                }
                for r in results
            ],
        })

    def _handle_search(self, data: dict[str, Any]) -> None:
        text = data.get("text", "")
        if not text:
            self._send_json({"error": "Missing 'text' field"}, 400)
            return

        # Core params
        top_k = data.get("top_k", 10)
        cascade_depth = data.get("cascade_depth", 2)
        contradiction_threshold = data.get("contradiction_threshold")
        enable_cascade = data.get("enable_cascade", True)
        enable_contradictions = data.get("enable_contradictions", False)

        # Advanced params
        enable_subgraph = data.get("enable_subgraph", False)
        enable_lexical = data.get("enable_lexical", False)
        enable_cross_encoder = data.get("enable_cross_encoder", False)
        enable_rerank = data.get("enable_rerank", True)

        # Topic sculpting — apply in-place, then reverse after search
        sculpt_state = None
        boost_topics = data.get("boost_topics", [])
        suppress_topics = data.get("suppress_topics", [])
        if boost_topics or suppress_topics:
            logger.info("Sculpting: boost=%s suppress=%s", boost_topics, suppress_topics)
            try:
                from resonance_lattice.field.dense import DenseField
                from resonance_lattice.pattern_injection import InterferenceSculptor
                field = self.lattice.field
                if isinstance(field, DenseField):
                    encoder = self.lattice.encoder
                    targets = []
                    for topic in boost_topics:
                        phase = encoder.encode(topic).vectors
                        targets.append((phase, 0.5, f"boost:{topic}"))
                    for topic in suppress_topics:
                        phase = encoder.encode(topic).vectors
                        targets.append((phase, -0.3, f"suppress:{topic}"))
                    sculpt_state = InterferenceSculptor.sculpt_multi(field, targets)
                    logger.info("Sculpting applied: %d patterns", len(targets))
                else:
                    logger.warning("Sculpting skipped: field is %s, not DenseField", type(field).__name__)
            except Exception:
                logger.exception("Topic sculpting failed")

        enriched = self.lattice.enriched_query(
            text=text,
            top_k=top_k,
            cascade_depth=cascade_depth,
            contradiction_threshold=contradiction_threshold,
            enable_cascade=enable_cascade,
            enable_contradictions=enable_contradictions,
            enable_subgraph=enable_subgraph,
            enable_lexical=enable_lexical,
            enable_cross_encoder=enable_cross_encoder,
            enable_rerank=enable_rerank,
        )

        # Reverse sculpting — restores field to original state exactly
        if sculpt_state is not None:
            try:
                from resonance_lattice.pattern_injection import InterferenceSculptor
                InterferenceSculptor.unsculpt(self.lattice.field, sculpt_state)
            except Exception as e:
                logger.warning("Failed to unsculpt: %s", e)

        self._send_json(enriched.to_dict())

    def _handle_add(self, data: dict[str, Any]) -> None:
        source_id = data.get("source_id", "")
        text = data.get("text", "")
        if not text:
            self._send_json({"error": "Missing 'text' field"}, 400)
            return

        salience = data.get("salience", 1.0)
        metadata = data.get("metadata")
        sid = self.lattice.add(source_id or "", text, salience=salience, metadata=metadata)
        self._send_json({"source_id": sid, "source_count": self.lattice.source_count})

    def _handle_remove(self, data: dict[str, Any]) -> None:
        source_id = data.get("source_id", "")
        if not source_id:
            self._send_json({"error": "Missing 'source_id' field"}, 400)
            return

        removed = self.lattice.remove(source_id)
        self._send_json({"removed": removed, "source_count": self.lattice.source_count})

    def _handle_xray(self) -> None:
        try:
            from resonance_lattice.field.dense import DenseField
            from resonance_lattice.xray import FieldXRay

            if not isinstance(self.lattice.field, DenseField):
                self._send_json({"error": "xray requires DenseField"}, 400)
                return

            result = FieldXRay.quick(
                field=self.lattice.field,
                source_count=self.lattice.source_count,
            )
            self._send_json(result.to_dict())
        except Exception as e:
            logger.exception("xray failed")
            self._send_json({"error": f"xray failed: {e}"}, 500)

    def _handle_locate(self, data: dict[str, Any]) -> None:
        text = data.get("text", "")
        if not text:
            self._send_json({"error": "Missing 'text' field"}, 400)
            return

        from resonance_lattice.field.dense import DenseField
        from resonance_lattice.locate import QueryLocator

        if not isinstance(self.lattice.field, DenseField):
            self._send_json({"error": "locate requires DenseField"}, 400)
            return

        phase = self.lattice.encoder.encode_query(text)
        q = phase.vectors if hasattr(phase, "vectors") else phase
        location = QueryLocator.locate(
            field=self.lattice.field,
            query_phase=q,
            query_text=text,
            registry=self.lattice.registry,
            store=self.lattice.store,
        )
        self._send_json(location.to_dict())

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)


def serve(lattice: Lattice, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the lattice HTTP server.

    Args:
        lattice: The Lattice instance to serve.
        host: Bind address.
        port: Port number.
    """
    LatticeHandler.lattice = lattice
    server = HTTPServer((host, port), LatticeHandler)
    print(f"Resonance Lattice server running on http://{host}:{port}")
    print(f"  Sources: {lattice.source_count}")
    print(f"  Field: {lattice.config.field_type.value} ({lattice.config.bands}B x {lattice.config.dim}D)")
    print("  Endpoints: POST /query, POST /search, POST /add, POST /remove, POST /locate, GET /xray, GET /info, GET /health")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
