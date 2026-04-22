# SPDX-License-Identifier: BUSL-1.1
"""Reader layer — last-mile synthesis from retrieved evidence to a
query answer with citations.

The semantic layer's three stages:

  field     -> which region of the corpus is relevant?
  retrieval -> what bytes do we return to the reader? (expand + hybrid)
  reader    -> what answer does the user see?  (this layer)

Layered cleanly so users can stop at any stage:

  rlat search ...                -> stage 1+2, raw evidence
  rlat ask --reader context ...  -> stage 2 + context-pack (no LLM)
  rlat ask --reader llm ...      -> full stack, synthesized answer

Modules in this package:

  base    (C1): Evidence / Citation / Answer dataclasses, Reader ABC,
               `build_context_pack` — the stable prompt format.
  local   (C2): on-device OpenVINO-backed reader (LocalReader).
  api     (C3): Anthropic / OpenAI-compatible reader (APIReader).
  citations (C5): structured citation bundle format with verification.
"""

from resonance_lattice.reader.base import (
    Answer,
    Citation,
    Evidence,
    Reader,
    build_context_pack,
)
from resonance_lattice.reader.citations import (
    CitationBundle,
    EnrichedCitation,
    build_bundle,
    bundle_to_dict,
)

__all__ = [
    "Answer",
    "Citation",
    "CitationBundle",
    "EnrichedCitation",
    "Evidence",
    "Reader",
    "build_bundle",
    "build_context_pack",
    "bundle_to_dict",
]

# LocalReader is re-exported lazily via module attribute access so
# importing `resonance_lattice.reader` doesn't eagerly pull in
# optimum-intel / transformers. Callers who want the class use
# `from resonance_lattice.reader.local import LocalReader` directly.
