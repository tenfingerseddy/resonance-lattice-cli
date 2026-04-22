# SPDX-License-Identifier: BUSL-1.1
"""OpenVINO-backed local reader (C2 of the v1.0.0 semantic-layer plan).

Runs synthesis on-device via `optimum.intel.OVModelForCausalLM` over
CPU, Intel Arc iGPU, or NPU — the same OpenVINO stack `encoder_openvino`
uses for embeddings. This keeps `rlat ask` working without a network
round-trip, without API cost, and without sending corpus text to a
third party.

Design contract:

1. **Lazy imports.** `optimum.intel` and `transformers` are heavy deps
   (~hundreds of MB). Module-level import would drag in GPUs and model
   hubs for every CLI invocation; we import inside the constructor so
   `from resonance_lattice.reader.local import LocalReader` is cheap.

2. **Grounded by prompt, validated by extraction.** The system prompt
   tells the model to cite evidence as `[N]`; `extract_citations`
   enforces it — any `[N]` outside `1..len(evidence)` is silently
   dropped so the public `Answer.citations` only references real
   evidence (Reader's grounding invariant from C1).

3. **Fail-fast on unavailable deps.** `is_available()` returns a
   diagnostic tuple so CLI callers can print a clear remediation step.
   Constructing a LocalReader without the deps raises immediately
   rather than crashing mid-inference.

4. **Stateful, closable.** Loaded models hold ~GB of device memory;
   `close()` releases them. The ABC's default close() would leave
   handles hanging until GC.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path

from resonance_lattice.reader.base import (
    Answer,
    Citation,
    Evidence,
    Reader,
    build_context_pack,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3

DEFAULT_SYSTEM_PROMPT = (
    "You are a grounded research assistant. Answer the user's question "
    "using ONLY the evidence provided below. "
    "Cite each factual claim with the evidence index in square brackets, "
    "for example [1] or [2]. "
    "If the evidence does not support an answer, say so plainly — do not "
    "guess, and do not invent sources. "
    "Be concise; avoid preamble."
)

# User-message template. Kept separate from the system prompt so the
# same LocalReader can swap prompts for different use-cases (research
# assistant vs. code reviewer) without rewriting the synthesis logic.
_USER_TEMPLATE = (
    "{context_pack}\n\n"
    "---\n\n"
    "Question: {query}\n\n"
    "Instructions: Answer the question above using ONLY the evidence. "
    "Cite each supporting claim as [N] where N is the evidence index. "
    "If unsure, say so."
)


def is_available() -> tuple[bool, str]:
    """Return (available, diagnostic) for the LocalReader stack.

    Callers can use this to decide between LocalReader and APIReader
    without catching ImportError. The diagnostic is a short string
    suitable for error messages (`rlat ask --reader llm` prints it
    when it has to fall back).
    """
    try:
        import openvino  # noqa: F401
    except ImportError as exc:
        return False, f"openvino not installed: {exc}"
    try:
        from optimum.intel import OVModelForCausalLM  # noqa: F401
    except ImportError as exc:
        return False, f"optimum-intel not installed: {exc}"
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError as exc:
        return False, f"transformers not installed: {exc}"
    return True, ""


def build_prompt(
    query: str,
    evidence: Sequence[Evidence],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build a chat-template-compatible message list for the reader.

    Returns the canonical [{'role': 'system', ...}, {'role': 'user', ...}]
    shape. Tokenisers that support `apply_chat_template` can consume
    it directly; callers that don't can stringify with a simple join.
    """
    context_pack = build_context_pack(query, evidence)
    user_content = _USER_TEMPLATE.format(
        context_pack=context_pack,
        query=query.strip() or "(empty)",
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# `[N]` where N is any positive integer. Matches `[1]`, `[12]`, etc.
# Looser patterns (e.g. `[1,2]`, `[1-3]`) aren't part of the instructed
# citation format and would ambiguate mapping.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(
    text: str, evidence: Sequence[Evidence],
) -> list[Citation]:
    """Parse `[N]` markers from generated text and map to evidence.

    Citations are returned in first-occurrence order. Indices outside
    `1..len(evidence)` are dropped (a reader that cites `[99]` when
    only 5 evidence items were given is hallucinating). Duplicate
    citations to the same evidence collapse to one entry so downstream
    renderers don't print the same footnote twice.

    The `quote` field carries a truncated preview of the evidence
    text so UIs can show the citation without re-reading the source
    file. The full span offsets are preserved for exact lookup.
    """
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for m in _CITATION_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if not (1 <= idx <= len(evidence)):
            continue
        ev = evidence[idx - 1]
        key = (ev.source_file, ev.char_offset)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(
            source_file=ev.source_file,
            char_offset=ev.char_offset,
            char_length=len(ev.text),
            quote=ev.text[:200],
        ))
    return citations


class LocalReader(Reader):
    """On-device reader via OpenVINO-exported causal LM.

    Args:
        model: HuggingFace model id (e.g. "Qwen/Qwen2.5-3B-Instruct")
            or a path to a local OpenVINO IR directory. HF ids
            trigger a one-time export on first use; IR dirs load
            directly.
        device: OpenVINO device ("AUTO", "CPU", "GPU", "NPU"). Default
            picks via `preferred_device()` from encoder_openvino, which
            favours GPU > NPU > CPU.
        max_new_tokens: Hard cap on generated tokens. Protects against
            pathological model loops.
        temperature: Sampling temperature. 0 = greedy (deterministic);
            higher = more varied. 0.3 is a reasonable default for
            grounded synthesis — enough variation to avoid repetition
            loops, conservative enough to stay on-evidence.
        system_prompt: Override the default grounded-assistant prompt.
            Use when wrapping the reader for a specialised task.

    Raises:
        RuntimeError: if OpenVINO / optimum-intel / transformers are
            unavailable. Check `LocalReader.is_available()` first if
            you want to avoid the exception.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        device: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str | None = None,
    ) -> None:
        ok, diag = is_available()
        if not ok:
            raise RuntimeError(
                f"LocalReader unavailable: {diag}. "
                f"Install `openvino optimum-intel transformers`, or use APIReader."
            )

        # Resolve device now so .name reflects what we'll actually run on.
        from resonance_lattice.encoder_openvino import preferred_device
        resolved_device = device or preferred_device() or "CPU"

        # Lazy heavy imports — kept inside __init__ so module import
        # stays cheap for CLI commands that don't instantiate a reader.
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer

        model_str = str(model)
        model_path = Path(model_str)

        # Heuristic: a directory with `openvino_model.xml` is a local IR
        # export; anything else we hand to HF with export=True.
        is_local_ir = (
            model_path.is_dir()
            and (model_path / "openvino_model.xml").exists()
        )

        self._model = OVModelForCausalLM.from_pretrained(
            model_str,
            export=not is_local_ir,
            device=resolved_device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_str)

        # Some tokenisers lack a pad token; reuse eos to avoid a
        # generate()-time warning and non-deterministic padding.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._device = resolved_device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.name = f"openvino-{_short_name(model_str)}"

    def answer(
        self, query: str, evidence: Sequence[Evidence],
    ) -> Answer:
        messages = build_prompt(query, evidence, self._system_prompt)
        t0 = time.perf_counter()
        generated_text = self._generate(messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        citations = extract_citations(generated_text, evidence)
        return Answer(
            query=query,
            text=generated_text.strip(),
            citations=citations,
            model=self.name,
            latency_ms=round(latency_ms, 2),
            evidence_used=len(evidence),
        )

    def close(self) -> None:
        # Best-effort release. OVModelForCausalLM wraps an openvino
        # CompiledModel; drop references so GC can reclaim device mem.
        self._model = None  # type: ignore[assignment]
        self._tokenizer = None  # type: ignore[assignment]

    # ── internals ────────────────────────────────────────────────────

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run generation and return the *new* tokens decoded as text.

        Split out so tests can monkeypatch `_generate` without having
        to mock the model/tokenizer at a lower level.
        """
        import torch

        tokenizer = self._tokenizer
        model = self._model

        # Prefer the tokeniser's chat template when present (most
        # instruct models ship one). Fall back to a naive system+user
        # concatenation so non-chat tokenisers still work.
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 — tokeniser variants differ
            text = "\n\n".join(m["content"] for m in messages)
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        prompt_len = input_ids.shape[-1]
        do_sample = self._temperature > 0.0

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=do_sample,
                temperature=max(self._temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _short_name(model: str) -> str:
    """Shorten a HF id or path for `Answer.model` tagging.

    `Qwen/Qwen2.5-3B-Instruct` -> `Qwen2.5-3B-Instruct`
    `/path/to/ir` -> `ir`
    """
    return Path(model.replace("\\", "/")).name or "local"
