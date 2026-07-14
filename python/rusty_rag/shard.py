"""Optional SHARD inference configuration with strictly local model loading.

The Rust extension provides the portable streaming quantizer. This module is
the opt-in bridge for the upstream SHARD Transformers cache, which remains
necessary for Llama-specific PCA/VQ compression and fused attention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LLM_BACKENDS = ("ollama", "shard")
DEFAULT_LLM_BACKEND = "ollama"


@dataclass(frozen=True)
class ShardSettings:
    """Configuration for an already-downloaded SHARD-compatible Llama model."""

    model_path: Path | None
    stream_bits: int
    max_new_tokens: int


def get_llm_backend(backend: str | None = None) -> str:
    """Resolve the active LLM backend."""
    value = (backend or os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND)).lower()
    if value not in LLM_BACKENDS:
        supported = ", ".join(LLM_BACKENDS)
        raise ValueError(f"Unsupported LLM backend '{value}'. Choose one of: {supported}.")
    return value


def get_shard_settings() -> ShardSettings:
    """Read and validate SHARD settings without accessing the network."""
    configured_path = os.getenv("SHARD_MODEL_PATH", "").strip()
    stream_bits = int(os.getenv("SHARD_STREAM_BITS", "8"))
    max_new_tokens = int(os.getenv("SHARD_MAX_NEW_TOKENS", "256"))
    if stream_bits not in {2, 3, 4, 8}:
        raise ValueError("SHARD_STREAM_BITS must be 2, 3, 4, or 8.")
    if max_new_tokens <= 0:
        raise ValueError("SHARD_MAX_NEW_TOKENS must be positive.")
    return ShardSettings(
        model_path=Path(configured_path).expanduser() if configured_path else None,
        stream_bits=stream_bits,
        max_new_tokens=max_new_tokens,
    )


def ask_with_shard(question: str, system: str) -> str:
    """Generate through an already-installed local SHARD Transformers runtime.

    Model and tokenizer loading use ``local_files_only=True`` by design. This
    feature therefore cannot download models as a side effect of a RAG query.
    """
    settings = get_shard_settings()
    if settings.model_path is None:
        raise RuntimeError(
            "LLM_BACKEND=shard requires SHARD_MODEL_PATH to point to an already-downloaded "
            "Llama-3.1-compatible Transformers model. No model was downloaded."
        )
    if not settings.model_path.is_dir():
        raise RuntimeError(f"SHARD_MODEL_PATH does not exist: {settings.model_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from shard import Cache, enable_llama_fused_attention
    except ImportError as exc:
        raise RuntimeError(
            "The SHARD backend needs local torch, transformers, and the upstream SHARD package. "
            "Install those separately, then point SHARD_MODEL_PATH at a local model checkout."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(settings.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    enable_llama_fused_attention(model)
    cache = Cache.from_model(model)
    cache._streaming = True
    cache._stream_bits = settings.stream_bits
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        inputs,
        max_new_tokens=settings.max_new_tokens,
        past_key_values=cache,
        do_sample=False,
    )
    return tokenizer.decode(generated[0][inputs.shape[-1] :], skip_special_tokens=True)
