"""LLM loader: LLaMA 3.2 3B (primary) and Mistral 7B (judge) with 4-bit BnB quantization."""
from __future__ import annotations

import torch
from functools import lru_cache
from typing import Optional

from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from app.core.config import settings


# ── BitsAndBytes 4-bit config ──────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── Generic pipeline loader ────────────────────────────────────────────────────

def _load_pipeline(
    model_id: str,
    *,
    max_new_tokens: int = 2048,
    use_quantization: bool = True,
) -> "pipeline":
    """Load a text-generation pipeline with optional 4-bit quantization."""
    logger.info(f"Loading model: {model_id}")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config: Optional[BitsAndBytesConfig] = None

    if use_quantization and torch.cuda.is_available():
        quantization_config = _bnb_config()
        logger.info("4-bit BnB quantization enabled")
    else:
        logger.warning(
            "GPU not available or quantization disabled — loading in full precision on CPU. "
            "This may be slow."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=settings.hf_token or None,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=settings.hf_token or None,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    logger.info(f"Pipeline ready: {model_id}")
    return pipe


# ── Public accessors (lazy singletons) ────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llama_pipeline():
    """LLaMA 3.2 3B Instruct — primary content generation model."""
    return _load_pipeline(settings.llama_model_id, max_new_tokens=2048)


@lru_cache(maxsize=1)
def get_mistral_pipeline():
    """Mistral 7B Instruct v0.3 — LLM-as-judge / validation model."""
    return _load_pipeline(settings.mistral_model_id, max_new_tokens=512)


# ── Convenience wrapper ────────────────────────────────────────────────────────

def generate_text(
    messages: list[dict],
    *,
    use_judge: bool = False,
    max_new_tokens: int = 1024,
) -> str:
    """Generate text using the appropriate pipeline.

    Args:
        messages: Chat-formatted list of dicts with ``role`` / ``content`` keys.
        use_judge: If True, use the Mistral judge model; otherwise LLaMA.
        max_new_tokens: Override default max new tokens.

    Returns:
        Generated text string.
    """
    pipe = get_mistral_pipeline() if use_judge else get_llama_pipeline()

    # Adjust generation config per call
    result = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        if isinstance(first, list) and first:
            return first[0].get("generated_text", "")

    logger.warning("Unexpected pipeline output format; returning empty string")
    return ""
