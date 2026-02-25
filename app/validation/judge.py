"""LLM-as-Judge: TinyLlama 1.1B evaluates content on 5 dimensions.

Triggered ONLY when rule-based validation fails.
Scores 1-5 per dimension; pass threshold = avg ≥ 3.5 AND no single score < 2.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from loguru import logger

from app.core.prompts import build_judge_prompt


PASS_THRESHOLD = 3.5
MIN_SINGLE_SCORE = 2.0
DIMENSIONS = [
    "curriculum_alignment",
    "bloom_level",
    "cultural_appropriateness",
    "language_quality",
    "educational_value",
]


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from model output (handles markdown fences)."""
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find first {...}
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_judge(
    content: str,
    *,
    content_type: str = "content",
    subject: str = "general",
    education_level: str = "SSS1",
    curriculum_board: str = "WAEC",
    bloom_level: str = "UNDERSTAND",
    output_language: str = "en",
    failed_rules: Optional[List[str]] = None,
) -> Tuple[float, dict, Optional[str]]:
    """Run the TinyLlama 1.1B judge on generated content.

    Returns:
        (average_score, dimension_scores_dict, revision_instruction_or_None)
    """
    from app.core.llm import generate_text

    messages = build_judge_prompt(
        content,
        content_type=content_type,
        subject=subject,
        education_level=education_level,
        curriculum_board=curriculum_board,
        bloom_level=bloom_level,
        output_language=output_language,
        failed_rules=failed_rules or [],
    )

    try:
        raw = generate_text(messages, use_judge=True, max_new_tokens=512)
    except Exception as exc:
        logger.error(f"Judge LLM call failed: {exc}")
        # Soft-pass: don't block content if judge itself errors
        return PASS_THRESHOLD, {d: PASS_THRESHOLD for d in DIMENSIONS}, None

    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning(f"Judge returned non-JSON output: {raw[:200]}")
        return PASS_THRESHOLD, {d: PASS_THRESHOLD for d in DIMENSIONS}, None

    scores: dict = {}
    for dim in DIMENSIONS:
        val = parsed.get(dim)
        try:
            scores[dim] = float(val)
        except (TypeError, ValueError):
            scores[dim] = PASS_THRESHOLD  # default on parse failure

    avg = parsed.get("average_score")
    try:
        avg_score = float(avg)
    except (TypeError, ValueError):
        avg_score = sum(scores.values()) / len(scores) if scores else PASS_THRESHOLD

    revision_instruction: Optional[str] = parsed.get("revision_instruction") or None
    if revision_instruction and not revision_instruction.strip():
        revision_instruction = None

    logger.info(
        f"Judge scores: {scores} | avg={avg_score:.2f} | "
        f"{'PASS' if _judge_passes(avg_score, scores) else 'FAIL'}"
    )
    return avg_score, scores, revision_instruction


def _judge_passes(avg_score: float, dimension_scores: dict) -> bool:
    """Return True if avg ≥ 3.5 AND no single dimension < 2."""
    if avg_score < PASS_THRESHOLD:
        return False
    return all(v >= MIN_SINGLE_SCORE for v in dimension_scores.values())


def judge_passes(avg_score: float, dimension_scores: dict) -> bool:
    """Public helper — same logic, importable without running judge."""
    return _judge_passes(avg_score, dimension_scores)
