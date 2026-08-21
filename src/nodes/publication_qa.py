"""
Publication QA node implementation for InkFlow-AI.

Responsibilities:
- Run deterministic publication QA validation on finalized article and planned assets.
- Set publication_status to PASS or FAIL.
- Block invalid publications from silent release.
"""

from __future__ import annotations

import logging
from typing import Any

from src.guardrails.publication_qa import validate_publication
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def publication_qa(state: BlogState) -> dict[str, Any]:
    """
    Execute publication QA validation on the final article.
    """
    logger.info("Running Publication QA gate node...")

    article_text = state.final_markdown or state.blog_markdown or ""

    qa_result = validate_publication(
        article_markdown=article_text,
        plan=state.plan,
        image_plan=state.image_plan,
        generated_images=state.generated_images,
        section_word_counts=state.section_word_counts,
        image_failures=state.image_failures,
        target_word_count=state.target_word_count,
        min_word_count=state.min_word_count,
        max_word_count=state.max_word_count,
    )

    logger.info("Publication QA result: %s (Failures: %d, Warnings: %d)", qa_result.status, len(qa_result.failures), len(qa_result.warnings))

    result: dict[str, Any] = {
        "publication_status": qa_result.status,
        "qa_result": qa_result.model_dump(),
    }

    if qa_result.status == "FAIL":
        error_msg = f"Publication QA Failed: {'; '.join(qa_result.failures)}"
        logger.warning(error_msg)
        result["error"] = error_msg

    return result
