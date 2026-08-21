"""
Editorial Review node implementation for InkFlow-AI.

Responsibilities:
- Perform Senior Technical Editor review of assembled article markdown.
- Smooth section transitions, remove duplicate ideas, standardize terminology.
- Enforce length preservation and context-aware closing section integrity.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.models.gateway import gateway
from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.prompts.base import PromptFactory
from src.prompts.prompts import SystemPrompts
from src.schemas.state import BlogState
from src.tools.word_counter import count_words

logger = logging.getLogger(__name__)


def editor(state: BlogState) -> dict[str, Any]:
    """
    Perform Senior Technical Editorial Review on assembled blog markdown while preserving length.
    """
    logger.info("Running Senior Editorial Review node...")

    if not state.blog_markdown:
        logger.warning("No blog markdown available for editorial review.")
        return {}

    start_time = time.perf_counter()
    config = get_node_config(NodeType.EDITOR)

    target_words = getattr(state, "target_word_count", 3500)
    min_words = getattr(state, "min_word_count", 2500)
    max_words = getattr(state, "max_word_count", 5000)

    closing_title = (
        state.plan.closing_section_title
        if state.plan and state.plan.closing_section_title
        else "Context-Aware Engineering Conclusion"
    )

    human_prompt = f"""Target Article Length: {target_words} words (Required range: {min_words} - {max_words} words)
Planner Closing Section Heading: {closing_title}

Article Markdown:

{{article}}
"""

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.EDITOR,
        human_prompt=human_prompt,
    )

    llm = gateway.chat(NodeType.EDITOR)
    chain = prompt | llm

    response = chain.invoke({"article": state.blog_markdown})
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metric = cost_tracker.extract_llm_metrics(
        response=response,
        node_name="editor",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    polished_markdown = str(response.content).strip()
    result: dict[str, Any] = {"metrics": [metric]}

    if polished_markdown:
        edited_words = count_words(polished_markdown)
        pre_edit_words = count_words(state.blog_markdown)

        # Ensure editor didn't severely truncate article below threshold
        if edited_words < min_words * 0.70 and pre_edit_words >= min_words * 0.70:
            logger.warning(
                "Editorial review dropped word count severely (%d -> %d words). Preserving pre-edit draft.",
                pre_edit_words,
                edited_words,
            )
            result["blog_markdown"] = state.blog_markdown
        else:
            result["blog_markdown"] = polished_markdown
            logger.info("Senior Editorial Review completed successfully (%d words).", edited_words)

    return result
