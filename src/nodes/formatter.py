"""
Markdown Formatter node implementation for InkFlow-AI.

Responsibilities:
- Standardize Markdown presentation, headings, blockquote callouts, and spacing.
- Ensure exact formatting compliance without rewriting article narrative.
"""

from __future__ import annotations

import logging
from typing import Any
import time

from src.models.gateway import gateway
from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.prompts.base import PromptFactory
from src.prompts.prompts import SystemPrompts
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def markdown_formatter(state: BlogState) -> BlogState:
    """
    Standardize Markdown presentation and callout formatting.
    """
    logger.info("Running Markdown Formatter node...")

    if not state.blog_markdown:
        logger.warning("No blog markdown available for Markdown Formatter.")
        return state

    start_time = time.perf_counter()
    config = get_node_config(NodeType.MARKDOWN)

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.MARKDOWN_FORMATTER,
        human_prompt="Article Content:\n\n{article}\n",
    )

    llm = gateway.chat(NodeType.MARKDOWN)
    chain = prompt | llm

    response = chain.invoke({"article": state.blog_markdown})
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metric = cost_tracker.extract_llm_metrics(
        response=response,
        node_name="markdown_formatter",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    formatted_markdown = str(response.content).strip()
    result: dict[str, Any] = {"metrics": [metric]}
    if formatted_markdown:
        result["blog_markdown"] = formatted_markdown
        logger.info("Markdown Formatter completed successfully.")

    return result
