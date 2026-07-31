"""
Editorial Review node implementation for InkFlow-AI.

Responsibilities:
- Perform Senior Technical Editor review of assembled article markdown.
- Smooth section transitions, remove duplicate ideas, standardize terminology.
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


def editor(state: BlogState) -> BlogState:
    """
    Perform Senior Technical Editorial Review on assembled blog markdown.
    """
    logger.info("Running Senior Editorial Review node...")

    if not state.blog_markdown:
        logger.warning("No blog markdown available for editorial review.")
        return state

    start_time = time.perf_counter()
    config = get_node_config(NodeType.EDITOR)

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.EDITOR,
        human_prompt="Article Markdown:\n\n{article}\n",
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
        result["blog_markdown"] = polished_markdown
        logger.info("Senior Editorial Review completed successfully.")

    return result
