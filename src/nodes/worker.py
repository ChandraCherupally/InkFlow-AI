"""
Writer worker node implementations for InkFlow-AI.

Responsibilities:
- Write individual blog sections in parallel.
- Assemble written sections into complete blog markdown.
"""

from __future__ import annotations

import logging

import time

from src.models.gateway import gateway
from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.prompts.base import PromptFactory
from src.prompts.prompts import SystemPrompts
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def worker_section(state: dict) -> dict:
    """
    Parallel worker node for writing a single blog section task efficiently.
    """
    topic = state.get("topic", "")
    task = state.get("task")

    if not task:
        return {"sections": []}

    logger.info("Writing section task %d: '%s'", task.id, task.title)
    start_time = time.perf_counter()
    config = get_node_config(NodeType.WRITER)

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.WRITER,
        human_prompt="""Topic: {topic}
Section Title: {title}
Goal: {goal}
Key Points to Cover:
{bullets}
""",
    )

    llm = gateway.chat(NodeType.WRITER)
    chain = prompt | llm

    response = chain.invoke(
        {
            "topic": topic,
            "title": task.title,
            "goal": task.goal,
            "bullets": "\n".join(f"- {b}" for b in task.bullets),
        }
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metric = cost_tracker.extract_llm_metrics(
        response=response,
        node_name="writer",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    section_markdown = str(response.content)
    return {
        "sections": [(task.id, section_markdown)],
        "metrics": [metric],
    }


def assemble_sections(state: BlogState) -> BlogState:
    """
    Sort and assemble parallel section outputs into complete blog markdown.
    """
    logger.info("Assembling written blog sections...")

    if not state.sections:
        logger.warning("No sections written.")
        return state

    sorted_sections = sorted(state.sections, key=lambda x: x[0])
    section_texts = [text for _, text in sorted_sections]

    state.blog_markdown = "\n\n".join(section_texts)
    logger.info("Successfully assembled %d blog sections.", len(section_texts))

    return state
