"""
Writer worker node implementations for InkFlow-AI.

Responsibilities:
- Write individual blog sections in parallel honoring explicit section word budgets.
- Validate section word counts and perform controlled single retry when outside tolerance.
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
from src.schemas.models import Task
from src.schemas.state import BlogState
from src.tools.word_counter import count_words

logger = logging.getLogger(__name__)


def worker_section(state: dict) -> dict:
    """
    Parallel worker node for writing a single blog section task with word-budget validation.
    """
    topic = state.get("topic", "")
    task_raw = state.get("task")
    plan_target = state.get("target_word_count", 3500)

    if not task_raw:
        return {"sections": [], "metrics": []}

    task = task_raw if isinstance(task_raw, Task) else Task(**task_raw)

    target_words = getattr(task, "target_words", 500)
    is_closing = getattr(task, "is_closing_section", False)

    # For closing section, standard range is 150-300 words
    if is_closing:
        min_allowed = 150
        max_allowed = 350
    else:
        min_allowed = max(50, int(target_words * 0.80))
        max_allowed = int(target_words * 1.20)

    logger.info(
        "Writing section task %d: '%s' (Target: %d words, Range: %d-%d)",
        task.id,
        task.title,
        target_words,
        min_allowed,
        max_allowed,
    )
    start_time = time.perf_counter()
    config = get_node_config(NodeType.WRITER)

    human_prompt = f"""Topic: {{topic}}
Article Target Words: {plan_target}
Section Title: {{title}}
Goal: {{goal}}
Target Section Words: {target_words} words (Allowed range: {min_allowed} - {max_allowed} words)
Technical Depth: {{technical_depth}}
Is Dedicated Closing Section: {is_closing}

Key Points to Cover:
{{bullets}}
"""

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.WRITER,
        human_prompt=human_prompt,
    )

    llm = gateway.chat(NodeType.WRITER)
    chain = prompt | llm

    response = chain.invoke(
        {
            "topic": topic,
            "title": task.title,
            "goal": task.goal,
            "technical_depth": getattr(task, "technical_depth", "deep_dive"),
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

    section_markdown = str(response.content).strip()
    actual_words = count_words(section_markdown)

    # ---------------------------------------------------------
    # Programmatic Section Word Validation & Controlled 1-Retry
    # ---------------------------------------------------------
    if (actual_words < min_allowed or actual_words > max_allowed) and not is_closing:
        logger.info(
            "Section %d word count (%d) outside allowed range [%d, %d]. Performing 1 controlled retry...",
            task.id,
            actual_words,
            min_allowed,
            max_allowed,
        )
        retry_prompt = PromptFactory.create(
            system_prompt=SystemPrompts.WRITER,
            human_prompt=f"""Topic: {{topic}}
Section Title: {{title}}
Goal: {{goal}}

IMPORTANT CORRECTION:
The previous draft was {actual_words} words.
The required section length is {min_allowed} to {max_allowed} words (Target: {target_words} words).
{"Expand the technical depth, add concrete architectural/code explanations, and explain WHY decisions were made." if actual_words < min_allowed else "Condense prose, remove repetition, and tighten explanations while preserving technical precision."}
Do NOT add generic filler.

Key Points to Cover:
{{bullets}}
""",
        )
        retry_chain = retry_prompt | llm
        try:
            retry_res = retry_chain.invoke(
                {
                    "topic": topic,
                    "title": task.title,
                    "goal": task.goal,
                    "bullets": "\n".join(f"- {b}" for b in task.bullets),
                }
            )
            retry_md = str(retry_res.content).strip()
            retry_words = count_words(retry_md)
            if retry_words > 0:
                section_markdown = retry_md
                actual_words = retry_words
                logger.info("Section %d retry completed with %d words.", task.id, retry_words)
        except Exception as err:
            logger.warning("Section %d retry failed: %s", task.id, err)

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

    # Calculate per-section word counts dictionary
    section_counts = {sec_id: count_words(text) for sec_id, text in sorted_sections}
    state.section_word_counts = section_counts

    state.blog_markdown = "\n\n".join(section_texts)
    total_words = count_words(state.blog_markdown)
    logger.info(
        "Successfully assembled %d blog sections (%d total prose words).",
        len(section_texts),
        total_words,
    )

    return state
