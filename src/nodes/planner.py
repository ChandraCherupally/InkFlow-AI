"""
Planner node implementation for InkFlow-AI.

Responsibilities:
- Consume topic and research evidence.
- Produce structured Plan outline matching target word budget.
- Distribute word counts deterministically across sections.
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
from src.schemas.models import Plan
from src.schemas.state import BlogState
from src.tools.word_counter import resolve_word_budget

logger = logging.getLogger(__name__)


def planner(state: BlogState) -> dict:
    """
    Generate a structured blog outline with optimized research context and explicit word budget.
    """
    logger.info("Running planner node...")
    start_time = time.perf_counter()
    config = get_node_config(NodeType.PLANNER)

    budget = resolve_word_budget(getattr(state, "target_word_count", 3500))
    target_words = budget["target_word_count"]
    min_words = budget["min_word_count"]
    max_words = budget["max_word_count"]

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.PLANNER,
        human_prompt="""Topic:
{topic}

Requested Article Length:
- Target: {target_word_count} words
- Minimum acceptable: {min_word_count} words
- Maximum acceptable: {max_word_count} words

Research Context:
{research}
""",
    )

    llm = gateway.chat(NodeType.PLANNER).with_structured_output(
        Plan, include_raw=True
    )
    chain = prompt | llm

    research_items = state.evidence.evidence[:6] if state.evidence and state.evidence.evidence else []
    research_context = "\n".join(
        f"- {item.title}: {item.snippet[:200] if item.snippet else ''}"
        for item in research_items
    ) if research_items else "No external research required."

    raw_res = chain.invoke(
        {
            "topic": state.topic,
            "target_word_count": target_words,
            "min_word_count": min_words,
            "max_word_count": max_words,
            "research": research_context,
        }
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if isinstance(raw_res, dict) and "parsed" in raw_res:
        plan = raw_res["parsed"]
        ai_msg = raw_res.get("raw")
    else:
        plan = raw_res
        ai_msg = None

    if plan is None:
        raise ValueError("Planner node failed to parse structured output.")

    # ---------------------------------------------------------
    # Hard Word Budget Enforcement & Section Distribution
    # ---------------------------------------------------------
    plan.target_word_count = target_words
    plan.min_word_count = min_words
    plan.max_word_count = max_words

    if plan.tasks:
        total_allocated = sum(t.target_words for t in plan.tasks)
        if total_allocated < min_words or total_allocated > max_words:
            logger.info(
                "Rebalancing section word budgets: total %d outside [%d, %d]. Target=%d",
                total_allocated,
                min_words,
                max_words,
                target_words,
            )
            # Rebalance proportionally so sum matches target_words
            scale_factor = target_words / max(1, total_allocated)
            rebalanced_sum = 0
            for t in plan.tasks[:-1]:
                new_target = max(50, min(5000, int(t.target_words * scale_factor)))
                t.target_words = new_target
                rebalanced_sum += new_target
            # Allocate remainder to the last task (or closing section)
            last_task = plan.tasks[-1]
            last_task.target_words = max(50, min(5000, target_words - rebalanced_sum))

        # Mark final section as closing section if not already marked
        for t in plan.tasks[:-1]:
            t.is_closing_section = False
        plan.tasks[-1].is_closing_section = True

        if not plan.closing_section_title:
            plan.closing_section_title = plan.tasks[-1].title

    metric = cost_tracker.extract_llm_metrics(
        response=ai_msg,
        node_name="planner",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    logger.info(
        "Generated outline with %d tasks. Planned total: %d words (target: %d).",
        len(plan.tasks),
        sum(t.target_words for t in plan.tasks),
        target_words,
    )
    return {
        "plan": plan,
        "target_word_count": target_words,
        "min_word_count": min_words,
        "max_word_count": max_words,
        "metrics": [metric],
    }
