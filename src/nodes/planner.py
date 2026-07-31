"""
Planner node implementation for InkFlow-AI.

Responsibilities:
- Consume topic and research evidence.
- Produce structured Plan outline efficiently.
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

logger = logging.getLogger(__name__)


def planner(state: BlogState) -> BlogState:
    """
    Generate a structured blog outline with optimized research context.
    """
    logger.info("Running planner node...")
    start_time = time.perf_counter()
    config = get_node_config(NodeType.PLANNER)

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.PLANNER,
        human_prompt="""Topic:
{topic}

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

    metric = cost_tracker.extract_llm_metrics(
        response=ai_msg,
        node_name="planner",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    logger.info("Generated outline with %d tasks.", len(plan.tasks))
    return {
        "plan": plan,
        "metrics": [metric],
    }
