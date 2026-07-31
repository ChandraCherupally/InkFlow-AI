"""
Router node implementation for InkFlow-AI.

Responsibilities:
- Determine routing mode and research requirements.
- Generate search queries if needed.
- Update BlogState.
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
from src.schemas.models import RouterDecision
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def router(state: BlogState) -> BlogState:
    """
    Decide whether the workflow requires external research.
    """
    logger.info("Running router node...")
    start_time = time.perf_counter()
    config = get_node_config(NodeType.ROUTER)

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.ROUTER,
        human_prompt="Topic:\n{topic}\n",
    )

    llm = gateway.chat(NodeType.ROUTER).with_structured_output(
        RouterDecision, include_raw=True
    )
    chain = prompt | llm

    raw_res = chain.invoke({"topic": state.topic})
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if isinstance(raw_res, dict) and "parsed" in raw_res:
        decision = raw_res["parsed"]
        ai_msg = raw_res.get("raw")
    else:
        decision = raw_res
        ai_msg = None

    if decision is None:
        raise ValueError("Router node failed to parse structured output.")

    metric = cost_tracker.extract_llm_metrics(
        response=ai_msg,
        node_name="router",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    logger.info("Router Mode=%s Research=%s", decision.mode, decision.needs_research)

    return {
        "routing_mode": decision.mode,
        "needs_research": decision.needs_research,
        "search_queries": decision.queries,
        "metrics": [metric],
    }


def route_after_router(state: BlogState) -> str:
    """
    Conditional edge logic after router.
    """
    if state.needs_research:
        return "research"
    return "planner"
