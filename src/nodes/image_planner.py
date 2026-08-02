"""
Image planner node implementation for InkFlow-AI.

Responsibilities:
- Inspect blog markdown article.
- Plan technical visual illustrations and prompts efficiently.
- Return GlobalImagePlan.
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
from src.schemas.models import GlobalImagePlan
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def image_planner(state: BlogState) -> BlogState:
    """
    Decide which technical illustrations/images should be created.
    """
    logger.info("Running image planner node...")

    if not state.blog_markdown:
        logger.warning("No blog markdown available for image planning.")
        state.image_plan = GlobalImagePlan()
        return state

    start_time = time.perf_counter()
    config = get_node_config(NodeType.IMAGE_PLANNER)

    if state.plan and state.plan.tasks:
        article_context = f"Topic: {state.topic}\nAudience: {state.plan.audience}\nFormat: {state.plan.blog_kind}\n\nArticle Sections:\n" + "\n".join(
            f"Section {t.id}: {t.title}\nGoal: {t.goal}\nKey Points: {', '.join(t.bullets[:3])}"
            for t in state.plan.tasks
        )
    else:
        article_context = f"Topic: {state.topic}\n\nContent:\n{state.blog_markdown[:1500]}"

    prompt = PromptFactory.create(
        system_prompt=SystemPrompts.IMAGE_PLANNER,
        human_prompt="Article Structure & Goals:\n\n{article}\n",
    )

    llm = gateway.chat(NodeType.IMAGE_PLANNER).with_structured_output(
        GlobalImagePlan, include_raw=True
    )
    chain = prompt | llm

    raw_res = chain.invoke({"article": article_context})
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if isinstance(raw_res, dict) and "parsed" in raw_res:
        image_plan = raw_res["parsed"]
        ai_msg = raw_res.get("raw")
    else:
        image_plan = raw_res
        ai_msg = None

    if image_plan is None:
        image_plan = GlobalImagePlan()

    metric = cost_tracker.extract_llm_metrics(
        response=ai_msg,
        node_name="image_planner",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
    )

    logger.info("Image planner returned %d image(s).", len(image_plan.images))

    return {
        "image_plan": image_plan,
        "metrics": [metric],
    }
