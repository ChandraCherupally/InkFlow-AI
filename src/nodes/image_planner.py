"""
Image planner node implementation for InkFlow-AI.

Responsibilities:
- Inspect blog markdown article and plan 1-3 content-grounded technical visual illustrations.
- Programmatically enforce max 3 images, deterministic filenames, and sequential placeholder binding.
- Return GlobalImagePlan.
"""

from __future__ import annotations

import logging
import re
import time

from src.models.gateway import gateway
from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.prompts.base import PromptFactory
from src.prompts.prompts import SystemPrompts
from src.schemas.models import GlobalImagePlan, ImageSpec
from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def _sanitize_image_filename(raw_filename: str, fallback_index: int) -> str:
    """Ensure image filename is deterministic, lowercase, and contains no timestamps or duplicate suffixes."""
    fn = raw_filename.strip().lower()
    # Remove extension if present
    if fn.endswith(".png"):
        fn = fn[:-4]
    # Remove illegal chars, timestamps, (1), etc.
    fn = re.sub(r"\(\d+\)", "", fn)
    fn = re.sub(r"[-_]?\d{8}[-_]\d{6}", "", fn)
    fn = re.sub(r"[-_]?\d{8}", "", fn)
    fn = re.sub(r"[^a-z0-9_-]", "_", fn)
    fn = re.sub(r"_+", "_", fn).strip("_-")
    if not fn:
        fn = f"technical_diagram_{fallback_index}"
    return f"{fn}.png"


def image_planner(state: BlogState) -> dict:
    """
    Decide which technical illustrations/images should be created with deterministic validation.
    """
    logger.info("Running image planner node...")

    if not state.blog_markdown:
        logger.warning("No blog markdown available for image planning.")
        return {"image_plan": GlobalImagePlan()}

    start_time = time.perf_counter()
    config = get_node_config(NodeType.IMAGE_PLANNER)

    if state.plan and state.plan.tasks:
        article_context = f"Topic: {state.topic}\nAudience: {state.plan.audience}\nFormat: {state.plan.blog_kind}\n\nArticle Sections:\n" + "\n".join(
            f"Section {t.id}: {t.title}\nGoal: {t.goal}\nKey Points: {', '.join(t.bullets[:3])}"
            for t in state.plan.tasks
        )
    else:
        article_context = f"Topic: {state.topic}\n\nContent:\n{state.blog_markdown[:2000]}"

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

    # ---------------------------------------------------------
    # Hard Deterministic Image Constraint Enforcement (Max 3)
    # ---------------------------------------------------------
    if image_plan.images and len(image_plan.images) > 3:
        logger.info("Capping planned images from %d down to maximum 3.", len(image_plan.images))
        image_plan.images = image_plan.images[:3]

    # Re-index placeholders and sanitize filenames sequentially
    sanitized_images: list[ImageSpec] = []
    seen_filenames = set()

    for idx, spec in enumerate(image_plan.images, 1):
        placeholder = f"[[IMAGE_{idx}]]"
        clean_fn = _sanitize_image_filename(spec.filename, idx)
        if clean_fn in seen_filenames:
            clean_fn = f"{clean_fn[:-4]}_{idx}.png"
        seen_filenames.add(clean_fn)

        sanitized_spec = ImageSpec(
            placeholder=placeholder,
            filename=clean_fn,
            alt=spec.alt or f"Figure {idx}: Technical diagram",
            caption=spec.caption or f"Figure {idx}: Visual explanation of the system",
            prompt=spec.prompt,
            size=spec.size or "2560x1440",
            quality=spec.quality or "medium",
        )
        sanitized_images.append(sanitized_spec)

    image_plan.images = sanitized_images

    # If markdown_with_placeholders is empty or missing placeholders, use blog_markdown
    if not image_plan.markdown_with_placeholders:
        image_plan.markdown_with_placeholders = state.blog_markdown

    logger.info("Image planner finalized %d image(s).", len(image_plan.images))

    return {
        "image_plan": image_plan,
        "metrics": [metric],
    }
