"""
Image generator node implementations for InkFlow-AI.

Responsibilities:
- Generate images in parallel using Send().
- Replace placeholders inline and build final Markdown deliverable.
"""

from __future__ import annotations

import logging

import time

from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.schemas.models import GeneratedImage, ImageSpec
from src.schemas.state import BlogState
from src.tools.image_generator import image_generator
from src.tools.markdown import markdown_builder

logger = logging.getLogger(__name__)


def image_worker(state: dict) -> dict:
    """
    Parallel worker for generating a single image specification.
    """
    spec_data = state.get("spec")

    if not spec_data:
        return {"generated_images": [], "metrics": []}

    start_time = time.perf_counter()
    config = get_node_config(NodeType.IMAGE_GENERATOR)

    try:
        spec = spec_data if isinstance(spec_data, ImageSpec) else ImageSpec(**spec_data)
        logger.info("Generating parallel image for placeholder: %s", spec.placeholder)

        generated = image_generator.generate(spec)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metric = cost_tracker.create_metric(
            node_name="image_generator",
            provider=config.primary.provider,
            model=config.primary.model,
            latency_ms=latency_ms,
            images_generated=1,
            resolution=spec.size or "1792x1024",
            estimated_cost=0.04,
            status="completed",
        )

        return {
            "generated_images": [generated],
            "metrics": [metric],
        }
    except Exception as e:
        logger.warning("Parallel image worker failed for spec: %s", e)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        metric = cost_tracker.create_metric(
            node_name="image_generator",
            provider=config.primary.provider,
            model=config.primary.model,
            latency_ms=latency_ms,
            images_generated=0,
            status="failed",
        )
        return {
            "generated_images": [],
            "metrics": [metric],
        }


def assemble_publishing(state: BlogState) -> BlogState:
    """
    Collect generated images and build the final markdown document with inline image placement.
    """
    logger.info("Assembling final publishing output with inline image placement...")

    sections = [state.blog_markdown] if state.blog_markdown else []
    markdown_with_placeholders = (
        state.image_plan.markdown_with_placeholders
        if state.image_plan and state.image_plan.markdown_with_placeholders
        else ""
    )

    if state.plan:
        state.final_markdown = markdown_builder.build(
            plan=state.plan,
            sections=sections,
            images=state.generated_images,
            markdown_with_placeholders=markdown_with_placeholders,
        )
    else:
        state.final_markdown = state.blog_markdown

    logger.info("Final Medium-grade Markdown document with inline images assembled successfully.")
    return state
