"""
PublishingGraph subgraph for InkFlow-AI.

Responsibilities:
- Validate content assembly.
- Execute Senior Editorial Review.
- Run Output Guardrails validation.
- Standardize Markdown presentation and formatting.
- Plan technical visual illustrations for the polished article.
- Generate images in parallel using Send() fanout.
- Format and produce the final Markdown deliverable.
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.guardrails.output_guardrails import output_guardrails
from src.nodes.editor import editor
from src.nodes.formatter import markdown_formatter
from src.nodes.image_generator import assemble_publishing, image_worker
from src.nodes.image_planner import image_planner
from src.nodes.merge import merge_content
from src.schemas.state import BlogState


def fanout_images(state: BlogState):
    """
    Fan out parallel image generation workers for each planned ImageSpec using Send().
    """
    if not state.image_plan or not state.image_plan.images:
        return "assemble_publishing"

    return [
        Send("image_worker", {"spec": spec})
        for spec in state.image_plan.images
    ]


def build_publishing_graph():
    """
    Build and compile the PublishingGraph.
    Flow: merge_content -> editor -> output_guardrails -> image_planner -> image_worker -> assemble_publishing
    """
    builder = StateGraph(BlogState)

    builder.add_node("merge_content", merge_content)
    builder.add_node("editor", editor)
    builder.add_node("output_guardrails", output_guardrails)
    builder.add_node("image_planner", image_planner)
    builder.add_node("image_worker", image_worker)
    builder.add_node("assemble_publishing", assemble_publishing)

    # Sequential Editorial Flow
    builder.add_edge(START, "merge_content")
    builder.add_edge("merge_content", "editor")
    builder.add_edge("editor", "output_guardrails")
    builder.add_edge("output_guardrails", "image_planner")

    # Image Generation Fanout
    builder.add_conditional_edges(
        "image_planner",
        fanout_images,
        ["image_worker", "assemble_publishing"],
    )

    builder.add_edge("image_worker", "assemble_publishing")
    builder.add_edge("assemble_publishing", END)

    return builder.compile()


publishing_graph = build_publishing_graph()

