"""
WritingGraph subgraph for InkFlow-AI.

Responsibilities:
- Write blog sections in parallel using Send() fanout.
- Assemble written section markdowns into a unified article.
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.nodes.worker import assemble_sections, worker_section
from src.schemas.state import BlogState


def fanout_tasks(state: BlogState):
    """
    Fan out parallel writer workers for each section task in the Plan using Send().
    """
    if not state.plan or not state.plan.tasks:
        return "assemble_sections"

    return [
        Send("worker_section", {"topic": state.topic, "task": task})
        for task in state.plan.tasks
    ]


def build_writing_graph():
    """
    Build and compile the WritingGraph.
    """
    builder = StateGraph(BlogState)

    builder.add_node("worker_section", worker_section)
    builder.add_node("assemble_sections", assemble_sections)

    builder.add_conditional_edges(
        START,
        fanout_tasks,
        ["worker_section", "assemble_sections"],
    )

    builder.add_edge("worker_section", "assemble_sections")
    builder.add_edge("assemble_sections", END)

    return builder.compile()


writing_graph = build_writing_graph()
