"""
Root workflow graph for InkFlow-AI.

Architecture:
START -> Input Guardrails -> RoutingGraph -> ResearchGraph -> Planning Node -> WritingGraph -> PublishingGraph -> END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.guardrails.input_guardrails import input_guardrails
from src.nodes.planner import planner
from src.schemas.state import BlogState
from src.subgraphs.publishing_graph import publishing_graph
from src.subgraphs.research_graph import research_graph
from src.subgraphs.routing_graph import routing_graph
from src.subgraphs.writing_graph import writing_graph


def route_after_input_guardrails(state: BlogState) -> str:
    """
    Determine transition after input guardrails based on error state.
    """
    if state.error:
        return "end"
    return "routing"


def route_after_routing_stage(state: BlogState) -> str:
    """
    Determine transition after RoutingGraph based on needs_research state.
    """
    if state.needs_research:
        return "research"
    return "planner"


def build_main_graph(checkpointer=None):
    """
    Construct and compile the root StateGraph orchestrating all subgraphs and nodes.
    """
    if checkpointer is None:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("src.schemas.models", "EvidencePack"),
                ("src.schemas.models", "EvidenceItem"),
                ("src.schemas.models", "Plan"),
                ("src.schemas.models", "GlobalImagePlan"),
                ("src.schemas.models", "GeneratedImage"),
                ("src.schemas.models", "Task"),
                ("src.schemas.models", "RouterDecision"),
                ("src.schemas.models", "ImageSpec"),
                ("src.schemas.state", "BlogState"),
                ("src.schemas.blog", "EvidencePack"),
                ("src.schemas.blog", "EvidenceItem"),
                ("src.schemas.blog", "Plan"),
                ("src.schemas.blog", "GlobalImagePlan"),
                ("src.schemas.blog", "GeneratedImage"),
                ("src.schemas.blog", "Task"),
                ("src.schemas.blog", "RouterDecision"),
                ("src.schemas.blog", "ImageSpec"),
                ("src.schemas.blog", "BlogState"),
            ]
        )
        checkpointer = MemorySaver(serde=serde)

    builder = StateGraph(BlogState)

    # ---------------------------------------------------------
    # Subgraphs & Nodes Registration
    # ---------------------------------------------------------
    builder.add_node("input_guardrails", input_guardrails)
    builder.add_node("routing", routing_graph)
    builder.add_node("research", research_graph)
    builder.add_node("planner", planner)
    builder.add_node("writing", writing_graph)
    builder.add_node("publishing", publishing_graph)

    # ---------------------------------------------------------
    # Transitions
    # ---------------------------------------------------------
    builder.add_edge(START, "input_guardrails")

    builder.add_conditional_edges(
        "input_guardrails",
        route_after_input_guardrails,
        {
            "end": END,
            "routing": "routing",
        },
    )

    builder.add_conditional_edges(
        "routing",
        route_after_routing_stage,
        {
            "research": "research",
            "planner": "planner",
        },
    )

    builder.add_edge("research", "planner")
    builder.add_edge("planner", "writing")
    builder.add_edge("writing", "publishing")
    builder.add_edge("publishing", END)

    return builder.compile(checkpointer=checkpointer)


main_graph = build_main_graph()

