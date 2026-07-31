"""
RoutingGraph subgraph for InkFlow-AI.

Responsibilities:
- Encapsulate routing decisions.
- Determine research requirements and queries.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.nodes.router import router
from src.schemas.state import BlogState


def build_routing_graph():
    """
    Build and compile the RoutingGraph.
    """
    builder = StateGraph(BlogState)

    builder.add_node("router", router)
    builder.add_edge(START, "router")
    builder.add_edge("router", END)

    return builder.compile()


routing_graph = build_routing_graph()
