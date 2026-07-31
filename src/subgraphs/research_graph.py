"""
ResearchGraph subgraph for InkFlow-AI.

Responsibilities:
- Perform parallel web searches using Send() fanout.
- Merge and deduplicate evidence items into EvidencePack.
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.nodes.research import merge_research, tavily_worker
from src.schemas.state import BlogState


def fanout_queries(state: BlogState):
    """
    Fan out parallel Tavily search workers for each generated search query using Send().
    """
    if not state.search_queries:
        return "merge_research"

    return [
        Send("tavily_worker", {"query": query})
        for query in state.search_queries
    ]


def build_research_graph():
    """
    Build and compile the ResearchGraph.
    """
    builder = StateGraph(BlogState)

    builder.add_node("tavily_worker", tavily_worker)
    builder.add_node("merge_research", merge_research)

    builder.add_conditional_edges(
        START,
        fanout_queries,
        ["tavily_worker", "merge_research"],
    )

    builder.add_edge("tavily_worker", "merge_research")
    builder.add_edge("merge_research", END)

    return builder.compile()


research_graph = build_research_graph()
