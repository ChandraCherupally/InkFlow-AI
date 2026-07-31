"""
Main test execution script for InkFlow-AI.
"""

import sys

from src.graph.nodes import BlogNodes
from src.graph.state import BlogState
from src.tools.web_search import web_search

# Configure UTF-8 encoding for stdout on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    state = BlogState(
        topic="Latest LangGraph features in 2026"
    )

    state = BlogNodes.router(state)

    print(f"Routing mode: {state.routing_mode}")
    print(f"Needs research: {state.needs_research}")
    print(f"Search queries: {state.search_queries}")

    if state.needs_research:
        evidence_pack = web_search.search(state.search_queries)
        print(f"Retrieved {len(evidence_pack.evidence)} evidence items.")
        for item in evidence_pack.evidence:
            print("-" * 80)
            print(f"Title: {item.title}")
            print(f"URL: {item.url}")
            print(f"Snippet: {item.snippet}")


if __name__ == "__main__":
    main()