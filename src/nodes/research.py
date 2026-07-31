"""
Research node implementations for InkFlow-AI.

Responsibilities:
- Perform parallel Tavily web searches using Send().
- Deduplicate evidence items and build EvidencePack.
"""

from __future__ import annotations

import logging

import time

from src.models.registry import get_node_config
from src.models.types import NodeType
from src.observability.cost_tracker import cost_tracker
from src.schemas.models import EvidenceItem, EvidencePack
from src.schemas.state import BlogState
from src.tools.web_search import web_search

logger = logging.getLogger(__name__)


def tavily_worker(state: dict) -> dict:
    """
    Parallel Tavily worker for a single query.
    """
    query = state.get("query", "")
    logger.info("Executing parallel Tavily search query: '%s'", query)

    start_time = time.perf_counter()
    config = get_node_config(NodeType.RESEARCH)

    evidence_pack = web_search.search(queries=[query])
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    items = evidence_pack.evidence if evidence_pack else []

    metric = cost_tracker.create_metric(
        node_name="research",
        provider=config.primary.provider,
        model=config.primary.model,
        latency_ms=latency_ms,
        estimated_cost=0.001,
        status="completed",
    )

    return {
        "raw_evidence_list": items,
        "metrics": [metric],
    }


def merge_research(state: BlogState) -> BlogState:
    """
    Merge and deduplicate parallel research evidence items.
    """
    logger.info("Merging parallel research results...")

    seen_urls: set[str] = set()
    deduped: list[EvidenceItem] = []

    for item in state.raw_evidence_list:
        url_str = str(item.url)
        if url_str not in seen_urls:
            seen_urls.add(url_str)
            deduped.append(item)

    logger.info("Prepared %d deduplicated evidence items.", len(deduped))
    state.evidence = EvidencePack(evidence=deduped)

    return state
