"""
Merge node implementation for InkFlow-AI.

Responsibilities:
- Validate content assembly prior to image planning.
"""

from __future__ import annotations

import logging

from src.schemas.state import BlogState

logger = logging.getLogger(__name__)


def merge_content(state: BlogState) -> BlogState:
    """
    Validate content assembly prior to image planning.
    """
    logger.info("Running merge_content node...")
    if not state.blog_markdown:
        logger.warning("blog_markdown is empty during content merge.")
    return state
