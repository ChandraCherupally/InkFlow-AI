"""
LangGraph application state.

Responsibilities:
- Define the shared graph state.
- Define reducer annotations.
- Define reusable state types.

This module MUST NOT contain:
- Node logic
- LLM calls
- Business logic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from src.schemas.blog import EvidencePack, GeneratedImage, GlobalImagePlan, Plan


@dataclass
class BlogState:
    """
    Global LangGraph state.

    Every node reads from and/or writes to this state.
    """

    # ------------------------------------------------------------------
    # User Input
    # ------------------------------------------------------------------
    topic: str

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    routing_mode: str = "closed_book"
    needs_research: bool = False
    search_queries: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------
    evidence: EvidencePack = field(default_factory=EvidencePack)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    plan: Plan | None = None

    # ------------------------------------------------------------------
    # Article Generation
    # ------------------------------------------------------------------
    blog_markdown: str = ""

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    image_plan: GlobalImagePlan | None = None
    generated_images: list[GeneratedImage] = field(default_factory=list)
    final_markdown: str = ""

    # ------------------------------------------------------------------
    # Metadata & Execution
    # ------------------------------------------------------------------
    thread_id: str = "default"
    model_used: str = ""
    image_model_used: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)