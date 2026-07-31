"""
LangGraph application state for InkFlow-AI.

Responsibilities:
- Define the single shared global graph state.
- Define Annotated reducers for parallel Send() workers.
- Maintain compatibility across subgraphs.
- Store real-time observability metrics and execution summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Annotated
from src.schemas.models import EvidenceItem, EvidencePack, GeneratedImage, GlobalImagePlan, Plan


@dataclass
class BlogState:
    """
    Global LangGraph state shared across main graph and all subgraphs.

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
    raw_evidence_list: Annotated[list[EvidenceItem], operator.add] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    plan: Plan | None = None

    # ------------------------------------------------------------------
    # Article Generation
    # ------------------------------------------------------------------
    blog_markdown: str = ""
    sections: Annotated[list[tuple[int, str]], operator.add] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    image_plan: GlobalImagePlan | None = None
    generated_images: Annotated[list[GeneratedImage], operator.add] = field(default_factory=list)
    final_markdown: str = ""

    # ------------------------------------------------------------------
    # Metadata & Observability Execution
    # ------------------------------------------------------------------
    thread_id: str = "default"
    model_used: str = ""
    image_model_used: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: Annotated[list[dict], operator.add] = field(default_factory=list)
    execution_summary: dict = field(default_factory=dict)
