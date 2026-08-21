"""
LangGraph application state for InkFlow-AI.

Responsibilities:
- Define the single shared global graph state.
- Define Annotated reducers for parallel Send() workers.
- Maintain compatibility across subgraphs.
- Store real-time observability metrics and execution summaries.
- Track deterministic word counts, publication QA results, and publication status.
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
    # User Input & Length Contract
    # ------------------------------------------------------------------
    topic: str
    target_word_count: int = 3500
    min_word_count: int = 2500
    max_word_count: int = 5000

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
    # Article Generation & Section Word Tracking
    # ------------------------------------------------------------------
    blog_markdown: str = ""
    sections: Annotated[list[tuple[int, str]], operator.add] = field(default_factory=list)
    section_word_counts: dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    image_plan: GlobalImagePlan | None = None
    generated_images: Annotated[list[GeneratedImage], operator.add] = field(default_factory=list)
    image_failures: list[str] = field(default_factory=list)
    final_markdown: str = ""

    # ------------------------------------------------------------------
    # Publication QA & Quality Gate
    # ------------------------------------------------------------------
    publication_status: str = "PENDING"  # "PASS", "FAIL", "PENDING"
    qa_result: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Metadata & Observability Execution
    # ------------------------------------------------------------------
    run_id: str = ""
    thread_id: str = "default"
    model_used: str = ""
    image_model_used: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    guardrail_warnings: Annotated[list[str], operator.add] = field(default_factory=list)
    guardrail_errors: Annotated[list[str], operator.add] = field(default_factory=list)
    metrics: Annotated[list[dict], operator.add] = field(default_factory=list)
    execution_summary: dict = field(default_factory=dict)
