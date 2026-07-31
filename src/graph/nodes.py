"""
LangGraph node implementations for InkFlow-AI.

Responsibilities
----------------
- Execute business workflow.
- Coordinate prompts and LLM calls.
- Update BlogState.
- Return updated state.

This module contains NO graph construction.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import JsonOutputParser

from src.graph.state import BlogState
from src.models.gateway import gateway
from src.prompts.base import PromptFactory
from src.prompts.prompts import SystemPrompts
from src.schemas.blog import (
    EvidencePack,
    GeneratedImage,
    GlobalImagePlan,
    Plan,
    RouterDecision,
)
from src.tools.image_generator import image_generator
from src.tools.markdown import markdown_builder
from src.tools.web_search import web_search

logger = logging.getLogger(__name__)


class BlogNodes:
    """Collection of LangGraph node implementations."""

    @staticmethod
    def router(state: BlogState) -> BlogState:
        """
        Decide whether the workflow requires external research.

        Workflow
        --------
        Topic
            ↓
        Router LLM
            ↓
        RouterDecision
            ↓
        Update BlogState
        """

        logger.info("Running router node...")

        prompt = PromptFactory.create(
            system_prompt=SystemPrompts.ROUTER,
            human_prompt="""Topic:
{topic}
""",
        )

        llm = gateway.chat().with_structured_output(RouterDecision)

        chain = prompt | llm

        decision = chain.invoke(
            {
                "topic": state.topic,
            }
        )

        logger.info("Router Mode=%s Research=%s", decision.mode, decision.needs_research)

        state.routing_mode = decision.mode
        state.needs_research = decision.needs_research
        state.search_queries = decision.queries

        return state


    @staticmethod
    def route_after_router(state: BlogState) -> str:
        """
        Conditional edge after router.

        Returns
        -------
        str
            "research" or "planner"
        """

        if state.needs_research:
            return "research"

        return "planner"

    # ==========================================================
    # Research Node
    # ==========================================================

    @staticmethod
    def research(state: BlogState) -> BlogState:
        """
        Execute external web research.

        Workflow

        Search Queries
                │
                ▼
        Web Search
                │
                ▼
        EvidencePack
                │
                ▼
        Update BlogState
        """

        logger.info("Running research node...")

        if not state.search_queries:
            logger.info("No search queries generated.")
            state.evidence = EvidencePack()
            return state

        evidence_pack = web_search.search(
            queries=state.search_queries,
        )

        logger.info(
            "Retrieved %d research evidence items.",
            len(evidence_pack.evidence),
        )

        state.evidence = evidence_pack

        return state

    # ==========================================================
    # Planner Node
    # ==========================================================

    @staticmethod
    def planner(state: BlogState) -> BlogState:
        """
        Generate a structured blog outline.

        Workflow

        Topic
            +
        Research Evidence
            │
            ▼
        Planner LLM
            │
            ▼
        Plan
            │
            ▼
        Update BlogState
        """

        logger.info("Running planner node...")

        prompt = PromptFactory.create(
            system_prompt=SystemPrompts.PLANNER,
            human_prompt="""
Topic:
{topic}

Research Context:
{research}
""",
        )

        llm = gateway.chat().with_structured_output(
            Plan
        )

        chain = prompt | llm

        research_context = "\n\n".join(
            f"- Title: {item.title}\n  URL: {item.url}\n  Snippet: {item.snippet or ''}"
            for item in state.evidence.evidence
        )

        plan = chain.invoke(
            {
                "topic": state.topic,
                "research": research_context,
            }
        )

        logger.info(
            "Generated outline with %d tasks.",
            len(plan.tasks),
        )

        state.plan = plan

        return state

    # ==========================================================
    # Writer Node
    # ==========================================================

    @staticmethod
    def writer(state: BlogState) -> BlogState:
        """
        Generate the complete markdown article.

        Workflow

        Plan
          │
          ▼
        Generate each task
          │
          ▼
        Markdown article
        """

        logger.info("Running writer node...")

        if not state.plan or not state.plan.tasks:
            logger.warning("No plan or tasks available for writing.")
            state.blog_markdown = ""
            return state

        prompt = PromptFactory.create(
            system_prompt=SystemPrompts.WRITER,
            human_prompt="""
Topic:
{topic}

Section Title:
{title}

Goal:
{goal}

Bullet Points:
{bullets}
""",
        )

        llm = gateway.chat()

        chain = prompt | llm

        sections: list[str] = []

        for task in state.plan.tasks:

            response = chain.invoke(
                {
                    "topic": state.topic,
                    "title": task.title,
                    "goal": task.goal,
                    "bullets": "\n".join(task.bullets),
                }
            )

            sections.append(response.content)

        state.blog_markdown = "\n\n".join(sections)

        logger.info(
            "Generated %d markdown sections.",
            len(sections),
        )

        return state

    # ==========================================================
    # Image Planner
    # ==========================================================

    @staticmethod
    def image_planner(state: BlogState) -> BlogState:
        """
        Decide which images should be created.
        """

        logger.info("Running image planner...")

        prompt = PromptFactory.create(
            system_prompt=SystemPrompts.IMAGE_PLANNER,
            human_prompt="""
Article

{article}
""",
        )

        llm = gateway.chat().with_structured_output(
            GlobalImagePlan,
        )

        chain = prompt | llm

        image_plan = chain.invoke(
            {
                "article": state.blog_markdown,
            }
        )

        state.image_plan = image_plan

        logger.info(
            "Image planner returned %d image(s).",
            len(image_plan.images),
        )

        return state

    # ==========================================================
    # Image Generator
    # ==========================================================

    @staticmethod
    def image_generator(state: BlogState) -> BlogState:
        """
        Generate all planned images.
        """

        logger.info("Running image generator...")

        if not state.image_plan or not state.image_plan.images:
            logger.info("No images requested.")

            state.generated_images = []
            if state.plan:
                sections = [state.blog_markdown] if state.blog_markdown else []
                state.final_markdown = markdown_builder.build(
                    state.plan, sections, []
                )
            else:
                state.final_markdown = state.blog_markdown

            return state

        generated_images: list[GeneratedImage] = []

        for image_spec in state.image_plan.images:

            generated = image_generator.generate(
                image_spec,
            )

            generated_images.append(generated)

        state.generated_images = generated_images

        logger.info(
            "Generated %d image(s).",
            len(generated_images),
        )

        if state.plan:
            sections = [state.blog_markdown] if state.blog_markdown else []
            state.final_markdown = markdown_builder.build(
                state.plan, sections, generated_images
            )
        else:
            state.final_markdown = state.blog_markdown

        return state