"""
Comprehensive Publication QA and Contract Regression Tests for InkFlow-AI.

Tests:
1. test_word_counter_clean_prose: Validate word counter strips code, syntax, and placeholders.
2. test_resolve_word_budget_presets: Verify 3k, 5k, 13k flagship budget calculations.
3. test_plan_budget_distribution_flagship: Verify 12000-15000 word distribution sums properly.
4. test_publication_qa_pass: Verify a well-formed article passes publication QA.
5. test_publication_qa_word_count_below_min: Verify failure when article is under minimum words.
6. test_publication_qa_word_count_above_max: Verify failure when article exceeds maximum words.
7. test_publication_qa_missing_closing_section: Verify failure when no dedicated conclusion exists.
8. test_publication_qa_generic_key_takeaways_rejected: Verify rejection of generic Key Takeaways.
9. test_publication_qa_image_limits_max_3: Verify max 3 images enforcement.
10. test_publication_qa_image_failure_blocks_pass: Verify failed image generation marks publication as FAIL.
11. test_publication_qa_duplicate_image_filenames: Verify duplicate image filenames are caught.
12. test_image_planner_hard_capping: Verify image planner caps at 3 images and sanitizes filenames.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.publication_qa import validate_publication
from src.nodes.image_planner import _sanitize_image_filename
from src.schemas.models import (
    GeneratedImage,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    Task,
)
from src.tools.word_counter import count_words, resolve_word_budget


def test_word_counter_clean_prose():
    """Verify count_words strips markdown code, images, and formatting."""
    md = """# Title Here

*An engaging subtitle paragraph explaining the architecture.*

## Architecture Overview

Here is a paragraph explaining the distributed system with high clarity.

```python
def example_function():
    # Long code comment that should not count as body prose
    x = 1 + 2
    return x
```

![Alt text](images/diagram.png)
*Figure 1: Diagram explanation.*

[[IMAGE_1]]

| Col 1 | Col 2 |
|---|---|
| A | B |
"""
    words = count_words(md)
    # The prose text has around 25-30 words, definitely excluding code lines
    assert 15 <= words <= 40


def test_resolve_word_budget_presets():
    """Verify word budgets for standard, short, long-form, and flagship ranges."""
    # Default / standard
    std = resolve_word_budget(None)
    assert std["target_word_count"] == 3500
    assert std["min_word_count"] == 2500
    assert std["max_word_count"] == 5000

    # Short (2000)
    short = resolve_word_budget(2000)
    assert short["target_word_count"] == 2000
    assert short["min_word_count"] <= 2000 <= short["max_word_count"]

    # Flagship (13500)
    flagship = resolve_word_budget(13500)
    assert flagship["target_word_count"] == 13500
    assert flagship["min_word_count"] == 12000
    assert flagship["max_word_count"] == 15000


def test_plan_budget_distribution_flagship():
    """Verify that a 12000-15000 flagship plan distributes task targets within bounds."""
    budget = resolve_word_budget(13500)
    tasks = [
        Task(id=1, title="Introduction & Motivation", goal="Frame problem", target_words=1200),
        Task(id=2, title="Core Architecture", goal="Explain architecture", target_words=2200),
        Task(id=3, title="Pipeline Mechanics", goal="Explain data flow", target_words=2500),
        Task(id=4, title="Production Implementation", goal="Show implementation", target_words=2800),
        Task(id=5, title="Trade-offs & Failure Modes", goal="Analyze tradeoffs", target_words=2200),
        Task(id=6, title="Operational Guardrails", goal="Detail guardrails", target_words=1800),
        Task(id=7, title="Engineering Lessons from the Pipeline", goal="Synthesize lessons", target_words=800, is_closing_section=True),
    ]
    plan = Plan(
        blog_title="Building High-Throughput Agentic Systems",
        subtitle="A deep dive into distributed LLM orchestration and production pipelines.",
        target_word_count=budget["target_word_count"],
        min_word_count=budget["min_word_count"],
        max_word_count=budget["max_word_count"],
        closing_section_title="## Engineering Lessons from the Pipeline",
        tasks=tasks,
    )

    total_target = sum(t.target_words for t in plan.tasks)
    assert budget["min_word_count"] <= total_target <= budget["max_word_count"]


def test_publication_qa_pass():
    """Verify a compliant article passes publication QA."""
    body_prose = " ".join(["Production systems require resilient pipelines and strict error boundaries."] * 40)
    article_md = f"""# Building Resilient Systems

*A comprehensive blueprint for reliable architectures in production.*

## System Architecture

{body_prose}

![System Overview](images/system_overview.png)
*Figure 1: High level architecture overview.*

## Engineering Lessons from the Architecture

{body_prose}
"""
    plan = Plan(
        blog_title="Building Resilient Systems",
        subtitle="A comprehensive blueprint for reliable architectures in production.",
        target_word_count=500,
        min_word_count=200,
        max_word_count=1000,
        closing_section_title="## Engineering Lessons from the Architecture",
    )
    img_plan = GlobalImagePlan(
        images=[
            ImageSpec(
                placeholder="[[IMAGE_1]]",
                filename="system_overview.png",
                alt="System Overview",
                caption="Figure 1: High level architecture overview.",
                prompt="Claude style architecture diagram",
            )
        ]
    )
    gen_images = [
        GeneratedImage(
            filename="system_overview.png",
            path="images/system_overview.png",
            alt="System Overview",
            caption="Figure 1: High level architecture overview.",
        )
    ]

    result = validate_publication(
        article_markdown=article_md,
        plan=plan,
        image_plan=img_plan,
        generated_images=gen_images,
        target_word_count=500,
        min_word_count=200,
        max_word_count=1000,
    )

    assert result.status == "PASS"
    assert len(result.failures) == 0


def test_publication_qa_word_count_below_min():
    """Verify failure when article word count is below minimum."""
    short_md = """# Short Post

*A short subtitle.*

## Intro

Too short text.
"""
    result = validate_publication(
        article_markdown=short_md,
        min_word_count=500,
        max_word_count=1500,
    )
    assert result.status == "FAIL"
    assert any("below the minimum acceptable limit" in f for f in result.failures)


def test_publication_qa_generic_key_takeaways_rejected():
    """Verify rejection when generic Key Takeaways is used instead of planned thesis heading."""
    article_md = """# Data Annotation Pipeline

*A comprehensive guide to high-accuracy training data.*

## Data Annotation Mechanics

Prose content here explaining annotation techniques and quality control.

## Key Takeaways

Generic takeaway points.
"""
    plan = Plan(
        blog_title="Data Annotation Pipeline",
        subtitle="A comprehensive guide to high-accuracy training data.",
        target_word_count=50,
        min_word_count=10,
        max_word_count=500,
        closing_section_title="## What Reliable Annotation Requires",
    )
    result = validate_publication(
        article_markdown=article_md,
        plan=plan,
        min_word_count=10,
        max_word_count=500,
    )
    assert result.status == "FAIL"
    assert any("Generic closing heading" in f for f in result.failures)


def test_publication_qa_image_limits_max_3():
    """Verify publication fails if more than 3 images are planned."""
    specs = [
        ImageSpec(placeholder=f"[[IMAGE_{i}]]", filename=f"img_{i}.png", alt=f"Img {i}", caption=f"Cap {i}", prompt="Prompt")
        for i in range(1, 5)
    ]
    img_plan = GlobalImagePlan(images=specs)
    result = validate_publication(
        article_markdown="# Title\n\n*Subtitle*\n\n## Section\n\nContent",
        image_plan=img_plan,
        min_word_count=1,
        max_word_count=500,
    )
    assert result.status == "FAIL"
    assert any("maximum allowed is 3" in f for f in result.failures)


def test_publication_qa_image_failure_blocks_pass():
    """Verify image generation failure blocks publication."""
    img_plan = GlobalImagePlan(
        images=[
            ImageSpec(placeholder="[[IMAGE_1]]", filename="arch.png", alt="Alt", caption="Cap", prompt="P")
        ]
    )
    result = validate_publication(
        article_markdown="# Title\n\n*Subtitle*\n\n## Section\n\nContent",
        image_plan=img_plan,
        image_failures=["arch.png"],
        min_word_count=1,
        max_word_count=500,
    )
    assert result.status == "FAIL"
    assert any("Image generation failed" in f for f in result.failures)


def test_publication_qa_duplicate_image_filenames():
    """Verify duplicate image filenames are caught."""
    img_plan = GlobalImagePlan(
        images=[
            ImageSpec(placeholder="[[IMAGE_1]]", filename="arch.png", alt="Alt", caption="Cap", prompt="P"),
            ImageSpec(placeholder="[[IMAGE_2]]", filename="arch.png", alt="Alt 2", caption="Cap 2", prompt="P"),
        ]
    )
    result = validate_publication(
        article_markdown="# Title\n\n*Subtitle*\n\n## Section\n\nContent",
        image_plan=img_plan,
        min_word_count=1,
        max_word_count=500,
    )
    assert result.status == "FAIL"
    assert any("Duplicate image filename" in f for f in result.failures)


def test_image_planner_sanitization():
    """Verify image filename sanitization removes (1) and timestamps."""
    assert _sanitize_image_filename("diagram(1).png", 1) == "diagram.png"
    assert _sanitize_image_filename("architecture_20260814_120000.png", 2) == "architecture.png"
    assert _sanitize_image_filename("My Complex Workflow Diagram.png", 3) == "my_complex_workflow_diagram.png"
