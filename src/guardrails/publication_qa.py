"""
Deterministic Publication QA Verification Layer for InkFlow-AI.

Responsibilities:
- Programmatically validate article length, narrative structure, section word compliance,
  closing section style, and image invariants.
- Serve as the final publication gate before saving deliverables.
- Return strongly-typed PublicationQAResult.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.schemas.models import (
    GeneratedImage,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    PublicationQAResult,
)
from src.tools.word_counter import count_words

logger = logging.getLogger(__name__)

# Prompt leakage detection regex
_LEAKAGE_PATTERNS = [
    r"\bas\s+an\s+ai\s+language\s+model\b",
    r"\bhere\s+is\s+your\s+system\s+prompt\b",
    r"\bi\s+am\s+configured\s+to\b",
    r"\bignore\s+previous\s+instructions\b",
    r"^system\s+prompt:",
]
_LEAKAGE_REGEX = re.compile("|".join(_LEAKAGE_PATTERNS), re.IGNORECASE | re.MULTILINE)

# Placeholder token patterns
_PLACEHOLDER_REGEX = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")
_DOUBLE_CURLY_REGEX = re.compile(r"\{\{.*?\}\}")

# Generic banned closing headings when domain-specific thesis is required
_GENERIC_CLOSING_HEADINGS = {
    "key takeaways",
    "final thoughts",
    "summary",
    "conclusion",
    "wrapping up",
    "in conclusion",
}

# Unstable filename patterns (e.g. image(1).png, diagram_20260814_120000.png)
_UNSTABLE_FILENAME_REGEX = re.compile(r"\(\d+\)|\b\d{8}[-_]\d{6}\b")


def validate_publication(
    article_markdown: str,
    plan: Plan | None = None,
    image_plan: GlobalImagePlan | None = None,
    generated_images: list[GeneratedImage] | None = None,
    section_word_counts: dict[int, int] | None = None,
    image_failures: list[str] | None = None,
    target_word_count: int = 3500,
    min_word_count: int = 2500,
    max_word_count: int = 5000,
) -> PublicationQAResult:
    """
    Perform deterministic publication QA checks on the finalized article.
    """
    failures: list[str] = []
    warnings: list[str] = []
    markdown_errors: list[str] = []
    section_results: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []

    clean_md = (article_markdown or "").strip()

    # -------------------------------------------------------------
    # 1. Title & Subtitle Structure
    # -------------------------------------------------------------
    lines = clean_md.splitlines()
    h1_headings = [line for line in lines if line.strip().startswith("# ") and not line.strip().startswith("## ")]

    if not h1_headings:
        failures.append("Missing H1 title (# Title).")
    elif len(h1_headings) > 1:
        failures.append(f"Multiple H1 headings detected ({len(h1_headings)} found).")

    # Check for subtitle (italicized paragraph near top)
    has_subtitle = False
    for line in lines[:8]:
        s = line.strip()
        if s.startswith("*") and s.endswith("*") and len(s) > 10:
            has_subtitle = True
            break
    if not has_subtitle and plan and plan.subtitle:
        has_subtitle = True

    if not has_subtitle:
        failures.append("Missing italicized subtitle (*Subtitle text...*).")

    # -------------------------------------------------------------
    # 2. Total Word Count Contract
    # -------------------------------------------------------------
    actual_words = count_words(clean_md)
    effective_target = plan.target_word_count if plan and plan.target_word_count else target_word_count
    effective_min = plan.min_word_count if plan and plan.min_word_count else min_word_count
    effective_max = plan.max_word_count if plan and plan.max_word_count else max_word_count

    if actual_words < effective_min:
        failures.append(
            f"Article word count ({actual_words} words) is below the minimum acceptable limit ({effective_min} words)."
        )
    elif actual_words > effective_max:
        failures.append(
            f"Article word count ({actual_words} words) exceeds the maximum acceptable limit ({effective_max} words)."
        )

    # -------------------------------------------------------------
    # 3. Section Word Count Compliance
    # -------------------------------------------------------------
    if plan and plan.tasks:
        sec_counts = section_word_counts or {}
        for task in plan.tasks:
            task_actual = sec_counts.get(task.id, 0)
            task_min = max(50, int(task.target_words * 0.70))  # generous tolerance for QA gate
            task_max = int(task.target_words * 1.30)
            passed_sec = task_actual == 0 or (task_min <= task_actual <= task_max)
            sec_res = {
                "id": task.id,
                "title": task.title,
                "target_words": task.target_words,
                "actual_words": task_actual,
                "status": "PASS" if passed_sec else "WARN",
            }
            section_results.append(sec_res)
            if task_actual > 0 and not passed_sec:
                warnings.append(
                    f"Section {task.id} ('{task.title}') word count ({task_actual}) deviates from target {task.target_words}."
                )

    # -------------------------------------------------------------
    # 4. Closing Section Verification
    # -------------------------------------------------------------
    h2_headings = [line.strip()[3:].strip() for line in lines if line.strip().startswith("## ")]
    detected_closing = h2_headings[-1] if h2_headings else ""

    if not h2_headings:
        failures.append("No H2 sections found in article.")
    else:
        # Check if closing section is generic when plan specifies a domain heading
        norm_closing = detected_closing.lower()
        if plan and plan.closing_section_title:
            expected_closing = plan.closing_section_title.replace("#", "").strip().lower()
            if norm_closing in _GENERIC_CLOSING_HEADINGS and expected_closing not in _GENERIC_CLOSING_HEADINGS:
                failures.append(
                    f"Generic closing heading ('{detected_closing}') used instead of domain-specific heading ('{plan.closing_section_title}')."
                )

    # -------------------------------------------------------------
    # 5. Image Plan & Placeholder Integrity
    # -------------------------------------------------------------
    specs: list[ImageSpec] = image_plan.images if image_plan and image_plan.images else []
    gen_images: list[GeneratedImage] = generated_images or []

    if len(specs) > 3:
        failures.append(f"Image plan contains {len(specs)} images (maximum allowed is 3).")

    # Check for image generation failures
    if image_failures:
        for failed_file in image_failures:
            failures.append(f"Image generation failed for planned asset '{failed_file}'.")

    # Check placeholder consistency
    placeholder_matches = re.findall(r"\[\[?IMAGE_(\d+)\]?\]", clean_md, re.IGNORECASE)
    seen_filenames = set()

    for idx, spec in enumerate(specs, 1):
        clean_fn = spec.filename.strip().lower()
        if not clean_fn.endswith(".png"):
            failures.append(f"Image filename '{spec.filename}' must end with .png.")

        if " " in clean_fn:
            failures.append(f"Image filename '{spec.filename}' must not contain spaces.")

        if _UNSTABLE_FILENAME_REGEX.search(clean_fn):
            failures.append(f"Unstable image filename detected ('{spec.filename}').")

        if clean_fn in seen_filenames:
            failures.append(f"Duplicate image filename detected ('{spec.filename}').")
        seen_filenames.add(clean_fn)

        # Image spec validation record
        image_results.append({
            "placeholder": spec.placeholder,
            "filename": spec.filename,
            "alt": spec.alt,
            "caption": spec.caption,
            "status": "PASS",
        })

    # Check if placeholder is inside a table
    table_lines = [line for line in lines if line.strip().startswith("|")]
    for t_line in table_lines:
        if re.search(r"\[\[?IMAGE_\d+\]?\]", t_line, re.IGNORECASE):
            failures.append("Image placeholder detected inside Markdown table.")
            break

    # -------------------------------------------------------------
    # 6. Syntax & Guardrails Errors
    # -------------------------------------------------------------
    # Unclosed code fences
    fence_count = sum(1 for line in lines if line.strip().startswith("```"))
    if fence_count % 2 != 0:
        failures.append("Unclosed markdown code block (odd number of code fences).")
        markdown_errors.append("Unclosed code fence.")

    # Unresolved tokens
    unresolved_tokens = _PLACEHOLDER_REGEX.findall(clean_md)
    if unresolved_tokens:
        failures.append(f"Unresolved placeholder tokens ({', '.join(sorted(set(unresolved_tokens)))}).")
        markdown_errors.append("Unresolved placeholders.")

    curly_matches = _DOUBLE_CURLY_REGEX.findall(clean_md)
    if curly_matches:
        failures.append(f"Unresolved double curly template tokens ({len(curly_matches)} item(s)).")
        markdown_errors.append("Unresolved double curlies.")

    # Prompt leakage
    leakage = _LEAKAGE_REGEX.search(clean_md)
    if leakage:
        failures.append(f"Prompt leakage detected: '{leakage.group(0)}'.")

    # -------------------------------------------------------------
    # QA Outcome Resolution
    # -------------------------------------------------------------
    status = "FAIL" if failures else "PASS"

    return PublicationQAResult(
        status=status,
        failures=failures,
        warnings=warnings,
        actual_word_count=actual_words,
        target_word_count=effective_target,
        min_word_count=effective_min,
        max_word_count=effective_max,
        section_results=section_results,
        image_count=len(specs),
        image_results=image_results,
        closing_section=detected_closing,
        markdown_errors=markdown_errors,
    )
