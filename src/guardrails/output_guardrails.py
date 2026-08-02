"""
Output guardrails implementation for InkFlow-AI.

Responsibilities:
- Validate generated blog markdown after Senior Editorial Review.
- Detect unresolved placeholders, prompt leakage, leaked secrets/API keys, stack traces, empty headings, and unclosed code fences.
- Pure Python deterministic validation with standard library regex and string operations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.schemas.state import BlogState

logger = logging.getLogger(__name__)

# Placeholder token patterns (TODO, TBD, FIXME, XXX)
_PLACEHOLDER_REGEX = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")

# Template tag placeholders {{ ... }}
_DOUBLE_CURLY_REGEX = re.compile(r"\{\{.*?\}\}")

# Image plan placeholders [[IMAGE_*]]
_IMAGE_PLACEHOLDER_REGEX = re.compile(r"\[\[IMAGE_[^\]]*\]\]")

# Prompt leakage phrases
_LEAKAGE_PATTERNS = [
    r"\bas\s+an\s+ai\s+language\s+model\b",
    r"\bhere\s+is\s+your\s+system\s+prompt\b",
    r"\bi\s+am\s+configured\s+to\b",
    r"\bignore\s+previous\s+instructions\b",
    r"^system\s+prompt:",
]
_LEAKAGE_REGEX = re.compile("|".join(_LEAKAGE_PATTERNS), re.IGNORECASE | re.MULTILINE)

# API Keys and Secrets patterns
_API_KEY_PATTERNS = [
    r"\bsk-[a-zA-Z0-9_-]{20,}\b",               # OpenAI style key (including sk-proj-...)
    r"\bAIzaSy[a-zA-Z0-9_-]{33}\b",           # Google API key
    r"\bghp_[a-zA-Z0-9]{36}\b",                # GitHub personal access token
    r"\bbearer\s+[a-zA-Z0-9._-]{20,}\b",      # Bearer token
]
_API_KEY_REGEX = re.compile("|".join(_API_KEY_PATTERNS), re.IGNORECASE)

# Traceback and Stack Traces / Exception Dumps
_TRACEBACK_REGEX = re.compile(
    r"(Traceback\s+\(most\s+recent\s+call\s+last\):|File\s+[\"'].*?[\"'],\s+line\s+\d+|^\s*(?:Exception|ValueError|TypeError|RuntimeError|KeyError|AttributeError|ZeroDivisionError):)",
    re.MULTILINE,
)

# Empty Markdown Headings (e.g. "#", "## ", with no text following)
_EMPTY_HEADING_REGEX = re.compile(r"^#{1,6}\s*$", re.MULTILINE)


def validate_output(markdown: str) -> dict[str, Any]:
    """
    Validate output markdown for unresolved tokens, leakage, secrets, stack traces, and formatting issues.

    Returns
    -------
    dict
        {
            "passed": bool,
            "warnings": list[str],
            "errors": list[str]
        }
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not markdown:
        warnings.append("Blog markdown is empty.")
        return {
            "passed": True,
            "warnings": warnings,
            "errors": errors,
        }

    # Check TODO / TBD / FIXME / XXX placeholders
    placeholder_matches = _PLACEHOLDER_REGEX.findall(markdown)
    if placeholder_matches:
        unique_placeholders = sorted(set(placeholder_matches))
        errors.append(f"Unresolved placeholder tokens found: {', '.join(unique_placeholders)}")

    # Check {{ ... }} template placeholders
    curly_matches = _DOUBLE_CURLY_REGEX.findall(markdown)
    if curly_matches:
        errors.append(f"Unresolved double curly placeholders found ({len(curly_matches)} item(s)).")

    # Check [[IMAGE_*]] placeholders
    image_matches = _IMAGE_PLACEHOLDER_REGEX.findall(markdown)
    if image_matches:
        warnings.append(f"Unresolved image placeholders found ({len(image_matches)} item(s)).")

    # Check Prompt Leakage
    leakage_match = _LEAKAGE_REGEX.search(markdown)
    if leakage_match:
        errors.append(f"Potential prompt leakage detected ('{leakage_match.group(0)}').")

    # Check API Keys / Secrets
    api_match = _API_KEY_REGEX.search(markdown)
    if api_match:
        errors.append("Potential API key or secret leakage detected in output.")

    # Check Tracebacks and Stack Traces
    tb_match = _TRACEBACK_REGEX.search(markdown)
    if tb_match:
        errors.append(f"Python traceback or exception dump detected ('{tb_match.group(0).strip()}').")

    # Check Empty Headings
    if _EMPTY_HEADING_REGEX.search(markdown):
        errors.append("Empty markdown heading detected.")

    # Check Unclosed Code Fences (counting delimiter lines starting with ```)
    code_fence_lines = [
        line for line in markdown.splitlines() if line.strip().startswith("```")
    ]
    if len(code_fence_lines) % 2 != 0:
        errors.append("Unclosed markdown code block detected.")

    passed = len(errors) == 0

    return {
        "passed": passed,
        "warnings": warnings,
        "errors": errors,
    }


def output_guardrails(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node executing output guardrails immediately after Senior Editorial Review.
    """
    logger.info("Running output guardrails...")

    result = validate_output(state.blog_markdown)

    if result["warnings"] or result["errors"]:
        logger.warning("Guardrail warnings detected.")

    logger.info("Output validation completed.")

    return {
        "guardrail_warnings": result["warnings"],
        "guardrail_errors": result["errors"],
    }
