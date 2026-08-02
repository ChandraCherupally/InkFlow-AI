"""
Input guardrails implementation for InkFlow-AI.

Responsibilities:
- Validate user input before passing it to the Routing Graph.
- Detect prompt injection, jailbreak attempts, empty prompts, and malformed inputs.
- Pure Python deterministic validation with standard library regex and string operations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.schemas.state import BlogState

logger = logging.getLogger(__name__)

# Default maximum prompt character length
MAX_PROMPT_LENGTH = 5000

# Common prompt injection and system prompt leak detection patterns
INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|above\s+|previous\s+)?instructions\b",
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\breveal\s+(?:your\s+)?instructions\b",
    r"\bjailbreak\b",
    r"\bpretend\s+you\s+are\b",
    r"\bsimulate\s+chatgpt\b",
    r"\bdisregard\s+previous\s+instructions\b",
    r"\boverride\s+instructions\b",
    r"\bbypass\s+safety\b",
]

# Compiled regex for injection patterns
_INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Suspicious control characters (excluding standard whitespace \n, \r, \t)
_CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

# Excessive repeated non-alphanumeric symbols (10 or more consecutive identical symbols)
_REPEATED_SYMBOLS_REGEX = re.compile(r"([^\w\s])\1{9,}")


def validate_input(prompt: str) -> dict[str, Any]:
    """
    Validate input prompt against security, length, and formatting rules.

    Returns
    -------
    dict
        {
            "is_valid": bool,
            "warnings": list[str],
            "error_message": str | None
        }
    """
    warnings: list[str] = []

    # Check empty or whitespace-only input
    if not prompt or not prompt.strip():
        return {
            "is_valid": False,
            "warnings": warnings,
            "error_message": "Input prompt cannot be empty or whitespace-only.",
        }

    # Check prompt length limit
    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "is_valid": False,
            "warnings": warnings,
            "error_message": f"Input prompt length ({len(prompt)}) exceeds maximum limit of {MAX_PROMPT_LENGTH} characters.",
        }

    # Check suspicious control characters
    if _CONTROL_CHAR_REGEX.search(prompt):
        return {
            "is_valid": False,
            "warnings": warnings,
            "error_message": "Input prompt contains suspicious control characters.",
        }

    # Check prompt injection patterns
    match = _INJECTION_REGEX.search(prompt)
    if match:
        return {
            "is_valid": False,
            "warnings": warnings,
            "error_message": f"Input prompt failed security check: potential prompt injection detected ('{match.group(0)}').",
        }

    # Check excessive repeated symbols
    if _REPEATED_SYMBOLS_REGEX.search(prompt):
        warnings.append("Excessive repeated symbols detected in input prompt.")

    return {
        "is_valid": True,
        "warnings": warnings,
        "error_message": None,
    }


def input_guardrails(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node executing input guardrails before the Router.
    """
    logger.info("Running input guardrails...")

    result = validate_input(state.topic)

    if not result["is_valid"]:
        logger.warning("Input validation failed.")
        error_msg = result["error_message"] or "Input validation failed."
        return {
            "error": error_msg,
            "guardrail_errors": [error_msg],
            "guardrail_warnings": result["warnings"],
        }

    logger.info("Input validation passed.")
    return {
        "guardrail_warnings": result["warnings"],
    }
