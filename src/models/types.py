"""
Core model infrastructure types for InkFlow-AI.

Responsibilities:
- Define NodeType Enum for workflow nodes.
- Define strongly-typed ModelProfile dataclass representing model capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    """Workflow node identifiers for node-aware model resolution."""

    ROUTER = "router"
    RESEARCH = "research"
    PLANNER = "planner"
    WRITER = "writer"
    EDITOR = "editor"
    MARKDOWN = "markdown"
    IMAGE_PLANNER = "image_planner"
    IMAGE_GENERATOR = "image_generator"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """
    Canonical, immutable specification of an LLM or Image model.
    """

    provider: str
    model: str
    supports_structured_output: bool = True
    supports_reasoning: bool = True
    supports_streaming: bool = True
    supports_images: bool = False
    max_context_tokens: int = 128000
    temperature: float = 1.0
    max_output_tokens: int | None = 8192
    reasoning_effort: str | None = None
    timeout_seconds: int = 60
    retryable: bool = True
    enabled: bool = True
