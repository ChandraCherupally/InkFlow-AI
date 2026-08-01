"""
Provider definitions and ModelProfile registry instances for InkFlow-AI.

Contains configuration ONLY. No business logic or graph code.
"""

from __future__ import annotations

from src.models.types import ModelProfile


class Providers:
    """Supported LLM & Image Provider identifiers."""

    VERTEX_AI = "vertex_ai"
    GEMINI = "gemini"
    OPENAI = "openai"


class ModelProfiles:
    """
    Central repository of strongly-typed ModelProfile definitions.
    """

    # ==========================================================
    # Text Models
    # ==========================================================

    # Gemini 2.5 Flash Lite (Router / Fast tasks via Vertex AI)
    GEMINI_2_5_FLASH_LITE = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="vertex_ai/gemini-2.5-flash-lite",
        supports_structured_output=True,
        supports_reasoning=False,
        supports_streaming=True,
        max_context_tokens=1048576,
        temperature=1.0,
    )

    # Gemini 2.5 Flash Lite (Router / Fast tasks via Vertex AI)
    GEMINI_3_1_PRO_PREVIEW = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="vertex_ai/gemini-3.1-pro-preview",
        supports_structured_output=True,
        supports_reasoning=False,
        supports_streaming=True,
        max_context_tokens=1048576,
        temperature=1.0,
    )

    # Gemini 2.5 Flash (Research / Markdown tasks via Vertex AI)
    GEMINI_2_5_FLASH = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="vertex_ai/gemini-2.5-flash",
        supports_structured_output=True,
        supports_reasoning=True,
        supports_streaming=True,
        max_context_tokens=1048576,
        temperature=1.0,
    )

    # Gemini 3.5 Flash / Vertex AI (Primary Chat & Writer)
    GEMINI_3_5_FLASH = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="vertex_ai/gemini-3.5-flash",
        supports_structured_output=True,
        supports_reasoning=True,
        supports_streaming=True,
        max_context_tokens=1048576,
        temperature=1.0,
    )

    # Gemini 2.5 Pro (Planner / Writer / Image Planner)
    GEMINI_2_5_PRO = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="vertex_ai/gemini-2.5-pro",
        supports_structured_output=True,
        supports_reasoning=True,
        supports_streaming=True,
        max_context_tokens=2097152,
        temperature=1.0,
    )

    # GPT-5 Mini / GPT-4o Mini (Fallback for lightweight tasks)
    GPT_5_MINI = ModelProfile(
        provider=Providers.OPENAI,
        model="openai/gpt-4o-mini",
        supports_structured_output=True,
        supports_reasoning=False,
        supports_streaming=True,
        max_context_tokens=128000,
        temperature=1.0,
    )

    # GPT-5 / GPT-4o (Fallback for complex reasoning/writing)
    GPT_5 = ModelProfile(
        provider=Providers.OPENAI,
        model="openai/gpt-4o",
        supports_structured_output=True,
        supports_reasoning=True,
        supports_streaming=True,
        max_context_tokens=128000,
        temperature=1.0,
    )

    # ==========================================================
    # Image Generation Models
    # ==========================================================

    # Primary: Gemini Flash Image (Google GenAI Enterprise SDK)imagen-3.0-generate-002"
    GEMINI_FLASH_IMAGE = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="gemini-3-pro-image",
        supports_structured_output=False,
        supports_reasoning=False,
        supports_streaming=False,
        supports_images=True,
    )

    # Fallback 1: Imagen 3
    IMAGEN_3 = ModelProfile(
        provider=Providers.VERTEX_AI,
        model="gemini-3.1-flash-image",
        supports_structured_output=False,
        supports_reasoning=False,
        supports_streaming=False,
        supports_images=True,
    )

    # Fallback 2: OpenAI DALL-E 3
    GPT_IMAGE_1 = ModelProfile(
        provider=Providers.OPENAI,
        model="openai/dall-e-3",
        supports_structured_output=False,
        supports_reasoning=False,
        supports_streaming=False,
        supports_images=True,
    )