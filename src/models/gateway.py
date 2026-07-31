"""
LLM & Image Gateway for InkFlow-AI.

Responsibilities
----------------
- Provide a single provider-agnostic interface for workflow nodes.
- Resolve models automatically based on NodeType.
- Build ChatLiteLLM instances with automatic fallback chains.
- Execute image generation using Google GenAI SDK (Vertex AI Enterprise) and image fallbacks.

This is the ONLY place in the application that instantiates LLM and Image model clients.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from src.models.registry import get_node_config
from src.models.types import ModelProfile, NodeType

os.environ["VERTEXAI_LOCATION"] = "global"

logger = logging.getLogger(__name__)


class LLMGateway:
    """
    Production-ready Node-Aware LLM & Image Gateway.

    Workflow nodes request models using NodeType (e.g. gateway.chat(NodeType.ROUTER)).
    The Gateway resolves provider, model name, temperature, and fallback chains automatically.
    """

    def __init__(self) -> None:
        self._cache: dict[NodeType, BaseChatModel] = {}

    @staticmethod
    def _create_chat_model(profile: ModelProfile) -> ChatLiteLLM:
        """
        Create a ChatLiteLLM instance from a ModelProfile.
        Omit temperature for Gemini 3+ models to avoid warning noise.
        """
        kwargs: dict[str, Any] = {}

        if profile.temperature is not None and "gemini-3" not in profile.model:
            kwargs["temperature"] = profile.temperature

        return ChatLiteLLM(model=profile.model, **kwargs)

    def chat(self, node_type: NodeType = NodeType.WRITER) -> BaseChatModel:
        """
        Return the compiled Chat model fallback chain for the specified NodeType.

        Parameters
        ----------
        node_type:
            The workflow node requesting an LLM client.

        Returns
        -------
        BaseChatModel
            LangChain BaseChatModel instance with fallbacks attached.
        """
        if node_type in self._cache:
            return self._cache[node_type]

        config = get_node_config(node_type)
        primary_model = self._create_chat_model(config.primary)

        if not config.fallbacks:
            self._cache[node_type] = primary_model
            return primary_model

        fallback_models = [
            self._create_chat_model(fb_profile)
            for fb_profile in config.fallbacks
        ]

        model_chain = primary_model.with_fallbacks(fallback_models)
        self._cache[node_type] = model_chain
        return model_chain

    def image(
        self,
        node_type: NodeType = NodeType.IMAGE_GENERATOR,
        prompt: str = "",
        size: str = "1024x1024",
        quality: str = "medium",
    ) -> bytes:
        """
        Generate an image for the specified NodeType trying primary model first, then fallbacks.

        Parameters
        ----------
        node_type:
            The image workflow node requesting generation.
        prompt:
            Image description prompt.
        size:
            Image resolution spec.
        quality:
            Quality mode.

        Returns
        -------
        bytes
            Generated PNG image bytes.
        """
        config = get_node_config(node_type)
        chain = [config.primary] + config.fallbacks

        for profile in chain:
            try:
                logger.info("Attempting image generation with model: %s", profile.model)
                image_bytes = self._call_genai_sdk(profile.model, prompt=prompt)
                if image_bytes:
                    logger.info("Successfully generated image with model: %s", profile.model)
                    return image_bytes
            except Exception as e:
                logger.warning("Image model %s failed: %s", profile.model, e)
                continue

        logger.warning(
            "All Vertex AI image models failed or were restricted. Generating local technical illustration visual..."
        )
        return self._create_placeholder_image(prompt, size)

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "medium",
    ) -> bytes:
        """Backward compatibility alias for image generation."""
        return self.image(node_type=NodeType.IMAGE_GENERATOR, prompt=prompt, size=size, quality=quality)

    def _call_genai_sdk(
        self,
        model_name: str,
        prompt: str,
    ) -> bytes:
        """
        Generate image using Google GenAI SDK (Vertex AI Enterprise Client).
        """
        from google import genai
        from google.genai.types import GenerateContentConfig, Modality

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or (os.getenv("PROJECT_ID") or "gen-lang-client-0579266941").strip('"' "' ")
        location = os.getenv("REGION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
        api_key = os.getenv("GOOGLE_API_KEY")
        os.environ["VERTEXAI_LOCATION"] = "global"

        # Initialize Enterprise GenAI Client (Vertex AI)
        try:
            client = genai.Client(enterprise=True, project=project_id, location=location)
        except Exception:
            client = genai.Client(api_key=api_key)

        full_prompt = (
            f"Generate a high-resolution, modern, professional technical infographic diagram/illustration: {prompt}. "
            "Style: sleek dark slate background (#0f172a), crisp vector graphic lines, glowing indigo and cyan accents (#6366f1, #38bdf8), ultra detailed 4K technical design."
        )

        # 1. Primary method: generate_content with response_modalities=[TEXT, IMAGE]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=GenerateContentConfig(
                    response_modalities=[Modality.TEXT, Modality.IMAGE],
                ),
            )
            if response and response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        return part.inline_data.data
        except Exception as err:
            logger.debug("generate_content failed for %s: %s", model_name, err)

        # 2. Fallback method: generate_images
        try:
            res = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=dict(number_of_images=1),
            )
            if res and hasattr(res, "generated_images") and res.generated_images:
                return res.generated_images[0].image.image_bytes
        except Exception as err:
            logger.debug("generate_images failed for %s: %s", model_name, err)

        raise ValueError(f"Could not generate image using model '{model_name}'.")

    def _create_placeholder_image(
        self,
        prompt: str,
        size: str,
    ) -> bytes:
        """
        Generate a clean technical illustration diagram PNG when cloud APIs are unavailable.
        """
        try:
            from PIL import Image, ImageDraw

            width, height = 1024, 576
            if "x" in size:
                try:
                    parts = size.split("x")
                    width, height = int(parts[0]), int(parts[1])
                except Exception:
                    pass

            image = Image.new("RGB", (width, height), color="#0f172a")
            draw = ImageDraw.Draw(image)

            grid_color = "#1e293b"
            for x in range(0, width, 40):
                draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
            for y in range(0, height, 40):
                draw.line([(0, y), (width, y)], fill=grid_color, width=1)

            card_box = [40, 40, width - 40, height - 40]
            draw.rectangle(card_box, fill="#1e1e2e", outline="#6366f1", width=3)

            draw.rectangle([40, 40, width - 40, 110], fill="#312e81")

            text_title = "InkFlow-AI Technical Visual"
            text_prompt = prompt[:100] + ("..." if len(prompt) > 100 else "")

            draw.text((60, 60), text_title, fill="#ffffff")
            draw.text((60, 140), "Illustration Prompt:", fill="#818cf8")
            draw.text((60, 170), text_prompt, fill="#e2e8f0")

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

        except Exception:
            return base64.b64decode(
                "iVBORw0KGgoAAAANSU0EUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )


gateway = LLMGateway()
image_gateway = gateway