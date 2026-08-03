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

    def image_with_details(
        self,
        node_type: NodeType = NodeType.IMAGE_GENERATOR,
        prompt: str = "",
        size: str = "2560x1440",
        quality: str = "medium",
    ) -> tuple[bytes, str, str, bool]:
        """
        Generate an image trying primary model first, then fallbacks.

        Returns
        -------
        tuple[bytes, str, str, bool]
            (image_bytes, used_model, used_provider, is_fallback)
        """
        config = get_node_config(node_type)
        chain = [config.primary] + config.fallbacks

        for idx, profile in enumerate(chain):
            try:
                logger.info("Attempting image generation with model: %s (provider=%s)", profile.model, profile.provider)
                if (
                    profile.provider == "openai"
                    or "dall-e" in profile.model.lower()
                    or profile.model.startswith("openai/")
                ):
                    image_bytes = self._call_openai_image(profile.model, prompt=prompt, size=size, quality=quality)
                else:
                    image_bytes = self._call_genai_sdk(profile.model, prompt=prompt)

                if image_bytes:
                    logger.info("Successfully generated image with model: %s", profile.model)
                    is_fallback = (idx > 0) or (profile.model != config.primary.model)
                    return image_bytes, profile.model, profile.provider, is_fallback
            except Exception as e:
                logger.warning("Image model %s failed: %s", profile.model, e)
                continue

        logger.warning(
            "All image models failed or were restricted. Generating local technical illustration visual..."
        )
        img_bytes = self._create_placeholder_image(prompt, size)
        return img_bytes, "local/placeholder-illustrator", "local", True

    def image(
        self,
        node_type: NodeType = NodeType.IMAGE_GENERATOR,
        prompt: str = "",
        size: str = "2560x1440",
        quality: str = "medium",
    ) -> bytes:
        """
        Generate an image for the specified NodeType trying primary model first, then fallbacks.
        Returns generated PNG image bytes.
        """
        image_bytes, _, _, _ = self.image_with_details(
            node_type=node_type,
            prompt=prompt,
            size=size,
            quality=quality,
        )
        return image_bytes

    def generate(
        self,
        prompt: str,
        size: str = "2560x1440",
        quality: str = "medium",
    ) -> bytes:
        """Backward compatibility alias for image generation."""
        return self.image(node_type=NodeType.IMAGE_GENERATOR, prompt=prompt, size=size, quality=quality)

    def _call_openai_image(
        self,
        model_name: str,
        prompt: str,
        size: str = "2560x1440",
        quality: str = "medium",
    ) -> bytes:
        """
        Generate image using OpenAI image models via LiteLLM or direct OpenAI API client.
        Automatically tries alternative model aliases (e.g. gpt-image-1) if dall-e-3 returns 400 invalid_value.
        """
        import base64
        import requests
        from litellm import image_generation

        clean_model = model_name.replace("openai/", "")
        valid_sizes = {"2560x1440", "1792x1024", "1024x1024", "1024x1792", "1792x1024"}
        img_size = size if size in valid_sizes else "2560x1440"

        candidate_models = [clean_model]
        if clean_model in ("dall-e-3", "dall-e-2"):
            candidate_models.extend(["gpt-image-1", "gpt-image-1-mini"])

        api_key = os.getenv("OPENAI_API_KEY")

        for m_name in candidate_models:
            # 1. Primary method: litellm.image_generation
            try:
                res = image_generation(
                    model=m_name,
                    prompt=prompt,
                    size=img_size,
                    n=1,
                    api_key=api_key,
                )
                if hasattr(res, "data") and res.data:
                    item = res.data[0]
                    if hasattr(item, "b64_json") and item.b64_json:
                        return base64.b64decode(item.b64_json)
                    elif hasattr(item, "url") and item.url:
                        return requests.get(item.url, timeout=30).content
            except Exception as err:
                logger.debug("litellm image_generation failed for %s: %s", m_name, err)

            # 2. Fallback method: direct OpenAI client
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                res = client.images.generate(
                    model=m_name,
                    prompt=prompt,
                    size=img_size,
                    n=1,
                )
                if res.data and res.data[0].url:
                    return requests.get(res.data[0].url, timeout=30).content
            except Exception as err:
                logger.debug("openai client images.generate failed for %s: %s", m_name, err)

        raise ValueError(f"Could not generate image using OpenAI model '{model_name}'.")

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
            "Style: Claude Design studio visual aesthetic, modern 3D translucent glassmorphism diagram on pristine soft off-white background (#FAFAFC) with subtle background infographics (faint dot grid, soft metric curves, delicate architectural flowlines), featuring light sky blue (#38BDF8, #60A5FA), soft lavender purple (#8B5CF6, #A78BFA), pastel mint green (#10B981, #A7F3D0), and warm butter yellow accents (#FBBF24), layered floating frosted glass cards, smooth 3D bezier curves, publication-ready."
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

            image = Image.new("RGB", (width, height), color="#FAFAFC")
            draw = ImageDraw.Draw(image)

            grid_color = "#F1F5F9"
            for x in range(0, width, 40):
                draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
            for y in range(0, height, 40):
                draw.line([(0, y), (width, y)], fill=grid_color, width=1)

            card_box = [40, 40, width - 40, height - 40]
            draw.rectangle(card_box, fill="#FFFFFF", outline="#8B5CF6", width=3)

            draw.rectangle([40, 40, width - 40, 110], fill="#F3E8FF")

            text_title = "InkFlow-AI Technical Visual (Claude Design Style)"
            text_prompt = prompt[:100] + ("..." if len(prompt) > 100 else "")

            draw.text((60, 60), text_title, fill="#5B21B6")
            draw.text((60, 140), "Illustration Prompt:", fill="#10B981")
            draw.text((60, 170), text_prompt, fill="#334155")

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

        except Exception:
            return base64.b64decode(
                "iVBORw0KGgoAAAANSU0EUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )

gateway = LLMGateway()