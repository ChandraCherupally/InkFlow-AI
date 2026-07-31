"""
Image generation tool for InkFlow-AI.

Responsibilities
----------------
- Save generated images.
- Delegate image generation to gateway using NodeType.IMAGE_GENERATOR.
- Return GeneratedImage metadata.

Contains NO graph logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import IMAGES_DIR
from src.models.gateway import gateway
from src.models.types import NodeType
from src.schemas.models import (
    GeneratedImage,
    ImageSpec,
)

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate blog images using Node-Aware Gateway with automatic provider fallbacks."""

    def __init__(self) -> None:
        self._output_dir = Path(IMAGES_DIR)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image: ImageSpec) -> GeneratedImage:
        """
        Generate a single image using the Gateway for NodeType.IMAGE_GENERATOR.

        Parameters
        ----------
        image:
            Image generation request spec.

        Returns
        -------
        GeneratedImage
        """
        logger.info("Generating image for placeholder: %s", image.placeholder)

        image_bytes = gateway.image(
            node_type=NodeType.IMAGE_GENERATOR,
            prompt=image.prompt,
            size=image.size,
            quality=image.quality,
        )

        filename = image.filename if image.filename.endswith(".png") else f"{image.filename}.png"
        filepath = self._output_dir / filename
        filepath.write_bytes(image_bytes)

        logger.info("Saved image: %s", filepath)

        return GeneratedImage(
            filename=filename,
            path=str(filepath),
            alt=image.alt,
            caption=image.caption,
        )


image_generator = ImageGenerator()
