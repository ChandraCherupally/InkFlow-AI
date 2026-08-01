"""
Node-Aware Model Registry for InkFlow-AI.

Responsibilities:
- Map every NodeType to primary ModelProfile, fallback profiles, and capability rules.
- Perform fast capability validation for node requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.models.providers import ModelProfiles
from src.models.types import ModelProfile, NodeType


@dataclass(frozen=True, slots=True)
class NodeModelConfig:
    """Model configuration for a single workflow node type."""

    node_type: NodeType
    primary: ModelProfile
    fallbacks: list[ModelProfile] = field(default_factory=list)
    required_capabilities: dict[str, bool] = field(default_factory=dict)


def validate_model_capabilities(
    node_type: NodeType,
    profile: ModelProfile,
    required_capabilities: dict[str, bool],
) -> None:
    """
    Validate that a ModelProfile satisfies the required capabilities of a NodeType.

    Raises
    ------
    ValueError
        If the profile fails capability validation.
    """
    for cap, required in required_capabilities.items():
        if required:
            val = getattr(profile, cap, False)
            if not val:
                raise ValueError(
                    f"Model '{profile.model}' (provider={profile.provider}) configured for NodeType.{node_type.name} "
                    f"does not satisfy required capability '{cap}=True'."
                )


# ==========================================================
# Node-Aware Model Registry
# ==========================================================

NODE_MODEL_REGISTRY: dict[NodeType, NodeModelConfig] = {
    # Router Node: Requires structured output
    NodeType.ROUTER: NodeModelConfig(
        node_type=NodeType.ROUTER,
        primary=ModelProfiles.GEMINI_3_1_PRO_PREVIEW,
        fallbacks=[ModelProfiles.GEMINI_2_5_FLASH_LITE, ModelProfiles.GPT_5_MINI],
        required_capabilities={"supports_structured_output": True},
    ),
    # Research Node: Structured output + search support
    NodeType.RESEARCH: NodeModelConfig(
        node_type=NodeType.RESEARCH,
        primary=ModelProfiles.GEMINI_2_5_FLASH,
        fallbacks=[ModelProfiles.GPT_5_MINI],
        required_capabilities={"supports_structured_output": True},
    ),
    # Planner Node: Requires structured output + reasoning
    NodeType.PLANNER: NodeModelConfig(
        node_type=NodeType.PLANNER,
        primary=ModelProfiles.GEMINI_3_5_FLASH,
        fallbacks=[ModelProfiles.GEMINI_2_5_PRO, ModelProfiles.GPT_5],
        required_capabilities={
            "supports_structured_output": True,
            "supports_reasoning": True,
        },
    ),
    # Writer Node: Requires reasoning
    NodeType.WRITER: NodeModelConfig(
        node_type=NodeType.WRITER,
        primary=ModelProfiles.GEMINI_3_5_FLASH,
        fallbacks=[ModelProfiles.GEMINI_2_5_PRO, ModelProfiles.GPT_5],
        required_capabilities={"supports_reasoning": True},
    ),
    # Editor Node: Requires reasoning
    NodeType.EDITOR: NodeModelConfig(
        node_type=NodeType.EDITOR,
        primary=ModelProfiles.GEMINI_2_5_PRO,
        fallbacks=[ModelProfiles.GPT_5],
        required_capabilities={"supports_reasoning": True},
    ),
    # Markdown Formatter Node
    NodeType.MARKDOWN: NodeModelConfig(
        node_type=NodeType.MARKDOWN,
        primary=ModelProfiles.GEMINI_2_5_FLASH,
        fallbacks=[ModelProfiles.GPT_5_MINI],
        required_capabilities={},
    ),
    # Image Planner Node: Requires structured output
    NodeType.IMAGE_PLANNER: NodeModelConfig(
        node_type=NodeType.IMAGE_PLANNER,
        primary=ModelProfiles.GEMINI_3_5_FLASH,
        fallbacks=[ModelProfiles.GEMINI_2_5_PRO, ModelProfiles.GPT_5],
        required_capabilities={"supports_structured_output": True},
    ),
    # Image Generator Node: Requires image generation capability
    NodeType.IMAGE_GENERATOR: NodeModelConfig(
        node_type=NodeType.IMAGE_GENERATOR,
        primary=ModelProfiles.GEMINI_FLASH_IMAGE,
        fallbacks=[ModelProfiles.IMAGEN_3, ModelProfiles.GPT_IMAGE_1],
        required_capabilities={"supports_images": True},
    ),
}

def get_node_config(node_type: NodeType) -> NodeModelConfig:
    """
    Retrieve and validate the model configuration for a given NodeType.
    """
    config = NODE_MODEL_REGISTRY.get(node_type)
    if not config:
        raise KeyError(f"No model configuration registered for NodeType: {node_type}")

    # Validate primary profile
    validate_model_capabilities(node_type, config.primary, config.required_capabilities)

    # Validate fallback profiles
    for fb in config.fallbacks:
        validate_model_capabilities(node_type, fb, config.required_capabilities)

    return config
