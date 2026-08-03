"""
Application settings and configuration for InkFlow-AI.

Responsibilities:
- Load environment variables.
- Define project directories.
- Expose application constants.
- Suppress verbose logging and deprecation warnings.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Warning & Logger Noise Suppression
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*Setting temperature.*")
warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*planned for removal.*")
warnings.filterwarnings("ignore", message=".*sampling guidance.*")


class SuppressLiteLLMDeprecationFilter(logging.Filter):
    """Filter out LiteLLM deprecation warnings and sampling messages for Gemini 3."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "planned for removal" in message or "sampling guidance" in message:
            return False
        if "Setting temperature" in message or "continue to function for Gemini 3" in message:
            return False
        return True


deprecation_filter = SuppressLiteLLMDeprecationFilter()
logging.getLogger().addFilter(deprecation_filter)

# Mute noisy third-party loggers and apply deprecation filter
for logger_name in [
    "LiteLLM",
    "LiteLLM Router",
    "LiteLLM Proxy",
    "litellm",
    "google",
    "google.genai",
    "google.auth",
    "google.cloud",
    "httpx",
    "urllib3",
]:
    log_obj = logging.getLogger(logger_name)
    log_obj.setLevel(logging.WARNING)
    log_obj.addFilter(deprecation_filter)

try:
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except Exception:
    pass


# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for directory in [DATA_DIR, IMAGES_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------
# Default Models
# ---------------------------------------------------------------------

DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemini-3.1-flash")
DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "gemini-3.1-flash-image")

# ---------------------------------------------------------------------
# Graph Configuration
# ---------------------------------------------------------------------

DEFAULT_THREAD_ID = "default"
MAX_RESEARCH_RESULTS = 6
MAX_IMAGE_COUNT = 3

# ---------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 8192
