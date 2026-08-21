"""
Word count and word budget utilities for InkFlow-AI.

Responsibilities:
- Provide robust, deterministic word counting for markdown sections and complete articles.
- Filter out code fences, image syntax, HTML tags, and markdown markup to measure readable prose accurately.
- Resolve user-requested target word counts into deterministic target, min, and max boundaries.
"""

from __future__ import annotations

import re


def count_words(text: str) -> int:
    """
    Count the number of readable words in a markdown string, excluding code blocks,
    image references, and markdown syntax noise.
    """
    if not text:
        return 0

    # Remove fenced code blocks
    clean = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code
    clean = re.sub(r"`[^`]*`", " ", clean)
    # Remove markdown image syntax ![alt](url)
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", clean)
    # Remove image placeholders [[IMAGE_X]]
    clean = re.sub(r"\[\[?IMAGE_\d+\]?\]", " ", clean)
    # Convert markdown links [text](url) to just the text
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r" \1 ", clean)
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    # Remove table formatting pipes and separator rows
    clean = re.sub(r"\|", " ", clean)
    clean = re.sub(r"[-:]{3,}", " ", clean)
    # Remove leading markdown header marks and blockquotes
    clean = re.sub(r"^[#>\s*+-]+", " ", clean, flags=re.MULTILINE)
    # Remove bold / italic symbols
    clean = re.sub(r"[*_~]{1,3}", " ", clean)

    # Split on whitespace and count non-empty words containing alphanumeric characters
    words = [w for w in re.split(r"\s+", clean) if re.search(r"\w", w)]
    return len(words)


def resolve_word_budget(target_word_count: int | None = None) -> dict[str, int]:
    """
    Resolve requested target word count into target, min, and max bounds deterministically.

    Standard guidelines:
    - short: 1500-2500 words
    - standard: 2500-5000 words (default target: 3500)
    - long-form: 5000-8000 words (target: 6500)
    - flagship: 12000-15000 words (target: 13500)
    """
    if target_word_count is None or target_word_count <= 0:
        target = 3500
        min_words = 2500
        max_words = 5000
    elif 12000 <= target_word_count <= 15000:
        target = target_word_count
        min_words = 12000
        max_words = 15000
    elif target_word_count > 15000:
        target = target_word_count
        min_words = int(target * 0.85)
        max_words = int(target * 1.15)
    elif 5000 <= target_word_count < 12000:
        target = target_word_count
        min_words = max(4500, int(target * 0.85))
        max_words = int(target * 1.15)
    elif 1500 <= target_word_count < 5000:
        target = target_word_count
        min_words = max(1200, int(target * 0.85))
        max_words = int(target * 1.15)
    else:
        target = target_word_count
        min_words = max(500, int(target * 0.85))
        max_words = int(target * 1.15)

    return {
        "target_word_count": target,
        "min_word_count": min_words,
        "max_word_count": max_words,
    }
