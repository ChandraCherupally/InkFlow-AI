"""
Markdown builder for InkFlow-AI.

Responsibilities
----------------
- Assemble final Medium-grade Markdown document.
- Insert technical images inline directly at their respective sections.
- Format headings, code blocks, and captions.

Contains NO LLM logic.
"""

from __future__ import annotations

import re
from src.schemas.models import GeneratedImage, Plan


class MarkdownBuilder:
    """Build the final blog markdown with inline section images."""

    def build(
        self,
        plan: Plan,
        sections: list[str],
        images: list[GeneratedImage],
        markdown_with_placeholders: str = "",
    ) -> str:
        """
        Build the final markdown document with inline section images.

        Parameters
        ----------
        plan
            Article Plan outline containing blog_title.
        sections
            Generated markdown sections.
        images
            Generated images metadata.
        markdown_with_placeholders
            Full markdown containing [[IMAGE_1]], [[IMAGE_2]] placeholders.

        Returns
        -------
        str
        """
        # Map generated images by index (1-based) and filename
        image_map: dict[str, GeneratedImage] = {}
        for idx, img in enumerate(images, 1):
            image_map[f"[[IMAGE_{idx}]]"] = img
            image_map[f"[IMAGE_{idx}]"] = img
            if hasattr(img, "filename") and img.filename:
                image_map[img.filename] = img

        base_text = markdown_with_placeholders if markdown_with_placeholders else ""

        if not base_text and sections:
            base_text = f"# {plan.blog_title}\n\n" + "\n\n".join(sections)
        elif base_text and not base_text.startswith("#"):
            base_text = f"# {plan.blog_title}\n\n" + base_text

        # --------------------------------------------------
        # Method 1: Replace [[IMAGE_X]] placeholders inline
        # --------------------------------------------------
        has_placeholders = bool(re.search(r"\[\[?IMAGE_\d+\]?\]", base_text, re.IGNORECASE))

        if has_placeholders:
            final_md = base_text
            for idx, img in enumerate(images, 1):
                placeholder_patterns = [f"[[IMAGE_{idx}]]", f"[IMAGE_{idx}]"]
                image_url = f"/images/{img.filename}"
                inline_md = f"\n\n![{img.alt}]({image_url})\n*{img.caption}*\n\n"

                replaced = False
                for pat in placeholder_patterns:
                    if pat in final_md:
                        final_md = final_md.replace(pat, inline_md, 1)
                        replaced = True
                        break

            # Strip any remaining unreplaced image placeholders cleanly
            final_md = re.sub(r"\n*\[\[?IMAGE_\d+\]?\]\n*", "\n\n", final_md)
            return final_md.strip()

        # --------------------------------------------------
        # Method 2: Fallback inline placement across sections
        # --------------------------------------------------
        if not images:
            return base_text.strip()

        lines = base_text.split("\n")
        output_lines: list[str] = []
        img_idx = 0

        # Insert Image 1 (Hero Visual) after title or first paragraph
        hero_inserted = False

        for line in lines:
            output_lines.append(line)

            # Insert Hero image right after the first level 2 heading or intro paragraph
            if not hero_inserted and img_idx < len(images):
                if line.startswith("## ") or (len(output_lines) > 5 and line == ""):
                    img = images[img_idx]
                    image_url = f"/images/{img.filename}"
                    output_lines.append("")
                    output_lines.append(f"![{img.alt}]({image_url})")
                    output_lines.append(f"*{img.caption}*")
                    output_lines.append("")
                    img_idx += 1
                    hero_inserted = True
                    continue

            # Insert subsequent images beneath next ## H2 headings
            if hero_inserted and img_idx < len(images) and line.startswith("## "):
                img = images[img_idx]
                image_url = f"/images/{img.filename}"
                output_lines.append("")
                output_lines.append(f"![{img.alt}]({image_url})")
                output_lines.append(f"*{img.caption}*")
                output_lines.append("")
                img_idx += 1

        # Append remaining unused images inline cleanly
        while img_idx < len(images):
            img = images[img_idx]
            image_url = f"/images/{img.filename}"
            output_lines.append("")
            output_lines.append(f"![{img.alt}]({image_url})")
            output_lines.append(f"*{img.caption}*")
            output_lines.append("")
            img_idx += 1

        return "\n".join(output_lines).strip()


markdown_builder = MarkdownBuilder()
