"""
Base Prompt Factory.

Responsibilities
----------------
- Create reusable LangChain prompt templates.
- Keep all prompts consistent across the project.

This module contains NO business logic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


class PromptFactory:
    """Factory for creating chat prompts."""

    @staticmethod
    def create(system_prompt: str, human_prompt: str) -> ChatPromptTemplate:
        """
        Create a standard chat prompt.

        Parameters
        ----------
        system_prompt
            Instructions for the AI assistant.

        human_prompt
            Template containing user variables.
        """

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )