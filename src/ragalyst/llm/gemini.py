"""Gemini LLM integration for RAG evaluation.

This module defines the `GeminiLlm` class, a wrapper around Google's
Gemini models using the `langchain_google_genai` integration.
It extends the `BaseLlm` abstraction and provides a convenient
interface for initializing and using Gemini models within
retrieval-augmented generation (RAG) pipelines.

Environment Variables:
    GOOGLE_API_KEY (str): Required. Must be set in a `.env` file or environment.
"""

import os

import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from ragalyst.llm.base import BaseLlm


class GeminiLlm(BaseLlm):
    """Wrapper for Google's Gemini models integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize a Gemini LLM instance.

        Args:
            cfg: Configuration object containing LLM parameters

        Raises:
            AssertionError: If the `GOOGLE_API_KEY` environment variable
                is not set in the `.env` file or system environment.
        """
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("GOOGLE_API_KEY") is not None, (
            "GOOGLE_API_KEY environment variable is not set. Please set it in your .env file."
        )

        self.api_key = SecretStr(os.environ["GOOGLE_API_KEY"])
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )
