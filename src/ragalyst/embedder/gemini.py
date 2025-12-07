"""Gemini embedder integration for RAG evaluation."""

import os
from typing import Coroutine, List

import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from omegaconf import DictConfig
from pydantic import SecretStr

from ragalyst.embedder.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Wrapper for Gemini embedders integrated into RAG pipelines."""

    def __init__(self, cfg: DictConfig):
        """Initialize a Gemini embedder instance."""
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("GOOGLE_API_KEY") is not None, (
            "GOOGLE_API_KEY environment variable is not set. Please set it in your .env file."
        )

        self.api_key = SecretStr(os.environ["GOOGLE_API_KEY"])
        self.model = GoogleGenerativeAIEmbeddings(
            model=self.model_name, task_type=cfg.embedder.task_type
        )

    async def embedding_func(self, texts: List[str]) -> Coroutine:
        """Asynchronous embedding function for Ragas integration."""
        # TODO: it is possible to use the OpenAI API to call Gemini models
        # return openai_embed(
        #     texts,
        #     embed_model=self.model_name,
        #     api_key=self.api_key.get_secret_value(),
        # )
        assert False, "Gemini embedding function is not implemented yet."
