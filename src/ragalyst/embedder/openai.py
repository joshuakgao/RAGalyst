"""OpenAI embedder integration for RAG evaluation."""

import os
from typing import Coroutine, List

import dotenv
from langchain_openai import OpenAIEmbeddings
from lightrag.llm.openai import openai_embed
from omegaconf import DictConfig
from pydantic import SecretStr

from ragalyst.embedder.base import BaseEmbedder


class OpenAiEmbedder(BaseEmbedder):
    """Wrapper for OpenAI embedders integrated into RAG pipelines."""

    def __init__(self, cfg: DictConfig):
        """Initialize a OpenAI embedder instance."""
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("OPENAI_API_KEY") is not None, (
            "OPENAI_API_KEY environment variable is not set. Please set it in your .env file."
        )

        self.api_key = SecretStr(os.environ["OPENAI_API_KEY"])
        self.model = OpenAIEmbeddings(model=self.model_name, api_key=self.api_key)

    async def embedding_func(self, texts: List[str]) -> Coroutine:
        """Asynchronous embedding function for Ragas integration."""
        return openai_embed(
            texts,
            embed_model=self.model_name,
            api_key=self.api_key,
        )
