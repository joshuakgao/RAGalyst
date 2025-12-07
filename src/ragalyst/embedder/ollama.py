"""Ollama embedder integration for RAG evaluation."""

from typing import Coroutine

from langchain_ollama import OllamaEmbeddings
from lightrag.llm.ollama import ollama_embed

from ragalyst.embedder.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Wrapper for Ollama embedders integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize a Ollama embedder instance."""
        super().__init__(cfg)

        # if cfg.embedder.device == "no_load":
        #     return

        self.model = OllamaEmbeddings(model=self.model_name)

    async def embedding_func(self, texts) -> Coroutine:
        """Asynchronous embedding function for Ragas integration."""
        return ollama_embed(
            texts,
            embed_model=self.model_name,
        )
