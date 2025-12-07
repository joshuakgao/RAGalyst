"""LightRag implementation of the BaseRag interface.

This module provides a concrete subclass of `BaseRag` that integrates with
the LightRAG library. It supports chunk-based text processing, embedding
with either Ollama or OpenAI models, and querying contexts using hybrid
retrieval.
"""

import asyncio
from typing import Iterator

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.llm.openai import gpt_4o_complete, gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from ragalyst.rag.base import BaseRag


class LightRag(BaseRag):
    """LightRag-based retrieval-augmented generation pipeline."""

    def __init__(self, cfg):
        """Initialize a LightRag instance.

        This constructor sets up the text processor, selects the appropriate
        LLM and embedder functions, initializes the LightRAG backend, and
        inserts preprocessed text chunks into the retrieval index.

        Args:
            cfg: A configuration object containing model and pipeline settings.

        Raises:
            AssertionError: If the text processor type is not "chunk" or if
                no chunks are found in the input data.
            ValueError: If the specified LLM or embedder type is unsupported.
        """
        assert cfg.data.text_processor.type == "chunk", (
            "LightRag only supports chunk text processing."
        )

        super().__init__(cfg)

        self.text_processor.process()
        assert self.text_processor.chunks, "No chunks found in the raw data."
        self.model_func = self.select_llm_func(cfg)
        self.embedding_func = self.select_embedder_func(cfg)
        self.rag = LightRAG(
            working_dir=".cache/lightrag_cache",
            llm_model_name=cfg.llm.model_name,
            llm_model_func=self.model_func,
            embedding_func=self.embedding_func,
        )

        asyncio.run(self.init_rag())

        # Convert chunk Documents to strings and insert them into the RAG instance
        self.rag.insert([doc.page_content for doc in self.text_processor.chunks])

    async def init_rag(self):
        """Initialize LightRAG storages and pipeline status asynchronously.

        This sets up storage backends and pipeline status tracking required
        by the LightRAG engine.
        """
        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    def select_llm_func(self, cfg):
        """Select the appropriate LLM function based on the configuration.

        Args:
            cfg: Configuration object specifying the LLM type and model name.

        Returns:
            Callable: A function that performs LLM completion.

        Raises:
            ValueError: If the LLM type or model name is unsupported.
        """
        if cfg.llm.type == "ollama":
            return ollama_model_complete
        elif cfg.llm.type == "openai":
            if cfg.llm.model_name == "gpt-4o-mini":
                return gpt_4o_mini_complete
            elif cfg.llm.model_name == "gpt-4o":
                return gpt_4o_complete
        else:
            raise ValueError(f"Unsupported LLM type: {cfg.llm.type}")

    def select_embedder_func(self, cfg):
        """Select the appropriate embedder function based on the configuration.

        Args:
            cfg: Configuration object specifying the embedder type and model.

        Returns:
            EmbeddingFunc: Wrapper around the embedding function, including
            metadata such as embedding dimension and max token size.

        Raises:
            ValueError: If the embedder type is unsupported.
        """
        func = None
        if cfg.embedder.type == "ollama":

            def ollama_embedding_function(texts):
                return ollama_embed(texts, embed_model=cfg.embedder.model_name)

            func = ollama_embedding_function
            embedding_dim = self.embedder.get_dim()
            max_token_size = 8192
        elif cfg.embedder.type == "openai":
            func = openai_embed
            embedding_dim = 1536
            max_token_size = 8192
        else:
            raise ValueError(f"Unsupported embedder type: {cfg.embedder.type}")

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=func,
        )

    def search(self, query) -> list[str]:
        """Retrieve contexts relevant to a query using LightRAG.

        Args:
            query (str): The input query string.

        Returns:
            list[str]: A list of retrieved context strings.
        """
        retrieved_str = self.rag.query(
            query,
            param=QueryParam(
                mode="hybrid", only_need_context=True, top_k=self.cfg.rag.top_k
            ),
        )

        # type conversion
        if isinstance(retrieved_str, Iterator):
            retrieved_str = list(retrieved_str)
        elif isinstance(retrieved_str, str):
            retrieved_str = [retrieved_str]

        return retrieved_str
