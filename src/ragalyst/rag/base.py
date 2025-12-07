"""Base class for Retrieval-Augmented Generation (RAG) implementations.

This module defines the abstract base class `BaseRag`, which provides core
infrastructure for RAG pipelines. It handles configuration, initialization
of components (LLM, embedder, text processor), cache management for retrieved
contexts and LLM responses, and utility methods for JSONL persistence.

Subclasses must implement the `search` method to define how contexts are
retrieved for a given query.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from ragalyst.utils.cache.cache_path import make_cache_path


class BaseRag(ABC):
    """Abstract base class for Retrieval-Augmented Generation (RAG)."""

    def __init__(self, cfg):
        """Initialize the BaseRag instance.

        This method sets up the embedder, LLM, and text processor
        from the configuration. It also creates cache paths for
        retrieved contexts and LLM responses, and loads any existing
        cache data into memory.

        Args:
            cfg: A configuration object containing model and pipeline settings.
        """
        from ragalyst.module_registry import (
            get_embedder,
            get_llm,
            get_text_processor,
        )

        self.cfg = cfg
        self.embedder = get_embedder(cfg)
        self.llm = get_llm(cfg)
        self.text_processor = get_text_processor(cfg)

        # Paths
        self.retrieved_contexts_cache_path = (
            make_cache_path(
                purpose="rag_retrieved_contexts",
                relevant_cfg_dict={
                    "embedder": {
                        "type": cfg.embedder.type,
                        "model_name": cfg.embedder.model_name,
                    },
                    "rag": {
                        "type": cfg.rag.type,
                        "top_k": cfg.rag.top_k,
                        "order_preserve": cfg.rag.get("order_preserve", False),
                        "index_type": cfg.rag.get("index_type", None),
                    },
                    "text_processor": {
                        "type": cfg.data.text_processor.type,
                        "chunk_size": cfg.data.text_processor.get("chunk_size", None),
                        "chunk_overlap": cfg.data.text_processor.get(
                            "chunk_overlap", None
                        ),
                        "source_column": cfg.data.text_processor.get("split_by", None),
                    },
                    "domain": cfg.domain,
                },
            )
            / "cache.jsonl"
        )
        self.llm_rag_response_cache_path = (
            make_cache_path(
                purpose="rag_llm_responses",
                relevant_cfg_dict={
                    "llm": {
                        "type": cfg.llm.type,
                        "model_name": cfg.llm.model_name,
                    },
                    "embedder": {
                        "type": cfg.embedder.type,
                        "model_name": cfg.embedder.model_name,
                    },
                    "rag": {
                        "type": cfg.rag.type,
                        "top_k": cfg.rag.top_k,
                        "order_preserve": cfg.rag.get("order_preserve", False),
                        "index_type": cfg.rag.get("index_type", None),
                    },
                    "text_processor": {
                        "type": cfg.data.text_processor.type,
                        "chunk_size": cfg.data.text_processor.get("chunk_size", None),
                        "chunk_overlap": cfg.data.text_processor.get(
                            "chunk_overlap", None
                        ),
                        "source_column": cfg.data.text_processor.get("split_by", None),
                    },
                    "domain": cfg.domain,
                },
            )
            / "cache.jsonl"
        )

        # Load caches into memory
        self.retrieved_contexts = {}
        if self.retrieved_contexts_cache_path.exists():
            print(
                f"Loading retrieved contexts from {self.retrieved_contexts_cache_path}"
            )
            self.retrieved_contexts = self._load_jsonl(
                self.retrieved_contexts_cache_path
            )

        self.llm_rag_responses = {}
        if self.llm_rag_response_cache_path.exists():
            print(f"Loading LLM RAG responses from {self.llm_rag_response_cache_path}")
            self.llm_rag_responses = self._load_jsonl(self.llm_rag_response_cache_path)

    def _load_jsonl(self, path: Path) -> dict:
        """Load a JSONL file into a dictionary.

        Args:
            path (Path): Path to the JSONL file.

        Returns:
            dict: A dictionary where keys are query strings or prompts and
            values are the corresponding cached results.
        """
        data = {}
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = list(obj.keys())[0]
                    data[key] = obj[key]
                except json.JSONDecodeError:
                    continue
        return data

    def _append_jsonl(self, path: Path, key: str, value: str | list[str]):
        """Append a single record to a JSONL file.

        Args:
            path (Path): Path to the JSONL file.
            key (str): The key representing the query or prompt.
            value (str | list[str]): The associated response or contexts.
        """
        with open(path, "a") as f:
            f.write(json.dumps({key: value}) + "\n")

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Search for relevant contexts given a query.

        Args:
            query (str): Query string.

        Returns:
            list[str]: A list of retrieved contexts.
        """
        pass

    def query_with_rag(self, query: str) -> tuple[str, list[str]]:
        """Run a query through RAG and return both response and contexts.

        This method retrieves contexts for the query, caches them if needed,
        constructs a prompt, and queries the LLM. The LLM response is also
        cached for reuse.

        Args:
            query (str): Query string.

        Returns:
            tuple[str, list[str]]: A tuple containing:
                - The LLM's response string.
                - The list of retrieved contexts.
        """
        if query in self.retrieved_contexts:
            retrieved_contexts = self.retrieved_contexts[query]
        else:
            retrieved_contexts = self.search(query)
            self.retrieved_contexts[query] = retrieved_contexts
            with open(self.retrieved_contexts_cache_path, "a") as f:
                f.write(json.dumps({query: retrieved_contexts}) + "\n")

        prompt = "\n".join(retrieved_contexts) + "\n" + query
        if prompt in self.llm_rag_responses:
            response = self.llm_rag_responses[prompt]
        else:
            response = self.llm.chat(retrieved_contexts, query)
            self.llm_rag_responses[prompt] = response
            with open(self.llm_rag_response_cache_path, "a") as f:
                f.write(json.dumps({prompt: response}) + "\n")

        return response, retrieved_contexts
