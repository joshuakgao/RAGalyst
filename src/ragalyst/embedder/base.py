"""Base class for embedders in RAG evaluation.

This module defines the abstract `BaseEmbedder` class, which provides
a common interface for embedding text data in retrieval-augmented
generation (RAG) workflows. Subclasses should implement specific
embedding models and asynchronous embedding functions. The class
also includes utilities for batch querying, dimensionality inference,
and Ragas wrapper integration.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Coroutine, List

import numpy as np
from langchain_core.embeddings import Embeddings
from omegaconf import DictConfig
from ragas.embeddings import LangchainEmbeddingsWrapper


class BaseEmbedder(ABC):
    """Abstract base class for text embedders in RAG pipelines.

    Provides common functionality for:
        - Querying single or batch embeddings
        - Getting embedding dimensionality
        - Wrapping the embedder for Ragas evaluation
        - Deep copying while excluding stateful attributes
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the embedder with configuration.

        Args:
            cfg (DictConfig): Configuration object containing embedder parameters,
                such as `model_name`.
        """
        self.cfg = cfg
        self.dim: int | None = None
        self.model_name: str = cfg.embedder.model_name
        self.model: Any = None

    def get_dim(self) -> int:
        """Get the dimensionality of the embeddings.

        Returns:
            int: Embedding vector dimension.

        Note:
            If `self.dim` is not already set, this method performs a single
            query with `"hello world"` to infer the dimension.
        """
        if self.dim is None:
            self.dim = len(self.query("hello world"))
        return self.dim

    def query(self, text: str) -> list[float]:
        """Query the embedder for a single text input.

        Args:
            text (str): Input text to embed.

        Returns:
            list[float]: Embedding vector.

        Raises:
            AssertionError: If `self.model` is not an instance of `Embeddings`.
        """
        assert isinstance(self.model, Embeddings), "Model must be an instance of Embeddings."
        response = self.model.embed_query(text)
        return response

    def batch_query(self, text: List[str]) -> list[list[float]]:
        """Query the embedder for a batch of text inputs.

        Args:
            text (List[str]): List of input strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors for each input.

        Raises:
            AssertionError: If `self.model` is not an instance of `Embeddings`.
        """
        assert isinstance(self.model, Embeddings), "Model must be an instance of Embeddings."
        response = self.model.embed_documents(text)
        return response

    @staticmethod
    def embedding(texts: List[str], model: Embeddings) -> np.ndarray:
        """Compute embeddings for a list of texts using the given model.

        Args:
            texts (List[str]): List of input strings.
            model (Embeddings): LangChain `Embeddings` instance to use.

        Returns:
            np.ndarray: Array of embedding vectors.
        """
        result: list = []
        for text in texts:
            response = model.embed_query(text)
            result.append(response)
        return np.array(result)

    def get_ragas_wrapper(self) -> LangchainEmbeddingsWrapper:
        """Get a Ragas-compatible wrapper around the embedder.

        Returns:
            LangchainEmbeddingsWrapper: Wrapper for Ragas evaluation.

        Raises:
            AssertionError: If `self.model` is not an instance of `Embeddings`.
        """
        assert isinstance(self.model, Embeddings), "Model must be an instance of Embeddings."
        return LangchainEmbeddingsWrapper(self.model)

    @abstractmethod
    async def embedding_func(self, texts: List[str]) -> Coroutine:
        """Asynchronous method to embed a list of texts.

        Subclasses must implement this method.

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            Coroutine: Awaitable object returning embeddings.
        """
        pass

    def __deepcopy__(self, memo):
        """Custom deep copy method excluding stateful or unpickleable attributes.

        This ensures that `model` (which may be stateful or unpickleable)
        is not copied, while other attributes are deep-copied.

        Args:
            memo (dict): Dictionary used by the `deepcopy` function to avoid
                duplicate copies.

        Returns:
            BaseEmbedder: Deep-copied instance of the embedder.
        """
        cls = self.__class__
        result = cls.__new__(cls)  # Create a new instance without calling __init__
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if key == "model":  # Exclude the model from being copied
                setattr(result, key, None)
            else:
                setattr(result, key, deepcopy(value, memo))  # Deep copy other attributes
        return result
