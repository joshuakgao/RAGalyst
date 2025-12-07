"""Base LLM abstraction for RAG evaluation.

This module defines the `BaseLlm` abstract class, which provides a common
interface and shared utilities for integrating large language models (LLMs)
into retrieval-augmented generation (RAG) workflows. It handles context
processing, chat message construction, synchronous and asynchronous querying,
and exposes a Ragas-compatible wrapper for evaluation.
"""

from abc import ABC

from langchain.base_language import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from ragas.llms import LangchainLLMWrapper
from transformers.pipelines.text_generation import TextGenerationPipeline


class BaseLlm(ABC):
    """Abstract base class for Language Models in the RAG evaluation framework.

    This class provides shared functionality for handling context, building
    chat messages, and invoking LLMs synchronously or asynchronously.
    Subclasses should define the specific model instance.
    """

    def __init__(self, cfg):
        """Initialize the base LLM wrapper.

        Args:
            cfg: Configuration object containing LLM parameters such as
                model name, system message, and temperature.
        """
        self.model_name = cfg.llm.model_name
        self.system_message = cfg.llm.system_message
        self.temperature = cfg.llm.temperature
        self.model: BaseLanguageModel | TextGenerationPipeline | None = (
            None  # defined in subclasses
        )

    def process_context(self, context: str | list[str] | list[Document]) -> str:
        """Process and normalize the input context into a single string.

        Args:
            context: The input context, which can be a string, list of strings,
                or list of LangChain `Document` objects.

        Returns:
            str: Concatenated string representation of the context.
        """
        if isinstance(context, str):
            return context

        combined_context = ""
        if isinstance(context, list):
            for i in context:
                if isinstance(i, Document):
                    combined_context += i.page_content + " "
                elif isinstance(i, str):
                    combined_context += i + " "
            return combined_context.strip()

    def build_chat(
        self, context: str | list[str] | list[Document], question: str
    ) -> list[tuple[str, str]]:
        """Construct a chat message history for the model.

        Args:
            context: The context for the model, as a string, list of strings,
                or list of `Document` objects.
            question: The user question to ask the model.

        Returns:
            list[tuple[str, str]]: A list of (role, message) tuples for the chat.
        """
        combined_context = self.process_context(context)
        messages = [
            ("system", self.system_message),
            ("system", "context:" + combined_context),
            ("user", question),
        ]
        return messages

    def chat(self, context: str | list[str] | list[Document], question: str) -> str:
        """Synchronously query the model with context and a question.

        Retries up to 5 times in case of transient errors.

        Args:
            context: The context for the model, as a string, list of strings,
                or list of `Document` objects.
            question: The user question to ask the model.

        Returns:
            str: The model's response.

        Raises:
            RuntimeError: If the model fails to return a valid response
                after 5 retries.
        """
        assert isinstance(self.model, BaseLanguageModel), (
            "Model must be an instance of langchain BaseLanguageModel."
        )
        messages = self.build_chat(context, question)
        last_exception = None
        for _ in range(5):
            try:
                response: BaseMessage = self.model.invoke(messages)
                assert isinstance(response.content, str), "Response content should be a string"
                return response.content
            except Exception as e:
                last_exception = e
        raise RuntimeError("Failed to get response from model after 5 retries") from last_exception

    async def chat_async(self, context: str | list[str] | list[Document], question: str) -> str:
        """Asynchronously query the model with context and a question.

        Args:
            context: The context for the model, as a string, list of strings,
                or list of `Document` objects.
            question: The user question to ask the model.

        Returns:
            str: The model's response.
        """
        assert isinstance(self.model, BaseLanguageModel), (
            "Model must be an instance of langchain BaseLanguageModel."
        )
        messages = self.build_chat(context, question)
        response: BaseMessage = await self.model.ainvoke(messages)
        assert isinstance(response.content, str), "Response content should be a string"
        return response.content

    def get_ragas_wrapper(self):
        """Get a Ragas-compatible LLM wrapper for evaluation.

        Returns:
            LangchainLLMWrapper: A wrapper around the LangChain LLM instance.

        Raises:
            AssertionError: If the model is not a `BaseLanguageModel`.
        """
        assert isinstance(self.model, BaseLanguageModel), (
            "Model must be an instance of langchain BaseLanguageModel."
        )
        return LangchainLLMWrapper(self.model)
