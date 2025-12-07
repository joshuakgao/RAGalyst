"""Ollama LLM integration for RAG evaluation."""

import ollama
from langchain_ollama import ChatOllama

from ragalyst.llm.base import BaseLlm


class OllamaLlm(BaseLlm):
    """Wrapper for Ollama models integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize an Ollama LLM instance."""
        super().__init__(cfg)

        # check if model is available
        assert self._is_model_pulled(self.model_name), (
            f"Model {self.model_name} is not pulled. Please pull the model using `ollama pull {self.model_name}`."
        )

        self.model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
        )

    def _is_model_pulled(self, model_name: str) -> bool:
        """Check if the specified Ollama model is pulled locally."""
        pulled_models = ollama.list()
        for model in pulled_models["models"]:
            if model["model"] == model_name:
                return True
        return False
