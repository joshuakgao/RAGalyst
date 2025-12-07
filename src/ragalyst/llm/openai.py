"""OpenAI LLM integration for RAG evaluation."""

import os

import dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ragalyst.llm.base import BaseLlm


class OpenAiLlm(BaseLlm):
    """Wrapper for OpenAI models integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize an OpenAI LLM instance."""
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("OPENAI_API_KEY") is not None, (
            "OPENAI_API_KEY environment variable is not set. Please set it in your .env file."
        )

        self.api_key = SecretStr(os.environ["OPENAI_API_KEY"])
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )
