"""Huggingface LLM integration for RAG evaluation."""

import os

import dotenv
from huggingface_hub import login
from langchain_core.documents import Document
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from ragalyst.llm.base import BaseLlm


class HuggingfaceLlm(BaseLlm):
    """Wrapper for Huggingface models integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize a Huggingface LLM instance."""
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("HF_TOKEN") is not None, (
            "Huggingface token not found. Please set the HF_TOKEN environment variable."
        )
        assert cfg.llm.device in [
            "cuda",
            "cpu",
            "auto",
        ], "Invalid device specified. Please use 'no_load', 'auto', 'cuda', or 'cpu'."

        login(token=os.environ.get("HF_TOKEN"))  # login to huggingface hub

        if cfg.llm.device == "no_load":
            return
        elif cfg.llm.device == "cpu":
            self.model = pipeline(
                "text-generation",
                model=cfg.llm.model_name,
                device=-1,  # use CPU
            )
        elif cfg.llm.device == "cuda":
            self.model = pipeline(
                "text-generation",
                model=cfg.llm.model_name,
                device=0,  # use first GPU
            )
        else:  # cfg.llm.device == "auto"
            # use all GPU's available
            self.model = pipeline(
                "text-generation",
                model=cfg.llm.model_name,
                device_map="auto",
            )

    def chat(self, context: str | list[str] | list[Document], question: str) -> str:
        """Generate a response from the model given context and a question.

        Args:
            context: The context for the model, as a string, list of strings,
                or list of `Document` objects.
            question: The question to ask the model.

        Returns:
            str: The model's generated response.
        """
        assert isinstance(self.model, TextGenerationPipeline), (
            "Model must be an instance of Huggingface pipeline."
        )
        combined_context = self.process_context(context)
        messages = [
            {
                "role": "system",
                "content": self.system_message,
            },
            {
                "role": "user",
                "content": "context:"
                + combined_context
                + " question:"
                + question
                + "/no_think",
            },
        ]
        response = self.model(messages)
        return response[0]["generated_text"][-1]["content"]  # type: ignore
