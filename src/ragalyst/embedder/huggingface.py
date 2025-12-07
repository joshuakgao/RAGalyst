"""Huggingface embedder integration for RAG evaluation."""

import os
from typing import Coroutine

import dotenv
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from lightrag.llm.hf import hf_embed
from transformers import AutoModel, AutoTokenizer

from ragalyst.embedder.base import BaseEmbedder


class HuggingfaceEmbedder(BaseEmbedder):
    """Wrapper for Huggingface embedders integrated into RAG pipelines."""

    def __init__(self, cfg):
        """Initialize a Huggingface embedder instance."""
        super().__init__(cfg)

        dotenv.load_dotenv()
        assert os.environ.get("HF_TOKEN") is not None, (
            "Huggingface token not found. Please set the HF_TOKEN environment variable."
        )
        assert cfg.embedder.device in [
            "no_load",
            "auto",
            "cuda",
            "cpu",
        ], "Invalid device specified. Please use 'no_load', 'auto', 'cuda', or 'cpu'."

        device = cfg.embedder.device
        if device == "no_load":
            return

        if device == "auto":
            # check to see if cuda is available
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model = HuggingFaceEmbeddings(
            model_name=cfg.embedder.model_name,
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"normalize_embeddings": False},
        )

    async def embedding_func(self, texts) -> Coroutine:
        """Asynchronous embedding function for Ragas integration."""
        return hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(self.cfg.embedder.model_name),
            embed_model=AutoModel.from_pretrained(self.cfg.embedder.model_name),
        )
