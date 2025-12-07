"""Chunk text processor for RAG."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragalyst.data.text_processor.base import BaseTextProcessor


class ChunkTextProcessor(BaseTextProcessor):
    """Chunk text processor for RAG."""

    def __init__(self, cfg):
        """Initialize a chunk text processor instance."""
        super().__init__(cfg)

        self.chunk_size = cfg.data.text_processor.chunk_size
        self.chunk_overlap = cfg.data.text_processor.chunk_overlap
        self.split_by = cfg.data.text_processor.split_by
        assert self.split_by in [
            "token",
            "character",
        ], "split_by must be 'token' or 'character'"

    def process(self) -> list[Document]:
        """Chunk text for rag.

        Note that resulting chunks will have variable chunk sizes depending on where document separators are.
        So you can get very short chunks and chunks that are longer than your chunk size.
        """
        if self.split_by == "token":
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        self.chunks = text_splitter.split_documents(self.documents)
        return self.chunks
