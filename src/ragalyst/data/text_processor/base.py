"""Base class for text processing, including reading and chunking files."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from ragas.testset.graph import KnowledgeGraph
from tqdm import tqdm

from ragalyst.utils.cache.cache_path import make_cache_path


class BaseTextProcessor(ABC):
    """Abstract base class for text processing, including reading and chunking files."""

    def __init__(self, cfg):
        """Initialize a text processor instance."""
        self.cfg = cfg
        self.dir = Path(cfg.domain)
        assert self.dir.is_dir(), f"{self.dir} is not a valid directory"
        self.cache_path = make_cache_path(
            purpose="text_processing",
            relevant_cfg_dict={
                "domain": cfg.domain,
                "text_processor": {
                    "type": cfg.data.text_processor.type,
                    "chunk_size": cfg.data.text_processor.get("chunk_size", None),
                    "chunk_overlap": cfg.data.text_processor.get("chunk_overlap", None),
                    "split_by": cfg.data.text_processor.get("split_by", None),
                },
            },
        )

        # Read and chunk file
        self.documents = self.read_files(self.dir)

        self.kg: None | KnowledgeGraph = None  # Defined in kg subclass
        self.chunks: None | list[Document] = None  # Defined in chunk subclass

    def read_files(self, dir: Path) -> list[Document]:
        """Read files from the specified directory and return a list of documents."""
        # Ensure cache directory exists
        cache_file = self.cache_path / "documents.pkl"

        # If cache exists, load it
        if cache_file.exists():
            print("Loading documents from cache...")
            with open(cache_file, "rb") as f:
                documents = pickle.load(f)

            return documents

        # Otherwise, process files normally
        docs = list(dir.glob("*"))
        documents = []
        for file_path in tqdm(
            docs, total=len(docs), desc="Processing source documents"
        ):
            if not file_path.is_file():
                continue

            if file_path.suffix not in [".txt", ".md", ".pdf"]:
                continue

            # Load text files
            if file_path.suffix[1:].lower() in ["txt", "md"]:
                loader = TextLoader(file_path=file_path)
                documents.extend(loader.load())
            # Load PDF files
            elif file_path.suffix[1:].lower() in ["pdf"]:
                loader = PyPDFLoader(file_path=file_path)
                documents.extend(loader.load())

        # Save to cache for next time
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)

        return documents

    @abstractmethod
    def process(self) -> list[Document] | KnowledgeGraph:
        """Process the raw text and return a list of documents.

        This method should be implemented by subclasses to handle specific processing logic.
        """
        pass
