"""Base class for dataset managers in RAG evaluation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseDatasetManager(ABC):
    """Abstract base class for dataset managers in RAG evaluation."""

    def __init__(self, cfg):
        """Initialize a dataset manager instance."""
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_metrics, get_text_processor

        self.cfg = cfg
        self.text_processor = get_text_processor(cfg)
        self.metrics = get_metrics(cfg)
        self.num_samples: int = cfg.data.dataset_manager.num_samples
        self.output_dir: Path = Path()

    @abstractmethod
    def generate(self) -> List[dict]:
        """This is the main method to generate a dataset based on the provided configuration.

        This method will use generate_questions and generate_answers methods to create the dataset.
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        """Evaluate the generated dataset using the metrics defined in the configuration.

        This method will use the metrics defined in the configuration to evaluate the dataset.
        """
        pass
