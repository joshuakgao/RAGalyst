"""Cosine similarity metric implementation."""

import numpy as np
from numpy.linalg import norm

from ragalyst.metrics.base import BaseMetric


class CosineSimilarityMetric(BaseMetric):
    """Implementation of Cosine Similarity metric using embeddings.

    Computes the cosine similarity between two text embeddings,
    using Qwen-embedding-8b as the default embedding model.
    """

    def __init__(self, cfg):
        """Initialize the CosineSimilarity metric.

        Args:
            cfg (DictConfig): Configuration settings.
        """
        super().__init__(cfg)

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=10,
    ) -> float:
        """Evaluate cosine similarity between two texts using embeddings.

        This method computes the cosine similarity between embeddings of two texts.
        Common use cases:
        - Compare answer vs ground_truth (answer similarity)
        - Compare response vs ground_truth (response quality)
        - Compare answer vs response (consistency check)
        - Compare question vs context (relevance check)

        Args:
            question (str, optional): Question text.
            answer (str, optional): Answer text.
            response (str, optional): Response text.
            ground_truth (str, optional): Ground truth text.
            context (str, optional): Context text.
            max_retries (int): Maximum number of retries (unused for this metric).

        Returns:
            float: Cosine similarity score between 0 and 1, where:
                   - 1.0 indicates identical semantic meaning
                   - 0.0 indicates no similarity
                   - Values closer to 1 indicate higher similarity

        Raises:
            ValueError: If insufficient texts are provided for comparison.
        """
        # Determine which texts to compare based on what's provided
        text1, text2 = self._select_texts(
            question, answer, response, ground_truth, context
        )

        if text1 is None or text2 is None:
            raise ValueError(
                "At least two texts must be provided for cosine similarity calculation. "
                "Common combinations: (answer, ground_truth), (response, ground_truth), "
                "(answer, response), (question, context)"
            )

        # Get embeddings for both texts
        embedding1 = np.array(self.embedder.query(text1))
        embedding2 = np.array(self.embedder.query(text2))

        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)

        return float(similarity)

    def _select_texts(self, question, answer, response, ground_truth, context):
        """Select which texts to compare based on what's provided.

        Priority order for comparison:
        1. answer vs ground_truth (most common for evaluation)
        2. response vs ground_truth
        3. answer vs response
        4. question vs context

        Args:
            question (str, optional): Question text.
            answer (str, optional): Answer text.
            response (str, optional): Response text.
            ground_truth (str, optional): Ground truth text.
            context (str, optional): Context text.

        Returns:
            tuple: (text1, text2) to compare.
        """
        # Priority 1: answer vs ground_truth
        if answer is not None and ground_truth is not None:
            return answer, ground_truth

        # Priority 2: response vs ground_truth
        if response is not None and ground_truth is not None:
            return response, ground_truth

        # Priority 3: answer vs response
        if answer is not None and response is not None:
            return answer, response

        # Priority 4: question vs context
        if question is not None and context is not None:
            return question, context

        # Fallback: any two available texts
        texts = [
            t
            for t in [question, answer, response, ground_truth, context]
            if t is not None
        ]
        if len(texts) >= 2:
            return texts[0], texts[1]

        return None, None

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First embedding vector.
            vec2 (np.ndarray): Second embedding vector.

        Returns:
            float: Cosine similarity score between -1 and 1.
                   Clipped to [0, 1] for consistency with other metrics.
        """
        # Handle zero vectors
        norm1 = norm(vec1)
        norm2 = norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Clip to [0, 1] range for consistency with other metrics
        # (cosine similarity can be negative for opposite directions)
        similarity = max(0.0, min(1.0, similarity))

        return similarity
