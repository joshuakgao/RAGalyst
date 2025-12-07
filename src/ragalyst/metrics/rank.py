"""Rank Metric Implementation."""

from ragalyst.metrics.base import BaseMetric


class RankMetric(BaseMetric):
    """Rank Metric to evaluate the rank of a ground truth in the context."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the rank of a ground truth in the context."""
        assert context is not None, "Context cannot be None"
        assert ground_truth is not None, "Ground truth cannot be None"

        k = len(context)
        try:
            rank = context.index(ground_truth) + 1
            return 1.0 / rank if rank <= k else 0.0
        except ValueError:
            return 0.0
