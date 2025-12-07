"""Hit Rate Metric Implementation."""

from ragalyst.metrics.base import BaseMetric


class HitRateMetric(BaseMetric):
    """Hit Rate Metric to evaluate the relevance of a response given a context."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the hit rate of a ground truth in the context."""
        assert context is not None, "Context cannot be None"
        assert ground_truth is not None, "Ground truth cannot be None"

        k = len(context)
        return 1.0 if ground_truth in context[:k] else 0.0
