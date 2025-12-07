"""Relevance Metric Implementation."""

from ragalyst.metrics.base import BaseMetric


class RelevanceMetric(BaseMetric):
    """Relevance Metric to evaluate the relevance of a question given a context."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the relevance of a question given the context."""
        assert question is not None, "Question cannot be None"
        assert context is not None, "Context cannot be None"

        template = f"""
            You will be given a user context and a question.
            Your task is to provide a 'total rating' representing how useful this question can be to the specified context.
            Give your answer on a scale of 0.0 to 1.0, where 0.0 means that the question is not useful at all, and 1.0 means that the question is extremely useful.


            You MUST provide values for 'relevance_score:' in your answer.

            Now here are the question and context.

            Question: {question}\n
            context: {context}\n

            Format the output as a single number such as relevance_score: 0.5 for example. Do not produce any other output.
            relevance_score: your rating, as a number between 0.0 and 1.0
        """
        for attempt in range(max_retries + 1):
            try:
                return self.extract_score(template, "relevance_score:")
            except Exception as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with relevance metric: {e}"
                )
                continue

        return 0.0
