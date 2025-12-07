"""Groundedness Metric Implementation."""

from ragalyst.metrics.base import BaseMetric


class GroundednessMetric(BaseMetric):
    """Groundedness Metric to evaluate how well a question can be answered given a context."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the groundedness of a question given the context."""
        assert question is not None, "Question cannot be None"
        assert context is not None, "Context cannot be None"

        template = f"""
            You will be given a context and a question.
            Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
            Give your answer on a scale of 0.0 to 1.0, where 0.0 means that the question is not answerable at all given the context, and 1.0 means that the question is clearly and unambiguously answerable with the context.
            High groundedness score (>0.5) indicates that the context should explicitly contain the information needed to answer the question without any further assumptions or external knowledge.
            Avoid using any prior knowledge, common knowledge or external sources to answer the question. The context should be self-contained.

            You MUST provide values for 'groundedness_score:' in your answer.

            Now here are the question and context.

            Question: {question}\n
            Context: {context}\n

            Format the output as a single number such as groundedness_score: 0.5 for example. Do not produce any other output.
            groundedness_score: your rating, as a number between 0.0 and 1.0
        """
        for attempt in range(max_retries + 1):
            try:
                return self.extract_score(template, "groundedness_score:")
            except Exception as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with groundedness metric: {e}"
                )
                continue

        return 0.0
